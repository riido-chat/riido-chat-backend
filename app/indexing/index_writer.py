"""검색 버전을 만들고 임베딩을 적재해 ACTIVE로 반영한다."""

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import DefaultDict, List, Optional, Sequence

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.error_message import sanitize_error_message
from app.core.hashing import sha256_hex
from app.database.models import (
    ChunkEmbedding,
    ContentNode,
    DocumentChunk,
    ExecutionStatus,
    IndexDocument,
    IndexOperationType,
    IndexRun,
    IndexRunStage,
    IndexVersion,
    IndexVersionStatus,
)
from app.document.chunking_config import get_or_create_chunking_config
from app.document.document_group import get_document_group
from app.document.document_store import DEFAULT_TRIGGER_TYPE, DocumentStore
from app.document.models import NormalizedDocument
from app.retrieval.embedding import build_embedding_text
from app.retrieval.embedding_config import (
    get_or_create_embedding_config,
    validate_embedding_dimension,
)
from app.retrieval.models import RetrievalChunk, StoredEmbedding


CURRENT_CANDIDATE_K = 10
CURRENT_RRF_RANK_CONSTANT = 60
CURRENT_FINAL_TOP_K = 5


class IndexWriter:
    """AsyncSession으로 검색 버전의 생성과 반영을 기록한다."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        # 전체 재색인은 문서 적재와 색인 적재를 이어서 수행한다.
        self._documents = DocumentStore(session)

    async def replace_all(
        self,
        items: Sequence[StoredEmbedding],
        documents: Sequence[NormalizedDocument],
    ) -> IndexVersion:
        """호출자 transaction 안에서 전체 적재 단계를 한 번에 실행한다.

        운영 CLI는 실패 이력을 보존하기 위해 단계별 메서드와 checkpoint commit을
        사용한다. 이 메서드는 transaction 단위 통합 테스트와 내부 조합용이다.
        """

        self._validate_new_index_items(items)
        self._validate_reindex_documents(items, documents)
        if self._session.in_transaction():
            return await self._replace_rows(items, documents)

        async with self._session.begin():
            return await self._replace_rows(items, documents)

    async def start_index(
        self,
        chunks: Sequence[RetrievalChunk],
        *,
        trigger_type: str = DEFAULT_TRIGGER_TYPE,
        actor_id: Optional[str] = None,
    ) -> IndexRun:
        """전체 corpus 색인을 BUILDING/PROCESSING 상태로 시작한다."""

        self._validate_persisted_chunks(chunks)
        if not trigger_type:
            raise ValueError("trigger_type은 비어 있을 수 없습니다.")

        now = datetime.now(timezone.utc)
        group = await get_document_group(self._session)
        chunking_config = await get_or_create_chunking_config(self._session, now)
        embedding_config = await get_or_create_embedding_config(self._session, now)
        index_version = IndexVersion(
            document_group_id=group.id,
            version=self._new_index_version_name(now),
            status=IndexVersionStatus.BUILDING,
            chunking_config_id=chunking_config.id,
            embedding_config_id=embedding_config.id,
            keyword_config={
                "analyzer": "KIWI",
                "algorithm": "BM25",
                "candidate_k": CURRENT_CANDIDATE_K,
                "k1": 1.5,
                "b": 0.75,
            },
            fusion_config={
                "algorithm": "RRF",
                "bm25_candidate_k": CURRENT_CANDIDATE_K,
                "vector_candidate_k": CURRENT_CANDIDATE_K,
                "rank_constant": CURRENT_RRF_RANK_CONSTANT,
                "final_top_k": CURRENT_FINAL_TOP_K,
            },
            created_at=now,
        )
        self._session.add(index_version)
        await self._session.flush()

        run = IndexRun(
            index_version_id=index_version.id,
            trigger_type=trigger_type,
            operation_type=IndexOperationType.BUILD_AND_APPLY,
            stage=IndexRunStage.BUILDING,
            actor_id=actor_id,
            status=ExecutionStatus.PROCESSING,
            summary={
                "stage": "BUILDING",
                "document_count": len(
                    {chunk.document_version_id for chunk in chunks}
                ),
                "chunk_count": len(chunks),
            },
            started_at=now,
        )
        self._session.add(run)
        await self._session.flush()
        return run

    async def list_chunks_missing_embedding(
        self,
        index_run_id: int,
        chunks: Sequence[RetrievalChunk],
    ) -> List[RetrievalChunk]:
        """색인 범위 Chunk 중 이 버전의 설정으로 만든 embedding이 없는 것만 고른다.

        embedding은 접수 시점에 만들어지므로 보통 비어 있다. 접수 이전에 적재된
        판이나 embedding 설정이 바뀐 경우에만 채울 것이 남는다.
        """

        self._validate_persisted_chunks(chunks)
        run = await self._get_processing_index_run(index_run_id)
        index_version = await self._session.get(IndexVersion, run.index_version_id)
        if index_version is None:
            raise ValueError(f"존재하지 않는 색인 버전입니다: {run.index_version_id}")

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        embedded_chunk_ids = set(
            (
                await self._session.execute(
                    select(ChunkEmbedding.chunk_id).where(
                        ChunkEmbedding.chunk_id.in_(chunk_ids),
                        ChunkEmbedding.embedding_config_id
                        == index_version.embedding_config_id,
                    )
                )
            )
            .scalars()
            .all()
        )
        return [
            chunk for chunk in chunks if chunk.chunk_id not in embedded_chunk_ids
        ]

    async def store_index_items(
        self,
        index_run_id: int,
        chunks: Sequence[RetrievalChunk],
        items: Sequence[StoredEmbedding] = (),
    ) -> IndexVersion:
        """색인 범위 문서를 연결하고 누락 embedding만 채운 뒤 VALIDATING으로 전이한다."""

        self._validate_persisted_chunks(chunks)
        if items:
            self._validate_index_items(items)
        run = await self._get_processing_index_run(index_run_id)
        index_version = await self._session.get(IndexVersion, run.index_version_id)
        if index_version is None:
            raise ValueError(f"존재하지 않는 색인 버전입니다: {run.index_version_id}")
        if index_version.status != IndexVersionStatus.BUILDING:
            raise ValueError(
                "BUILDING 색인 버전만 저장할 수 있습니다: "
                f"{index_version.status}"
            )

        now = datetime.now(timezone.utc)
        embeddings = [
            ChunkEmbedding(
                chunk_id=chunk.chunk_id,
                embedding_config_id=index_version.embedding_config_id,
                embedding=list(embedding),
                embedding_input_hash=sha256_hex(build_embedding_text(chunk)),
                created_at=now,
            )
            for chunk, embedding in items
        ]
        document_version_ids = {chunk.document_version_id for chunk in chunks}

        expected_documents = (run.summary or {}).get("document_count")
        expected_chunks = (run.summary or {}).get("chunk_count")
        if (
            len(document_version_ids) != expected_documents
            or len(chunks) != expected_chunks
        ):
            raise RuntimeError(
                "색인 입력 건수가 시작 시점과 일치하지 않습니다: "
                f"문서 {len(document_version_ids)}/{expected_documents}, "
                f"Chunk {len(chunks)}/{expected_chunks}"
            )

        indexed_chunk_ids = {chunk.chunk_id for chunk in chunks}
        outside = [
            chunk for chunk, _ in items if chunk.chunk_id not in indexed_chunk_ids
        ]
        if outside:
            raise ValueError("색인 범위 밖 Chunk의 embedding은 저장할 수 없습니다.")

        self._session.add_all(embeddings)
        self._session.add_all(
            IndexDocument(
                index_version_id=index_version.id,
                document_version_id=document_version_id,
            )
            for document_version_id in document_version_ids
        )
        index_version.status = IndexVersionStatus.VALIDATING
        run.stage = IndexRunStage.VALIDATING
        run.summary = {
            "stage": "VALIDATING",
            "document_count": len(document_version_ids),
            "chunk_count": len(chunks),
            "embedding_count": len(embeddings),
        }
        await self._session.flush()
        return index_version

    async def mark_index_ready(self, index_run_id: int) -> IndexVersion:
        """저장 건수를 검증하고 적용 후보를 READY로 전이한다.

        version_no는 이 시점에 그룹 안에서 부여한다. 빌드에 실패한 후보에는
        번호가 남지 않아야 하기 때문이다.
        """

        run = await self._get_processing_index_run(index_run_id)
        index_version = await self._session.get(
            IndexVersion,
            run.index_version_id,
            with_for_update=True,
        )
        if index_version is None:
            raise ValueError(f"존재하지 않는 색인 버전입니다: {run.index_version_id}")
        if index_version.status != IndexVersionStatus.VALIDATING:
            raise ValueError(
                "VALIDATING 색인 버전만 READY로 전이할 수 있습니다: "
                f"{index_version.status}"
            )

        await self._validate_stored_index(index_version, run.summary or {})
        current_version_no = await self._session.scalar(
            select(func.max(IndexVersion.version_no)).where(
                IndexVersion.document_group_id == index_version.document_group_id
            )
        )
        index_version.version_no = (current_version_no or 0) + 1
        index_version.status = IndexVersionStatus.READY
        run.summary = {**(run.summary or {}), "stage": "READY"}
        await self._session.flush()
        return index_version

    async def start_apply_run(
        self,
        index_version_id: int,
        *,
        trigger_type: str = DEFAULT_TRIGGER_TYPE,
        actor_id: Optional[str] = None,
    ) -> IndexRun:
        """READY 후보에 적용 전용 실행을 새로 만든다.

        적용에 실패해도 후보는 READY로 남으므로 같은 검색 버전에 실행만
        다시 붙여 재시도한다.
        """

        if not trigger_type:
            raise ValueError("trigger_type은 비어 있을 수 없습니다.")

        index_version = await self._session.get(
            IndexVersion,
            index_version_id,
            with_for_update=True,
        )
        if index_version is None:
            raise ValueError(f"존재하지 않는 색인 버전입니다: {index_version_id}")
        if index_version.status != IndexVersionStatus.READY:
            raise ValueError(
                "READY 색인 버전만 적용할 수 있습니다: "
                f"{index_version.status}"
            )

        now = datetime.now(timezone.utc)
        run = IndexRun(
            index_version_id=index_version.id,
            trigger_type=trigger_type,
            operation_type=IndexOperationType.APPLY,
            stage=IndexRunStage.APPLYING,
            actor_id=actor_id,
            status=ExecutionStatus.PROCESSING,
            summary={"stage": "APPLYING"},
            started_at=now,
        )
        self._session.add(run)
        await self._session.flush()
        return run

    async def apply_index(self, index_run_id: int) -> IndexVersion:
        """READY 후보를 ACTIVE로 원자 전환한다.

        같은 문서 그룹의 기존 ACTIVE만 INACTIVE로 내린다.
        """

        run = await self._get_processing_index_run(index_run_id)
        index_version = await self._session.get(
            IndexVersion,
            run.index_version_id,
            with_for_update=True,
        )
        if index_version is None:
            raise ValueError(f"존재하지 않는 색인 버전입니다: {run.index_version_id}")
        if index_version.status != IndexVersionStatus.READY:
            raise ValueError(
                "READY 색인 버전만 활성화할 수 있습니다: "
                f"{index_version.status}"
            )

        now = datetime.now(timezone.utc)
        run.stage = IndexRunStage.APPLYING
        await self._session.execute(
            update(IndexVersion)
            .where(
                IndexVersion.status == IndexVersionStatus.ACTIVE,
                IndexVersion.document_group_id == index_version.document_group_id,
                IndexVersion.id != index_version.id,
            )
            .values(status=IndexVersionStatus.INACTIVE)
        )
        index_version.status = IndexVersionStatus.ACTIVE
        index_version.activated_at = now
        await self._session.flush()
        return index_version

    async def finish_apply_run(self, index_run_id: int) -> IndexRun:
        """적용과 corpus 재적재까지 끝난 실행을 SUCCESS로 마감한다."""

        run = await self._get_processing_index_run(index_run_id)
        now = datetime.now(timezone.utc)
        run.status = ExecutionStatus.SUCCESS
        run.summary = {**(run.summary or {}), "stage": "ACTIVE"}
        run.finished_at = now
        await self._session.flush()
        return run

    async def fail_index(
        self,
        index_run_id: int,
        error: Exception,
        *,
        failed_stage: str,
        error_code: Optional[str] = None,
    ) -> IndexRun:
        """PROCESSING 색인 실행을 FAILED로 마감한다.

        READY 후보는 그대로 남긴다. 적용에 실패해도 다시 시도할 수 있어야
        하므로 실행만 마감하고 검색 버전은 건드리지 않는다.
        """

        run = await self._get_processing_index_run(index_run_id)
        index_version = await self._session.get(
            IndexVersion,
            run.index_version_id,
            with_for_update=True,
        )
        if index_version is None:
            raise ValueError(f"존재하지 않는 색인 버전입니다: {run.index_version_id}")
        if index_version.status == IndexVersionStatus.ACTIVE:
            raise ValueError("ACTIVE 색인 버전은 실패 처리할 수 없습니다.")

        now = datetime.now(timezone.utc)
        if index_version.status != IndexVersionStatus.READY:
            # 적용 단계 실패는 후보를 READY로 남겨 재시도할 수 있게 한다.
            # 빌드나 검증 단계 실패는 후보 자체를 쓸 수 없으므로 마감한다.
            index_version.status = IndexVersionStatus.FAILED
        run.status = ExecutionStatus.FAILED
        run.summary = {
            **(run.summary or {}),
            "stage": "FAILED",
            "failed_stage": failed_stage,
        }
        run.error_code = error_code
        run.error_message = sanitize_error_message(error)
        run.finished_at = now
        await self._session.flush()
        return run

    async def _replace_rows(
        self,
        items: Sequence[StoredEmbedding],
        documents: Sequence[NormalizedDocument],
    ) -> IndexVersion:
        items_by_document: DefaultDict[str, List[StoredEmbedding]] = defaultdict(list)
        for item in items:
            items_by_document[item[0].document_id].append(item)

        # embedding은 접수 시점에 함께 적재되므로 색인 단계에서 채울 것이 남지 않는다.
        persisted_chunks: List[RetrievalChunk] = []
        for document in documents:
            document_items = items_by_document[document.document_id]
            ingestion_run = await self._documents.start_ingestion(document)
            persisted_chunks.extend(
                await self._documents.complete_ingestion(
                    ingestion_run.id,
                    document,
                    [chunk for chunk, _ in document_items],
                    [embedding for _, embedding in document_items],
                )
            )

        index_run = await self.start_index(persisted_chunks)
        await self.store_index_items(index_run.id, persisted_chunks)
        await self.mark_index_ready(index_run.id)
        index_version = await self.apply_index(index_run.id)
        await self.finish_apply_run(index_run.id)
        return index_version

    async def _get_processing_index_run(self, index_run_id: int) -> IndexRun:
        run = await self._session.get(
            IndexRun,
            index_run_id,
            with_for_update=True,
        )
        if run is None:
            raise ValueError(f"존재하지 않는 색인 실행입니다: {index_run_id}")
        if run.status != ExecutionStatus.PROCESSING:
            raise ValueError(
                "PROCESSING 색인 실행만 마감할 수 있습니다: "
                f"{run.status}"
            )
        return run

    async def _validate_stored_index(
        self,
        index_version: IndexVersion,
        summary: dict,
    ) -> None:
        expected_documents = summary.get("document_count")
        expected_chunks = summary.get("chunk_count")
        document_count = await self._session.scalar(
            select(func.count())
            .select_from(IndexDocument)
            .where(IndexDocument.index_version_id == index_version.id)
        )
        embedding_count = await self._session.scalar(
            select(func.count())
            .select_from(ChunkEmbedding)
            .join(DocumentChunk, DocumentChunk.id == ChunkEmbedding.chunk_id)
            .join(ContentNode, ContentNode.id == DocumentChunk.id)
            .join(
                IndexDocument,
                and_(
                    IndexDocument.document_version_id
                    == ContentNode.document_version_id,
                    IndexDocument.index_version_id == index_version.id,
                ),
            )
            .where(
                ChunkEmbedding.embedding_config_id
                == index_version.embedding_config_id,
                DocumentChunk.chunking_config_id
                == index_version.chunking_config_id,
            )
        )
        if document_count != expected_documents or embedding_count != expected_chunks:
            raise RuntimeError(
                "색인 검증 건수가 일치하지 않습니다: "
                f"문서 {document_count}/{expected_documents}, "
                f"embedding {embedding_count}/{expected_chunks}"
            )

    @staticmethod
    def _validate_index_items(items: Sequence[StoredEmbedding]) -> None:
        IndexWriter._validate_persisted_chunks(
            [chunk for chunk, _ in items]
        )
        for _, embedding in items:
            validate_embedding_dimension(embedding)

    @staticmethod
    def _validate_new_index_items(items: Sequence[StoredEmbedding]) -> None:
        if not items:
            raise ValueError("전체 재색인할 Chunk가 하나 이상이어야 합니다.")

        section_ids = set()
        for chunk, embedding in items:
            validate_embedding_dimension(embedding)
            if (
                chunk.chunk_id is not None
                or chunk.document_version_id is not None
                or chunk.index_version_id is not None
            ):
                raise ValueError("재색인 입력 Chunk에는 DB 식별자를 지정할 수 없습니다.")
            if chunk.section_id in section_ids:
                raise ValueError(f"중복 section_id입니다: {chunk.section_id}")
            section_ids.add(chunk.section_id)

    @staticmethod
    def _validate_reindex_documents(
        items: Sequence[StoredEmbedding],
        documents: Sequence[NormalizedDocument],
    ) -> None:
        document_ids = [document.document_id for document in documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("재색인 입력에 중복 document_id가 있습니다.")

        item_document_ids = {chunk.document_id for chunk, _ in items}
        if set(document_ids) != item_document_ids:
            raise ValueError("재색인 문서와 Chunk의 document_id가 일치하지 않습니다.")

    @staticmethod
    def _validate_persisted_chunks(chunks: Sequence[RetrievalChunk]) -> None:
        if not chunks:
            raise ValueError("전체 재색인할 Chunk가 하나 이상이어야 합니다.")

        section_ids = set()
        chunk_ids = set()
        for chunk in chunks:
            if chunk.chunk_id is None or chunk.document_version_id is None:
                raise ValueError("색인 입력 Chunk에는 DB 식별자가 필요합니다.")
            if chunk.index_version_id is not None:
                raise ValueError("색인 입력 Chunk에는 index_version_id를 지정할 수 없습니다.")
            if chunk.section_id in section_ids:
                raise ValueError(f"중복 section_id입니다: {chunk.section_id}")
            if chunk.chunk_id in chunk_ids:
                raise ValueError(f"중복 chunk_id입니다: {chunk.chunk_id}")
            section_ids.add(chunk.section_id)
            chunk_ids.add(chunk.chunk_id)

    @staticmethod
    def _new_index_version_name(now: datetime) -> str:
        timestamp = now.strftime("%Y%m%dT%H%M%S%fZ")
        return f"idx-{timestamp}-{uuid.uuid4().hex[:8]}"
