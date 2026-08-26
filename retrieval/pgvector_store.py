"""신규 ERD 기반 전체 색인 적재와 pgvector 검색을 관리한다."""

import hashlib
import re
import uuid
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from typing import DefaultDict, List, Optional, Sequence, Tuple

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    ChunkEmbedding,
    ChunkingConfig,
    ContentNode,
    DocumentChunk,
    DocumentSource,
    DocumentVersion,
    DocumentVersionStatus,
    EmbeddingConfig,
    ExecutionStatus,
    IngestionRun,
    IndexDocument,
    IndexRun,
    IndexVersion,
    IndexVersionStatus,
)
from pipeline.document.models import NormalizedDocument
from pipeline.document.section_parser import create_section_identity_hash
from retrieval.embedding import (
    OPENAI_EMBEDDING_DIMENSIONS,
    OPENAI_EMBEDDING_MODEL,
    build_embedding_text,
)
from retrieval.models import RetrievalChunk


CHUNKING_CONFIG_VERSION = "section-v1"
CHUNKING_STRATEGY = "SECTION"
EMBEDDING_CONFIG_VERSION = "openai-text-embedding-3-large-1536-v1"
EMBEDDING_INPUT_TEMPLATE_VERSION = "document-section-content-v1"
CURRENT_CANDIDATE_K = 10
CURRENT_RRF_RANK_CONSTANT = 60
CURRENT_FINAL_TOP_K = 5
DEFAULT_TRIGGER_TYPE = "MANUAL"
PARSER_NAME = "gitbook-markdown"
PARSER_VERSION = "1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

StoredEmbedding = Tuple[RetrievalChunk, Sequence[float]]
SimilarityResult = Tuple[RetrievalChunk, float]


class ActiveIndexNotFoundError(RuntimeError):
    """검색 가능한 ACTIVE index version이 없을 때 발생한다."""


class PgVectorStore:
    """AsyncSession으로 신규 ERD 색인을 적재하고 ACTIVE index를 검색한다."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

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

    async def start_ingestion(
        self,
        document: NormalizedDocument,
        *,
        trigger_type: str = DEFAULT_TRIGGER_TYPE,
    ) -> IngestionRun:
        """문서 한 건의 DB 적재 실행을 PROCESSING 상태로 시작한다."""

        if not trigger_type:
            raise ValueError("trigger_type은 비어 있을 수 없습니다.")

        now = datetime.now(timezone.utc)
        source = await self._get_or_create_document_source(document, now)
        run = IngestionRun(
            document_source_id=source.id,
            trigger_type=trigger_type,
            parser_name=PARSER_NAME,
            parser_version=PARSER_VERSION,
            status=ExecutionStatus.PROCESSING,
            summary={"stage": "PROCESSING"},
            started_at=now,
        )
        self._session.add(run)
        await self._session.flush()
        return run

    async def complete_ingestion(
        self,
        ingestion_run_id: int,
        document: NormalizedDocument,
        chunks: Sequence[RetrievalChunk],
    ) -> List[RetrievalChunk]:
        """정제 문서와 Chunk를 적재하고 실행을 SUCCESS로 마감한다."""

        self._validate_document_chunks(document, chunks)
        run = await self._get_processing_ingestion_run(ingestion_run_id)
        source = await self._session.get(DocumentSource, run.document_source_id)
        if source is None or source.canonical_uri != document.source_url:
            raise ValueError("수집 실행과 문서 원본이 일치하지 않습니다.")

        now = datetime.now(timezone.utc)
        chunking_config = await self._get_or_create_chunking_config(now)
        current_version_no = await self._session.scalar(
            select(func.max(DocumentVersion.version_no)).where(
                DocumentVersion.document_source_id == source.id
            )
        )
        document_version = DocumentVersion(
            document_source_id=source.id,
            version_no=(current_version_no or 0) + 1,
            raw_content_uri=document.raw_content_uri,
            mime_type="text/markdown",
            raw_content_hash=document.raw_content_hash,
            normalized_content_hash=document.normalized_content_hash,
            parser_name=PARSER_NAME,
            parser_version=PARSER_VERSION,
            status=DocumentVersionStatus.PROCESSING,
            collected_at=now,
            created_at=now,
        )
        self._session.add(document_version)
        await self._session.flush()

        persisted_chunks = await self._create_document_chunks(
            chunks,
            document_version,
            chunking_config,
            now,
        )
        document_version.status = DocumentVersionStatus.READY
        run.produced_version_id = document_version.id
        run.status = ExecutionStatus.SUCCESS
        run.summary = {
            "stage": "COMPLETED",
            "section_count": len(chunks),
            "chunk_count": len(persisted_chunks),
        }
        run.finished_at = now
        await self._session.flush()
        return persisted_chunks

    async def fail_ingestion(
        self,
        ingestion_run_id: int,
        error: Exception,
    ) -> IngestionRun:
        """PROCESSING 수집 실행을 FAILED로 마감한다."""

        run = await self._get_processing_ingestion_run(ingestion_run_id)
        run.status = ExecutionStatus.FAILED
        run.summary = {"stage": "FAILED"}
        run.error_message = self._safe_error_message(error)
        run.finished_at = datetime.now(timezone.utc)
        await self._session.flush()
        return run

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
        chunking_config = await self._get_or_create_chunking_config(now)
        embedding_config = await self._get_or_create_embedding_config(now)
        index_version = IndexVersion(
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

    async def store_index_items(
        self,
        index_run_id: int,
        items: Sequence[StoredEmbedding],
    ) -> IndexVersion:
        """Chunk Embedding과 포함 문서를 저장하고 VALIDATING으로 전이한다."""

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
        embeddings = []
        document_version_ids = set()
        for chunk, embedding in items:
            embedding_input = build_embedding_text(chunk)
            embeddings.append(
                ChunkEmbedding(
                    chunk_id=chunk.chunk_id,
                    embedding_config_id=index_version.embedding_config_id,
                    embedding=list(embedding),
                    embedding_input_hash=self._sha256(embedding_input),
                    created_at=now,
                )
            )
            document_version_ids.add(chunk.document_version_id)

        expected_documents = (run.summary or {}).get("document_count")
        expected_chunks = (run.summary or {}).get("chunk_count")
        if (
            len(document_version_ids) != expected_documents
            or len(items) != expected_chunks
        ):
            raise RuntimeError(
                "색인 입력 건수가 시작 시점과 일치하지 않습니다: "
                f"문서 {len(document_version_ids)}/{expected_documents}, "
                f"Chunk {len(items)}/{expected_chunks}"
            )

        self._session.add_all(embeddings)
        self._session.add_all(
            IndexDocument(
                index_version_id=index_version.id,
                document_version_id=document_version_id,
            )
            for document_version_id in document_version_ids
        )
        index_version.status = IndexVersionStatus.VALIDATING
        run.summary = {
            "stage": "VALIDATING",
            "document_count": len(document_version_ids),
            "chunk_count": len(items),
            "embedding_count": len(embeddings),
        }
        await self._session.flush()
        return index_version

    async def activate_index(self, index_run_id: int) -> IndexVersion:
        """저장 건수를 검증하고 새 색인을 ACTIVE로 원자 전환한다."""

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
                "VALIDATING 색인 버전만 활성화할 수 있습니다: "
                f"{index_version.status}"
            )

        await self._validate_stored_index(index_version, run.summary or {})
        now = datetime.now(timezone.utc)
        await self._session.execute(
            update(IndexVersion)
            .where(
                IndexVersion.status == IndexVersionStatus.ACTIVE,
                IndexVersion.id != index_version.id,
            )
            .values(status=IndexVersionStatus.INACTIVE)
        )
        index_version.status = IndexVersionStatus.ACTIVE
        index_version.activated_at = now
        run.status = ExecutionStatus.SUCCESS
        run.summary = {**(run.summary or {}), "stage": "ACTIVE"}
        run.finished_at = now
        await self._session.flush()
        return index_version

    async def fail_index(
        self,
        index_run_id: int,
        error: Exception,
        *,
        failed_stage: str,
    ) -> IndexRun:
        """PROCESSING 색인 실행과 연결 버전을 FAILED로 마감한다."""

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
        index_version.status = IndexVersionStatus.FAILED
        run.status = ExecutionStatus.FAILED
        run.summary = {
            **(run.summary or {}),
            "stage": "FAILED",
            "failed_stage": failed_stage,
        }
        run.error_message = self._safe_error_message(error)
        run.finished_at = now
        await self._session.flush()
        return run

    async def get_active_index_version_id(self) -> int:
        """유일한 ACTIVE index version의 DB 식별자를 반환한다."""

        return (await self._get_active_index_version()).id

    async def load_active_chunks(self) -> List[RetrievalChunk]:
        """ACTIVE index의 전체 Chunk를 BM25 corpus 순서로 복원한다."""

        active_index = await self._get_active_index_version()
        statement = (
            select(DocumentChunk, ContentNode, DocumentVersion, DocumentSource)
            .join(ContentNode, ContentNode.id == DocumentChunk.id)
            .join(
                DocumentVersion,
                DocumentVersion.id == ContentNode.document_version_id,
            )
            .join(
                DocumentSource,
                DocumentSource.id == DocumentVersion.document_source_id,
            )
            .join(
                IndexDocument,
                and_(
                    IndexDocument.document_version_id == DocumentVersion.id,
                    IndexDocument.index_version_id == active_index.id,
                ),
            )
            .join(
                ChunkEmbedding,
                and_(
                    ChunkEmbedding.chunk_id == DocumentChunk.id,
                    ChunkEmbedding.embedding_config_id
                    == active_index.embedding_config_id,
                ),
            )
            .where(
                DocumentChunk.chunking_config_id
                == active_index.chunking_config_id
            )
            .order_by(DocumentVersion.id, ContentNode.node_order)
        )
        rows = (await self._session.execute(statement)).all()
        return [
            self._to_retrieval_chunk(
                document_chunk,
                content_node,
                document_version,
                document_source,
                active_index.id,
            )
            for document_chunk, content_node, document_version, document_source in rows
        ]

    async def similarity_search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 10,
    ) -> List[SimilarityResult]:
        """ACTIVE index에서 Query vector와 가까운 Chunk를 cosine 순으로 반환한다."""

        if top_k <= 0:
            raise ValueError("top_k는 1 이상이어야 합니다.")
        self._validate_embedding_dimension(query_embedding)

        active_index = await self._get_active_index_version()
        cosine_distance = ChunkEmbedding.embedding.cosine_distance(
            list(query_embedding)
        ).label("cosine_distance")
        statement = (
            select(
                DocumentChunk,
                ContentNode,
                DocumentVersion,
                DocumentSource,
                cosine_distance,
            )
            .join(ContentNode, ContentNode.id == DocumentChunk.id)
            .join(
                DocumentVersion,
                DocumentVersion.id == ContentNode.document_version_id,
            )
            .join(
                DocumentSource,
                DocumentSource.id == DocumentVersion.document_source_id,
            )
            .join(
                IndexDocument,
                and_(
                    IndexDocument.document_version_id == DocumentVersion.id,
                    IndexDocument.index_version_id == active_index.id,
                ),
            )
            .join(
                ChunkEmbedding,
                and_(
                    ChunkEmbedding.chunk_id == DocumentChunk.id,
                    ChunkEmbedding.embedding_config_id
                    == active_index.embedding_config_id,
                ),
            )
            .where(
                DocumentChunk.chunking_config_id
                == active_index.chunking_config_id
            )
            .order_by(cosine_distance.asc())
            .limit(top_k)
        )

        rows = (await self._session.execute(statement)).all()
        return [
            (
                self._to_retrieval_chunk(
                    document_chunk,
                    content_node,
                    document_version,
                    document_source,
                    active_index.id,
                ),
                1.0 - float(distance),
            )
            for (
                document_chunk,
                content_node,
                document_version,
                document_source,
                distance,
            ) in rows
        ]

    async def _replace_rows(
        self,
        items: Sequence[StoredEmbedding],
        documents: Sequence[NormalizedDocument],
    ) -> IndexVersion:
        items_by_document: DefaultDict[str, List[StoredEmbedding]] = defaultdict(list)
        for item in items:
            items_by_document[item[0].document_id].append(item)

        persisted_items = []
        for document in documents:
            document_items = items_by_document[document.document_id]
            ingestion_run = await self.start_ingestion(document)
            persisted_chunks = await self.complete_ingestion(
                ingestion_run.id,
                document,
                [chunk for chunk, _ in document_items],
            )
            persisted_items.extend(
                (persisted_chunk, embedding)
                for persisted_chunk, (_, embedding) in zip(
                    persisted_chunks,
                    document_items,
                )
            )

        index_run = await self.start_index(
            [chunk for chunk, _ in persisted_items]
        )
        await self.store_index_items(index_run.id, persisted_items)
        return await self.activate_index(index_run.id)

    async def _get_or_create_chunking_config(
        self,
        now: datetime,
    ) -> ChunkingConfig:
        config = await self._session.scalar(
            select(ChunkingConfig).where(
                ChunkingConfig.version == CHUNKING_CONFIG_VERSION
            )
        )
        if config is not None:
            if (
                config.strategy != CHUNKING_STRATEGY
                or config.max_tokens != 0
                or config.overlap_tokens != 0
            ):
                raise ValueError("기존 ChunkingConfig가 현재 SECTION 설정과 다릅니다.")
            return config

        config = ChunkingConfig(
            version=CHUNKING_CONFIG_VERSION,
            strategy=CHUNKING_STRATEGY,
            max_tokens=0,
            overlap_tokens=0,
            parameters={"split": False, "section_boundary": "H2"},
            created_at=now,
        )
        self._session.add(config)
        await self._session.flush()
        return config

    async def _get_or_create_embedding_config(
        self,
        now: datetime,
    ) -> EmbeddingConfig:
        config = await self._session.scalar(
            select(EmbeddingConfig).where(
                EmbeddingConfig.version == EMBEDDING_CONFIG_VERSION
            )
        )
        if config is not None:
            if (
                config.provider != "openai"
                or config.model_name != OPENAI_EMBEDDING_MODEL
                or config.dimensions != OPENAI_EMBEDDING_DIMENSIONS
                or config.input_template_version
                != EMBEDDING_INPUT_TEMPLATE_VERSION
            ):
                raise ValueError("기존 EmbeddingConfig가 현재 OpenAI 설정과 다릅니다.")
            return config

        config = EmbeddingConfig(
            version=EMBEDDING_CONFIG_VERSION,
            provider="openai",
            model_name=OPENAI_EMBEDDING_MODEL,
            dimensions=OPENAI_EMBEDDING_DIMENSIONS,
            input_template_version=EMBEDDING_INPUT_TEMPLATE_VERSION,
            parameters={"encoding_format": "float"},
            created_at=now,
        )
        self._session.add(config)
        await self._session.flush()
        return config

    async def _create_document_chunks(
        self,
        chunks: Sequence[RetrievalChunk],
        document_version: DocumentVersion,
        chunking_config: ChunkingConfig,
        now: datetime,
    ) -> List[RetrievalChunk]:
        content_nodes = []
        embedding_inputs = []
        for node_order, chunk in enumerate(chunks):
            embedding_input = build_embedding_text(chunk)
            embedding_inputs.append(embedding_input)
            content_nodes.append(
                ContentNode(
                    document_version_id=document_version.id,
                    node_type="SECTION",
                    node_path=" > ".join(chunk.section_path),
                    node_order=node_order,
                    title=chunk.section_path[-1],
                    normalized_content=chunk.content,
                    source_locator={
                        "source_url": chunk.source_url,
                        "section_id": chunk.section_id,
                    },
                    content_hash=self._sha256(chunk.content),
                    node_identity_hash=create_section_identity_hash(
                        chunk.document_id,
                        chunk.section_path[1:],
                    ),
                    node_identity_kind="path",
                    metadata_={
                        "document_id": chunk.document_id,
                        "section_id": chunk.section_id,
                        "section_path": list(chunk.section_path),
                    },
                    created_at=now,
                )
            )

        self._session.add_all(content_nodes)
        await self._session.flush()

        document_chunks = []
        persisted_chunks = []
        for node, embedding_input, chunk in zip(
            content_nodes,
            embedding_inputs,
            chunks,
        ):
            input_hash = self._sha256(embedding_input)
            document_chunks.append(
                DocumentChunk(
                    id=node.id,
                    chunking_config_id=chunking_config.id,
                    chunk_index=node.node_order,
                    embedding_input_hash=input_hash,
                    keyword_search_text=embedding_input,
                    created_at=now,
                )
            )
            persisted_chunks.append(
                replace(
                    chunk,
                    chunk_id=node.id,
                    document_version_id=document_version.id,
                )
            )

        self._session.add_all(document_chunks)
        await self._session.flush()
        return persisted_chunks

    async def _get_or_create_document_source(
        self,
        document: NormalizedDocument,
        now: datetime,
    ) -> DocumentSource:
        source = await self._session.scalar(
            select(DocumentSource).where(
                DocumentSource.canonical_uri == document.source_url
            )
        )
        source_metadata = {
            "document_id": document.document_id,
            "category": document.category,
        }
        if source is None:
            source = DocumentSource(
                source_type="GITBOOK_MARKDOWN",
                canonical_uri=document.source_url,
                title=document.title,
                metadata_=source_metadata,
                enabled=True,
                created_at=now,
                updated_at=now,
            )
            self._session.add(source)
            await self._session.flush()
            return source

        source.title = document.title
        source.metadata_ = source_metadata
        source.enabled = True
        source.updated_at = now
        await self._session.flush()
        return source

    async def _get_processing_ingestion_run(
        self,
        ingestion_run_id: int,
    ) -> IngestionRun:
        run = await self._session.get(
            IngestionRun,
            ingestion_run_id,
            with_for_update=True,
        )
        if run is None:
            raise ValueError(f"존재하지 않는 수집 실행입니다: {ingestion_run_id}")
        if run.status != ExecutionStatus.PROCESSING:
            raise ValueError(
                "PROCESSING 수집 실행만 마감할 수 있습니다: "
                f"{run.status}"
            )
        return run

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

    async def _get_active_index_version(self) -> IndexVersion:
        result = await self._session.execute(
            select(IndexVersion)
            .where(IndexVersion.status == IndexVersionStatus.ACTIVE)
            .order_by(IndexVersion.activated_at.desc(), IndexVersion.id.desc())
            .limit(2)
        )
        active_versions = list(result.scalars().all())
        if not active_versions:
            raise ActiveIndexNotFoundError("ACTIVE index version이 없습니다.")
        if len(active_versions) > 1:
            raise RuntimeError("ACTIVE index version이 둘 이상 존재합니다.")
        return active_versions[0]

    @staticmethod
    def _to_retrieval_chunk(
        document_chunk: DocumentChunk,
        content_node: ContentNode,
        document_version: DocumentVersion,
        document_source: DocumentSource,
        index_version_id: int,
    ) -> RetrievalChunk:
        metadata = content_node.metadata_ or {}
        document_id = metadata.get("document_id")
        section_id = metadata.get("section_id")
        section_path = metadata.get("section_path")
        if not isinstance(document_id, str) or not document_id:
            raise RuntimeError("ContentNode에 document_id metadata가 없습니다.")
        if not isinstance(section_id, str) or not section_id:
            raise RuntimeError("ContentNode에 section_id metadata가 없습니다.")
        if not isinstance(section_path, list) or not all(
            isinstance(part, str) for part in section_path
        ):
            raise RuntimeError("ContentNode에 section_path metadata가 없습니다.")
        if not document_source.title:
            raise RuntimeError("DocumentSource에 title이 없습니다.")

        source_metadata = document_source.metadata_ or {}
        category = source_metadata.get("category")
        if category is not None and not isinstance(category, str):
            raise RuntimeError("DocumentSource category metadata 형식이 올바르지 않습니다.")

        return RetrievalChunk(
            document_id=document_id,
            section_id=section_id,
            document_title=document_source.title,
            section_path=tuple(section_path),
            source_url=document_source.canonical_uri,
            category=category,
            content=content_node.normalized_content,
            chunk_id=document_chunk.id,
            document_version_id=document_version.id,
            index_version_id=index_version_id,
        )

    @staticmethod
    def _validate_index_items(items: Sequence[StoredEmbedding]) -> None:
        PgVectorStore._validate_persisted_chunks(
            [chunk for chunk, _ in items]
        )
        for _, embedding in items:
            PgVectorStore._validate_embedding_dimension(embedding)

    @staticmethod
    def _validate_new_index_items(items: Sequence[StoredEmbedding]) -> None:
        if not items:
            raise ValueError("전체 재색인할 Chunk가 하나 이상이어야 합니다.")

        section_ids = set()
        for chunk, embedding in items:
            PgVectorStore._validate_embedding_dimension(embedding)
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
    def _validate_document_chunks(
        document: NormalizedDocument,
        chunks: Sequence[RetrievalChunk],
    ) -> None:
        if not chunks:
            raise ValueError("수집할 Chunk가 하나 이상이어야 합니다.")
        if not document.raw_content_uri:
            raise ValueError("원문 보관 위치가 필요합니다.")
        if not SHA256_PATTERN.fullmatch(document.raw_content_hash):
            raise ValueError("원문 hash는 SHA-256 형식이어야 합니다.")
        if not SHA256_PATTERN.fullmatch(document.normalized_content_hash):
            raise ValueError("정제 문서 hash는 SHA-256 형식이어야 합니다.")
        if PgVectorStore._sha256(document.content) != document.normalized_content_hash:
            raise ValueError("정제 문서 내용과 hash가 일치하지 않습니다.")

        section_ids = set()
        for chunk in chunks:
            if (
                chunk.document_id != document.document_id
                or chunk.document_title != document.title
                or chunk.source_url != document.source_url
                or not chunk.section_path
                or chunk.section_path[0] != document.title
            ):
                raise ValueError("문서와 Chunk metadata가 일치하지 않습니다.")
            if (
                chunk.chunk_id is not None
                or chunk.document_version_id is not None
                or chunk.index_version_id is not None
            ):
                raise ValueError("수집 입력 Chunk에는 DB 식별자를 지정할 수 없습니다.")
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
    def _validate_embedding_dimension(embedding: Sequence[float]) -> None:
        if len(embedding) != OPENAI_EMBEDDING_DIMENSIONS:
            raise ValueError(
                "embedding은 "
                f"{OPENAI_EMBEDDING_DIMENSIONS}차원이어야 합니다."
            )

    @staticmethod
    def _sha256(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _safe_error_message(error: Exception) -> str:
        message = re.sub(
            r"sk-[A-Za-z0-9_-]{8,}",
            "sk-***REDACTED***",
            str(error),
        )
        return message[:4000]

    @staticmethod
    def _new_index_version_name(now: datetime) -> str:
        timestamp = now.strftime("%Y%m%dT%H%M%S%fZ")
        return f"idx-{timestamp}-{uuid.uuid4().hex[:8]}"
