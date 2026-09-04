"""문서 원본과 판, 청크를 적재하고 수집 실행을 마감한다."""

import re
from dataclasses import replace
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.error_message import sanitize_error_message
from app.core.hashing import sha256_hex
from app.database.models import (
    ChunkEmbedding,
    ChunkingConfig,
    ContentNode,
    DocumentChunk,
    DocumentSource,
    DocumentVersion,
    DocumentVersionStatus,
    ExecutionStatus,
    IngestionRun,
    ModelCall,
    ModelCallPurpose,
)
from app.document.chunking_config import get_or_create_chunking_config
from app.document.document_group import get_document_group
from app.document.document_key import (
    SOURCE_TYPE_GITBOOK,
    build_gitbook_document_key,
)
from app.document.models import NormalizedDocument
from app.document.section_parser import create_section_identity_hash
from app.retrieval.embedding import (
    OPENAI_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_PROVIDER,
    build_embedding_text,
)
from app.retrieval.embedding_config import (
    EMBEDDING_INPUT_TEMPLATE_VERSION,
    get_or_create_embedding_config,
    validate_embedding_dimension,
)
from app.retrieval.models import RetrievalChunk


DEFAULT_TRIGGER_TYPE = "MANUAL"
PARSER_NAME = "gitbook-markdown"
PARSER_VERSION = "1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DocumentStore:
    """AsyncSession으로 문서 수집 실행과 판을 기록한다."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

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
        embeddings: Sequence[Sequence[float]],
    ) -> List[RetrievalChunk]:
        """정제 문서와 Chunk, embedding을 적재하고 실행을 SUCCESS로 마감한다.

        문서 판은 청크와 embedding이 모두 준비된 뒤에만 READY가 된다.
        """

        self._validate_document_chunks(document, chunks)
        self._validate_chunk_embeddings(chunks, embeddings)
        run = await self._get_processing_ingestion_run(ingestion_run_id)
        source = await self._session.get(DocumentSource, run.document_source_id)
        if source is None or source.canonical_uri != document.source_url:
            raise ValueError("수집 실행과 문서 원본이 일치하지 않습니다.")

        now = datetime.now(timezone.utc)
        chunking_config = await get_or_create_chunking_config(self._session, now)
        embedding_config = await get_or_create_embedding_config(self._session, now)
        current_version_no = await self._session.scalar(
            select(func.max(DocumentVersion.version_no)).where(
                DocumentVersion.document_source_id == source.id
            )
        )
        document_version = DocumentVersion(
            document_source_id=source.id,
            version_no=(current_version_no or 0) + 1,
            raw_content_uri=document.raw_content_uri,
            raw_content=document.raw_content,
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
        self._session.add_all(
            ChunkEmbedding(
                chunk_id=chunk.chunk_id,
                embedding_config_id=embedding_config.id,
                embedding=list(embedding),
                embedding_input_hash=sha256_hex(build_embedding_text(chunk)),
                created_at=now,
            )
            for chunk, embedding in zip(persisted_chunks, embeddings)
        )
        await self._session.flush()
        document_version.status = DocumentVersionStatus.READY
        run.produced_version_id = document_version.id
        run.status = ExecutionStatus.SUCCESS
        run.summary = {
            "stage": "COMPLETED",
            "section_count": len(chunks),
            "chunk_count": len(persisted_chunks),
            "embedding_count": len(embeddings),
        }
        run.finished_at = now
        await self._session.flush()
        return persisted_chunks

    async def load_reusable_embeddings(
        self,
        ingestion_run_id: int,
        embedding_input_hashes: Sequence[str],
    ) -> Dict[str, List[float]]:
        """같은 문서 원본의 이전 판에서 재사용 가능한 embedding을 찾는다.

        embedding 입력이 같으면 결과도 같으므로 다시 생성하지 않고 복사한다.
        """

        if not embedding_input_hashes:
            return {}

        run = await self._session.get(IngestionRun, ingestion_run_id)
        if run is None:
            raise ValueError(f"존재하지 않는 수집 실행입니다: {ingestion_run_id}")

        now = datetime.now(timezone.utc)
        embedding_config = await get_or_create_embedding_config(self._session, now)
        statement = (
            select(
                ChunkEmbedding.embedding_input_hash,
                ChunkEmbedding.embedding,
            )
            .join(DocumentChunk, DocumentChunk.id == ChunkEmbedding.chunk_id)
            .join(ContentNode, ContentNode.id == DocumentChunk.id)
            .join(
                DocumentVersion,
                DocumentVersion.id == ContentNode.document_version_id,
            )
            .where(
                DocumentVersion.document_source_id == run.document_source_id,
                ChunkEmbedding.embedding_config_id == embedding_config.id,
                ChunkEmbedding.embedding_input_hash.in_(
                    set(embedding_input_hashes)
                ),
            )
            .order_by(DocumentVersion.id.desc())
        )
        reusable: Dict[str, List[float]] = {}
        for input_hash, embedding in (await self._session.execute(statement)).all():
            reusable.setdefault(input_hash, list(embedding))
        return reusable

    async def record_embedding_model_call(
        self,
        ingestion_run_id: int,
        *,
        input_tokens: Optional[int],
        retry_count: int,
        latency_ms: int,
    ) -> ModelCall:
        """접수 중 발생한 청크 embedding 호출을 수집 실행에 붙여 기록한다."""

        call = ModelCall(
            ingestion_run_id=ingestion_run_id,
            purpose=ModelCallPurpose.CHUNK_EMBEDDING,
            provider=OPENAI_EMBEDDING_PROVIDER,
            model_name=OPENAI_EMBEDDING_MODEL,
            prompt_version=EMBEDDING_INPUT_TEMPLATE_VERSION,
            input_tokens=input_tokens,
            status=ExecutionStatus.SUCCESS,
            retry_count=retry_count,
            latency_ms=latency_ms,
            created_at=datetime.now(timezone.utc),
        )
        self._session.add(call)
        await self._session.flush()
        return call

    async def fail_ingestion(
        self,
        ingestion_run_id: int,
        error: Exception,
        *,
        failed_stage: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> IngestionRun:
        """PROCESSING 수집 실행을 FAILED로 마감한다."""

        run = await self._get_processing_ingestion_run(ingestion_run_id)
        run.status = ExecutionStatus.FAILED
        run.summary = {
            "stage": "FAILED",
            **(
                {"failed_stage": failed_stage}
                if failed_stage is not None
                else {}
            ),
            **({"error_code": error_code} if error_code is not None else {}),
        }
        run.error_message = sanitize_error_message(error)
        run.finished_at = datetime.now(timezone.utc)
        await self._session.flush()
        return run

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
                    content_hash=sha256_hex(chunk.content),
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
            input_hash = sha256_hex(embedding_input)
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
        group = await get_document_group(self._session)
        document_key = build_gitbook_document_key(document.source_url)
        source = await self._session.scalar(
            select(DocumentSource).where(
                DocumentSource.document_group_id == group.id,
                DocumentSource.document_key == document_key,
            )
        )
        source_metadata = {
            "document_id": document.document_id,
            "category": document.category,
        }
        if source is None:
            source = DocumentSource(
                document_group_id=group.id,
                document_key=document_key,
                source_type=SOURCE_TYPE_GITBOOK,
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

        source.canonical_uri = document.source_url
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

    @staticmethod
    def _validate_chunk_embeddings(
        chunks: Sequence[RetrievalChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunk 개수와 embedding 개수가 일치하지 않습니다: "
                f"Chunk {len(chunks)}개, embedding {len(embeddings)}개"
            )
        for embedding in embeddings:
            validate_embedding_dimension(embedding)

    @staticmethod
    def _validate_document_chunks(
        document: NormalizedDocument,
        chunks: Sequence[RetrievalChunk],
    ) -> None:
        if not chunks:
            raise ValueError("수집할 Chunk가 하나 이상이어야 합니다.")
        if document.raw_content_uri is None and document.raw_content is None:
            raise ValueError("원문 보관 위치 또는 inline 원문이 필요합니다.")
        if not SHA256_PATTERN.fullmatch(document.raw_content_hash):
            raise ValueError("원문 hash는 SHA-256 형식이어야 합니다.")
        if not SHA256_PATTERN.fullmatch(document.normalized_content_hash):
            raise ValueError("정제 문서 hash는 SHA-256 형식이어야 합니다.")
        if sha256_hex(document.content) != document.normalized_content_hash:
            raise ValueError("정제 문서 내용과 hash가 일치하지 않습니다.")
        if (
            document.raw_content is not None
            and sha256_hex(document.raw_content)
            != document.raw_content_hash
        ):
            raise ValueError("inline 원문 내용과 hash가 일치하지 않습니다.")

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
