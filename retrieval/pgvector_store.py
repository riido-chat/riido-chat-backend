"""신규 ERD 기반 전체 색인 적재와 pgvector 검색을 관리한다."""

import hashlib
import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import DefaultDict, List, Sequence, Tuple

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
    IndexDocument,
    IndexVersion,
    IndexVersionStatus,
)
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
    ) -> IndexVersion:
        """전체 corpus를 새 IndexVersion으로 적재한 뒤 원자적으로 ACTIVE 전환한다."""

        self._validate_index_items(items)

        if self._session.in_transaction():
            return await self._replace_rows(items)

        async with self._session.begin():
            return await self._replace_rows(items)

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
    ) -> IndexVersion:
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

        items_by_document: DefaultDict[str, List[StoredEmbedding]] = defaultdict(list)
        for item in items:
            items_by_document[item[0].document_id].append(item)

        for document_items in items_by_document.values():
            document_version = await self._create_document_version(
                document_items,
                now,
            )
            await self._create_document_chunks(
                document_items,
                document_version,
                chunking_config,
                embedding_config,
                now,
            )
            document_version.status = DocumentVersionStatus.READY
            self._session.add(
                IndexDocument(
                    index_version_id=index_version.id,
                    document_version_id=document_version.id,
                )
            )

        index_version.status = IndexVersionStatus.VALIDATING
        await self._session.flush()
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
        await self._session.flush()
        return index_version

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

    async def _create_document_version(
        self,
        items: Sequence[StoredEmbedding],
        now: datetime,
    ) -> DocumentVersion:
        first_chunk = items[0][0]
        for chunk, _ in items:
            if (
                chunk.document_id != first_chunk.document_id
                or chunk.document_title != first_chunk.document_title
                or chunk.source_url != first_chunk.source_url
            ):
                raise ValueError("동일 문서의 Chunk metadata가 일치하지 않습니다.")

        source = await self._session.scalar(
            select(DocumentSource).where(
                DocumentSource.canonical_uri == first_chunk.source_url
            )
        )
        source_metadata = {
            "document_id": first_chunk.document_id,
            "category": first_chunk.category,
        }
        if source is None:
            source = DocumentSource(
                source_type="GITBOOK_MARKDOWN",
                canonical_uri=first_chunk.source_url,
                title=first_chunk.document_title,
                metadata_=source_metadata,
                enabled=True,
                created_at=now,
                updated_at=now,
            )
            self._session.add(source)
            await self._session.flush()
        else:
            source.title = first_chunk.document_title
            source.metadata_ = source_metadata
            source.enabled = True
            source.updated_at = now

        current_version_no = await self._session.scalar(
            select(func.max(DocumentVersion.version_no)).where(
                DocumentVersion.document_source_id == source.id
            )
        )
        content_hash = self._document_content_hash(items)
        document_version = DocumentVersion(
            document_source_id=source.id,
            version_no=(current_version_no or 0) + 1,
            raw_content_uri=first_chunk.source_url,
            mime_type="text/markdown",
            raw_content_hash=content_hash,
            normalized_content_hash=content_hash,
            parser_name="gitbook-markdown",
            parser_version="1",
            status=DocumentVersionStatus.PROCESSING,
            collected_at=now,
            created_at=now,
        )
        self._session.add(document_version)
        await self._session.flush()
        return document_version

    async def _create_document_chunks(
        self,
        items: Sequence[StoredEmbedding],
        document_version: DocumentVersion,
        chunking_config: ChunkingConfig,
        embedding_config: EmbeddingConfig,
        now: datetime,
    ) -> None:
        content_nodes = []
        embedding_inputs = []
        for node_order, (chunk, _) in enumerate(items):
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
                    node_identity_hash=self._sha256(chunk.section_id),
                    node_identity_kind="SECTION_ID",
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
        chunk_embeddings = []
        for node, embedding_input, (_, embedding) in zip(
            content_nodes,
            embedding_inputs,
            items,
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
            chunk_embeddings.append(
                ChunkEmbedding(
                    chunk_id=node.id,
                    embedding_config_id=embedding_config.id,
                    embedding=list(embedding),
                    embedding_input_hash=input_hash,
                    created_at=now,
                )
            )

        self._session.add_all(document_chunks)
        await self._session.flush()
        self._session.add_all(chunk_embeddings)
        await self._session.flush()

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
    def _validate_embedding_dimension(embedding: Sequence[float]) -> None:
        if len(embedding) != OPENAI_EMBEDDING_DIMENSIONS:
            raise ValueError(
                "embedding은 "
                f"{OPENAI_EMBEDDING_DIMENSIONS}차원이어야 합니다."
            )

    @staticmethod
    def _document_content_hash(items: Sequence[StoredEmbedding]) -> str:
        serialized = json.dumps(
            [
                {
                    "section_id": chunk.section_id,
                    "section_path": chunk.section_path,
                    "content": chunk.content,
                }
                for chunk, _ in items
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return PgVectorStore._sha256(serialized)

    @staticmethod
    def _sha256(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _new_index_version_name(now: datetime) -> str:
        timestamp = now.strftime("%Y%m%dT%H%M%S%fZ")
        return f"idx-{timestamp}-{uuid.uuid4().hex[:8]}"
