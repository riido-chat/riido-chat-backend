"""ACTIVE 색인에서 검색에 필요한 Chunk를 읽는다."""

from typing import List, Optional, Sequence

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    ChunkEmbedding,
    ContentNode,
    DocumentChunk,
    DocumentSource,
    DocumentVersion,
    IndexDocument,
    IndexVersion,
    IndexVersionStatus,
)
from app.retrieval.embedding_config import validate_embedding_dimension
from app.retrieval.models import RetrievalChunk, SimilarityResult


class ActiveIndexNotFoundError(RuntimeError):
    """검색 가능한 ACTIVE index version이 없을 때 발생한다."""


class SearchReader:
    """AsyncSession으로 ACTIVE index를 조회한다."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

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
        validate_embedding_dimension(query_embedding)

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
        index_version_id: Optional[int],
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
