"""PostgreSQL + pgvector를 사용하는 Chunk 저장과 유사도 검색을 관리한다."""

from typing import List, Sequence, Tuple

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import LegacyChunkEmbedding as ChunkEmbedding
from app.database.models import LegacyDocumentChunk as DocumentChunk
from retrieval.embedding import OPENAI_EMBEDDING_DIMENSIONS
from retrieval.models import RetrievalChunk


StoredEmbedding = Tuple[RetrievalChunk, Sequence[float]]
SimilarityResult = Tuple[RetrievalChunk, float]


class PgVectorStore:
    """AsyncSession을 사용해 Vector corpus를 저장하고 검색한다."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace_all(
        self,
        items: Sequence[StoredEmbedding],
    ) -> None:
        """기존 Vector corpus를 완성된 새 corpus로 원자적으로 교체한다."""

        if not items:
            raise ValueError("전체 재색인할 Chunk가 하나 이상이어야 합니다.")

        for _, embedding in items:
            self._validate_embedding_dimension(embedding)

        if self._session.in_transaction():
            await self._replace_rows(items)
            return

        async with self._session.begin():
            await self._replace_rows(items)

    async def similarity_search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 10,
    ) -> List[SimilarityResult]:
        """Query vector와 가까운 Chunk를 cosine similarity 순으로 반환한다."""

        if top_k <= 0:
            raise ValueError("top_k는 1 이상이어야 합니다.")
        self._validate_embedding_dimension(query_embedding)

        cosine_distance = ChunkEmbedding.embedding.cosine_distance(
            list(query_embedding)
        ).label("cosine_distance")
        statement = (
            select(DocumentChunk, cosine_distance)
            .join(
                ChunkEmbedding,
                ChunkEmbedding.chunk_id == DocumentChunk.chunk_id,
            )
            .order_by(cosine_distance.asc())
            .limit(top_k)
        )

        rows = (await self._session.execute(statement)).all()
        return [
            (
                self._to_retrieval_chunk(document_chunk),
                1.0 - float(distance),
            )
            for document_chunk, distance in rows
        ]

    async def _replace_rows(
        self,
        items: Sequence[StoredEmbedding],
    ) -> None:
        await self._session.execute(delete(DocumentChunk))

        document_chunks = [
            DocumentChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                section_id=chunk.section_id,
                document_title=chunk.document_title,
                section_path=list(chunk.section_path),
                source_url=chunk.source_url,
                category=chunk.category,
                content=chunk.content,
            )
            for chunk, _ in items
        ]
        self._session.add_all(document_chunks)
        await self._session.flush()

        chunk_embeddings = [
            ChunkEmbedding(
                chunk_id=chunk.chunk_id,
                embedding=list(embedding),
            )
            for chunk, embedding in items
        ]
        self._session.add_all(chunk_embeddings)
        await self._session.flush()

    @staticmethod
    def _to_retrieval_chunk(document_chunk: DocumentChunk) -> RetrievalChunk:
        return RetrievalChunk(
            chunk_id=document_chunk.chunk_id,
            document_id=document_chunk.document_id,
            section_id=document_chunk.section_id,
            document_title=document_chunk.document_title,
            section_path=tuple(document_chunk.section_path),
            source_url=document_chunk.source_url,
            category=document_chunk.category,
            content=document_chunk.content,
        )

    @staticmethod
    def _validate_embedding_dimension(embedding: Sequence[float]) -> None:
        if len(embedding) != OPENAI_EMBEDDING_DIMENSIONS:
            raise ValueError(
                "embedding은 "
                f"{OPENAI_EMBEDDING_DIMENSIONS}차원이어야 합니다."
            )
