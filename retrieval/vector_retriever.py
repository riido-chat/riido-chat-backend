"""OpenAI query embedding과 pgvector 검색을 연결한다."""

import asyncio
from typing import List

from retrieval.embedding import OpenAIEmbedder
from retrieval.models import RetrievalResult
from retrieval.pgvector_store import PgVectorStore


class VectorRetriever:
    """사용자 Query를 embedding하고 Vector 검색 결과를 조립한다."""

    def __init__(
        self,
        embedder: OpenAIEmbedder,
        store: PgVectorStore,
    ) -> None:
        self._embedder = embedder
        self._store = store

    async def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Query와 유사한 Chunk를 score와 rank가 포함된 결과로 반환한다."""

        if top_k <= 0:
            raise ValueError("top_k는 1 이상이어야 합니다.")
        if not query.strip():
            return []

        query_embedding = await asyncio.to_thread(
            self._embedder.embed,
            query,
        )
        matches = await self._store.similarity_search(
            query_embedding,
            top_k,
        )

        return [
            RetrievalResult(
                chunk=chunk,
                score=score,
                rank=rank,
            )
            for rank, (chunk, score) in enumerate(matches, start=1)
        ]
