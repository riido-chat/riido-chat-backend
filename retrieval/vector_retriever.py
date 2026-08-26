"""OpenAI query embedding과 pgvector 검색을 연결한다."""

import asyncio
import time
from typing import List, Optional

from app.rag.model_trace import BeforeModelCallHook, ModelCallTrace
from retrieval.embedding import (
    OPENAI_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_PROVIDER,
    OpenAIEmbedder,
)
from retrieval.models import RetrievalResult, VectorSearchCall
from retrieval.pgvector_store import PgVectorStore


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


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

        call = await self.search_with_trace(query, top_k)
        if call.error is not None:
            raise call.error
        return list(call.results)

    async def search_with_trace(
        self,
        query: str,
        top_k: int = 10,
        *,
        before_model_call: Optional[BeforeModelCallHook] = None,
    ) -> VectorSearchCall:
        """검색 결과와 함께 Query embedding 호출 관측값을 반환한다."""

        if top_k <= 0:
            raise ValueError("top_k는 1 이상이어야 합니다.")
        if not query.strip():
            return VectorSearchCall()

        if before_model_call is not None:
            await before_model_call(
                OPENAI_EMBEDDING_PROVIDER,
                OPENAI_EMBEDDING_MODEL,
                None,
            )

        started = time.perf_counter()
        try:
            response = await asyncio.to_thread(
                self._embedder.embed_many_with_usage,
                [query],
            )
        except Exception as error:
            return VectorSearchCall(
                latency_ms=_elapsed_ms(started),
                embedding_call=self._embedding_call(started, error=error),
                error=error,
            )

        embedding_call = self._embedding_call(
            started,
            input_tokens=response.input_tokens,
            retry_count=response.retry_count,
        )

        try:
            matches = await self._store.similarity_search(
                response.embeddings[0],
                top_k,
            )
        except Exception as error:
            return VectorSearchCall(
                latency_ms=_elapsed_ms(started),
                embedding_call=embedding_call,
                error=error,
            )

        return VectorSearchCall(
            results=tuple(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                )
                for rank, (chunk, score) in enumerate(matches, start=1)
            ),
            latency_ms=_elapsed_ms(started),
            embedding_call=embedding_call,
        )

    @staticmethod
    def _embedding_call(
        started: float,
        input_tokens: Optional[int] = None,
        retry_count: int = 0,
        error: Optional[Exception] = None,
    ) -> ModelCallTrace:
        return ModelCallTrace(
            provider=OPENAI_EMBEDDING_PROVIDER,
            model_name=OPENAI_EMBEDDING_MODEL,
            succeeded=error is None,
            latency_ms=_elapsed_ms(started),
            retry_count=retry_count,
            input_tokens=input_tokens,
            error_message=None if error is None else str(error),
        )
