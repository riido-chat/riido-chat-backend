"""BM25와 Vector 검색 결과를 RRF로 결합한다."""

import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

from app.core.model_trace import BeforeModelCallHook
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.models import (
    HybridRetrievalResult,
    HybridSearchCall,
    RetrievalChunk,
    RetrievalResult,
)
from app.retrieval.vector_retriever import VectorRetriever


CANDIDATE_K = 10
DEFAULT_FINAL_TOP_K = 5
RRF_RANK_CONSTANT = 60


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def fuse_rrf_results(
    bm25_results: Sequence[RetrievalResult],
    vector_results: Sequence[RetrievalResult],
    top_k: int = DEFAULT_FINAL_TOP_K,
) -> List[HybridRetrievalResult]:
    """두 검색 결과를 chunk_id 기준으로 병합해 RRF 순위를 만든다."""

    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    chunks: Dict[int, RetrievalChunk] = {}
    bm25_ranks: Dict[int, int] = {}
    vector_ranks: Dict[int, int] = {}

    index_version_ids = {
        result.chunk.index_version_id
        for result in (*bm25_results, *vector_results)
    }
    if None in index_version_ids:
        raise ValueError("Hybrid Retrieval에는 DB 식별자가 있는 Chunk가 필요합니다.")
    if len(index_version_ids) > 1:
        raise ValueError("BM25와 Vector의 ACTIVE index version이 일치하지 않습니다.")

    for result in bm25_results:
        chunk_id = result.chunk.chunk_id
        if chunk_id is None:
            raise ValueError("BM25 결과에 신규 document_chunks.id가 없습니다.")
        chunks.setdefault(chunk_id, result.chunk)
        bm25_ranks.setdefault(chunk_id, result.rank)

    for result in vector_results:
        chunk_id = result.chunk.chunk_id
        if chunk_id is None:
            raise ValueError("Vector 결과에 신규 document_chunks.id가 없습니다.")
        existing = chunks.get(chunk_id)
        if existing is not None and (
            existing.document_version_id != result.chunk.document_version_id
            or existing.section_id != result.chunk.section_id
        ):
            raise ValueError("동일 Chunk PK의 문서 식별 정보가 일치하지 않습니다.")
        chunks.setdefault(chunk_id, result.chunk)
        vector_ranks.setdefault(chunk_id, result.rank)

    def rrf_score(chunk_id: int) -> float:
        score = 0.0
        bm25_rank = bm25_ranks.get(chunk_id)
        vector_rank = vector_ranks.get(chunk_id)

        if bm25_rank is not None:
            score += 1.0 / (RRF_RANK_CONSTANT + bm25_rank)
        if vector_rank is not None:
            score += 1.0 / (RRF_RANK_CONSTANT + vector_rank)

        return score

    ranked_chunk_ids = sorted(
        chunks,
        key=lambda chunk_id: (-rrf_score(chunk_id), chunk_id),
    )[:top_k]

    return [
        HybridRetrievalResult(
            chunk=chunks[chunk_id],
            rrf_score=rrf_score(chunk_id),
            final_rank=final_rank,
            bm25_rank=bm25_ranks.get(chunk_id),
            vector_rank=vector_ranks.get(chunk_id),
        )
        for final_rank, chunk_id in enumerate(ranked_chunk_ids, start=1)
    ]


class HybridRetriever:
    """기존 BM25와 Vector Retriever의 후보를 RRF로 결합한다."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
    ) -> None:
        self._bm25_retriever = bm25_retriever
        self._vector_retriever = vector_retriever

    async def search(
        self,
        query: str,
        top_k: int = DEFAULT_FINAL_TOP_K,
    ) -> List[HybridRetrievalResult]:
        """각 Retriever의 Top-10을 조회해 RRF 결과를 반환한다."""

        call = await self.search_with_trace(query, top_k=top_k)
        if call.error is not None:
            raise call.error
        return list(call.fused_results)

    async def search_with_trace(
        self,
        query: str,
        top_k: int = DEFAULT_FINAL_TOP_K,
        *,
        before_model_call: Optional[BeforeModelCallHook] = None,
    ) -> HybridSearchCall:
        """융합 결과와 함께 검색기별 후보 전체와 모델 호출 관측값을 반환한다.

        RAG 실행 로그는 융합에서 탈락한 후보까지 남겨야 검색 품질을 되짚을 수
        있으므로, 여기서 버리지 않고 호출자에게 그대로 올린다.
        """

        if top_k <= 0:
            raise ValueError("top_k는 1 이상이어야 합니다.")

        bm25_started = time.perf_counter()
        try:
            bm25_results = self._bm25_retriever.search(
                query,
                top_k=CANDIDATE_K,
            )
        except Exception as error:
            return HybridSearchCall(
                bm25_latency_ms=_elapsed_ms(bm25_started),
                error=error,
            )
        bm25_latency_ms = _elapsed_ms(bm25_started)

        vector_search_kwargs = {}
        if before_model_call is not None:
            vector_search_kwargs["before_model_call"] = before_model_call
        vector_call = await self._vector_retriever.search_with_trace(
            query,
            top_k=CANDIDATE_K,
            **vector_search_kwargs,
        )
        partial = HybridSearchCall(
            bm25_results=tuple(bm25_results),
            vector_results=vector_call.results,
            bm25_latency_ms=bm25_latency_ms,
            vector_latency_ms=vector_call.latency_ms,
            embedding_call=vector_call.embedding_call,
        )
        if vector_call.error is not None:
            return replace(partial, error=vector_call.error)

        try:
            fused_results = fuse_rrf_results(
                bm25_results,
                vector_call.results,
                top_k=top_k,
            )
        except Exception as error:
            return replace(partial, error=error)

        return replace(partial, fused_results=tuple(fused_results))
