"""BM25와 Vector 검색 결과를 RRF로 결합한다."""

from typing import Dict, List, Sequence

from retrieval.bm25_retriever import BM25Retriever
from retrieval.models import (
    HybridRetrievalResult,
    RetrievalChunk,
    RetrievalResult,
)
from retrieval.vector_retriever import VectorRetriever


CANDIDATE_K = 10
DEFAULT_FINAL_TOP_K = 5
RRF_RANK_CONSTANT = 60


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

        if top_k <= 0:
            raise ValueError("top_k는 1 이상이어야 합니다.")

        bm25_results = self._bm25_retriever.search(
            query,
            top_k=CANDIDATE_K,
        )
        vector_results = await self._vector_retriever.search(
            query,
            top_k=CANDIDATE_K,
        )
        return fuse_rrf_results(
            bm25_results,
            vector_results,
            top_k=top_k,
        )
