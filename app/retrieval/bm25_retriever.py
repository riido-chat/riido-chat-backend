"""Kiwi 토큰과 BM25Okapi를 사용하는 메모리 기반 검색기."""

from typing import List, Optional, Sequence

from rank_bm25 import BM25Okapi

from app.retrieval.analyzer import KiwiAnalyzer
from app.retrieval.models import RetrievalChunk, RetrievalResult


class BM25Retriever:
    """RetrievalChunk를 색인하고 BM25 점수순으로 검색한다."""

    def __init__(
        self,
        chunks: Sequence[RetrievalChunk],
        analyzer: Optional[KiwiAnalyzer] = None,
    ) -> None:
        if not chunks:
            raise ValueError("BM25 index를 생성하려면 Chunk가 하나 이상 필요합니다.")

        self._chunks = tuple(chunks)
        self._analyzer = analyzer if analyzer is not None else KiwiAnalyzer()
        tokenized_corpus = [
            self._analyzer.tokenize(self._create_search_text(chunk))
            for chunk in self._chunks
        ]

        if not any(tokenized_corpus):
            raise ValueError("BM25 index를 생성할 검색 토큰이 없습니다.")

        self._index = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Query와 관련된 Chunk를 BM25 점수순으로 최대 top_k개 반환한다."""

        if top_k <= 0:
            raise ValueError("top_k는 1 이상이어야 합니다.")

        query_tokens = self._analyzer.tokenize(query)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(self._chunks)),
            key=lambda index: (-float(scores[index]), index),
        )[:top_k]

        return [
            RetrievalResult(
                chunk=self._chunks[index],
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(ranked_indices, start=1)
        ]

    @staticmethod
    def _create_search_text(chunk: RetrievalChunk) -> str:
        parts = [chunk.document_title, *chunk.section_path[1:], chunk.content]
        return "\n".join(part for part in parts if part)
