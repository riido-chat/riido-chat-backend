import unittest
from unittest.mock import AsyncMock, Mock

from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import (
    CANDIDATE_K,
    DEFAULT_FINAL_TOP_K,
    RRF_RANK_CONSTANT,
    HybridRetriever,
    fuse_rrf_results,
)
from retrieval.models import RetrievalChunk, RetrievalResult
from retrieval.vector_retriever import VectorRetriever


class HybridRetrieverTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.bm25_retriever = Mock(spec=BM25Retriever)
        self.vector_retriever = AsyncMock(spec=VectorRetriever)
        self.retriever = HybridRetriever(
            self.bm25_retriever,
            self.vector_retriever,
        )

    async def test_fuses_candidates_by_chunk_id_with_rrf(self) -> None:
        chunk_a = self._chunk("a")
        chunk_b = self._chunk("b")
        chunk_c = self._chunk("c")
        self.bm25_retriever.search.return_value = [
            self._result(chunk_a, score=100.0, rank=1),
            self._result(chunk_b, score=50.0, rank=2),
        ]
        self.vector_retriever.search.return_value = [
            self._result(chunk_b, score=0.9, rank=1),
            self._result(chunk_c, score=0.8, rank=2),
        ]

        results = await self.retriever.search("사용자 질문", top_k=10)

        self.bm25_retriever.search.assert_called_once_with(
            "사용자 질문",
            top_k=CANDIDATE_K,
        )
        self.vector_retriever.search.assert_awaited_once_with(
            "사용자 질문",
            top_k=CANDIDATE_K,
        )
        self.assertEqual(
            [chunk_b, chunk_a, chunk_c],
            [result.chunk for result in results],
        )
        self.assertEqual([1, 2, 3], [result.final_rank for result in results])
        self.assertEqual([2, 1, None], [result.bm25_rank for result in results])
        self.assertEqual([1, None, 2], [result.vector_rank for result in results])
        self.assertAlmostEqual(
            1.0 / (RRF_RANK_CONSTANT + 2)
            + 1.0 / (RRF_RANK_CONSTANT + 1),
            results[0].rrf_score,
        )
        self.assertAlmostEqual(
            1.0 / (RRF_RANK_CONSTANT + 1),
            results[1].rrf_score,
        )

    async def test_uses_default_final_top_five(self) -> None:
        self.bm25_retriever.search.return_value = [
            self._result(self._chunk(str(index)), score=1.0, rank=index)
            for index in range(1, 11)
        ]
        self.vector_retriever.search.return_value = []

        results = await self.retriever.search("질문")

        self.assertEqual(DEFAULT_FINAL_TOP_K, len(results))
        self.assertEqual(
            list(range(1, DEFAULT_FINAL_TOP_K + 1)),
            [result.final_rank for result in results],
        )

    async def test_allows_top_ten_for_evaluation(self) -> None:
        self.bm25_retriever.search.return_value = [
            self._result(self._chunk(str(index)), score=1.0, rank=index)
            for index in range(1, 11)
        ]
        self.vector_retriever.search.return_value = []

        results = await self.retriever.search("평가 질문", top_k=10)

        self.assertEqual(10, len(results))
        self.assertEqual(list(range(1, 11)), [result.final_rank for result in results])

    async def test_rejects_invalid_top_k_before_search(self) -> None:
        for top_k in (0, -1):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, "top_k"):
                    await self.retriever.search("질문", top_k=top_k)

        self.bm25_retriever.search.assert_not_called()
        self.vector_retriever.search.assert_not_awaited()

    def test_keeps_different_chunk_ids_with_same_section_id(self) -> None:
        bm25_chunk = self._chunk("a", section_id="shared-section")
        vector_chunk = self._chunk("b", section_id="shared-section")

        results = fuse_rrf_results(
            [self._result(bm25_chunk, score=1.0, rank=1)],
            [self._result(vector_chunk, score=1.0, rank=1)],
            top_k=2,
        )

        self.assertEqual(
            ["chunk-a", "chunk-b"],
            [result.chunk.chunk_id for result in results],
        )

    def test_uses_chunk_id_for_deterministic_rrf_ties(self) -> None:
        chunk_a = self._chunk("a")
        chunk_b = self._chunk("b")

        results = fuse_rrf_results(
            [self._result(chunk_b, score=100.0, rank=1)],
            [self._result(chunk_a, score=-100.0, rank=1)],
            top_k=2,
        )

        self.assertEqual(
            ["chunk-a", "chunk-b"],
            [result.chunk.chunk_id for result in results],
        )
        self.assertEqual(results[0].rrf_score, results[1].rrf_score)

    @staticmethod
    def _result(
        chunk: RetrievalChunk,
        score: float,
        rank: int,
    ) -> RetrievalResult:
        return RetrievalResult(chunk=chunk, score=score, rank=rank)

    @staticmethod
    def _chunk(
        name: str,
        section_id: str = "",
    ) -> RetrievalChunk:
        return RetrievalChunk(
            chunk_id=f"chunk-{name}",
            document_id=f"document-{name}",
            section_id=section_id or f"section-{name}",
            document_title=f"문서 {name}",
            section_path=(f"문서 {name}", f"섹션 {name}"),
            source_url=f"https://docs.riido.io/document-{name}.md",
            category="guide",
            content=f"본문 {name}",
        )


if __name__ == "__main__":
    unittest.main()
