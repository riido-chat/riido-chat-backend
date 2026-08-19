import unittest
from unittest.mock import AsyncMock

from evaluation.run_vector_evaluation import run_vector_evaluation
from retrieval.models import RetrievalChunk, RetrievalResult
from retrieval.vector_retriever import VectorRetriever


class VectorEvaluationTest(unittest.IsolatedAsyncioTestCase):
    async def test_converts_async_results_to_existing_candidate_shape(
        self,
    ) -> None:
        retriever = AsyncMock(spec=VectorRetriever)
        retriever.search.return_value = [
            RetrievalResult(
                chunk=RetrievalChunk(
                    chunk_id="chunk-1",
                    document_id="document-1",
                    section_id="section-1",
                    document_title="테스트 문서",
                    section_path=("테스트 문서", "테스트 섹션"),
                    source_url="https://example.com/test.md",
                    category="guide",
                    content="테스트 본문",
                ),
                score=0.875,
                rank=1,
            )
        ]
        questions = [
            {
                "id": "Q01",
                "question": "테스트 질문인가요?",
            }
        ]

        candidate_items = await run_vector_evaluation(
            questions,
            retriever,
        )

        retriever.search.assert_awaited_once_with(
            "테스트 질문인가요?",
            top_k=10,
        )
        self.assertEqual(
            [
                {
                    "question_id": "Q01",
                    "question": "테스트 질문인가요?",
                    "candidates": [
                        {
                            "rank": 1,
                            "section_id": "section-1",
                            "document_title": "테스트 문서",
                            "section_path": "테스트 문서 > 테스트 섹션",
                            "score": 0.875,
                        }
                    ],
                }
            ],
            candidate_items,
        )


if __name__ == "__main__":
    unittest.main()
