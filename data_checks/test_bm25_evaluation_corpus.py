"""BM25 평가 파이프라인이 실제 corpus 위에서 동작하는지 검증한다.
질문 30개의 기대 문서·섹션 실존, retrieval chunk 142개 생성, 결과 출력 내용.
"""

import io
import unittest
from contextlib import redirect_stdout

from evaluation.run_bm25_evaluation import (
    load_questions,
    print_evaluation_results,
)
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.corpus import build_retrieval_chunks


class BM25EvaluationCorpusCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.questions = load_questions()
        cls.retrieval_chunks = build_retrieval_chunks()
        cls.retriever = BM25Retriever(cls.retrieval_chunks)

    def test_builds_retrieval_chunks_for_canonical_corpus(self) -> None:
        self.assertEqual(142, len(self.retrieval_chunks))
        self.assertTrue(
            all(chunk.document_title for chunk in self.retrieval_chunks)
        )
        self.assertTrue(all(chunk.source_url for chunk in self.retrieval_chunks))

    def test_expected_document_sections_exist_in_canonical_corpus(self) -> None:
        document_sections = {
            (chunk.document_title, chunk.section_path[-1])
            for chunk in self.retrieval_chunks
        }

        for question in self.questions:
            expected = (
                question["expected_document"],
                question["expected_section"],
            )
            self.assertIn(expected, document_sections, question["id"])

    def test_prints_question_and_top_ten_result_fields(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            print_evaluation_results(self.questions[:1], self.retriever)

        text = output.getvalue()
        self.assertIn("질문 ID: Q01", text)
        self.assertIn(self.questions[0]["question"], text)
        self.assertEqual(10, text.count("\nrank: "))
        self.assertIn("score:", text)
        self.assertIn("section_id:", text)
        self.assertIn("document_title:", text)
        self.assertIn("section_path:", text)
        self.assertIn("source_url:", text)
        self.assertEqual(10, text.count("\ncontent:\n"))
        first_result = self.retriever.search(
            self.questions[0]["question"],
            top_k=10,
        )[0]
        self.assertIn(first_result.chunk.content, text)


if __name__ == "__main__":
    unittest.main()
