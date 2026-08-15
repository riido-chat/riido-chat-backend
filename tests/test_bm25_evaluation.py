import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.run_bm25_evaluation import (
    load_questions,
    print_evaluation_results,
    save_evaluation_results,
)
from retrieval.bm25_retriever import BM25Retriever
from retrieval.corpus import build_retrieval_chunks


class BM25EvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.questions = load_questions()
        cls.retrieval_chunks = build_retrieval_chunks()
        cls.retriever = BM25Retriever(cls.retrieval_chunks)

    def test_loads_thirty_questions_with_required_fields(self) -> None:
        required_fields = {
            "id",
            "question",
            "expected_document",
            "expected_section",
            "evaluation_purpose",
        }

        self.assertEqual(30, len(self.questions))
        self.assertEqual(
            len(self.questions),
            len({question["id"] for question in self.questions}),
        )
        self.assertTrue(
            all(set(question) == required_fields for question in self.questions)
        )

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

    def test_saves_evaluation_results_as_utf8_text(self) -> None:
        result_text = "질문 ID: Q01\ncontent:\n실제 청크 내용\n"

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "evaluation_result.txt"
            saved_path = save_evaluation_results(result_text, output_path)

            self.assertEqual(output_path, saved_path)
            self.assertEqual(
                result_text,
                saved_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
