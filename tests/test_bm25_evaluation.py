import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.run_bm25_evaluation import (
    load_questions,
    save_evaluation_results,
)


class BM25EvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.questions = load_questions()

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
