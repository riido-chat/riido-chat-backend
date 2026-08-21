import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from evaluation.evaluate_retrieval import (
    METRIC_KEYS,
    calculate_mrr_at_10,
    calculate_ndcg_at_10,
    calculate_recall_at_k,
    evaluate_retrieval,
    load_evaluation_data,
    save_evaluation_metrics,
)


def create_section(
    name: str,
    rank: int = 1,
    section_id: Optional[str] = None,
):
    return {
        "rank": rank,
        "section_id": section_id or name,
        "document_title": "문서",
        "section_path": f"문서 > {name}",
    }


class RetrievalMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.relevant_sections = [
            create_section("정답 A"),
            create_section("정답 B"),
            create_section("정답 C"),
        ]
        self.candidates = [
            create_section("오답", rank=1),
            create_section("정답 A", rank=2),
            create_section("다른 오답", rank=3),
            create_section("정답 B", rank=4),
        ]

    def test_calculates_recall_against_all_relevant_sections(self) -> None:
        self.assertEqual(
            0.0,
            calculate_recall_at_k(
                self.candidates, self.relevant_sections, 1
            ),
        )
        self.assertAlmostEqual(
            1.0 / 3.0,
            calculate_recall_at_k(
                self.candidates, self.relevant_sections, 3
            ),
        )
        self.assertAlmostEqual(
            2.0 / 3.0,
            calculate_recall_at_k(
                self.candidates, self.relevant_sections, 5
            ),
        )

    def test_calculates_mrr_from_first_relevant_section(self) -> None:
        self.assertEqual(
            0.5,
            calculate_mrr_at_10(
                self.candidates,
                self.relevant_sections,
            ),
        )

    def test_calculates_binary_ndcg(self) -> None:
        dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
        idcg = 1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4)

        self.assertAlmostEqual(
            dcg / idcg,
            calculate_ndcg_at_10(
                self.candidates,
                self.relevant_sections,
            ),
        )

    def test_returns_zero_when_top_ten_has_no_relevant_section(self) -> None:
        candidates = [create_section("오답", rank=1)]

        self.assertEqual(
            0.0,
            calculate_recall_at_k(candidates, self.relevant_sections, 10),
        )
        self.assertEqual(
            0.0,
            calculate_mrr_at_10(candidates, self.relevant_sections),
        )
        self.assertEqual(
            0.0,
            calculate_ndcg_at_10(candidates, self.relevant_sections),
        )

    def test_counts_duplicate_relevant_section_only_once(self) -> None:
        candidates = [
            create_section("정답 A", rank=1),
            create_section("정답 A", rank=2),
        ]

        self.assertAlmostEqual(
            1.0 / 3.0,
            calculate_recall_at_k(candidates, self.relevant_sections, 10),
        )
        self.assertAlmostEqual(
            1.0 / (
                1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4)
            ),
            calculate_ndcg_at_10(candidates, self.relevant_sections),
        )

    def test_distinguishes_same_display_path_by_section_id(self) -> None:
        candidates = [
            create_section("개요", rank=1, section_id="document-a:0"),
            create_section("개요", rank=2, section_id="document-b:0"),
        ]
        relevant_sections = [
            create_section("개요", section_id="document-b:0"),
        ]

        self.assertEqual(
            0.0,
            calculate_recall_at_k(candidates, relevant_sections, 1),
        )
        self.assertEqual(
            1.0,
            calculate_recall_at_k(candidates, relevant_sections, 2),
        )

    def test_evaluates_current_thirty_questions(self) -> None:
        candidates, ground_truth = load_evaluation_data()

        result = evaluate_retrieval(candidates, ground_truth)

        self.assertEqual(30, result["summary"]["question_count"])
        self.assertEqual(38, result["summary"]["relevant_section_count"])
        self.assertEqual(30, len(result["questions"]))
        self.assertEqual(
            set(METRIC_KEYS),
            set(result["summary"]["average_metrics"]),
        )
        self.assertTrue(
            all(
                0.0 <= metric <= 1.0
                for metric in result["summary"][
                    "average_metrics"
                ].values()
            )
        )
        self.assertEqual(
            [
                question["question_id"]
                for question in result["questions"]
                if question["metrics"]["recall_at_10"] == 0.0
            ],
            result["summary"]["top_10_miss_question_ids"],
        )

    def test_saves_evaluation_result_as_utf8_json(self) -> None:
        evaluation_result = {
            "summary": {"question_count": 1},
            "questions": [{"question_id": "Q01"}],
        }

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "metrics.json"
            saved_path = save_evaluation_metrics(
                evaluation_result,
                output_path,
            )
            saved_result = json.loads(
                saved_path.read_text(encoding="utf-8")
            )

        self.assertEqual(evaluation_result, saved_result)


if __name__ == "__main__":
    unittest.main()
