import json
import unittest
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_PATH = PROJECT_ROOT / "evaluation/ground_truth.json"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation/bm25_questions.json"
CANDIDATES_PATH = (
    PROJECT_ROOT / "evaluation/retrieval_labeling_candidates.json"
)


class GroundTruthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ground_truth = json.loads(
            GROUND_TRUTH_PATH.read_text(encoding="utf-8")
        )
        cls.questions = json.loads(
            QUESTIONS_PATH.read_text(encoding="utf-8")
        )
        cls.candidates = json.loads(
            CANDIDATES_PATH.read_text(encoding="utf-8")
        )

    def test_has_ground_truth_for_every_question(self) -> None:
        question_ids = [question["id"] for question in self.questions]
        ground_truth_ids = [
            item["question_id"] for item in self.ground_truth
        ]

        self.assertEqual(30, len(self.ground_truth))
        self.assertEqual(question_ids, ground_truth_ids)
        self.assertEqual(len(ground_truth_ids), len(set(ground_truth_ids)))

    def test_uses_section_id_with_human_readable_fields(self) -> None:
        candidate_identity_fields = {"section_id"}
        display_fields = {"document_title", "section_path"}
        actual_candidate_fields = set(
            self.candidates[0]["candidates"][0]
        )

        self.assertTrue(
            candidate_identity_fields.issubset(actual_candidate_fields)
        )
        for item in self.ground_truth:
            self.assertEqual(
                {"question_id", "relevant_sections"},
                set(item),
            )
            self.assertTrue(item["relevant_sections"])
            for section in item["relevant_sections"]:
                self.assertEqual(
                    candidate_identity_fields | display_fields,
                    set(section),
                )




    def test_has_no_duplicate_sections_per_question(self) -> None:
        for item in self.ground_truth:
            identities = [
                section["section_id"]
                for section in item["relevant_sections"]
            ]
            self.assertEqual(
                len(identities),
                len(set(identities)),
                item["question_id"],
            )


if __name__ == "__main__":
    unittest.main()
