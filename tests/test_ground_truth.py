import json
import unittest
from pathlib import Path

from retrieval.corpus import build_retrieval_chunks


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

    def test_relevant_sections_exist_in_canonical_corpus(self) -> None:
        corpus_section_ids = {
            chunk.section_id for chunk in build_retrieval_chunks()
        }

        for item in self.ground_truth:
            for section in item["relevant_sections"]:
                self.assertIn(
                    section["section_id"],
                    corpus_section_ids,
                    item["question_id"],
                )

    def test_section_ids_are_unique_in_canonical_corpus(self) -> None:
        chunks = build_retrieval_chunks()

        self.assertEqual(
            len(chunks),
            len({chunk.section_id for chunk in chunks}),
        )

    def test_includes_each_questions_expected_section(self) -> None:
        chunks = build_retrieval_chunks()
        ground_truth_by_id = {
            item["question_id"]: {
                section["section_id"] for section in item["relevant_sections"]
            }
            for item in self.ground_truth
        }

        for question in self.questions:
            matches = [
                chunk
                for chunk in chunks
                if chunk.document_title == question["expected_document"]
                and chunk.section_path[-1] == question["expected_section"]
            ]
            self.assertEqual(1, len(matches), question["id"])
            self.assertIn(
                matches[0].section_id,
                ground_truth_by_id[question["id"]],
                question["id"],
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
