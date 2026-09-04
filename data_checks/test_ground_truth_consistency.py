"""Ground truth가 가리키는 섹션이 실제 corpus에 존재하고 section_id가
corpus 전역에서 유일한지, 질문별 기대 섹션이 ground truth에 있는지 검증한다.
"""

import json
import unittest
from pathlib import Path

from app.retrieval.corpus import build_retrieval_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_PATH = PROJECT_ROOT / "evaluation/ground_truth.json"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation/bm25_questions.json"


class GroundTruthCorpusConsistencyCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ground_truth = json.loads(
            GROUND_TRUTH_PATH.read_text(encoding="utf-8")
        )
        cls.questions = json.loads(
            QUESTIONS_PATH.read_text(encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
