import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.create_labeling_candidates import (
    create_labeling_candidates,
    parse_labeling_candidates,
    save_labeling_candidates,
)


class LabelingCandidatesTest(unittest.TestCase):
    def test_parses_only_required_candidate_fields(self) -> None:
        text = """================================================================================
질문 ID: Q01
질문: 테스트 질문인가요?

rank: 1
score: 12.345678
document_title: 테스트 문서
section_path: 테스트 문서 > 테스트 섹션
source_url: https://example.com/test.md
content:
본문에 rank: 99라는 문자열이 있어도 결과로 읽지 않습니다.
================================================================================
"""

        questions = parse_labeling_candidates(text)

        self.assertEqual(
            [
                {
                    "question_id": "Q01",
                    "question": "테스트 질문인가요?",
                    "candidates": [
                        {
                            "rank": 1,
                            "document_title": "테스트 문서",
                            "section_path": "테스트 문서 > 테스트 섹션",
                            "score": 12.345678,
                        }
                    ],
                }
            ],
            questions,
        )

    def test_extracts_all_questions_and_top_ten_candidates(self) -> None:
        questions = create_labeling_candidates()

        self.assertEqual(30, len(questions))
        self.assertTrue(
            all(len(question["candidates"]) == 10 for question in questions)
        )
        self.assertEqual(
            list(range(1, 11)),
            [candidate["rank"] for candidate in questions[0]["candidates"]],
        )

    def test_raises_when_input_has_no_questions(self) -> None:
        with self.assertRaises(ValueError):
            parse_labeling_candidates("검색 결과가 없는 문자열")

    def test_saves_utf8_json_without_content_and_source_url(self) -> None:
        questions = create_labeling_candidates()

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "candidates.json"
            saved_path = save_labeling_candidates(questions, output_path)
            saved_questions = json.loads(
                saved_path.read_text(encoding="utf-8")
            )

        self.assertEqual(questions, saved_questions)
        self.assertEqual(
            {"rank", "document_title", "section_path", "score"},
            set(saved_questions[0]["candidates"][0]),
        )


if __name__ == "__main__":
    unittest.main()
