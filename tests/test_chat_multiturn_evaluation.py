import unittest

from evaluation.run_chat_multiturn_evaluation import (
    DEFAULT_CASES_PATH,
    DEFAULT_CS_CASES_PATH,
    answer_lead_paragraph,
    answer_lead_sentence,
    evaluate_turn,
    load_cases,
    recheck_saved_results,
    response_answer_markdown,
    response_citation_titles,
    select_cases,
    selected_turn_nos,
)


class ChatMultiTurnEvaluationTest(unittest.TestCase):
    def test_loads_versioned_fixed_cases(self) -> None:
        payload = load_cases(DEFAULT_CASES_PATH)

        self.assertEqual("v3", payload["version"])
        self.assertEqual(
            [f"MT{number:02d}" for number in range(1, 19)],
            [case["id"] for case in payload["cases"]],
        )

    def test_loads_versioned_cs_cases(self) -> None:
        payload = load_cases(DEFAULT_CS_CASES_PATH)

        self.assertEqual("cs-v2", payload["version"])
        self.assertEqual(
            [f"CS{number:02d}" for number in range(1, 16)],
            [case["id"] for case in payload["cases"]],
        )
        for case in payload["cases"]:
            self.assertEqual(1, len(case["turns"]))
            turn = case["turns"][0]
            self.assertTrue(turn["expectedAnswerConceptGroups"])
            self.assertTrue(turn["expectedCitationDocumentTitlesAny"])

    def test_extracts_selected_turn_numbers_from_v1_snapshot(self) -> None:
        self.assertEqual(
            [1, 3],
            selected_turn_nos(
                {
                    "schemaVersion": "v1",
                    "selectedTurns": [
                        {"turnNo": 1},
                        {"turnNo": 3},
                    ],
                }
            ),
        )
        self.assertEqual([], selected_turn_nos(None))

    def test_extracts_answer_and_citation_titles(self) -> None:
        response = {
            "answer": {"answerMarkdown": "쉬운 설명"},
            "citations": [
                {"documentTitle": "디스코드"},
                {"documentTitle": "핵심 개념"},
            ],
        }

        self.assertEqual("쉬운 설명", response_answer_markdown(response))
        self.assertEqual(
            ["디스코드", "핵심 개념"],
            response_citation_titles(response),
        )
        self.assertEqual(
            "첫 문단입니다.",
            answer_lead_paragraph("첫 문단입니다.\n\n두 번째 문단입니다."),
        )
        self.assertEqual(
            "첫 문장입니다.",
            answer_lead_sentence("첫 문장입니다. 두 번째 문장입니다."),
        )

    def test_selects_requested_cases_in_requested_order(self) -> None:
        payload = load_cases(DEFAULT_CASES_PATH)

        selected = select_cases(payload, ["MT09", "MT01", "MT09"])

        self.assertEqual(
            ["MT09", "MT01"],
            [case["id"] for case in selected["cases"]],
        )

    def test_accepts_matching_api_and_db_follow_up(self) -> None:
        failures = evaluate_turn(
            {
                "expectedStatus": "COMPLETED",
                "expectedContextStrategy": "FOLLOW_UP_WINDOW",
                "expectedSelectedTurnNos": [1],
                "resolvedQueryKeywords": ["스프린트", "설정"],
            },
            200,
            {"status": "COMPLETED"},
            {
                "status": "COMPLETED",
                "contextStrategy": "FOLLOW_UP_WINDOW",
                "selectedTurnNos": [1],
                "resolvedQuery": "스프린트 설정 방법",
            },
        )

        self.assertEqual([], failures)

    def test_reports_context_and_resolved_query_mismatches(self) -> None:
        failures = evaluate_turn(
            {
                "expectedStatus": "COMPLETED",
                "expectedContextStrategy": "FOLLOW_UP_WINDOW",
                "expectedSelectedTurnNos": [1],
                "resolvedQueryKeywords": ["스프린트"],
            },
            200,
            {"status": "COMPLETED"},
            {
                "status": "COMPLETED",
                "contextStrategy": "NEW_TOPIC",
                "selectedTurnNos": [],
                "resolvedQuery": "설정 방법",
            },
        )

        self.assertEqual(3, len(failures))

    def test_accepts_grounded_cs_answer_with_expected_concepts(self) -> None:
        failures = evaluate_turn(
            {
                "expectedStatus": "COMPLETED",
                "expectedDefinitionSentenceConceptGroups": [
                    ["소통", "커뮤니케이션"]
                ],
                "expectedAnswerConceptGroups": [
                    ["뤼이도"],
                    ["알림", "업데이트"],
                ],
                "minimumCitationCount": 1,
                "expectedCitationDocumentTitlesAny": ["디스코드"],
            },
            200,
            {
                "status": "COMPLETED",
                "answer": {
                    "answerMarkdown": (
                        "디스코드는 팀 소통 도구입니다. "
                        "뤼이도 업데이트를 디스코드 알림으로 받을 수 있어요."
                    )
                },
                "citations": [{"documentTitle": "디스코드"}],
            },
            {
                "status": "COMPLETED",
                "contextStrategy": "NEW_TOPIC",
                "selectedTurnNos": [],
                "resolvedQuery": "디스코드는 뭐야?",
            },
        )

        self.assertEqual([], failures)

    def test_reports_missing_cs_concept_and_wrong_citation(self) -> None:
        failures = evaluate_turn(
            {
                "expectedStatus": "COMPLETED",
                "expectedAnswerConceptGroups": [["연동"]],
                "minimumCitationCount": 1,
                "expectedCitationDocumentTitlesAny": ["디스코드"],
            },
            200,
            {
                "status": "COMPLETED",
                "answer": {"answerMarkdown": "메신저입니다."},
                "citations": [{"documentTitle": "슬랙"}],
            },
            {
                "status": "COMPLETED",
                "contextStrategy": "NEW_TOPIC",
                "selectedTurnNos": [],
                "resolvedQuery": "디스코드는 뭐야?",
            },
        )

        self.assertEqual(2, len(failures))

    def test_reports_missing_definition_in_first_sentence(self) -> None:
        failures = evaluate_turn(
            {
                "expectedStatus": "COMPLETED",
                "expectedDefinitionSentenceConceptGroups": [
                    ["팀", "사람"],
                    ["소통", "커뮤니케이션"],
                ],
            },
            200,
            {
                "status": "COMPLETED",
                "answer": {
                    "answerMarkdown": (
                        "디스코드는 뤼이도와 연동해서 사용하는 도구입니다."
                    )
                },
                "citations": [{"documentTitle": "디스코드"}],
            },
            {
                "status": "COMPLETED",
                "contextStrategy": "NEW_TOPIC",
                "selectedTurnNos": [],
                "resolvedQuery": "디스코드는 뭐야?",
            },
        )

        self.assertEqual(2, len(failures))

    def test_rechecks_saved_snapshot_without_model_call(self) -> None:
        cases = {
            "version": "test-v2",
            "cases": [
                {
                    "id": "MT01",
                    "description": "수정된 기대값",
                    "turns": [
                        {
                            "question": "질문",
                            "expectedStatus": "COMPLETED",
                            "resolvedQueryKeywords": ["구독", "취소"],
                        }
                    ],
                }
            ],
        }
        saved = {
            "casesVersion": "test-v1",
            "summary": {},
            "results": [
                {
                    "id": "MT01",
                    "description": "과거 기대값",
                    "passed": False,
                    "turns": [
                        {
                            "turnNo": 1,
                            "question": "질문",
                            "passed": False,
                            "failures": ["과거 실패"],
                            "httpStatus": 200,
                            "response": {"status": "COMPLETED"},
                            "db": {
                                "status": "COMPLETED",
                                "contextStrategy": "NEW_TOPIC",
                                "selectedTurnNos": [],
                                "resolvedQuery": "구독을 취소하면 어떻게 되나요?",
                            },
                        }
                    ],
                }
            ],
        }

        rechecked = recheck_saved_results(cases, saved)

        self.assertEqual("test-v2", rechecked["casesVersion"])
        self.assertEqual(1, rechecked["summary"]["passedCaseCount"])
        self.assertTrue(rechecked["results"][0]["passed"])


if __name__ == "__main__":
    unittest.main()
    answer_lead_paragraph,
