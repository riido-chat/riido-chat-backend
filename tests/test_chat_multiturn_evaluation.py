import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from app.answering.models import (
    GenerationAnswerType,
    GenerationAnswerScope,
    GenerationContextSource,
    GenerationEvidenceRequirement,
    GenerationResult,
    GenerationSourcePlan,
    GenerationStageTrace,
    GenerationStatus,
)
from app.core.model_trace import ModelCallTrace
from app.retrieval.models import RetrievalChunk
from evaluation.run_chat_multiturn_evaluation import (
    DEFAULT_ANSWER_CONSISTENCY_CASES_PATH,
    DEFAULT_CASES_PATH,
    DEFAULT_CS_CASES_PATH,
    answer_lead_paragraph,
    answer_lead_sentence,
    build_parser,
    evaluate_turn,
    generation_stage_trace_snapshot,
    load_cases,
    recheck_saved_results,
    response_answer_markdown,
    response_citation_titles,
    run_case_with_executor,
    select_cases,
    selected_turn_nos,
    summarize_runs,
    unique_output_path,
)


class ChatMultiTurnEvaluationTest(unittest.IsolatedAsyncioTestCase):
    def test_loads_balanced_answer_consistency_cases(self) -> None:
        payload = load_cases(DEFAULT_ANSWER_CONSISTENCY_CASES_PATH)

        self.assertEqual("answer-consistency-v1", payload["version"])
        self.assertEqual(
            [f"AC{number:02d}" for number in range(1, 11)],
            [case["id"] for case in payload["cases"]],
        )
        self.assertEqual(
            {
                "NEW_CONVERSATION": 6,
                "CONTINUOUS_CONVERSATION": 2,
                "FIXED_CONTEXT": 2,
            },
            {
                mode: sum(case["mode"] == mode for case in payload["cases"])
                for mode in {
                    "NEW_CONVERSATION",
                    "CONTINUOUS_CONVERSATION",
                    "FIXED_CONTEXT",
                }
            },
        )

    def test_loads_versioned_fixed_cases(self) -> None:
        payload = load_cases(DEFAULT_CASES_PATH)

        self.assertEqual("v3", payload["version"])
        self.assertEqual(
            [f"MT{number:02d}" for number in range(1, 19)],
            [case["id"] for case in payload["cases"]],
        )

    def test_loads_versioned_cs_cases(self) -> None:
        payload = load_cases(DEFAULT_CS_CASES_PATH)

        self.assertEqual("cs-v3", payload["version"])
        self.assertEqual(
            [f"CS{number:02d}" for number in range(1, 26)],
            [case["id"] for case in payload["cases"]],
        )
        for case in payload["cases"]:
            self.assertEqual(1, len(case["turns"]))
            turn = case["turns"][0]
            self.assertTrue(turn["expectedAnswerConceptGroups"])
            self.assertTrue(turn["expectedCitationDocumentTitlesAny"])

        definition_case_count = sum(
            "expectedDefinitionSentenceConceptGroups" in case["turns"][0]
            for case in payload["cases"]
        )
        self.assertGreaterEqual(definition_case_count, 10)

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

    def test_parses_repeat_and_target_revision(self) -> None:
        args = build_parser().parse_args(
            [
                "--repeat",
                "5",
                "--execution-mode",
                "in-process",
                "--target-repository-revision",
                "server-sha",
            ]
        )

        self.assertEqual(5, args.repeat)
        self.assertEqual("in-process", args.execution_mode)
        self.assertEqual("server-sha", args.target_repository_revision)

    def test_serializes_generation_stage_trace_for_evaluation(self) -> None:
        plan = GenerationSourcePlan(
            status=GenerationStatus.ANSWERABLE,
            answer_type=GenerationAnswerType.PROCEDURE,
            answer_scope=GenerationAnswerScope.SUMMARY,
            evidence_requirements=[
                GenerationEvidenceRequirement(
                    information_unit="설정 방법",
                    source_ids=["SOURCE_1"],
                )
            ],
            withheld_reason=None,
        )
        generated = GenerationResult(
            status=GenerationStatus.ANSWERABLE,
            answer_markdown="답변 [SOURCE_1]",
            withheld_reason=None,
        )
        source = GenerationContextSource(
            source_id="SOURCE_1",
            chunk=RetrievalChunk(
                document_id="document-1",
                section_id="section-1",
                document_title="문서",
                section_path=("문서", "섹션"),
                source_url="https://docs.riido.io/1",
                category="guide",
                content="근거 본문",
                chunk_id=1,
                document_version_id=2,
                index_version_id=3,
            ),
        )

        snapshot = generation_stage_trace_snapshot(
            GenerationStageTrace(
                source_plan=plan,
                initial_source_plan=plan,
                selected_sources=(source,),
                pre_validation_result=generated,
                validation_error="잘못된 marker",
                validation_errors=("잘못된 marker",),
                planning_attempt_count=2,
                planning_regeneration_count=1,
                planning_regeneration_model_call=ModelCallTrace(
                    provider="openai",
                    model_name="gpt-5.4-mini",
                    succeeded=True,
                    latency_ms=90,
                    retry_count=0,
                    input_tokens=8,
                    output_tokens=12,
                    prompt_version="v10-plan-repair-1",
                ),
                planning_regeneration_result=plan,
                answer_attempt_count=2,
                validation_regeneration_count=1,
                validation_regeneration_model_call=ModelCallTrace(
                    provider="openai",
                    model_name="gpt-5.4-mini",
                    succeeded=True,
                    latency_ms=100,
                    retry_count=0,
                    input_tokens=10,
                    output_tokens=20,
                    prompt_version="v11-repair-1",
                ),
                validation_regeneration_result=generated,
            )
        )

        self.assertEqual("ANSWERABLE", snapshot["planningOutput"]["status"])
        self.assertEqual("SOURCE_1", snapshot["selectedSources"][0]["sourceId"])
        self.assertEqual(
            "답변 [SOURCE_1]",
            snapshot["preValidationResult"]["answer_markdown"],
        )
        self.assertEqual("잘못된 marker", snapshot["validationError"])
        self.assertEqual(["잘못된 marker"], snapshot["validationErrors"])
        self.assertEqual(2, snapshot["planningAttemptCount"])
        self.assertEqual(1, snapshot["planningRegenerationCount"])
        self.assertEqual(
            "ANSWERABLE",
            snapshot["initialPlanningOutput"]["status"],
        )
        self.assertEqual(
            "SOURCE_PLAN_REGENERATION",
            snapshot["planningRegenerationModelCall"]["purpose"],
        )
        self.assertEqual(
            "ANSWERABLE",
            snapshot["planningRegenerationResult"]["status"],
        )
        self.assertEqual(2, snapshot["answerAttemptCount"])
        self.assertEqual(1, snapshot["validationRegenerationCount"])
        self.assertEqual(
            "ANSWER_VALIDATION_REGENERATION",
            snapshot["validationRegenerationModelCall"]["purpose"],
        )
        self.assertEqual(
            "답변 [SOURCE_1]",
            snapshot["validationRegenerationResult"]["answer_markdown"],
        )

    async def test_passes_in_process_trace_to_db_snapshot(self) -> None:
        trace = GenerationStageTrace()
        execute_turn = AsyncMock(
            return_value=(
                200,
                {
                    "status": "COMPLETED",
                    "conversationId": "conversation-id",
                    "ragRunId": "rag-run-id",
                },
                trace,
            )
        )
        db_snapshot = {
            "status": "COMPLETED",
            "contextStrategy": "NEW_TOPIC",
            "selectedTurnNos": [],
            "resolvedQuery": "질문",
        }
        case = {
            "id": "CS01",
            "turns": [{"question": "질문", "expectedStatus": "COMPLETED"}],
        }

        with patch(
            "evaluation.run_chat_multiturn_evaluation.load_db_snapshot",
            new=AsyncMock(return_value=db_snapshot),
        ) as load_snapshot:
            result = await run_case_with_executor(execute_turn, case)

        self.assertTrue(result["passed"])
        load_snapshot.assert_awaited_once_with("rag-run-id", trace)

    def test_creates_unique_timestamped_output_when_file_exists(self) -> None:
        with TemporaryDirectory() as directory:
            requested = Path(directory) / "result.json"
            requested.touch()

            actual = unique_output_path(
                requested,
                now=datetime(2026, 9, 5, 1, 2, 3, tzinfo=timezone.utc),
            )

        self.assertEqual("result-20260905T010203Z.json", actual.name)

    def test_summarizes_repeated_case_executions(self) -> None:
        summary = summarize_runs(
            [
                {
                    "repeatNo": 1,
                    "results": [
                        {"id": "CS01", "passed": True},
                        {"id": "CS02", "passed": False},
                    ],
                },
                {
                    "repeatNo": 2,
                    "results": [
                        {"id": "CS01", "passed": False},
                        {"id": "CS02", "passed": True},
                    ],
                },
            ]
        )

        self.assertEqual(2, summary["repetitionCount"])
        self.assertEqual(4, summary["totalCaseExecutionCount"])
        self.assertEqual(2, summary["passedCaseExecutionCount"])
        self.assertEqual(
            [
                {
                    "id": "CS01",
                    "executionCount": 2,
                    "passedCount": 1,
                    "failedCount": 1,
                },
                {
                    "id": "CS02",
                    "executionCount": 2,
                    "passedCount": 1,
                    "failedCount": 1,
                },
            ],
            summary["perCase"],
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

    def test_accepts_matching_planning_status(self) -> None:
        failures = evaluate_turn(
            {
                "expectedStatus": "WITHHELD",
                "expectedPlanningStatus": "WITHHELD",
            },
            200,
            {"status": "WITHHELD"},
            {
                "status": "WITHHELD",
                "contextStrategy": "NEW_TOPIC",
                "selectedTurnNos": [],
                "resolvedQuery": "질문",
                "stageTrace": {
                    "generation": {
                        "planningOutput": {
                            "availability": "AVAILABLE",
                            "value": {"status": "WITHHELD"},
                        }
                    }
                },
            },
        )

        self.assertEqual([], failures)

    def test_reports_planning_status_mismatch(self) -> None:
        failures = evaluate_turn(
            {
                "expectedStatus": "WITHHELD",
                "expectedPlanningStatus": "WITHHELD",
            },
            200,
            {"status": "WITHHELD"},
            {
                "status": "WITHHELD",
                "contextStrategy": "NEW_TOPIC",
                "selectedTurnNos": [],
                "resolvedQuery": "질문",
                "stageTrace": {
                    "generation": {
                        "planningOutput": {
                            "availability": "AVAILABLE",
                            "value": {"status": "ANSWERABLE"},
                        }
                    }
                },
            },
        )

        self.assertEqual(1, len(failures))
        self.assertIn("Planning status 불일치", failures[0])

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

    def test_rechecks_each_repeat_without_collapsing_same_case_id(self) -> None:
        cases = {
            "version": "test-v2",
            "cases": [
                {
                    "id": "CS01",
                    "turns": [
                        {
                            "question": "질문",
                            "expectedStatus": "COMPLETED",
                        }
                    ],
                }
            ],
        }
        saved_turn = {
            "turnNo": 1,
            "question": "질문",
            "httpStatus": 200,
            "response": {"status": "COMPLETED"},
            "db": {
                "status": "COMPLETED",
                "contextStrategy": "NEW_TOPIC",
                "selectedTurnNos": [],
                "resolvedQuery": "질문",
            },
        }
        saved = {
            "schemaVersion": "v2",
            "casesVersion": "test-v1",
            "summary": {},
            "runs": [
                {
                    "repeatNo": repeat_no,
                    "results": [
                        {
                            "id": "CS01",
                            "passed": False,
                            "turns": [dict(saved_turn)],
                        }
                    ],
                }
                for repeat_no in (1, 2)
            ],
        }

        rechecked = recheck_saved_results(cases, saved)

        self.assertEqual(2, len(rechecked["runs"]))
        self.assertEqual(2, rechecked["summary"]["passedCaseExecutionCount"])
        self.assertTrue(rechecked["runs"][0]["results"][0]["passed"])
        self.assertTrue(rechecked["runs"][1]["results"][0]["passed"])


if __name__ == "__main__":
    unittest.main()
    answer_lead_paragraph,
