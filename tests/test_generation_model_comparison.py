import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from app.answering.models import (
    Citation,
    CitationSourceKind,
    FinalAnswerStatus,
    FinalGenerationResult,
    FinalWithheldReason,
)
from evaluation.run_generation_model_comparison import (
    CANDIDATE_CONFIGS,
    EXCLUDED_CASE_IDS,
    calculate_cost_usd,
    build_parser,
    evaluate_result,
    load_frozen_turns,
    run_comparison,
    select_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELATED_GUIDANCE_CASES_PATH = (
    PROJECT_ROOT / "evaluation/source_planning_related_guidance_cases.json"
)
CUSTOMER_SUPPORT_FIXTURE_PATH = PROJECT_ROOT / (
    "evaluation/baselines/customer-support-v3-final-v33-full.json"
)


class GenerationModelComparisonTest(unittest.TestCase):
    def test_defines_agreed_candidate_models(self) -> None:
        self.assertEqual(
            [
                ("A", "gpt-5.4-mini"),
                ("B", "gpt-5.6-luna"),
                ("C", "gpt-5.6-terra"),
                ("D", "gpt-5.6-sol"),
                ("E", "gpt-6-astra"),
            ],
            [(config.id, config.model) for config in CANDIDATE_CONFIGS],
        )

    def test_excludes_only_agreed_known_limit_cases(self) -> None:
        self.assertEqual({"MQ027", "MQ040", "MQ066"}, EXCLUDED_CASE_IDS)

    def test_loads_frozen_generation_inputs_without_known_limits(self) -> None:
        turns, skipped = load_frozen_turns()

        self.assertEqual(95, len(turns))
        self.assertNotIn(
            True,
            [turn.case_id in EXCLUDED_CASE_IDS for turn in turns],
        )
        self.assertTrue(all(turn.retrieval_results for turn in turns))
        self.assertIn(
            {"caseId": "MQ078", "turnNo": 2, "reason": "GENERATION_NOT_REACHED"},
            skipped,
        )

    def test_loads_related_guidance_frozen_inputs(self) -> None:
        turns, skipped = load_frozen_turns(
            cases_path=RELATED_GUIDANCE_CASES_PATH,
            fixture_source_path=CUSTOMER_SUPPORT_FIXTURE_PATH,
            supplemental_fixture_paths=(),
            excluded_case_ids=(),
        )

        self.assertEqual(8, len(turns))
        self.assertEqual([], skipped)
        self.assertEqual(
            [
                ("RC006", 1),
                ("RC007", 1),
                ("RC018", 2),
                ("RC040", 1),
                ("RC012", 2),
                ("RC010", 1),
                ("RC023", 1),
                ("RC008", 1),
            ],
            [(turn.case_id, turn.turn_no) for turn in turns],
        )

    def test_parser_accepts_custom_frozen_inputs(self) -> None:
        args = build_parser().parse_args(
            [
                "--cases",
                str(RELATED_GUIDANCE_CASES_PATH),
                "--fixture-source",
                str(CUSTOMER_SUPPORT_FIXTURE_PATH),
                "--supplemental-fixtures",
                "--models",
                "C",
                "--repeat",
                "3",
            ]
        )

        self.assertEqual(RELATED_GUIDANCE_CASES_PATH, args.cases)
        self.assertEqual(CUSTOMER_SUPPORT_FIXTURE_PATH, args.fixture_source)
        self.assertEqual([], args.supplemental_fixtures)
        self.assertEqual(["C"], args.models)
        self.assertEqual(3, args.repeat)

    def test_comparison_accepts_project_relative_paths(self) -> None:
        turns, _ = load_frozen_turns(
            cases_path=RELATED_GUIDANCE_CASES_PATH,
            fixture_source_path=CUSTOMER_SUPPORT_FIXTURE_PATH,
            supplemental_fixture_paths=(),
            excluded_case_ids=(),
        )
        output_path = PROJECT_ROOT / "evaluation/unused-test-output.json"

        with patch(
            "evaluation.run_generation_model_comparison.get_settings"
        ) as settings:
            settings.return_value.openai_api_key = None
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                asyncio.run(
                    run_comparison(
                        candidates=[CANDIDATE_CONFIGS[2]],
                        turns=turns[:1],
                        repeat=1,
                        output_path=output_path,
                        cases_path=Path(
                            "evaluation/"
                            "source_planning_related_guidance_cases.json"
                        ),
                        fixture_source_path=Path(
                            "evaluation/baselines/"
                            "customer-support-v3-final-v33-full.json"
                        ),
                        supplemental_fixture_paths=(),
                    )
                )

    def test_selects_candidates_by_id_or_model(self) -> None:
        selected = select_candidates(["A", "gpt-5.6-terra", "A"])

        self.assertEqual(["A", "C"], [config.id for config in selected])

    def test_calculates_input_and_output_cost(self) -> None:
        config = CANDIDATE_CONFIGS[0]

        self.assertAlmostEqual(1.2, calculate_cost_usd(1_000_000, 100_000, config))

    def test_evaluates_status_concepts_and_citation(self) -> None:
        result = FinalGenerationResult(
            status=FinalAnswerStatus.COMPLETED,
            answer_markdown="워크스페이스는 하나의 조직입니다. [1]",
            citations=(
                Citation(
                    citation_number=1,
                    document_title="워크스페이스",
                    section_path=("개요",),
                    source_url="https://docs.riido.io/workspaces/overview.md",
                    source_kind=CitationSourceKind.GITBOOK,
                ),
            ),
        )
        expected = {
            "expectedStatus": "COMPLETED",
            "expectedAnswerConceptGroups": [["조직", "회사"]],
            "minimumCitationCount": 1,
            "expectedCitationDocumentTitlesAny": ["워크스페이스"],
        }

        self.assertEqual([], evaluate_result(expected, result))

    def test_reports_wrong_withheld_reason(self) -> None:
        result = FinalGenerationResult(
            status=FinalAnswerStatus.WITHHELD,
            answer_markdown="보류",
            citations=(),
            withheld_reason=FinalWithheldReason.OUT_OF_SCOPE,
        )

        failures = evaluate_result(
            {
                "expectedStatus": "WITHHELD",
                "expectedWithheldReason": "INSUFFICIENT_EVIDENCE",
            },
            result,
        )

        self.assertEqual(1, len(failures))
        self.assertIn("withheld reason", failures[0])

    def test_reports_wrong_planning_status_and_forbidden_phrase(self) -> None:
        result = FinalGenerationResult(
            status=FinalAnswerStatus.COMPLETED,
            answer_markdown="프로젝트를 백로그로 이동할 수 있습니다. [1]",
            citations=(),
        )

        failures = evaluate_result(
            {
                "expectedStatus": "COMPLETED",
                "expectedPlanningStatus": "RELATED_GUIDANCE",
                "forbiddenAnswerPhrases": [
                    "프로젝트를 백로그로 이동할 수 있습니다"
                ],
            },
            result,
        )

        self.assertEqual(2, len(failures))
        self.assertIn("planning status", failures[0])
        self.assertIn("금지 문구", failures[1])


if __name__ == "__main__":
    unittest.main()
