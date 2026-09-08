import unittest

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
    evaluate_result,
    load_frozen_turns,
    select_candidates,
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


if __name__ == "__main__":
    unittest.main()
