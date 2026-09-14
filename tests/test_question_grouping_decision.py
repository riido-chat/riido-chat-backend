import json
import unittest
import uuid
from typing import Any, Dict, Optional

from app.core.model_trace import ModelCallTrace
from app.database.models import (
    AttributionSource,
    ClassificationDecision,
    QuestionProblemGroupKind,
    QuestionSubproblemServingState,
)
from app.question_grouping.constants import (
    INVALID_EMPTY_OUTPUT,
    INVALID_UNKNOWN_SUBPROBLEM_KEY,
    INVALID_UNPARSEABLE_OUTPUT,
)
from app.question_grouping.decision import (
    DOC_VIOLATION_CANDIDATE_ID,
    DOC_VIOLATION_DECISION,
    DOC_VIOLATION_RATIONALE_CODE,
    SUBPROBLEM_VIOLATION_GROUP_ID,
    SUBPROBLEM_VIOLATION_SUBPROBLEM_ID,
    build_turn_judgment,
    criteria_report,
    normalize_judgment,
)
from app.question_grouping.models import (
    DocumentCandidate,
    DocumentDecision,
    DocumentOutline,
    JudgeCall,
    JudgeFailure,
    JudgeFailureKind,
    SubproblemCandidate,
    SubproblemCatalogItem,
)
from app.question_grouping.payload import build_judge_presentation


def _subproblem(index: int) -> SubproblemCandidate:
    return SubproblemCandidate(
        item=SubproblemCatalogItem(
            subproblem_id=uuid.UUID(int=index),
            key=f"notifications.sp-{index}",
            name=f"세부 문제 {index}",
            inclusion_criteria=("포함 하나", "포함 둘"),
            exclusion_criteria=("제외 하나",),
            current_version=3,
            problem_group_id=uuid.UUID(int=100 + index),
            document_source_id=10 + index,
            document_key=f"notifications/doc-{index}",
            serving_state=QuestionSubproblemServingState.SERVING,
        ),
        similarity=0.5,
        retrieval_rank=index,
    )


def _document(index: int) -> DocumentCandidate:
    return DocumentCandidate(
        outline=DocumentOutline(
            document_source_id=50 + index,
            document_version_id=70 + index,
            document_key=f"docs/doc-{index}",
            title=f"문서 {index}",
            parent_path="docs",
            headings=(),
        ),
        score=0.01,
        retrieval_rank=index,
    )


PRESENTATION = build_judge_presentation(
    "알림을 끄고 싶어요",
    [_subproblem(1), _subproblem(2)],
    [_document(1), _document(2)],
    seed="decision-test",
)
SP1 = PRESENTATION.subproblem_by_key()["notifications.sp-1"]
SP2 = PRESENTATION.subproblem_by_key()["notifications.sp-2"]
D1 = PRESENTATION.document_by_id()["D1"]


def _connect(**overrides: Any) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "decision": "CONNECT",
        "groupId": SP1.document_key,
        "subproblemId": SP1.key,
        "confidence": 0.82,
        "rationaleCode": "OUTCOME_MATCH",
        "matchedCriteria": ["I1"],
        "conflictingCriteria": [f"{SP2.key}:OUTCOME"],
        "ambiguityReason": None,
        "documentDecision": "SUBPROBLEM_DOCUMENT",
        "documentCandidateId": None,
        "documentRationaleCode": None,
    }
    output.update(overrides)
    return output


def _separate(**overrides: Any) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "decision": "SEPARATE",
        "groupId": None,
        "subproblemId": None,
        "confidence": 0.6,
        "rationaleCode": "NO_CANDIDATE",
        "matchedCriteria": [],
        "conflictingCriteria": [f"{SP1.key}:QUALIFIER", f"{SP2.key}:E1"],
        "ambiguityReason": None,
        "documentDecision": "MATCHED",
        "documentCandidateId": "D1",
        "documentRationaleCode": "ASK_COVERED",
    }
    output.update(overrides)
    return output


def _normalize(output: Optional[Dict[str, Any]]):
    return normalize_judgment(
        None if output is None else json.dumps(output, ensure_ascii=False),
        PRESENTATION,
    )


def _trace(succeeded: bool = True) -> ModelCallTrace:
    return ModelCallTrace(
        provider="openai",
        model_name="gpt-5.6-luna",
        succeeded=succeeded,
        latency_ms=10,
    )


class InvalidOutputTest(unittest.TestCase):
    def test_empty_output_is_invalid(self) -> None:
        for text in (None, "", "   \n"):
            with self.subTest(text=text):
                result = normalize_judgment(text, PRESENTATION)

                self.assertFalse(result.valid)
                self.assertEqual(INVALID_EMPTY_OUTPUT, result.invalid_reason)
                self.assertEqual(ClassificationDecision.UNCLASSIFIED, result.decision)
                self.assertIsNone(result.raw_output)

    def test_unparseable_output_is_invalid(self) -> None:
        for text in ('{"decision": "CONNECT"', "[1, 2]", '"CONNECT"'):
            with self.subTest(text=text):
                result = normalize_judgment(text, PRESENTATION)

                self.assertEqual(INVALID_UNPARSEABLE_OUTPUT, result.invalid_reason)
                self.assertIsNone(result.subproblem)

    def test_unknown_decision_value_is_unparseable(self) -> None:
        result = _normalize(_connect(decision="MAYBE"))

        self.assertEqual(INVALID_UNPARSEABLE_OUTPUT, result.invalid_reason)
        self.assertEqual("MAYBE", result.raw_output["decision"])

    def test_connect_to_unpresented_key_is_invalid(self) -> None:
        for key in ("notifications.sp-9", None, str(SP1.subproblem_id)):
            with self.subTest(key=key):
                result = _normalize(_connect(subproblemId=key))

                self.assertEqual(INVALID_UNKNOWN_SUBPROBLEM_KEY, result.invalid_reason)
                self.assertEqual(ClassificationDecision.UNCLASSIFIED, result.decision)
                self.assertIsNone(result.subproblem)
                self.assertEqual(key, result.raw_output["subproblemId"])


class ConnectNormalizationTest(unittest.TestCase):
    def test_clean_connect_maps_key_to_presented_subproblem(self) -> None:
        result = _normalize(_connect())

        self.assertTrue(result.valid)
        self.assertEqual(ClassificationDecision.CONNECT, result.decision)
        self.assertEqual(SP1, result.subproblem)
        self.assertEqual(DocumentDecision.SUBPROBLEM_DOCUMENT, result.document_decision)
        self.assertIsNone(result.document)
        self.assertTrue(result.document_fields_ignored)
        self.assertEqual((), result.document_field_violations)
        self.assertFalse(result.group_id_mismatch)
        self.assertEqual(0.82, result.confidence)
        self.assertEqual("OUTCOME_MATCH", result.rationale_code)

    def test_connect_ignores_document_fields_and_records_violations(self) -> None:
        result = _normalize(
            _connect(
                documentDecision="MATCHED",
                documentCandidateId="D2",
                documentRationaleCode="ASK_COVERED",
            )
        )

        self.assertTrue(result.valid)
        self.assertEqual(DocumentDecision.SUBPROBLEM_DOCUMENT, result.document_decision)
        self.assertIsNone(result.document)
        self.assertEqual(
            (
                DOC_VIOLATION_DECISION,
                DOC_VIOLATION_CANDIDATE_ID,
                DOC_VIOLATION_RATIONALE_CODE,
            ),
            result.document_field_violations,
        )
        self.assertFalse(result.unknown_document_id)

    def test_connect_group_mismatch_uses_subproblem_group(self) -> None:
        for group_id in (SP2.document_key, "unknown/doc", None):
            with self.subTest(group_id=group_id):
                result = _normalize(_connect(groupId=group_id))

                self.assertTrue(result.valid)
                self.assertTrue(result.group_id_mismatch)
                self.assertEqual(SP1, result.subproblem)

    def test_connect_with_bad_confidence_keeps_decision(self) -> None:
        for confidence in (1.4, -0.1, "high", True):
            with self.subTest(confidence=confidence):
                result = _normalize(_connect(confidence=confidence))

                self.assertTrue(result.valid)
                self.assertIsNone(result.confidence)


class SeparateNormalizationTest(unittest.TestCase):
    def test_matched_document_in_list_is_kept(self) -> None:
        result = _normalize(_separate())

        self.assertTrue(result.valid)
        self.assertEqual(ClassificationDecision.SEPARATE, result.decision)
        self.assertIsNone(result.subproblem)
        self.assertEqual(DocumentDecision.MATCHED, result.document_decision)
        self.assertEqual(D1, result.document)
        self.assertEqual((), result.document_field_violations)
        self.assertFalse(result.document_fields_ignored)

    def test_matched_document_not_in_list_becomes_none(self) -> None:
        for candidate_id in ("D9", "docs/doc-1", None):
            with self.subTest(candidate_id=candidate_id):
                result = _normalize(_separate(documentCandidateId=candidate_id))

                self.assertTrue(result.valid)
                self.assertEqual(DocumentDecision.NONE, result.document_decision)
                self.assertIsNone(result.document)
                self.assertTrue(result.unknown_document_id)
                self.assertIn(DOC_VIOLATION_CANDIDATE_ID, result.document_field_violations)

    def test_matched_with_wrong_rationale_keeps_document(self) -> None:
        result = _normalize(_separate(documentRationaleCode="NO_CANDIDATE_COVERS"))

        self.assertEqual(DocumentDecision.MATCHED, result.document_decision)
        self.assertEqual(D1, result.document)
        self.assertEqual((DOC_VIOLATION_RATIONALE_CODE,), result.document_field_violations)

    def test_none_document_is_kept(self) -> None:
        result = _normalize(
            _separate(
                decision="UNCLASSIFIED",
                ambiguityReason="OUT_OF_SCOPE",
                documentDecision="NONE",
                documentCandidateId=None,
                documentRationaleCode="NO_CANDIDATE_COVERS",
            )
        )

        self.assertEqual(ClassificationDecision.UNCLASSIFIED, result.decision)
        self.assertTrue(result.valid)
        self.assertEqual(DocumentDecision.NONE, result.document_decision)
        self.assertEqual("OUT_OF_SCOPE", result.ambiguity_reason)
        self.assertEqual((), result.document_field_violations)

    def test_subproblem_document_without_connect_becomes_none(self) -> None:
        result = _normalize(
            _separate(
                documentDecision="SUBPROBLEM_DOCUMENT",
                documentCandidateId="D1",
                documentRationaleCode=None,
            )
        )

        self.assertEqual(DocumentDecision.NONE, result.document_decision)
        self.assertIsNone(result.document)
        self.assertEqual(
            (DOC_VIOLATION_DECISION, DOC_VIOLATION_CANDIDATE_ID),
            result.document_field_violations,
        )

    def test_none_with_candidate_id_ignores_the_id(self) -> None:
        result = _normalize(
            _separate(documentDecision="NONE", documentRationaleCode="TOO_VAGUE_TO_PLACE")
        )

        self.assertEqual(DocumentDecision.NONE, result.document_decision)
        self.assertIsNone(result.document)
        self.assertEqual((DOC_VIOLATION_CANDIDATE_ID,), result.document_field_violations)
        self.assertFalse(result.unknown_document_id)

    def test_unknown_document_decision_becomes_none(self) -> None:
        result = _normalize(_separate(documentDecision="MAYBE", documentCandidateId=None))

        self.assertTrue(result.valid)
        self.assertEqual(DocumentDecision.NONE, result.document_decision)
        self.assertEqual((DOC_VIOLATION_DECISION,), result.document_field_violations)

    def test_non_connect_subproblem_ids_are_ignored(self) -> None:
        result = _normalize(
            _separate(decision="UNCLASSIFIED", groupId=SP1.document_key, subproblemId=SP1.key)
        )

        self.assertTrue(result.valid)
        self.assertEqual(ClassificationDecision.UNCLASSIFIED, result.decision)
        self.assertIsNone(result.subproblem)
        self.assertEqual(
            (SUBPROBLEM_VIOLATION_GROUP_ID, SUBPROBLEM_VIOLATION_SUBPROBLEM_ID),
            result.subproblem_field_violations,
        )


class CriteriaReportTest(unittest.TestCase):
    def test_counts_malformed_and_unknown_entries_without_invalidating(self) -> None:
        output = _connect(
            matchedCriteria=["I1", "I3", "포함 하나", 7],
            conflictingCriteria=[
                f"{SP2.key}:OUTCOME",
                f"{SP2.key}:E1",
                f"{SP2.key}:E2",
                "notifications.sp-9:TIE",
                f"{SP2.key}: 이유를 서술",
            ],
        )

        result = _normalize(output)
        report = result.criteria_report

        self.assertTrue(result.valid)
        self.assertEqual(
            {"entries": 4, "malformed": 2, "unknownReference": 1},
            report["matchedCriteria"],
        )
        self.assertEqual(
            {"entries": 5, "malformed": 1, "unknownReference": 2},
            report["conflictingCriteria"],
        )
        self.assertEqual(6, report["violations"])
        self.assertEqual(["I1"], report["matched"])
        self.assertEqual([f"{SP2.key}:OUTCOME", f"{SP2.key}:E1"], report["conflicting"])
        self.assertEqual({"E": 1, "OUTCOME": 1}, report["conflictCodes"])

    def test_report_keeps_no_free_text(self) -> None:
        report = criteria_report(
            _separate(matchedCriteria="I1", conflictingCriteria=["알림 질문 원문"]),
            PRESENTATION,
        )

        self.assertEqual(0, report["matchedCriteria"]["entries"])
        self.assertEqual(1, report["conflictingCriteria"]["malformed"])
        self.assertNotIn("알림 질문 원문", json.dumps(report, ensure_ascii=False))

    def test_invalid_output_still_reports_when_object_parsed(self) -> None:
        result = _normalize(_connect(subproblemId="notifications.sp-9"))

        self.assertFalse(result.valid)
        self.assertIsNotNone(result.criteria_report)
        normalization = result.normalization_judgment_input()
        self.assertEqual(INVALID_UNKNOWN_SUBPROBLEM_KEY, normalization["invalidReason"])
        json.dumps(normalization)


class TurnJudgmentTest(unittest.TestCase):
    def test_connect_attributes_to_subproblem_group(self) -> None:
        call = JudgeCall(trace=_trace(), output_text=json.dumps(_connect()))

        turn = build_turn_judgment(call, PRESENTATION)

        self.assertFalse(turn.failed)
        self.assertEqual(ClassificationDecision.CONNECT, turn.decision)
        self.assertEqual(SP1, turn.subproblem)
        self.assertEqual(3, turn.subproblem_version)
        self.assertEqual(AttributionSource.SUBPROBLEM, turn.attribution.attribution_source)
        self.assertEqual(SP1.problem_group_id, turn.attribution.problem_group_id)

    def test_matched_attributes_to_document(self) -> None:
        call = JudgeCall(trace=_trace(), output_text=json.dumps(_separate()))

        turn = build_turn_judgment(call, PRESENTATION)

        self.assertEqual(AttributionSource.DOCUMENT, turn.attribution.attribution_source)
        self.assertEqual(QuestionProblemGroupKind.DOCUMENT, turn.attribution.group_kind)
        self.assertEqual(D1.document_source_id, turn.attribution.document_source_id)
        self.assertIsNone(turn.subproblem_version)

    def test_unknown_document_falls_back_to_no_document(self) -> None:
        call = JudgeCall(
            trace=_trace(),
            output_text=json.dumps(_separate(documentCandidateId="D7")),
        )

        turn = build_turn_judgment(call, PRESENTATION)

        self.assertFalse(turn.failed)
        self.assertEqual(AttributionSource.NONE, turn.attribution.attribution_source)
        self.assertEqual(QuestionProblemGroupKind.NO_DOCUMENT, turn.attribution.group_kind)

    def test_call_failure_is_unclassified_without_normalization(self) -> None:
        failure = JudgeFailure(JudgeFailureKind.API_ERROR, "OpenAI 판별 호출 실패: HTTP 500")
        call = JudgeCall(trace=_trace(False), failure=failure)

        turn = build_turn_judgment(call, PRESENTATION)

        self.assertTrue(turn.failed)
        self.assertIs(failure, turn.failure)
        self.assertIsNone(turn.normalized)
        self.assertEqual(ClassificationDecision.UNCLASSIFIED, turn.decision)
        self.assertEqual(AttributionSource.NONE, turn.attribution.attribution_source)

    def test_invalid_output_is_a_judgment_failure_that_keeps_raw_output(self) -> None:
        for text, reason in (
            ("", INVALID_EMPTY_OUTPUT),
            (json.dumps(_connect(subproblemId="x")), INVALID_UNKNOWN_SUBPROBLEM_KEY),
        ):
            with self.subTest(reason=reason):
                turn = build_turn_judgment(
                    JudgeCall(trace=_trace(), output_text=text), PRESENTATION
                )

                self.assertTrue(turn.failed)
                self.assertEqual(JudgeFailureKind.INVALID_OUTPUT, turn.failure.kind)
                self.assertIn(reason, turn.failure.safe_message)
                self.assertEqual(reason, turn.normalized.invalid_reason)
                self.assertIsNone(turn.subproblem)
                self.assertEqual(
                    QuestionProblemGroupKind.NO_DOCUMENT, turn.attribution.group_kind
                )


if __name__ == "__main__":
    unittest.main()
