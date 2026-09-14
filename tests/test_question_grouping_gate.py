import json
import unittest
import uuid
from dataclasses import replace
from typing import Optional, Sequence

from app.database.models import (
    CacheAttemptOutcome,
    ClassificationDecision,
    QuestionSubproblemServingState,
    QuestionSubproblemStatus,
)
from app.question_grouping.attribution import (
    initial_attribution,
    no_document_attribution,
)
from app.question_grouping.constants import (
    REJECT_CANONICAL_ANSWER_NOT_FOUND,
    REJECT_CANONICAL_CITATION_MISSING,
    REJECT_CITED_DOCUMENT_NOT_INDEXED,
    REJECT_CITED_SECTION_CHANGED,
    REJECT_CLASSIFICATION_NOT_CONNECTED,
    REJECT_SUBPROBLEM_NOT_APPROVED,
    REJECT_SUBPROBLEM_STOPPED,
    REJECT_SUBPROBLEM_UNUSED,
    REJECT_SUBPROBLEM_VERSION_MISMATCH,
)
from app.question_grouping.gate import (
    evaluate_cache_gate,
    gate_judgment_input,
    resolve_citation,
    resolve_citations,
)
from app.question_grouping.models import (
    CanonicalCitationSnapshot,
    CitationIndexContext,
    CitationResolution,
    GateCanonicalAnswer,
    GateSubproblemState,
    IndexedSection,
    JudgeFailure,
    JudgeFailureKind,
    PresentedSubproblem,
    TurnJudgment,
)


SUBPROBLEM_ID = uuid.UUID(int=1)
CANONICAL_ID = uuid.UUID(int=2)
PRESENTED = PresentedSubproblem(
    key="billing.cancel",
    subproblem_id=SUBPROBLEM_ID,
    subproblem_version=2,
    problem_group_id=uuid.UUID(int=3),
    document_source_id=7,
    document_key="workspaces/plans-and-billing",
    canonical_answer_id=CANONICAL_ID,
    similarity=0.7,
    retrieval_rank=1,
    presented_order=1,
    inclusion_count=1,
    exclusion_count=0,
)


def _citation(
    order: int = 1,
    *,
    chunk_id: int = 101,
    version_id: int = 11,
    source_id: int = 7,
    content_hash: str = "content-a",
    identity_hash: Optional[str] = "identity-a",
    node_order: int = 2,
) -> CanonicalCitationSnapshot:
    return CanonicalCitationSnapshot(
        citation_order=order,
        chunk_id=chunk_id,
        document_version_id=version_id,
        document_source_id=source_id,
        content_hash=content_hash,
        node_order=node_order,
        node_identity_hash=identity_hash,
    )


def _section(
    chunk_id: int,
    *,
    version_id: int = 12,
    content_hash: str = "content-a",
    identity_hash: Optional[str] = "identity-a",
    node_order: int = 2,
) -> IndexedSection:
    return IndexedSection(
        chunk_id=chunk_id,
        document_version_id=version_id,
        content_hash=content_hash,
        node_order=node_order,
        node_identity_hash=identity_hash,
        document_title="구독 및 결제",
        node_path="구독 및 결제 > 구독 취소",
        source_uri="https://docs.riido.io/workspaces/plans-and-billing",
    )


def _context(version_id: Optional[int], *sections: IndexedSection) -> CitationIndexContext:
    return CitationIndexContext(indexed_document_version_id=version_id, sections=sections)


class ResolveCitationTest(unittest.TestCase):
    def test_step1_same_version_and_chunk_is_used_as_is(self) -> None:
        section = _section(101, version_id=11)

        result = resolve_citation(_citation(), _context(11, _section(102, version_id=11), section))

        self.assertTrue(result.passed)
        self.assertEqual(1, result.step)
        self.assertIs(section, result.section)

    def test_step2_new_version_with_same_identity_and_content(self) -> None:
        moved = _section(201, identity_hash="identity-a", node_order=5)
        decoy = _section(202, identity_hash="identity-b", node_order=2)

        result = resolve_citation(_citation(), _context(12, decoy, moved))

        self.assertEqual(2, result.step)
        self.assertEqual(201, result.section.chunk_id)

    def test_step2_handles_changed_chunking_config_of_same_version(self) -> None:
        rechunked = _section(301, version_id=11)

        result = resolve_citation(_citation(), _context(11, rechunked))

        self.assertEqual(2, result.step)
        self.assertEqual(301, result.section.chunk_id)

    def test_step3_same_content_picks_closest_node_order(self) -> None:
        far = _section(401, identity_hash="renamed-1", node_order=9)
        near_after = _section(402, identity_hash="renamed-2", node_order=3)
        near_before = _section(403, identity_hash="renamed-3", node_order=1)

        result = resolve_citation(_citation(node_order=2), _context(12, far, near_after, near_before))

        self.assertEqual(3, result.step)
        self.assertEqual(403, result.section.chunk_id)

    def test_step3_when_old_identity_hash_is_missing(self) -> None:
        section = _section(501)

        result = resolve_citation(_citation(identity_hash=None), _context(12, section))

        self.assertEqual(3, result.step)

    def test_identity_match_with_changed_content_fails(self) -> None:
        changed = _section(601, content_hash="content-b")

        result = resolve_citation(_citation(), _context(12, changed))

        self.assertFalse(result.passed)
        self.assertEqual(REJECT_CITED_SECTION_CHANGED, result.rejection_reason)
        self.assertIsNone(result.step)

    def test_document_not_in_index_fails(self) -> None:
        result = resolve_citation(_citation(), _context(None))

        self.assertEqual(REJECT_CITED_DOCUMENT_NOT_INDEXED, result.rejection_reason)

    def test_ignores_sections_of_other_versions(self) -> None:
        stray = _section(701, version_id=99)

        result = resolve_citation(_citation(), _context(12, stray))

        self.assertEqual(REJECT_CITED_SECTION_CHANGED, result.rejection_reason)

    def test_resolves_all_citations_in_order(self) -> None:
        citations = [
            _citation(2, chunk_id=102, source_id=8),
            _citation(1, chunk_id=101, source_id=7),
        ]

        results = resolve_citations(citations, {7: _context(12, _section(201))})

        self.assertEqual([1, 2], [item.citation_order for item in results])
        self.assertTrue(results[0].passed)
        self.assertEqual(REJECT_CITED_DOCUMENT_NOT_INDEXED, results[1].rejection_reason)


def _connect() -> TurnJudgment:
    return TurnJudgment(
        decision=ClassificationDecision.CONNECT,
        attribution=initial_attribution(ClassificationDecision.CONNECT, subproblem=PRESENTED),
        subproblem=PRESENTED,
    )


def _state(
    serving_state: QuestionSubproblemServingState = QuestionSubproblemServingState.SERVING,
    *,
    status: QuestionSubproblemStatus = QuestionSubproblemStatus.APPROVED,
    current_version: int = 2,
) -> GateSubproblemState:
    return GateSubproblemState(
        subproblem_id=SUBPROBLEM_ID,
        status=status,
        serving_state=serving_state,
        current_version=current_version,
    )


CANONICAL = GateCanonicalAnswer(
    canonical_answer_id=CANONICAL_ID,
    subproblem_version=2,
    content_markdown="구독은 설정에서 취소합니다 [1].",
)
PASSED = (CitationResolution(citation_order=1, section=_section(201), step=2),)


def _gate(
    judgment: Optional[TurnJudgment] = None,
    *,
    subproblem: Optional[GateSubproblemState] = None,
    canonical: Optional[GateCanonicalAnswer] = CANONICAL,
    resolutions: Sequence[CitationResolution] = PASSED,
    semantic_cache_enabled: bool = True,
    use_default_state: bool = True,
):
    return evaluate_cache_gate(
        judgment or _connect(),
        subproblem=subproblem if subproblem is not None or not use_default_state else _state(),
        canonical_answer=canonical,
        citation_resolutions=resolutions,
        semantic_cache_enabled=semantic_cache_enabled,
    )


class CacheGateTest(unittest.TestCase):
    def test_served_carries_canonical_and_resolved_citations(self) -> None:
        second = CitationResolution(citation_order=2, section=_section(202), step=1)

        result = _gate(resolutions=(second, PASSED[0]))

        self.assertEqual(CacheAttemptOutcome.SERVED, result.outcome)
        self.assertEqual(CANONICAL_ID, result.canonical_answer_id)
        self.assertEqual((), result.rejection_reasons)
        self.assertEqual([1, 2], [item.citation_order for item in result.served_citations])

    def test_judgment_failure_is_failed_first(self) -> None:
        failed = TurnJudgment(
            decision=ClassificationDecision.UNCLASSIFIED,
            attribution=no_document_attribution(),
            failure=JudgeFailure(JudgeFailureKind.API_ERROR, "HTTP 500"),
        )

        result = _gate(failed, canonical=None, resolutions=())

        self.assertEqual(CacheAttemptOutcome.FAILED, result.outcome)
        self.assertIsNone(result.canonical_answer_id)
        self.assertEqual((), result.rejection_reasons)

    def test_not_connected_is_rejected_before_other_checks(self) -> None:
        for decision in (ClassificationDecision.SEPARATE, ClassificationDecision.UNCLASSIFIED):
            with self.subTest(decision=decision):
                judgment = TurnJudgment(decision=decision, attribution=no_document_attribution())

                result = _gate(judgment, canonical=None, resolutions=())

                self.assertEqual(CacheAttemptOutcome.REJECTED, result.outcome)
                self.assertEqual((REJECT_CLASSIFICATION_NOT_CONNECTED,), result.rejection_reasons)
                self.assertIsNone(result.canonical_answer_id)

    def test_subproblem_missing_or_not_approved(self) -> None:
        cases = (
            None,
            _state(status=QuestionSubproblemStatus.ARCHIVED),
            replace(_state(), subproblem_id=uuid.UUID(int=99)),
        )
        for state in cases:
            with self.subTest(state=state):
                result = _gate(subproblem=state, use_default_state=False)

                self.assertEqual((REJECT_SUBPROBLEM_NOT_APPROVED,), result.rejection_reasons)

    def test_missing_canonical_answer(self) -> None:
        result = _gate(canonical=None, resolutions=())

        self.assertEqual(CacheAttemptOutcome.REJECTED, result.outcome)
        self.assertEqual((REJECT_CANONICAL_ANSWER_NOT_FOUND,), result.rejection_reasons)

    def test_version_and_citation_failures_are_collected(self) -> None:
        resolutions = (
            CitationResolution(citation_order=1, rejection_reason=REJECT_CITED_SECTION_CHANGED),
            CitationResolution(citation_order=2, rejection_reason=REJECT_CITED_DOCUMENT_NOT_INDEXED),
            CitationResolution(citation_order=3, rejection_reason=REJECT_CITED_SECTION_CHANGED),
            PASSED[0],
        )

        result = _gate(
            subproblem=_state(current_version=3),
            resolutions=resolutions,
        )

        self.assertEqual(CacheAttemptOutcome.REJECTED, result.outcome)
        self.assertEqual(
            (
                REJECT_SUBPROBLEM_VERSION_MISMATCH,
                REJECT_CITED_SECTION_CHANGED,
                REJECT_CITED_DOCUMENT_NOT_INDEXED,
            ),
            result.rejection_reasons,
        )
        self.assertIsNone(result.canonical_answer_id)

    def test_canonical_without_citations_is_rejected(self) -> None:
        result = _gate(resolutions=())

        self.assertEqual((REJECT_CANONICAL_CITATION_MISSING,), result.rejection_reasons)

    def test_version_check_precedes_shadow_and_serving_state(self) -> None:
        for serving_state in QuestionSubproblemServingState:
            with self.subTest(serving_state=serving_state):
                result = _gate(subproblem=_state(serving_state, current_version=5))

                self.assertEqual((REJECT_SUBPROBLEM_VERSION_MISMATCH,), result.rejection_reasons)

    def test_shadow_precedes_group_disabled(self) -> None:
        result = _gate(
            subproblem=_state(QuestionSubproblemServingState.SHADOW),
            semantic_cache_enabled=False,
        )

        self.assertEqual(CacheAttemptOutcome.SHADOW, result.outcome)
        self.assertEqual(CANONICAL_ID, result.canonical_answer_id)
        self.assertEqual((), result.served_citations)

    def test_unused_and_stopped_are_rejected_before_group_disabled(self) -> None:
        for serving_state, reason in (
            (QuestionSubproblemServingState.UNUSED, REJECT_SUBPROBLEM_UNUSED),
            (QuestionSubproblemServingState.STOPPED, REJECT_SUBPROBLEM_STOPPED),
        ):
            with self.subTest(serving_state=serving_state):
                result = _gate(subproblem=_state(serving_state), semantic_cache_enabled=False)

                self.assertEqual(CacheAttemptOutcome.REJECTED, result.outcome)
                self.assertEqual((reason,), result.rejection_reasons)
                self.assertIsNone(result.canonical_answer_id)

    def test_group_disabled_when_profile_cache_is_off(self) -> None:
        result = _gate(semantic_cache_enabled=False)

        self.assertEqual(CacheAttemptOutcome.GROUP_DISABLED, result.outcome)
        self.assertEqual(CANONICAL_ID, result.canonical_answer_id)
        self.assertEqual((), result.served_citations)

    def test_result_shapes_satisfy_cache_attempt_constraints(self) -> None:
        results = [
            _gate(),
            _gate(semantic_cache_enabled=False),
            _gate(subproblem=_state(QuestionSubproblemServingState.SHADOW)),
            _gate(subproblem=_state(QuestionSubproblemServingState.UNUSED)),
            _gate(canonical=None),
            _gate(
                TurnJudgment(
                    decision=ClassificationDecision.UNCLASSIFIED,
                    attribution=no_document_attribution(),
                    failure=JudgeFailure(JudgeFailureKind.INVALID_OUTPUT, "invalid"),
                )
            ),
        ]
        with_canonical = {
            CacheAttemptOutcome.SERVED,
            CacheAttemptOutcome.SHADOW,
            CacheAttemptOutcome.GROUP_DISABLED,
        }
        for result in results:
            with self.subTest(outcome=result.outcome):
                self.assertEqual(
                    result.outcome in with_canonical,
                    result.canonical_answer_id is not None,
                )
                self.assertEqual(
                    result.outcome == CacheAttemptOutcome.REJECTED,
                    len(result.rejection_reasons) > 0,
                )
                self.assertTrue(all(len(reason) <= 50 for reason in result.rejection_reasons))



class GateJudgmentInputTest(unittest.TestCase):
    def test_served_uses_result_canonical_and_citations(self) -> None:
        second = CitationResolution(citation_order=2, section=_section(202), step=3)
        result = _gate(resolutions=(second, PASSED[0]))

        value = gate_judgment_input(result)

        self.assertEqual("SERVED", value["outcome"])
        self.assertEqual(str(CANONICAL_ID), value["canonicalAnswerId"])
        self.assertEqual([], value["rejectionReasons"])
        self.assertEqual(
            [
                {
                    "citationOrder": 1,
                    "step": PASSED[0].step,
                    "rejectionReason": None,
                    "chunkId": PASSED[0].section.chunk_id,
                    "documentVersionId": PASSED[0].section.document_version_id,
                },
                {"citationOrder": 2, "step": 3, "rejectionReason": None, "chunkId": 202, "documentVersionId": 12},
            ],
            value["citationResolutions"],
        )
        json.dumps(value)

    def test_rejected_keeps_read_canonical_id_and_failed_resolutions(self) -> None:
        failed = CitationResolution(citation_order=1, rejection_reason=REJECT_CITED_SECTION_CHANGED)
        result = _gate(resolutions=(failed,))
        self.assertEqual(CacheAttemptOutcome.REJECTED, result.outcome)
        self.assertIsNone(result.canonical_answer_id)

        value = gate_judgment_input(result, citation_resolutions=(failed,), canonical_answer_id=CANONICAL_ID)

        self.assertEqual(
            {
                "outcome": "REJECTED",
                "canonicalAnswerId": str(CANONICAL_ID),
                "rejectionReasons": [REJECT_CITED_SECTION_CHANGED],
                "citationResolutions": [
                    {
                        "citationOrder": 1,
                        "step": None,
                        "rejectionReason": REJECT_CITED_SECTION_CHANGED,
                        "chunkId": None,
                        "documentVersionId": None,
                    }
                ],
            },
            value,
        )


if __name__ == "__main__":
    unittest.main()
