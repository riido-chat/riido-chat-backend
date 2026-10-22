import json
import unittest
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
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
    EXACT_SOURCE_CANONICAL_CHANGED,
    EXACT_SOURCE_CANONICAL_UNVERIFIED,
    EXACT_SOURCE_SUBPROBLEM_VERSION_STALE,
    REJECT_CANONICAL_ANSWER_CHANGED,
    REJECT_CANONICAL_ANSWER_NOT_FOUND,
    REJECT_CANONICAL_CITATION_MISSING,
    REJECT_CITED_DOCUMENT_DISABLED,
    REJECT_CITED_DOCUMENT_NOT_INDEXED,
    REJECT_CITED_SECTION_CHANGED,
    REJECT_CLASSIFICATION_NOT_CONNECTED,
    REJECT_SUBPROBLEM_DOCUMENT_DISABLED,
    REJECT_SUBPROBLEM_NOT_APPROVED,
    REJECT_SUBPROBLEM_STOPPED,
    REJECT_SUBPROBLEM_UNUSED,
    REJECT_SUBPROBLEM_VERSION_MISMATCH,
)
from app.question_grouping.gate import (
    evaluate_cache_gate,
    exact_source_fallthrough_reasons,
    gate_judgment_input,
    resolve_citation,
    resolve_citations,
)
from app.question_grouping.models import (
    CanonicalCitationSnapshot,
    CitationIndexContext,
    CitationResolution,
    ExactQuestionLogMatch,
    GateCanonicalAnswer,
    GateInputs,
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

    def test_disabled_document_fails_even_when_section_is_indexed(self) -> None:
        context = CitationIndexContext(
            indexed_document_version_id=11,
            sections=(_section(101, version_id=11),),
            document_enabled=False,
        )

        result = resolve_citation(_citation(), context)

        self.assertFalse(result.passed)
        self.assertEqual(REJECT_CITED_DOCUMENT_DISABLED, result.rejection_reason)

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


    def test_judge_presented_canonical_must_match_served_canonical(self) -> None:
        cases = (
            replace(PRESENTED, canonical_answer_id=uuid.UUID(int=77)),
            replace(PRESENTED, canonical_answer_id=None),
        )
        for presented in cases:
            with self.subTest(presented=presented.canonical_answer_id):
                judgment = replace(_connect(), subproblem=presented)

                result = _gate(judgment)

                self.assertEqual(CacheAttemptOutcome.REJECTED, result.outcome)
                self.assertEqual((REJECT_CANONICAL_ANSWER_CHANGED,), result.rejection_reasons)

    def test_judged_subproblem_version_must_be_current(self) -> None:
        judgment = replace(_connect(), subproblem=replace(PRESENTED, subproblem_version=1))

        result = _gate(judgment)

        self.assertEqual((REJECT_SUBPROBLEM_VERSION_MISMATCH,), result.rejection_reasons)

    def test_disabled_subproblem_document_is_rejected_before_serving_state(self) -> None:
        for serving_state in QuestionSubproblemServingState:
            with self.subTest(serving_state=serving_state):
                state = replace(_state(serving_state), document_enabled=False)

                result = _gate(subproblem=state)

                self.assertEqual(CacheAttemptOutcome.REJECTED, result.outcome)
                self.assertEqual((REJECT_SUBPROBLEM_DOCUMENT_DISABLED,), result.rejection_reasons)

    def test_disabled_cited_document_is_rejected(self) -> None:
        disabled = CitationResolution(citation_order=1, rejection_reason=REJECT_CITED_DOCUMENT_DISABLED)

        result = _gate(resolutions=(disabled,))

        self.assertEqual(CacheAttemptOutcome.REJECTED, result.outcome)
        self.assertEqual((REJECT_CITED_DOCUMENT_DISABLED,), result.rejection_reasons)


def _match(
    *,
    source_version: int = 2,
    current_version: int = 2,
    recorded: bool = False,
    canonical_id: Optional[uuid.UUID] = None,
    approved: bool = False,
    effective_from: Optional[datetime] = None,
) -> ExactQuestionLogMatch:
    return ExactQuestionLogMatch(
        subproblem_id=SUBPROBLEM_ID,
        key="billing.cancel",
        problem_group_id=uuid.UUID(int=3),
        current_version=current_version,
        document_source_id=7,
        document_key="workspaces/plans-and-billing",
        normalized_question="구독 취소",
        source_rag_run_id=uuid.UUID(int=10),
        classification_id=9,
        matched_count=1,
        source_subproblem_version=source_version,
        source_canonical_recorded=recorded,
        source_canonical_answer_id=canonical_id,
        source_exact_cache_approved=approved,
        source_effective_from=effective_from,
    )


def _inputs(
    *,
    current_version: int = 2,
    canonical: Optional[GateCanonicalAnswer] = CANONICAL,
) -> GateInputs:
    return GateInputs(subproblem=_state(current_version=current_version), canonical_answer=canonical)


class ExactSourceFallthroughTest(unittest.TestCase):
    def test_same_revision_and_recorded_canonical_is_reused(self) -> None:
        self.assertEqual((), exact_source_fallthrough_reasons(_match(recorded=True, canonical_id=CANONICAL_ID), _inputs()))

    def test_stale_subproblem_revision_falls_through(self) -> None:
        result = exact_source_fallthrough_reasons(
            _match(source_version=1, recorded=True, canonical_id=CANONICAL_ID), _inputs()
        )

        self.assertEqual((EXACT_SOURCE_SUBPROBLEM_VERSION_STALE,), result)

    def test_revision_is_compared_with_reread_state(self) -> None:
        # 조회 때 읽은 개정이 같아도 게이트 입력의 현재 개정이 바뀌었으면 낡은 것이다.
        result = exact_source_fallthrough_reasons(_match(current_version=2), _inputs(current_version=3))

        self.assertEqual((EXACT_SOURCE_SUBPROBLEM_VERSION_STALE,), result)

    def test_recorded_canonical_change_falls_through(self) -> None:
        cases = (
            (uuid.UUID(int=77), CANONICAL),
            (None, CANONICAL),
            (CANONICAL_ID, None),
        )
        for source_id, canonical in cases:
            with self.subTest(source_id=source_id, canonical=canonical):
                result = exact_source_fallthrough_reasons(
                    _match(recorded=True, canonical_id=source_id), _inputs(canonical=canonical)
                )

                self.assertEqual((EXACT_SOURCE_CANONICAL_CHANGED,), result)

    def test_unrecorded_canonical_with_applicability_rules_falls_through(self) -> None:
        ruled = replace(CANONICAL, applicability_rules=("환불 금액 문의는 다루지 않는다",))

        result = exact_source_fallthrough_reasons(_match(), _inputs(canonical=ruled))

        self.assertEqual((EXACT_SOURCE_CANONICAL_UNVERIFIED,), result)

    def test_unrecorded_canonical_without_rules_is_reused(self) -> None:
        for canonical in (CANONICAL, None):
            with self.subTest(canonical=canonical):
                self.assertEqual((), exact_source_fallthrough_reasons(_match(), _inputs(canonical=canonical)))

    def test_approved_source_after_ruled_canonical_creation_is_reused(self) -> None:
        # DEV 추천 질문 원천 모양(#220): 제시 목록 없음, 운영자 승인, 현재 정본에 규칙 2개,
        # 승인(effective_from)이 현재 정본 생성 뒤다. 같은 시각도 정본이 있던 때로 본다.
        created = datetime(2026, 9, 30, 9, 0, tzinfo=timezone.utc)
        ruled = replace(CANONICAL, applicability_rules=("규칙 1", "규칙 2"), created_at=created)
        for effective_from in (created, created + timedelta(hours=3)):
            with self.subTest(effective_from=effective_from):
                result = exact_source_fallthrough_reasons(
                    _match(approved=True, effective_from=effective_from), _inputs(canonical=ruled)
                )

                self.assertEqual((), result)

    def test_approved_source_before_ruled_canonical_creation_falls_through(self) -> None:
        # 승인 뒤에 정본이 새로 만들어졌으면 승인이 그 정본을 본 것이 아니다.
        created = datetime(2026, 9, 30, 9, 0, tzinfo=timezone.utc)
        ruled = replace(CANONICAL, applicability_rules=("규칙 1", "규칙 2"), created_at=created)

        result = exact_source_fallthrough_reasons(
            _match(approved=True, effective_from=created - timedelta(seconds=1)), _inputs(canonical=ruled)
        )

        self.assertEqual((EXACT_SOURCE_CANONICAL_UNVERIFIED,), result)

    def test_approval_time_rule_needs_operator_approval_and_timestamps(self) -> None:
        created = datetime(2026, 9, 30, 9, 0, tzinfo=timezone.utc)
        later = created + timedelta(hours=1)
        ruled = replace(CANONICAL, applicability_rules=("규칙 1",), created_at=created)
        cases = (
            ("승인 아님", _match(approved=False, effective_from=later), ruled),
            ("원천 시각 없음", _match(approved=True), ruled),
            ("정본 생성 시각 없음", _match(approved=True, effective_from=later), replace(ruled, created_at=None)),
        )
        for name, match, canonical in cases:
            with self.subTest(name):
                self.assertEqual(
                    (EXACT_SOURCE_CANONICAL_UNVERIFIED,),
                    exact_source_fallthrough_reasons(match, _inputs(canonical=canonical)),
                )

    def test_recorded_mismatch_is_not_rescued_by_approval_time(self) -> None:
        # 본 정본이 기록돼 있으면 승인 시각과 관계없이 그 기록으로 판단한다.
        created = datetime(2026, 9, 30, 9, 0, tzinfo=timezone.utc)
        ruled = replace(CANONICAL, applicability_rules=("규칙 1",), created_at=created)

        result = exact_source_fallthrough_reasons(
            _match(
                recorded=True,
                canonical_id=uuid.UUID(int=77),
                approved=True,
                effective_from=created + timedelta(hours=1),
            ),
            _inputs(canonical=ruled),
        )

        self.assertEqual((EXACT_SOURCE_CANONICAL_CHANGED,), result)

    def test_stale_revision_falls_through_before_approval_time_rule(self) -> None:
        created = datetime(2026, 9, 30, 9, 0, tzinfo=timezone.utc)
        ruled = replace(CANONICAL, applicability_rules=("규칙 1",), created_at=created)

        result = exact_source_fallthrough_reasons(
            _match(source_version=1, approved=True, effective_from=created + timedelta(hours=1)),
            _inputs(canonical=ruled),
        )

        self.assertEqual((EXACT_SOURCE_SUBPROBLEM_VERSION_STALE,), result)

    def test_without_gate_inputs_only_revision_from_lookup_is_checked(self) -> None:
        self.assertEqual((), exact_source_fallthrough_reasons(_match(), None))
        self.assertEqual(
            (EXACT_SOURCE_SUBPROBLEM_VERSION_STALE,),
            exact_source_fallthrough_reasons(_match(source_version=1), None),
        )


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
