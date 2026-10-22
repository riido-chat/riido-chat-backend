import unittest
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional

from app.chat.log_store import CitationLog
from app.database.models import (
    AttributionSource,
    CacheAttemptOutcome,
    ClassificationDecision,
    QuestionProblemGroupKind,
)
from app.question_grouping.attribution import (
    document_attribution,
    initial_attribution,
    no_document_attribution,
)
from app.question_grouping.constants import (
    REJECT_CITED_SECTION_CHANGED,
    REJECT_CLASSIFICATION_NOT_CONNECTED,
)
from app.question_grouping.decision import failed_turn_judgment
from app.question_grouping.models import (
    AttributionTarget,
    CitationResolution,
    GateResult,
    IndexedSection,
    JudgeFailure,
    JudgeFailureKind,
    PresentedSubproblem,
    TurnJudgment,
)
from app.question_grouping.store import (
    QuestionGroupingStore,
    presented_canonical_answer,
    presented_canonical_answer_id,
    served_citation_logs,
    validate_gate_result,
    validate_turn_judgment,
)

CANONICAL_ID = uuid.UUID(int=2)
PRESENTED = PresentedSubproblem(
    key="billing.cancel",
    subproblem_id=uuid.UUID(int=1),
    subproblem_version=3,
    problem_group_id=uuid.UUID(int=4),
    document_source_id=7,
    document_key="workspaces/plans-and-billing",
    canonical_answer_id=CANONICAL_ID,
    similarity=0.8,
    retrieval_rank=1,
    presented_order=1,
    inclusion_count=2,
    exclusion_count=0,
)


def _section(chunk_id: int, version_id: int = 12) -> IndexedSection:
    return IndexedSection(
        chunk_id=chunk_id,
        document_version_id=version_id,
        content_hash="c",
        node_order=1,
        node_identity_hash="i",
        document_title="구독 및 결제",
        node_path=f"구독 및 결제 > 절 {chunk_id}",
        source_uri="https://docs.riido.io/billing",
    )


def _connect() -> TurnJudgment:
    return TurnJudgment(
        decision=ClassificationDecision.CONNECT,
        attribution=initial_attribution(ClassificationDecision.CONNECT, subproblem=PRESENTED),
        subproblem=PRESENTED,
    )


class ValidateTurnJudgmentTest(unittest.TestCase):
    def test_accepts_initial_attribution_shapes(self) -> None:
        judgments = (
            _connect(),
            TurnJudgment(
                decision=ClassificationDecision.SEPARATE,
                attribution=document_attribution(7),
            ),
            TurnJudgment(
                decision=ClassificationDecision.UNCLASSIFIED,
                attribution=no_document_attribution(),
            ),
            failed_turn_judgment(JudgeFailure(JudgeFailureKind.API_ERROR, "HTTP 500")),
        )
        for judgment in judgments:
            with self.subTest(decision=judgment.decision):
                validate_turn_judgment(judgment)

    def test_rejects_shapes_that_break_checks_or_rules(self) -> None:
        other_group = AttributionTarget(
            attribution_source=AttributionSource.SUBPROBLEM,
            group_kind=QuestionProblemGroupKind.DOCUMENT,
            problem_group_id=uuid.UUID(int=99),
        )
        cases = {
            "connect without subproblem": TurnJudgment(
                decision=ClassificationDecision.CONNECT,
                attribution=_connect().attribution,
            ),
            "separate with subproblem": TurnJudgment(
                decision=ClassificationDecision.SEPARATE,
                attribution=no_document_attribution(),
                subproblem=PRESENTED,
            ),
            "connect with document attribution": TurnJudgment(
                decision=ClassificationDecision.CONNECT,
                attribution=document_attribution(7),
                subproblem=PRESENTED,
            ),
            "citation at insert": TurnJudgment(
                decision=ClassificationDecision.SEPARATE,
                attribution=document_attribution(7, AttributionSource.CITATION),
            ),
            "subproblem group mismatch": TurnJudgment(
                decision=ClassificationDecision.CONNECT,
                attribution=other_group,
                subproblem=PRESENTED,
            ),
            "document without source": TurnJudgment(
                decision=ClassificationDecision.SEPARATE,
                attribution=AttributionTarget(
                    attribution_source=AttributionSource.DOCUMENT,
                    group_kind=QuestionProblemGroupKind.DOCUMENT,
                ),
            ),
            "failure with document": TurnJudgment(
                decision=ClassificationDecision.UNCLASSIFIED,
                attribution=document_attribution(7),
                failure=JudgeFailure(JudgeFailureKind.API_ERROR, "HTTP 500"),
            ),
            "failure not unclassified": TurnJudgment(
                decision=ClassificationDecision.SEPARATE,
                attribution=no_document_attribution(),
                failure=JudgeFailure(JudgeFailureKind.API_ERROR, "HTTP 500"),
            ),
        }
        for name, judgment in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                validate_turn_judgment(judgment)


class ValidateGateResultTest(unittest.TestCase):
    def test_accepts_every_used_outcome(self) -> None:
        for gate in (
            GateResult(CacheAttemptOutcome.SERVED, canonical_answer_id=CANONICAL_ID),
            GateResult(CacheAttemptOutcome.SHADOW, canonical_answer_id=CANONICAL_ID),
            GateResult(CacheAttemptOutcome.GROUP_DISABLED, canonical_answer_id=CANONICAL_ID),
            GateResult(CacheAttemptOutcome.REJECTED, rejection_reasons=(REJECT_CLASSIFICATION_NOT_CONNECTED,)),
            GateResult(CacheAttemptOutcome.FAILED),
        ):
            with self.subTest(outcome=gate.outcome):
                validate_gate_result(gate)

    def test_rejects_check_violations_and_skipped(self) -> None:
        for gate in (
            GateResult(CacheAttemptOutcome.SERVED),
            GateResult(CacheAttemptOutcome.SHADOW, canonical_answer_id=CANONICAL_ID, rejection_reasons=("X",)),
            GateResult(CacheAttemptOutcome.REJECTED),
            GateResult(CacheAttemptOutcome.REJECTED, canonical_answer_id=CANONICAL_ID, rejection_reasons=("X",)),
            GateResult(CacheAttemptOutcome.REJECTED, rejection_reasons=("X" * 51,)),
            GateResult(CacheAttemptOutcome.FAILED, canonical_answer_id=CANONICAL_ID),
            GateResult(CacheAttemptOutcome.FAILED, rejection_reasons=("X",)),
            GateResult(CacheAttemptOutcome.SKIPPED),
        ):
            with self.subTest(gate=gate), self.assertRaises(ValueError):
                validate_gate_result(gate)


class ServedCitationLogsTest(unittest.TestCase):
    def test_copies_current_sections_in_citation_order(self) -> None:
        gate = GateResult(
            CacheAttemptOutcome.SERVED,
            canonical_answer_id=CANONICAL_ID,
            served_citations=(
                CitationResolution(citation_order=2, section=_section(202, 13), step=3),
                CitationResolution(citation_order=1, section=_section(201), step=2),
            ),
        )

        logs = served_citation_logs(gate)

        self.assertEqual(
            (
                CitationLog(
                    chunk_id=201,
                    document_version_id=12,
                    citation_order=1,
                    document_title_snapshot="구독 및 결제",
                    node_path_snapshot="구독 및 결제 > 절 201",
                    source_uri_snapshot="https://docs.riido.io/billing",
                ),
                CitationLog(
                    chunk_id=202,
                    document_version_id=13,
                    citation_order=2,
                    document_title_snapshot="구독 및 결제",
                    node_path_snapshot="구독 및 결제 > 절 202",
                    source_uri_snapshot="https://docs.riido.io/billing",
                ),
            ),
            logs,
        )

    def test_rejects_non_served_empty_unresolved_or_duplicate(self) -> None:
        passed = CitationResolution(citation_order=1, section=_section(201), step=1)
        cases = (
            GateResult(CacheAttemptOutcome.SHADOW, canonical_answer_id=CANONICAL_ID, served_citations=(passed,)),
            GateResult(CacheAttemptOutcome.SERVED, canonical_answer_id=CANONICAL_ID),
            GateResult(
                CacheAttemptOutcome.SERVED,
                canonical_answer_id=CANONICAL_ID,
                served_citations=(CitationResolution(citation_order=1, rejection_reason=REJECT_CITED_SECTION_CHANGED),),
            ),
            GateResult(CacheAttemptOutcome.SERVED, canonical_answer_id=CANONICAL_ID, served_citations=(passed, passed)),
        )
        for gate in cases:
            with self.subTest(gate=gate), self.assertRaises(ValueError):
                served_citation_logs(gate)


class PresentedCanonicalAnswerTest(unittest.TestCase):
    SUBPROBLEM_ID = uuid.UUID(int=1)

    def test_presented_item_gives_recorded_canonical(self) -> None:
        items = [
            {"subproblemId": str(uuid.UUID(int=9)), "canonicalAnswerId": str(uuid.UUID(int=8))},
            PRESENTED.to_judgment_input(),
        ]

        self.assertEqual((True, CANONICAL_ID), presented_canonical_answer(items, self.SUBPROBLEM_ID))

    def test_presented_without_canonical_is_recorded_as_none(self) -> None:
        items = [{"subproblemId": str(self.SUBPROBLEM_ID), "canonicalAnswerId": None}]

        self.assertEqual((True, None), presented_canonical_answer(items, self.SUBPROBLEM_ID))

    def test_missing_or_malformed_presentation_is_unknown(self) -> None:
        cases = (
            None,
            {"items": []},
            [{"subproblemId": str(uuid.UUID(int=9)), "canonicalAnswerId": None}],
            [{"subproblemId": str(self.SUBPROBLEM_ID)}],
            [{"subproblemId": str(self.SUBPROBLEM_ID), "canonicalAnswerId": "not-a-uuid"}],
        )
        for items in cases:
            with self.subTest(items=items):
                self.assertEqual((False, None), presented_canonical_answer(items, self.SUBPROBLEM_ID))



class PresentedCanonicalAnswerIdTest(unittest.TestCase):
    """판별 행 presented_canonical_answer_id 는 judgment_input 제시 목록에서 읽는다."""

    def _input(self, items) -> dict:
        return {"gate": {"outcome": "SERVED"}, "subproblemCandidates": {"seed": "s", "items": items}}

    def test_connect_row_records_canonical_shown_for_chosen_subproblem(self) -> None:
        other = {"subproblemId": str(uuid.UUID(int=9)), "canonicalAnswerId": str(uuid.UUID(int=8))}
        judgment_input = self._input([other, PRESENTED.to_judgment_input()])

        self.assertEqual(CANONICAL_ID, presented_canonical_answer_id(_connect(), judgment_input))

    def test_shown_without_canonical_is_null(self) -> None:
        judgment_input = self._input(
            [{"subproblemId": str(PRESENTED.subproblem_id), "canonicalAnswerId": None}]
        )

        self.assertIsNone(presented_canonical_answer_id(_connect(), judgment_input))

    def test_rows_without_judge_presentation_or_subproblem_are_null(self) -> None:
        separate = TurnJudgment(
            decision=ClassificationDecision.SEPARATE,
            attribution=no_document_attribution(),
        )
        cases = (
            # 정확 일치 재사용 행은 제시 목록이 없다(build_judgment_input 이 None 으로 둔다).
            (_connect(), {"gate": {}, "subproblemCandidates": None}),
            (_connect(), {"gate": {}}),
            (_connect(), self._input([])),
            # 세부 문제를 고르지 않은 행은 보여 준 정본이 여럿이어도 비운다.
            (separate, self._input([PRESENTED.to_judgment_input()])),
        )
        for judgment, judgment_input in cases:
            with self.subTest(judgment_input=judgment_input, decision=judgment.decision):
                self.assertIsNone(presented_canonical_answer_id(judgment, judgment_input))


class _RowsResult:
    def __init__(self, rows) -> None:
        self._rows = rows

    def all(self):
        return list(self._rows)


class _RowsSession:
    """find_exact_question_log_match 가 읽는 조회 결과 행만 돌려주는 가짜 세션."""

    def __init__(self, rows) -> None:
        self._rows = rows

    async def execute(self, statement: Any) -> _RowsResult:
        return _RowsResult(self._rows)


class ExactQuestionLogMatchRowTest(unittest.IsolatedAsyncioTestCase):
    """정확 일치 원천 행의 제시 정본 기록 여부와 승인 시각을 읽는 규칙(#220)."""

    GROUP_ID = 5
    APPROVED_AT = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)

    def _row(self, *, presented_items: Any = None, column: Optional[uuid.UUID] = None) -> SimpleNamespace:
        return SimpleNamespace(
            classification_id=9,
            effective_from=self.APPROVED_AT,
            source_subproblem_version=3,
            exact_cache_approved=True,
            presented_canonical_column=column,
            presented_items=presented_items,
            rag_run_id=uuid.UUID(int=10),
            user_query="추천 질문",
            subproblem_id=PRESENTED.subproblem_id,
            key=PRESENTED.key,
            problem_group_id=PRESENTED.problem_group_id,
            current_version=3,
            kind=QuestionProblemGroupKind.DOCUMENT,
            no_document_owner_id=None,
            document_source_id=7,
            document_key=PRESENTED.document_key,
            document_owner_id=self.GROUP_ID,
        )

    async def _lookup(self, row: SimpleNamespace):
        store = QuestionGroupingStore(_RowsSession([row]))
        return await store.find_exact_question_log_match(
            self.GROUP_ID, "추천 질문", exclude_rag_run_id=uuid.uuid4()
        )

    async def test_json_null_candidates_are_not_recorded(self) -> None:
        # DEV 승인 원천은 judgment_input.subproblemCandidates 가 JSON null 이라 그 아래 items
        # 조회 결과가 널이다. 제시 정본 칸도 널이면 본 정본을 모르는 것으로 둔다.
        match = await self._lookup(self._row(presented_items=None))

        self.assertEqual((False, None), (match.source_canonical_recorded, match.source_canonical_answer_id))
        self.assertTrue(match.source_exact_cache_approved)
        self.assertEqual(self.APPROVED_AT, match.source_effective_from)

    async def test_presented_canonical_column_is_used_without_candidates(self) -> None:
        match = await self._lookup(self._row(presented_items=None, column=CANONICAL_ID))

        self.assertEqual((True, CANONICAL_ID), (match.source_canonical_recorded, match.source_canonical_answer_id))

    async def test_candidates_take_precedence_over_column(self) -> None:
        items = [PRESENTED.to_judgment_input()]

        match = await self._lookup(self._row(presented_items=items, column=uuid.UUID(int=77)))

        self.assertEqual((True, CANONICAL_ID), (match.source_canonical_recorded, match.source_canonical_answer_id))


if __name__ == "__main__":
    unittest.main()
