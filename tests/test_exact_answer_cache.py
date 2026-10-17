"""Deterministic decision-path checks for the exact-answer cache boundary.

These tests use fixed result objects and an in-memory grouping double. They do not
exercise database readers, chat requests, or model clients.
"""

import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.database.models import CacheAttemptOutcome
from app.question_grouping.exact_cache import ExactAnswerCache
from app.question_grouping.models import GateResult
from app.question_grouping.service import (
    ExactQuestionFallthrough,
    ExactQuestionResult,
    GroupingTurn,
    RecordedJudgment,
)
from app.question_grouping.store import QuestionGroupingStore


TURN = GroupingTurn(
    rag_run_id=uuid.UUID("00000000-0000-0000-0000-000000000101"),
    document_group_id=17,
    index_version_id=29,
    classification_run_id=41,
)


def exact_result(outcome: CacheAttemptOutcome) -> ExactQuestionResult:
    served = outcome is CacheAttemptOutcome.SERVED
    rejected = outcome is CacheAttemptOutcome.REJECTED
    recorded = RecordedJudgment(
        classification_id=5,
        cache_attempt_id=8,
        judgment=SimpleNamespace(),
        gate=GateResult(
            outcome=outcome,
            canonical_answer_id=(uuid.UUID("00000000-0000-0000-0000-000000000202") if served else None),
            rejection_reasons=("EXACT_GATE_REJECTED",) if rejected else (),
        ),
        latency_ms=1,
        served_answer_markdown="fixed approved answer" if served else None,
    )
    return ExactQuestionResult(
        recorded=recorded,
        prepared=SimpleNamespace(),
        judged=SimpleNamespace(),
    )


class ExactAnswerCacheDecisionTests(unittest.IsolatedAsyncioTestCase):
    def _cache(self, result):
        grouping = SimpleNamespace(record_exact_question=AsyncMock(return_value=result))
        return ExactAnswerCache(grouping), grouping

    async def test_approved_exact_match_reuses_recorded_answer_and_stops(self):
        cache, grouping = self._cache(exact_result(CacheAttemptOutcome.SERVED))
        next_steps = []

        async def handle(decision):
            if decision.recorded is not None and decision.recorded.served:
                return decision.recorded.served_answer_markdown
            return None

        async def next_step(decision):
            next_steps.append(decision)
            return "generated"

        execution = await cache.execute(
            TURN, "fixed exact question", semantic_cache_enabled=True,
            exact_cache_enabled=True, handle_decision=handle, next_step=next_step,
        )

        self.assertTrue(execution.cache_handled)
        self.assertEqual("fixed approved answer", execution.result)
        self.assertEqual([], next_steps)
        grouping.record_exact_question.assert_awaited_once_with(
            TURN, "fixed exact question", semantic_cache_enabled=True,
            exact_cache_enabled=True,
        )

    async def test_exact_miss_allows_judgment_and_fallback(self):
        await self._assert_fallthrough(None, expect_judge=True)

    async def test_stale_source_allows_judgment_and_preserves_reason(self):
        stale = ExactQuestionFallthrough(
            rejection_reasons=("EXACT_SOURCE_SUBPROBLEM_VERSION_STALE",),
            judgment_input={"exactQuestionFallthrough": {"reason": "stale"}},
        )
        decision = await self._assert_fallthrough(stale, expect_judge=True)
        self.assertIs(stale, decision.fallthrough)

    async def test_gate_rejection_is_recorded_and_allows_generation_without_judgment(self):
        decision = await self._assert_fallthrough(
            exact_result(CacheAttemptOutcome.REJECTED), expect_judge=False
        )
        self.assertEqual(CacheAttemptOutcome.REJECTED, decision.recorded.gate.outcome)

    async def test_disabled_flag_is_forwarded_and_does_not_reuse_a_result(self):
        # The chat orchestration bypasses this wrapper when exact caching is off;
        # this contract checks the boundary's disabled flag behavior when inspected.
        cache, grouping = self._cache(None)
        decision = await cache.inspect(
            TURN, "fixed exact question", semantic_cache_enabled=True,
            exact_cache_enabled=False,
        )
        self.assertIsNone(decision.result)
        self.assertIsNone(decision.recorded)
        self.assertTrue(decision.allow_similar_judgment)
        grouping.record_exact_question.assert_awaited_once_with(
            TURN, "fixed exact question", semantic_cache_enabled=True,
            exact_cache_enabled=False,
        )

    async def test_different_service_scopes_do_not_share_exact_results(self):
        first_scope = GroupingTurn(
            rag_run_id=TURN.rag_run_id, document_group_id=17,
            index_version_id=29, classification_run_id=41,
        )
        second_scope = GroupingTurn(
            rag_run_id=uuid.UUID("00000000-0000-0000-0000-000000000102"),
            document_group_id=18, index_version_id=30, classification_run_id=42,
        )
        first, first_grouping = self._cache(exact_result(CacheAttemptOutcome.SERVED))
        second, second_grouping = self._cache(None)

        first_decision = await first.inspect(
            first_scope, "same fixed question", semantic_cache_enabled=True,
            exact_cache_enabled=True,
        )
        second_decision = await second.inspect(
            second_scope, "same fixed question", semantic_cache_enabled=True,
            exact_cache_enabled=True,
        )

        self.assertEqual(CacheAttemptOutcome.SERVED, first_decision.recorded.gate.outcome)
        self.assertTrue(first_decision.recorded.served)
        self.assertIsNone(second_decision.recorded)
        self.assertIsNone(second_decision.result)
        first_grouping.record_exact_question.assert_awaited_once()
        second_grouping.record_exact_question.assert_awaited_once_with(
            second_scope, "same fixed question", semantic_cache_enabled=True,
            exact_cache_enabled=True,
        )

    async def test_production_exact_lookup_filters_by_requested_document_group(self):
        class EmptyRows:
            def all(self):
                return []

        class CapturingSession:
            statement = None

            async def execute(self, statement):
                self.statement = statement
                return EmptyRows()

        session = CapturingSession()
        store = QuestionGroupingStore(session)
        await store.find_exact_question_log_match(
            18, "fixed exact question", exclude_rag_run_id=TURN.rag_run_id
        )

        self.assertIsNotNone(session.statement)
        compiled = session.statement.compile()
        self.assertIn(
            "index_versions.document_group_id =", str(compiled)
        )
        scope_parameters = {
            key: value for key, value in compiled.params.items()
            if "document_group_id" in key
        }
        self.assertTrue(scope_parameters)
        self.assertEqual({18}, set(scope_parameters.values()))

    async def _assert_fallthrough(self, result, *, expect_judge):
        cache, _grouping = self._cache(result)
        next_steps = []

        async def handle(decision):
            return decision.recorded.served_answer_markdown if (
                decision.recorded is not None and decision.recorded.served
            ) else None

        async def next_step(decision):
            next_steps.append(decision)
            return "fixed fallback"

        execution = await cache.execute(
            TURN, "fixed exact question", semantic_cache_enabled=True,
            exact_cache_enabled=True, handle_decision=handle, next_step=next_step,
        )
        self.assertFalse(execution.cache_handled)
        self.assertEqual("fixed fallback", execution.result)
        self.assertEqual(1, len(next_steps))
        self.assertEqual(expect_judge, next_steps[0].allow_similar_judgment)
        return next_steps[0]


if __name__ == "__main__":
    unittest.main()
