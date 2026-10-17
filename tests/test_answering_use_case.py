"""Tests for stage-selective execution of the shared answer use case."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.answering.use_case import AnswerExecutionTarget, AnswerQuestion
from app.answering.trace import ExecutionStage, ReplayContext, StageOutcome
from app.retrieval.models import HybridSearchCall


class AnswerQuestionExecutionTargetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.generation = SimpleNamespace(
            generate_answer=AsyncMock(
                return_value=SimpleNamespace(stage_trace=None)
            )
        )
        self.use_case = AnswerQuestion(self.generation)
        self.resolve_query = AsyncMock(return_value="resolved customer question")
        self.retrieve = AsyncMock(return_value=HybridSearchCall())
        self.choose_source = AsyncMock(return_value={"outcome": "REJECTED"})

    async def test_cache_decision_stops_after_cache_miss_without_generation(self):
        result = await self.use_case.execute(
            resolve_query=self.resolve_query,
            retrieve=self.retrieve,
            choose_source=self.choose_source,
            before_model_call=AsyncMock(),
            target=AnswerExecutionTarget.CACHE_DECISION,
        )

        self.assertTrue(result.generation_skipped)
        self.assertFalse(result.source_handled)
        self.assertEqual({"outcome": "REJECTED"}, result.source_decision)
        self.generation.generate_answer.assert_not_awaited()

    async def test_cache_hit_stops_without_generation(self):
        handle_source = AsyncMock(return_value=True)
        self.choose_source.return_value = {"outcome": "SERVED"}

        result = await self.use_case.execute(
            resolve_query=self.resolve_query,
            retrieve=self.retrieve,
            choose_source=self.choose_source,
            handle_source=handle_source,
            before_model_call=AsyncMock(),
            target=AnswerExecutionTarget.CACHE_DECISION,
        )

        self.assertFalse(result.source_handled)
        self.assertTrue(result.generation_skipped)
        handle_source.assert_not_awaited()
        self.generation.generate_answer.assert_not_awaited()

    async def test_full_answer_target_still_generates_after_cache_miss(self):
        result = await self.use_case.execute(
            resolve_query=self.resolve_query,
            retrieve=self.retrieve,
            choose_source=self.choose_source,
            handle_source=AsyncMock(return_value=False),
            before_model_call=AsyncMock(),
        )

        self.assertFalse(result.generation_skipped)
        self.generation.generate_answer.assert_awaited_once()

    async def test_full_answer_target_still_handles_cache_hit(self):
        handle_source = AsyncMock(return_value=True)
        self.choose_source.return_value = {"outcome": "SERVED"}

        result = await self.use_case.execute(
            resolve_query=self.resolve_query,
            retrieve=self.retrieve,
            choose_source=self.choose_source,
            handle_source=handle_source,
            before_model_call=AsyncMock(),
        )

        self.assertTrue(result.source_handled)
        self.assertTrue(result.generation_skipped)
        handle_source.assert_awaited_once()
        self.generation.generate_answer.assert_not_awaited()

    async def test_cache_decision_requires_cache_selection(self):
        with self.assertRaisesRegex(ValueError, "캐시 판정 함수"):
            await self.use_case.execute(
                resolve_query=self.resolve_query,
                retrieve=self.retrieve,
                before_model_call=AsyncMock(),
                target=AnswerExecutionTarget.CACHE_DECISION,
            )

    async def test_stage_failure_is_recorded_without_input_text(self):
        recorder = SimpleNamespace(record=lambda trace: setattr(recorder, "trace", trace))
        self.resolve_query.side_effect = RuntimeError("private question failed")
        context = ReplayContext(
            run_id="run-1",
            item_id="SYN-2",
            turn_no=1,
            golden_version="golden-v1",
            source_corpus_version=None,
            service_version="service-a",
            code_revision="abc123",
            cache_policy_version="policy-v1",
        )

        with self.assertRaisesRegex(RuntimeError, "private question failed"):
            await self.use_case.execute(
                resolve_query=self.resolve_query,
                retrieve=self.retrieve,
                choose_source=self.choose_source,
                before_model_call=AsyncMock(),
                replay_context=context,
                trace_recorder=recorder,
            )

        self.assertEqual(
            (ExecutionStage.QUERY_RESOLUTION, StageOutcome.FAILED),
            (recorder.trace.events[0].stage, recorder.trace.events[0].outcome),
        )
        self.assertEqual("RuntimeError", recorder.trace.events[0].detail_code)
        self.assertNotIn("private question", repr(recorder.trace))

