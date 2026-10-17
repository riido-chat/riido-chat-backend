"""Tests for stage-selective execution of the shared answer use case."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.answering.use_case import AnswerExecutionTarget, AnswerQuestion
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

