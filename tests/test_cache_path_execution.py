"""Tests for the cache-only evaluation adapter's safety boundary."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from app.answering.use_case import AnswerQuestion
from app.retrieval.models import HybridSearchCall
from evaluation.cache_path_eval.execution import (
    CacheDecisionCallbacks,
    CacheDecisionExecutionAdapter,
    CacheDecisionMetrics,
)


class CacheDecisionExecutionAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_shared_cache_decision_and_emits_text_free_row(self):
        generation = SimpleNamespace(generate_answer=AsyncMock())
        adapter = CacheDecisionExecutionAdapter(AnswerQuestion(generation))
        resolve_query = AsyncMock(return_value="private resolved question")
        retrieve = AsyncMock(return_value=HybridSearchCall())
        choose_source = AsyncMock(return_value=SimpleNamespace(outcome="REJECTED"))
        summarize = Mock(
            return_value=CacheDecisionMetrics(
                outcome="REJECTED",
                next_stage="NOT_APPLICABLE",
                model_calls=1,
                embedding_calls=1,
            )
        )

        observation = await adapter.execute(
            item_id="SYN-1",
            turn_no=2,
            callbacks=CacheDecisionCallbacks(
                resolve_query=resolve_query,
                retrieve=retrieve,
                choose_source=choose_source,
                summarize=summarize,
            ),
        )

        self.assertEqual(
            {
                "itemId": "SYN-1",
                "turnNo": 2,
                "outcome": "REJECTED",
                "nextStage": "NOT_APPLICABLE",
                "latencyMs": observation.latency_ms,
                "modelCalls": 1,
                "embeddingCalls": 1,
            },
            observation.to_row(),
        )
        self.assertNotIn("private resolved question", repr(observation.to_row()))
        generation.generate_answer.assert_not_awaited()
        summarize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
