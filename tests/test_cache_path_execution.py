"""Tests for the cache-only evaluation adapter's safety boundary."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from app.answering.use_case import AnswerQuestion
from app.answering.trace import ExecutionStage, ReplayContext, StageOutcome
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
        replay_context = ReplayContext(
            run_id="run-2026-10-17-01",
            item_id="SYN-1",
            turn_no=2,
            golden_version="golden-cache-v3",
            source_corpus_version="corpus-2026-10-10",
            service_version="service-a",
            code_revision="abc123",
            cache_policy_version="cache-policy-v2",
            embedding_model_version="embedding-v1",
        )
        trace_recorder = Mock()

        observation = await adapter.execute(
            item_id="SYN-1",
            turn_no=2,
            replay_context=replay_context,
            trace_recorder=trace_recorder,
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
        self.assertEqual("abc123", observation.execution_trace.replay_context.code_revision)
        self.assertEqual(
            [
                (ExecutionStage.QUERY_RESOLUTION, StageOutcome.SUCCEEDED),
                (ExecutionStage.RETRIEVAL, StageOutcome.SUCCEEDED),
                (ExecutionStage.SOURCE_SELECTION, StageOutcome.SUCCEEDED),
                (ExecutionStage.SOURCE_HANDLING, StageOutcome.SKIPPED),
                (ExecutionStage.ANSWER_GENERATION, StageOutcome.SKIPPED),
            ],
            [(event.stage, event.outcome) for event in observation.execution_trace.events],
        )
        self.assertNotIn("private resolved question", repr(observation.execution_trace))
        trace_recorder.record.assert_called_once_with(observation.execution_trace)
        generation.generate_answer.assert_not_awaited()
        summarize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
