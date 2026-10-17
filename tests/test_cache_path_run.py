"""The per-turn runner binds profile callbacks to the private run store."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from app.answering.trace import ReplayContext
from app.answering.use_case import AnswerQuestion
from app.retrieval.models import HybridSearchCall
from evaluation.cache_path_eval.execution import (
    CacheDecisionCallbacks,
    CacheDecisionExecutionAdapter,
    CacheDecisionMetrics,
)
from evaluation.cache_path_eval.run import execute_cache_path_run
from evaluation.cache_path_eval.store import CachePathRunMetadata


class CachePathRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_executes_selected_turns_once_and_persists_private_summary(self):
        metadata = CachePathRunMetadata(
            run_id="run-3",
            golden_version="1.0.1",
            golden_commit="c" * 40,
            source_corpus_version="cache-drafts-v1",
            app_build_commit="a" * 40,
            service_key="service-a",
            profile_revision_id=4,
            document_group_id=7,
            index_version_id=12,
            embedding_config_id=3,
            cache_policy_version="policy-v1",
            embedding_model_version="embed-v1",
            judge_model_version="judge-v1",
            judge_prompt_version="prompt-v1",
            executed_at="2026-10-17T12:00:00Z",
        )
        manifest = {
            "group": "cache-path",
            "goldenVersion": "1.0.1",
            "inputs": {"draftsSha256": None, "judgeLabelsSha256": None},
            "turns": [
                {
                    "itemId": "CACHE-1",
                    "path": "P1",
                    "turnNo": 1,
                    "kind": "exact_hit",
                    "expectedExactHit": True,
                    "expectedServed": True,
                }
            ],
        }
        callback_bundle = CacheDecisionCallbacks(
            resolve_query=AsyncMock(return_value="private question"),
            retrieve=AsyncMock(return_value=HybridSearchCall()),
            choose_source=AsyncMock(return_value=object()),
            summarize=Mock(
                return_value=CacheDecisionMetrics(
                    outcome="SERVED",
                    next_stage="NOT_APPLICABLE",
                    model_calls=1,
                    embedding_calls=0,
                )
            ),
        )

        class Profile:
            async def callbacks_for_turn(self, _selected_turn):
                return callback_bundle

            def replay_context_for_turn(self, run_metadata, selected_turn):
                return ReplayContext(
                    run_id=run_metadata.run_id,
                    item_id=selected_turn["itemId"],
                    turn_no=selected_turn["turnNo"],
                    golden_version=run_metadata.golden_version,
                    source_corpus_version=run_metadata.source_corpus_version,
                    service_version=run_metadata.service_key,
                    code_revision=run_metadata.app_build_commit,
                    cache_policy_version=run_metadata.cache_policy_version,
                    embedding_model_version=run_metadata.embedding_model_version,
                )

        class Store:
            def validate_manifest(self, _metadata, received_manifest):
                return received_manifest["turns"]

            def reserve(self, _metadata):
                self.reserved = True

            def checkpoint(self, _metadata, observation):
                self.checkpoints = getattr(self, "checkpoints", []) + [observation]

            def fail(self, _metadata, failure):
                self.failure = failure

            def begin_finalization(self, _metadata):
                self.finalizing = True

            def succeed(self, _metadata):
                self.succeeded = True

            def write_run(self, received_metadata, **kwargs):
                self.metadata = received_metadata
                self.record = kwargs
                return "/private/riido-evals/evals/cache-path/run-3.json"

        store = Store()
        output = await execute_cache_path_run(
            manifest=manifest,
            metadata=metadata,
            profile=Profile(),
            store=store,
            execution_adapter=CacheDecisionExecutionAdapter(
                AnswerQuestion(SimpleNamespace(generate_answer=AsyncMock()))
            ),
        )

        self.assertEqual("/private/riido-evals/evals/cache-path/run-3.json", output.result_path)
        self.assertEqual(1, len(output.observations))
        self.assertEqual(1, store.record["summary"]["all"]["turns"])
        callback_bundle.resolve_query.assert_awaited_once()
        self.assertNotIn("private question", repr(store.record["observations"][0].to_row()))

    async def test_persists_failed_stage_trace_without_exception_message(self):
        metadata = CachePathRunMetadata(
            run_id="run-failed",
            golden_version="1.0.1",
            golden_commit="c" * 40,
            source_corpus_version=None,
            app_build_commit="a" * 40,
            service_key="service-a",
            profile_revision_id=4,
            document_group_id=7,
            index_version_id=12,
            embedding_config_id=3,
            cache_policy_version="policy-v1",
            embedding_model_version="embed-v1",
            judge_model_version="judge-v1",
            judge_prompt_version="prompt-v1",
            executed_at="2026-10-17T12:00:00Z",
        )
        manifest = {
            "group": "cache-path",
            "goldenVersion": "1.0.1",
            "inputs": {"draftsSha256": None, "judgeLabelsSha256": None},
            "turns": [
                {
                    "itemId": "CACHE-FAIL",
                    "path": "P1",
                    "turnNo": 1,
                    "kind": "exact_hit",
                    "expectedExactHit": True,
                    "expectedServed": True,
                }
            ],
        }
        failure = RuntimeError("private question in exception")
        callback_bundle = CacheDecisionCallbacks(
            resolve_query=AsyncMock(side_effect=failure),
            retrieve=AsyncMock(return_value=HybridSearchCall()),
            choose_source=AsyncMock(return_value=object()),
            summarize=Mock(
                return_value=CacheDecisionMetrics(
                    outcome="FAILED",
                    next_stage="NOT_APPLICABLE",
                    model_calls=0,
                    embedding_calls=0,
                )
            ),
        )

        class Profile:
            async def callbacks_for_turn(self, _selected_turn):
                return callback_bundle

            def replay_context_for_turn(self, run_metadata, selected_turn):
                return ReplayContext(
                    run_id=run_metadata.run_id,
                    item_id=selected_turn["itemId"],
                    turn_no=selected_turn["turnNo"],
                    golden_version=run_metadata.golden_version,
                    source_corpus_version=run_metadata.source_corpus_version,
                    service_version=run_metadata.service_key,
                    code_revision=run_metadata.app_build_commit,
                    cache_policy_version=run_metadata.cache_policy_version,
                    embedding_model_version=run_metadata.embedding_model_version,
                )

        class Store:
            def validate_manifest(self, _metadata, received_manifest):
                return received_manifest["turns"]

            def reserve(self, _metadata):
                self.reserved = True

            def checkpoint(self, _metadata, observation):
                self.checkpoints = getattr(self, "checkpoints", []) + [observation]

            def fail(self, _metadata, failure):
                self.failure = failure

            def begin_finalization(self, _metadata):
                self.finalizing = True

            def succeed(self, _metadata):
                self.succeeded = True

        store = Store()
        with self.assertRaisesRegex(RuntimeError, "private question"):
            await execute_cache_path_run(
                manifest=manifest,
                metadata=metadata,
                profile=Profile(),
                store=store,
                execution_adapter=CacheDecisionExecutionAdapter(
                    AnswerQuestion(SimpleNamespace(generate_answer=AsyncMock()))
                ),
            )

        trace = store.failure.detail["trace"]
        failed = [event for event in trace.events if event.outcome.value == "FAILED"]
        self.assertEqual("QUERY_RESOLUTION", failed[0].stage.value)
        self.assertEqual("RuntimeError", failed[0].detail_code)
        self.assertNotIn("private question", repr(trace))


if __name__ == "__main__":
    unittest.main()
