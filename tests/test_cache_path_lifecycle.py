"""Lifecycle ordering and durable checkpoint behavior for cache-path runs."""

import asyncio
import unittest
import json
import tempfile
import uuid
from pathlib import Path

from app.answering.trace import (
    AnswerExecutionTrace,
    ExecutionStage,
    ExecutionTraceEvent,
    ReplayContext,
    StageOutcome,
)
from evaluation.cache_path_eval.execution import CacheDecisionObservation
from evaluation.cache_path_eval.run import execute_cache_path_run
from evaluation.cache_path_eval.store import CachePathRunMetadata, CachePathRunStore


def _metadata(run_id="lifecycle-run"):
    return CachePathRunMetadata(
        run_id=run_id,
        golden_version="g-v1",
        golden_commit="c" * 40,
        source_corpus_version=None,
        app_build_commit="a" * 40,
        service_key="service-a",
        profile_revision_id=1,
        document_group_id=2,
        index_version_id=3,
        embedding_config_id=4,
        cache_policy_version="policy-v1",
        embedding_model_version=None,
        judge_model_version=None,
        judge_prompt_version=None,
        executed_at="2026-10-17T12:00:00Z",
    )


def _manifest(turns=None):
    base = {
        "itemId": "ITEM-1",
        "path": "P1",
        "turnNo": 1,
        "kind": "exact_hit",
        "expectedExactHit": True,
        "expectedServed": True,
    }
    rows = [base] if turns is None else turns
    return {
        "group": "cache-path",
        "goldenVersion": "g-v1",
        "inputs": {"draftsSha256": None, "judgeLabelsSha256": None},
        "turns": rows,
    }


class _Profile:
    def __init__(self, fail_callback=False, fail_context_on=None):
        self.callback_calls = []
        self.context_calls = []
        self.callback_turn_keys = []
        self.context_turn_keys = []
        self.fail_callback = fail_callback
        self.fail_context_on = fail_context_on

    def replay_context_for_turn(self, metadata, turn):
        self.context_calls.append(turn["itemId"])
        self.context_turn_keys.append(set(turn))
        if turn["itemId"] == self.fail_context_on:
            raise RuntimeError("private context preparation failure")
        return ReplayContext(
            run_id=metadata.run_id,
            item_id=turn["itemId"],
            turn_no=turn["turnNo"],
            golden_version=metadata.golden_version,
            source_corpus_version=metadata.source_corpus_version,
            service_version=metadata.service_key,
            code_revision=metadata.app_build_commit,
            cache_policy_version=metadata.cache_policy_version,
            embedding_model_version=metadata.embedding_model_version,
        )

    async def callbacks_for_turn(self, turn):
        self.callback_calls.append(turn["itemId"])
        self.callback_turn_keys.append(set(turn))
        if self.fail_callback:
            raise RuntimeError("private question must not be stored")
        return object()


class _Store:
    def __init__(self, duplicate=False):
        self.duplicate = duplicate
        self.checkpoints = []
        self.failures = []

    def validate_manifest(self, _metadata, manifest):
        # Use the real store validation rules relevant to these tests.
        turns = manifest.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError("turns required")
        seen = set()
        for turn in turns:
            if not isinstance(turn, dict) or turn.get("path") not in {"P1", "P3", "P4", "P6"}:
                raise ValueError("invalid turn")
            key = turn.get("itemId"), turn.get("turnNo")
            if key in seen:
                raise ValueError("duplicate turn")
            seen.add(key)
        return turns

    def reserve(self, _metadata):
        if self.duplicate:
            raise ValueError("duplicate run")
        self.reserved = True

    def checkpoint(self, _metadata, observation):
        self.checkpoints.append(observation)

    def fail(self, _metadata, failure):
        self.failures.append(failure)

    def begin_finalization(self, _metadata):
        self.finalizing = True

    def succeed(self, _metadata):
        self.succeeded = True

    def write_run(self, _metadata, *, summary, **_kwargs):
        self.summary = summary
        return Path("/tmp/cache-path-test-result.json")


class _Execution:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    async def execute(self, *, item_id, turn_no, **_kwargs):
        self.calls.append(item_id)
        if item_id == self.fail_on:
            raise RuntimeError("private answer in execution error")
        return CacheDecisionObservation(
            item_id=item_id,
            turn_no=turn_no,
            outcome="SERVED",
            next_stage="NOT_APPLICABLE",
            latency_ms=1,
            model_calls=1,
            embedding_calls=0,
        )


class CachePathLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_profile_context_and_callbacks_receive_only_turn_references(self):
        profile, store = _Profile(), _Store()
        await execute_cache_path_run(
            manifest=_manifest(), metadata=_metadata(), profile=profile,
            store=store, execution_adapter=_Execution(),
        )
        self.assertEqual([{"itemId", "turnNo"}], profile.context_turn_keys)
        self.assertEqual([{"itemId", "turnNo"}], profile.callback_turn_keys)

    async def test_concurrent_same_run_id_executes_only_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git_dir = root / ".git"
            git_dir.mkdir()
            (git_dir / "config").write_text(
                '[remote "origin"]\n\turl = git@github.com:riido/riido-evals.git\n',
                encoding="utf-8",
            )
            (root / "README.md").write_text("# riido-evals\n", encoding="utf-8")
            (root / "golden").mkdir()
            (root / "golden" / "manifest.json").write_text("{}", encoding="utf-8")
            metadata = _metadata("concurrent-run")
            stores = (CachePathRunStore(root), CachePathRunStore(root))

            class TracedExecution:
                def __init__(self):
                    self.calls = 0

                async def execute(self, *, item_id, turn_no, replay_context, **_kwargs):
                    self.calls += 1
                    await asyncio.sleep(0)
                    trace = AnswerExecutionTrace(
                        trace_id=uuid.uuid4(),
                        target="CACHE_DECISION",
                        replay_context=replay_context,
                        events=(ExecutionTraceEvent(ExecutionStage.RETRIEVAL, StageOutcome.SUCCEEDED),),
                    )
                    return CacheDecisionObservation(
                        item_id=item_id,
                        turn_no=turn_no,
                        outcome="SERVED",
                        next_stage="NOT_APPLICABLE",
                        latency_ms=1,
                        model_calls=0,
                        embedding_calls=0,
                        execution_trace=trace,
                    )

            execution = TracedExecution()
            outcomes = await asyncio.gather(
                *(
                    execute_cache_path_run(
                        manifest=_manifest(), metadata=metadata, profile=_Profile(),
                        store=store, execution_adapter=execution,
                    )
                    for store in stores
                ),
                return_exceptions=True,
            )
            self.assertEqual(1, execution.calls)
            self.assertEqual(1, sum(not isinstance(value, BaseException) for value in outcomes))
            self.assertEqual(1, sum(isinstance(value, BaseException) for value in outcomes))

    async def test_cancellation_is_recorded_and_propagated(self):
        class CancelledExecution:
            async def execute(self, **_kwargs):
                raise asyncio.CancelledError("cancelled")

        profile, store = _Profile(), _Store()
        with self.assertRaises(asyncio.CancelledError):
            await execute_cache_path_run(
                manifest=_manifest(), metadata=_metadata(), profile=profile,
                store=store, execution_adapter=CancelledExecution(),
            )
        self.assertEqual(1, len(store.failures))
        self.assertEqual("execute", store.failures[0].stage)
        self.assertEqual("CancelledError", store.failures[0].error_type)

    async def test_callback_cannot_mutate_scoring_expectations(self):
        class MutatingProfile(_Profile):
            async def callbacks_for_turn(self, turn):
                turn["expectedServed"] = False
                return object()

        profile, store = MutatingProfile(), _Store()
        output = await execute_cache_path_run(
            manifest=_manifest(), metadata=_metadata(), profile=profile,
            store=store, execution_adapter=_Execution(),
        )
        self.assertEqual(0, output.summary["all"]["ineligibleServed"])

    async def test_invalid_later_turn_fails_before_any_profile_callback(self):
        rows = _manifest()["turns"] + [
            {**_manifest()["turns"][0], "itemId": "ITEM-2", "path": "P9", "turnNo": 2}
        ]
        profile, store, execution = _Profile(), _Store(), _Execution()
        with self.assertRaisesRegex(ValueError, "invalid turn"):
            await execute_cache_path_run(
                manifest=_manifest(rows), metadata=_metadata(), profile=profile,
                store=store, execution_adapter=execution,
            )
        self.assertEqual([], profile.callback_calls)
        self.assertEqual([], execution.calls)

    async def test_duplicate_run_id_fails_before_callbacks_or_execution(self):
        profile, store, execution = _Profile(), _Store(duplicate=True), _Execution()
        with self.assertRaisesRegex(ValueError, "duplicate run"):
            await execute_cache_path_run(
                manifest=_manifest(), metadata=_metadata(), profile=profile,
                store=store, execution_adapter=execution,
            )
        self.assertEqual([], profile.callback_calls)
        self.assertEqual([], execution.calls)

    async def test_mid_run_failure_keeps_prior_checkpoint_and_sanitizes_error(self):
        rows = [
            _manifest()["turns"][0],
            {**_manifest()["turns"][0], "itemId": "ITEM-2", "turnNo": 2},
        ]
        profile, store, execution = _Profile(), _Store(), _Execution(fail_on="ITEM-2")
        with self.assertRaisesRegex(RuntimeError, "private answer"):
            await execute_cache_path_run(
                manifest=_manifest(rows), metadata=_metadata(), profile=profile,
                store=store, execution_adapter=execution,
            )
        self.assertEqual(["ITEM-1"], [row.item_id for row in store.checkpoints])
        self.assertEqual(1, len(store.failures))
        failure = store.failures[0]
        self.assertEqual("execute", failure.stage)
        self.assertEqual("RuntimeError", failure.error_type)
        self.assertNotIn("private answer", repr(failure))

    async def test_profile_callback_failure_is_recorded_as_prepare_failure(self):
        profile, store = _Profile(fail_callback=True), _Store()
        with self.assertRaisesRegex(RuntimeError, "private question"):
            await execute_cache_path_run(
                manifest=_manifest(), metadata=_metadata(), profile=profile,
                store=store, execution_adapter=_Execution(),
            )
        self.assertEqual("prepare", store.failures[0].stage)
        self.assertEqual("RuntimeError", store.failures[0].error_type)
        self.assertNotIn("private question", repr(store.failures[0]))

    async def test_context_preparation_failure_is_recorded_before_callbacks(self):
        rows = [
            _manifest()["turns"][0],
            {**_manifest()["turns"][0], "itemId": "ITEM-2", "turnNo": 2},
        ]
        profile = _Profile(fail_context_on="ITEM-2")
        store = _Store()
        with self.assertRaisesRegex(RuntimeError, "private context"):
            await execute_cache_path_run(
                manifest=_manifest(rows), metadata=_metadata(), profile=profile,
                store=store, execution_adapter=_Execution(),
            )
        self.assertEqual([], profile.callback_calls)
        self.assertEqual("context_preparation", store.failures[0].stage)
        self.assertEqual("RuntimeError", store.failures[0].error_type)
        self.assertNotIn("private context", repr(store.failures[0]))

    async def test_real_store_keeps_checkpoint_and_marks_run_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git_dir = root / ".git"
            git_dir.mkdir()
            (git_dir / "config").write_text(
                '[remote "origin"]\n\turl = git@github.com:riido/riido-evals.git\n',
                encoding="utf-8",
            )
            (root / "README.md").write_text("# riido-evals\n", encoding="utf-8")
            (root / "golden").mkdir()
            (root / "golden" / "manifest.json").write_text("{}", encoding="utf-8")
            metadata = _metadata("persisted-failure")

            class TracedExecution:
                def __init__(self):
                    self.calls = 0

                async def execute(self, *, item_id, turn_no, replay_context, **_kwargs):
                    self.calls += 1
                    if self.calls == 2:
                        raise RuntimeError("do not persist this raw message")
                    trace = AnswerExecutionTrace(
                        trace_id=uuid.uuid4(),
                        target="CACHE_DECISION",
                        replay_context=replay_context,
                        events=(ExecutionTraceEvent(ExecutionStage.RETRIEVAL, StageOutcome.SUCCEEDED),),
                    )
                    return CacheDecisionObservation(
                        item_id=item_id,
                        turn_no=turn_no,
                        outcome="SERVED",
                        next_stage="NOT_APPLICABLE",
                        latency_ms=1,
                        model_calls=0,
                        embedding_calls=0,
                        execution_trace=trace,
                    )

            store = CachePathRunStore(root)
            rows = [_manifest()["turns"][0], {**_manifest()["turns"][0], "itemId": "ITEM-2", "turnNo": 2}]
            with self.assertRaisesRegex(RuntimeError, "do not persist"):
                await execute_cache_path_run(
                    manifest=_manifest(rows),
                    metadata=metadata,
                    profile=_Profile(),
                    store=store,
                    execution_adapter=TracedExecution(),
                )
            state_path = root / "evals/cache-path/state/persisted-failure.json"
            failure_path = root / "evals/cache-path/failures/persisted-failure.json"
            checkpoint_files = list((root / "evals/cache-path/checkpoints/persisted-failure").glob("*.json"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            self.assertEqual("FAILED", state["status"])
            self.assertEqual(1, state["checkpointCount"])
            self.assertEqual("execute", failure["failureStage"])
            self.assertEqual(1, len(checkpoint_files))
            self.assertNotIn(
                "do not persist",
                repr(state) + repr(failure) + checkpoint_files[0].read_text(encoding="utf-8"),
            )

    async def test_final_result_precedes_failed_success_state_transition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git_dir = root / ".git"
            git_dir.mkdir()
            (git_dir / "config").write_text(
                '[remote "origin"]\n\turl = git@github.com:riido/riido-evals.git\n',
                encoding="utf-8",
            )
            (root / "README.md").write_text("# riido-evals\n", encoding="utf-8")
            (root / "golden").mkdir()
            (root / "golden" / "manifest.json").write_text("{}", encoding="utf-8")
            metadata = _metadata("finalization-failure")
            base_store = CachePathRunStore(root)

            class SuccessTransitionFailureStore:
                def __getattr__(self, name):
                    return getattr(base_store, name)

                def succeed(self, _metadata):
                    raise OSError("state finalization write failed")

            class SuccessfulExecution:
                async def execute(self, *, item_id, turn_no, replay_context, **_kwargs):
                    trace = AnswerExecutionTrace(
                        trace_id=uuid.uuid4(),
                        target="CACHE_DECISION",
                        replay_context=replay_context,
                        events=(ExecutionTraceEvent(ExecutionStage.RETRIEVAL, StageOutcome.SUCCEEDED),),
                    )
                    return CacheDecisionObservation(
                        item_id=item_id,
                        turn_no=turn_no,
                        outcome="SERVED",
                        next_stage="NOT_APPLICABLE",
                        latency_ms=1,
                        model_calls=0,
                        embedding_calls=0,
                        execution_trace=trace,
                    )

            with self.assertRaisesRegex(OSError, "state finalization"):
                await execute_cache_path_run(
                    manifest=_manifest(),
                    metadata=metadata,
                    profile=_Profile(),
                    store=SuccessTransitionFailureStore(),
                    execution_adapter=SuccessfulExecution(),
                )
            result_path = root / "evals/cache-path/finalization-failure.json"
            state = json.loads(
                (root / "evals/cache-path/state/finalization-failure.json").read_text(encoding="utf-8")
            )
            failure = json.loads(
                (root / "evals/cache-path/failures/finalization-failure.json").read_text(encoding="utf-8")
            )
            self.assertTrue(result_path.is_file())
            self.assertEqual("FINALIZATION_INCOMPLETE", state["status"])
            self.assertEqual("FINALIZATION_INCOMPLETE", failure["status"])


if __name__ == "__main__":
    unittest.main()
