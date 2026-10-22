"""Privacy and lifecycle checks for private evaluation plan indexes."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evaluation.cache_path_eval.plan_store import CachePathPlanStore, EvaluationPlanStoreError
from evaluation.core.plan import (
    EvaluationPlan,
    EvaluationVariant,
    EvaluationVariantResult,
    PreparedEvaluationInputs,
)


class EvaluationPlanStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text(
            '[remote "origin"]\n\turl = git@github.com:swyg/riido-evals.git\n', encoding="utf-8"
        )
        (self.root / "README.md").write_text("# riido-evals\n", encoding="utf-8")
        (self.root / "golden").mkdir()
        (self.root / "golden" / "manifest.json").write_text("{}", encoding="utf-8")
        self.store = CachePathPlanStore(self.root)
        self.plan = EvaluationPlan(
            plan_id="plan-1",
            group="cache-path",
            selection_sha256="a" * 64,
            variants=(EvaluationVariant(
                variant_id="variant-1",
                run_id="run-1",
                configuration={"cacheEnabled": True},
            ),),
        )
        self.prepared = PreparedEvaluationInputs(
            input_sha256="c" * 64,
            value="private input payload",
            preparation_evidence={"source": "fixture-v1"},
        )

    def _result(self, *, run_id="run-1", path="evals/cache-path/run-1.json", contents=None):
        if contents is None:
            contents = json.dumps({"group": "cache-path", "runId": run_id, "safe": True}).encode()
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
        return EvaluationVariantResult(
            variant_id="variant-1",
            run_id=run_id,
            result_path=path,
            result_sha256=hashlib.sha256(contents).hexdigest(),
        )

    def test_reservation_is_create_only_and_prepared_payload_is_never_serialized(self):
        self.store.reserve(self.plan)
        state_path = self.root / "evals/cache-path/plans/state/plan-1.json"
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.reserve(self.plan)
        self.store.prepared(self.plan, self.prepared)
        rendered = state_path.read_text(encoding="utf-8")
        self.assertIn('"inputSha256"', rendered)
        self.assertIn('"preparationEvidence":{"source":"fixture-v1"}', rendered)
        self.assertNotIn("private input payload", rendered)

    def test_reservation_rejects_existing_final_plan_without_state(self):
        final_path = self.root / "evals/cache-path/plans/plan-1.json"
        final_path.parent.mkdir(parents=True)
        final_path.write_text("{}", encoding="utf-8")
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.reserve(self.plan)
        self.assertFalse((self.root / "evals/cache-path/plans/state/plan-1.json").exists())

    def test_reserved_plan_fingerprint_binds_selection_configuration_and_run_ids(self):
        self.store.reserve(self.plan)
        changed_plan = EvaluationPlan(
            plan_id="plan-1",
            group="cache-path",
            selection_sha256="b" * 64,
            variants=(EvaluationVariant(
                variant_id="variant-1",
                run_id="run-other",
                configuration={"cacheEnabled": False},
            ),),
        )
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.prepared(changed_plan, self.prepared)
        self.store.prepared(self.plan, self.prepared)
        changed_result = self._result(
            run_id="run-other",
            path="evals/cache-path/run-other.json",
        )
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.checkpoint(changed_plan, changed_result)
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.succeed(changed_plan, self.prepared, [changed_result])
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.fail(changed_plan, "execute", None, "ValueError")

    def test_checkpoint_and_success_validate_child_hash_and_create_index(self):
        self.store.reserve(self.plan)
        self.store.prepared(self.plan, self.prepared)
        result = self._result()
        self.store.checkpoint(self.plan, result)
        result_path = self.store.succeed(self.plan, self.prepared, [result])
        record = json.loads(Path(result_path).read_text(encoding="utf-8"))
        self.assertEqual(record["variants"][0]["variantId"], "variant-1")
        self.assertEqual(record["variants"][0]["status"], "SUCCEEDED")
        self.assertEqual(record["preparationEvidence"], {"source": "fixture-v1"})
        self.assertNotIn("value", record)
        state_path = self.root / "evals/cache-path/plans/state/plan-1.json"
        self.assertEqual(json.loads(state_path.read_text())["status"], "SUCCEEDED")
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.fail(self.plan, "execute", None, "ValueError")
        self.assertEqual(json.loads(state_path.read_text())["status"], "SUCCEEDED")

    def test_rejects_missing_or_changed_child_results(self):
        self.store.reserve(self.plan)
        self.store.prepared(self.plan, self.prepared)
        missing = EvaluationVariantResult(
            variant_id="variant-1",
            run_id="run-1",
            result_path="evals/cache-path/run-1.json",
            result_sha256="d" * 64,
        )
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.checkpoint(self.plan, missing)
        result = self._result(contents=b'{"safe":true}')
        (self.root / result.result_path).write_bytes(b'{"changed":true}')
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.checkpoint(self.plan, result)

    def test_rejects_child_reference_from_another_group(self):
        self.store.reserve(self.plan)
        self.store.prepared(self.plan, self.prepared)
        result = self._result(path="evals/other/run-1.json")
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.checkpoint(self.plan, result)

    def test_rejects_child_body_with_wrong_identity_or_invalid_json(self):
        self.store.reserve(self.plan)
        self.store.prepared(self.plan, self.prepared)
        for body in (
            {"group": "cache-path", "runId": "some-other-run"},
            {"group": "some-other-group", "runId": "run-1"},
        ):
            with self.subTest(body=body):
                result = self._result(contents=json.dumps(body).encode())
                with self.assertRaises(EvaluationPlanStoreError):
                    self.store.checkpoint(self.plan, result)
        malformed = self._result(contents=b"{not-json")
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.checkpoint(self.plan, malformed)

    def test_success_requires_checkpoints_to_match_final_refs_exactly(self):
        self.store.reserve(self.plan)
        self.store.prepared(self.plan, self.prepared)
        result = self._result()
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.succeed(self.plan, self.prepared, [result])
        self.store.checkpoint(self.plan, result)
        replacement = self._result(contents=json.dumps({"group": "cache-path", "runId": "run-1", "safe": False}).encode())
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.succeed(self.plan, self.prepared, [replacement])

    def test_state_write_failure_records_valid_incomplete_terminal_state(self):
        self.store.reserve(self.plan)
        self.store.prepared(self.plan, self.prepared)
        result = self._result()
        self.store.checkpoint(self.plan, result)
        original_write = self.store._write_atomic
        calls = 0

        def fail_once(destination, value):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("simulated state write failure")
            original_write(destination, value)

        self.store._write_atomic = fail_once
        with self.assertRaisesRegex(OSError, "simulated state write failure"):
            self.store.succeed(self.plan, self.prepared, [result])
        state_path = self.root / "evals/cache-path/plans/state/plan-1.json"
        serialized = state_path.read_text(encoding="utf-8")
        state = json.loads(serialized)
        self.assertEqual(state["status"], "FINALIZATION_INCOMPLETE")
        self.assertEqual(state["failureStage"], "succeed")
        self.assertEqual(state["errorType"], "FinalizationError")
        self.assertIn("failureStage", state)
        self.assertIn("errorType", state)

        # The core may report the same finalization exception through fail();
        # that follow-up preserves this terminal record after verifying plan identity.
        self.store.fail(self.plan, "succeed", None, "OSError")
        self.assertEqual(state_path.read_text(encoding="utf-8"), serialized)
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.succeed(self.plan, self.prepared, [result])

    def test_rejects_path_traversal_wrong_run_and_duplicate_checkpoint(self):
        self.store.reserve(self.plan)
        self.store.prepared(self.plan, self.prepared)
        traversal_path = "evals/cache-path/../other/run-1.json"
        traversal_bytes = b'{"safe":true}'
        traversal = SimpleNamespace(
            variant_id="variant-1",
            run_id="run-1",
            result_path=traversal_path,
            result_sha256=hashlib.sha256(traversal_bytes).hexdigest(),
        )
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.checkpoint(self.plan, traversal)
        wrong_run = self._result(run_id="run-other", path="evals/cache-path/run-other.json")
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.checkpoint(self.plan, wrong_run)
        good = self._result()
        self.store.checkpoint(self.plan, good)
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.checkpoint(self.plan, good)

    def test_failure_keeps_only_safe_exception_class_and_completed_refs(self):
        self.store.reserve(self.plan)
        self.store.prepared(self.plan, self.prepared)
        result = self._result()
        self.store.checkpoint(self.plan, result)
        self.store.fail(self.plan, "execute private stage", "variant-1", "ValueError: raw private exception")
        state_path = self.root / "evals/cache-path/plans/state/plan-1.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "FAILED")
        self.assertEqual(state["errorType"], "Exception")
        self.assertEqual(state["failureStage"], "unknown")
        self.assertEqual(len(state["completedVariants"]), 1)
        self.assertNotIn("raw private exception", state_path.read_text(encoding="utf-8"))
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.succeed(self.plan, self.prepared, [result])
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.checkpoint(self.plan, result)

    def test_plan_and_configuration_validation(self):
        bad = SimpleNamespace(
            plan_id="../outside",
            group="cache-path",
            selection_sha256="a" * 64,
            variants=self.plan.variants,
        )
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.reserve(bad)
        bad = SimpleNamespace(
            plan_id="plan-1",
            group="cache-path",
            selection_sha256="a" * 64,
            variants=(SimpleNamespace(
                variant_id="variant-1",
                run_id="run-1",
                configuration={"payload": object()},
                config_sha256="b" * 64,
            ),),
        )
        with self.assertRaises(EvaluationPlanStoreError):
            self.store.reserve(bad)


if __name__ == "__main__":
    unittest.main()
