import asyncio
import hashlib
import unittest

from evaluation.core.plan import (
    EvaluationPlan,
    EvaluationPlanOutput,
    EvaluationVariant,
    EvaluationVariantResult,
    PreparedEvaluationInputs,
    run_evaluation_plan,
)


_SELECTION = hashlib.sha256(b"selection").hexdigest()
_INPUT = hashlib.sha256(b"inputs").hexdigest()
_RESULT = hashlib.sha256(b"result").hexdigest()


def make_variant(number: int) -> EvaluationVariant:
    return EvaluationVariant(
        variant_id=f"v{number}",
        run_id=f"run{number}",
        configuration={"model": "model-v1", "params": {"temperature": 0.0, "enabled": True}},
    )


def make_plan(count: int = 1) -> EvaluationPlan:
    return EvaluationPlan("plan1", "group1", _SELECTION, tuple(make_variant(i) for i in range(count)))


class Executor:
    def __init__(self):
        self.preflight_calls = 0
        self.prepare_calls = 0
        self.finalize_calls = 0
        self.seen_prepared = []
        self.executed = []
        self.prepare_error = None
        self.execute_error_at = None
        self.finalize_error = None
        self.block = None

    def preflight(self, plan_value):
        self.preflight_calls += 1

    async def prepare(self, plan_value):
        self.prepare_calls += 1
        if self.prepare_error:
            raise self.prepare_error
        return PreparedEvaluationInputs(_INPUT, object())

    async def execute(self, plan_value, variant_value, prepared):
        self.executed.append(variant_value.variant_id)
        self.seen_prepared.append(prepared)
        if self.block is not None:
            await self.block.wait()
        if self.execute_error_at == variant_value.variant_id:
            raise RuntimeError("child failed")
        return EvaluationVariantResult(
            variant_value.variant_id,
            variant_value.run_id,
            f"evals/{plan_value.group}/{variant_value.run_id}.json",
            _RESULT,
        )

    async def finalize(self, prepared):
        self.finalize_calls += 1
        if self.finalize_error:
            raise self.finalize_error


class Store:
    def __init__(self):
        self.calls = []
        self.fail_error = None
        self.succeed_error = None

    def reserve(self, plan_value):
        self.calls.append(("reserve",))

    def prepared(self, plan_value, prepared):
        self.calls.append(("prepared", prepared.input_sha256))

    def checkpoint(self, plan_value, result):
        self.calls.append(("checkpoint", result.variant_id))

    def succeed(self, plan_value, prepared, results):
        self.calls.append(("succeed", tuple(result.variant_id for result in results)))
        if self.succeed_error:
            raise self.succeed_error
        return f"evals/{plan_value.group}/{plan_value.plan_id}.json"

    def fail(self, plan_value, stage, variant_id, error_type):
        self.calls.append(("fail", stage, variant_id, error_type))
        if self.fail_error:
            raise self.fail_error


class EvaluationVariantTests(unittest.TestCase):
    def test_config_is_validated_snapshotted_and_copied_on_export(self):
        source = {"model": "model-v1", "params": {"tags": ["a", "b"], "count": 2}}
        item = EvaluationVariant("variant1", "run1", source)
        digest = item.config_sha256
        source["model"] = "changed"
        source["params"]["tags"].append("c")
        exported = item.to_dict()
        exported["configuration"]["params"]["tags"].append("d")

        self.assertEqual(item.config_sha256, digest)
        self.assertEqual(item.configuration["model"], "model-v1")
        self.assertEqual(item.configuration["params"]["tags"], ("a", "b"))
        self.assertEqual(item.to_dict()["configSha256"], digest)
        with self.assertRaises(TypeError):
            item.configuration["model"] = "changed"
        with self.assertRaises(TypeError):
            item.configuration["params"]._data["x"] = "y"
        with self.assertRaises(AttributeError):
            item.configuration["params"]._data = {}

    def test_variant_rejects_unsafe_fields(self):
        invalid = [
            {"variant_id": "../escape"},
            {"run_id": "has space"},
            {"configuration": {"prompt": "please answer this question"}},
            {"configuration": {"params": {"apiKey": "abc"}}},
            {"configuration": {"params": {"authorization": "bearer"}}},
            {"configuration": {"params": {"temperature": float("inf")}}},
            {"configuration": {"params": {"temperature": float("nan")}}},
            {"configuration": {"params": {"value": object()}}},
            {"configuration": {"params": {"value": "x" * 201}}},
            {"configuration": {"模型": "v1"}},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                args = {"variant_id": "v1", "run_id": "r1", "configuration": {"model": "m1"}}
                args.update(kwargs)
                with self.assertRaises(ValueError):
                    EvaluationVariant(**args)

    def test_variant_rejects_configuration_over_64_kib(self):
        with self.assertRaisesRegex(ValueError, "64 KiB"):
            EvaluationVariant("v1", "r1", {"items": ["x" * 200] * 400})


class EvaluationPlanTests(unittest.TestCase):
    def test_plan_rejects_empty_and_duplicate_variant_or_run_ids(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            EvaluationPlan("plan1", "group1", _SELECTION, ())
        with self.assertRaisesRegex(ValueError, "variant_id"):
            EvaluationPlan("plan1", "group1", _SELECTION, (make_variant(1), make_variant(1)))
        duplicate_run = EvaluationVariant("v2", "run1", {"model": "model-v1"})
        with self.assertRaisesRegex(ValueError, "run_id"):
            EvaluationPlan("plan1", "group1", _SELECTION, (make_variant(1), duplicate_run))

    def test_plan_and_result_reject_bad_identifiers_hashes_and_paths(self):
        with self.assertRaises(ValueError):
            EvaluationPlan("nested/path", "group1", _SELECTION, (make_variant(1),))
        with self.assertRaises(ValueError):
            EvaluationPlan("plan1", "../group", _SELECTION, (make_variant(1),))
        with self.assertRaises(ValueError):
            EvaluationPlan("plan1", "group1", "A" * 64, (make_variant(1),))
        with self.assertRaises(ValueError):
            EvaluationVariantResult("v1", "r1", "evals/g/../r1.json", _RESULT)

    def test_to_dict_uses_camel_case_keys(self):
        item = make_variant(1)
        plan_value = make_plan()
        self.assertEqual(set(item.to_dict()), {"variantId", "runId", "configuration", "configSha256"})
        self.assertEqual(
            set(plan_value.to_dict()), {"planId", "group", "selectionSha256", "variants"}
        )


class PreparedEvaluationInputsTests(unittest.TestCase):
    def test_optional_evidence_is_snapshotted_and_exported_as_a_fresh_copy(self):
        source = {"calls": {"model-a": 2}, "latency_ms": 12.5, "error_code": "none"}
        prepared = PreparedEvaluationInputs(_INPUT, object(), source)
        source["calls"]["model-a"] = 99
        exported = prepared.preparation_evidence_dict()
        exported["calls"]["model-a"] = 100

        self.assertEqual(prepared.preparation_evidence_dict(), {
            "calls": {"model-a": 2}, "latency_ms": 12.5, "error_code": "none"
        })

    def test_two_argument_construction_defaults_to_empty_evidence_and_rejects_secrets(self):
        prepared = PreparedEvaluationInputs(_INPUT, object())
        self.assertEqual(prepared.preparation_evidence_dict(), {})
        with self.assertRaises(ValueError):
            PreparedEvaluationInputs(_INPUT, object(), {"provider": {"api_key": "hidden"}})

    def test_repr_does_not_include_opaque_value(self):
        prepared = PreparedEvaluationInputs(_INPUT, "private question text")
        self.assertNotIn("private question text", repr(prepared))


class EvaluationPlanExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_prepares_once_executes_sequentially_and_checkpoints_each_variant(self):
        for count in (1, 3):
            with self.subTest(count=count):
                executor, store = Executor(), Store()
                output = await run_evaluation_plan(
                    plan=make_plan(count), executor=executor, store=store
                )
                self.assertIsInstance(output, EvaluationPlanOutput)
                self.assertEqual(output.result_path, "evals/group1/plan1.json")
                self.assertEqual([result.variant_id for result in output.results], [f"v{i}" for i in range(count)])
                self.assertEqual(executor.prepare_calls, 1)
                self.assertEqual(executor.finalize_calls, 1)
                self.assertEqual(len(executor.seen_prepared), count)
                self.assertTrue(all(value is executor.seen_prepared[0] for value in executor.seen_prepared))
                self.assertEqual(
                    [call for call in store.calls if call[0] == "checkpoint"],
                    [("checkpoint", f"v{i}") for i in range(count)],
                )
                self.assertEqual(store.calls[-1], ("succeed", tuple(f"v{i}" for i in range(count))))

    async def test_preflight_failure_happens_before_reservation_or_preparation(self):
        class BadPreflight(Executor):
            def preflight(self, plan_value):
                super().preflight(plan_value)
                raise ValueError("preflight")

        executor, store = BadPreflight(), Store()
        with self.assertRaisesRegex(ValueError, "preflight"):
            await run_evaluation_plan(plan=make_plan(), executor=executor, store=store)
        self.assertEqual(executor.prepare_calls, 0)
        self.assertEqual(store.calls, [])

    async def test_prepare_failure_is_recorded_without_finalize(self):
        executor, store = Executor(), Store()
        executor.prepare_error = LookupError("private details")
        with self.assertRaisesRegex(LookupError, "private details"):
            await run_evaluation_plan(plan=make_plan(), executor=executor, store=store)
        self.assertEqual(executor.finalize_calls, 0)
        self.assertEqual(store.calls[-1], ("fail", "prepare", None, "LookupError"))
        self.assertNotIn("private details", repr(store.calls))

    async def test_child_failure_finalizes_and_records_variant_without_error_text(self):
        executor, store = Executor(), Store()
        executor.execute_error_at = "v1"
        with self.assertRaisesRegex(RuntimeError, "child failed"):
            await run_evaluation_plan(plan=make_plan(3), executor=executor, store=store)
        self.assertEqual(executor.executed, ["v0", "v1"])
        self.assertEqual(executor.finalize_calls, 1)
        self.assertEqual(store.calls[-1], ("fail", "execute", "v1", "RuntimeError"))

    async def test_mismatched_result_path_is_rejected_before_checkpoint(self):
        class BadPathExecutor(Executor):
            async def execute(self, plan_value, variant_value, prepared):
                result = await super().execute(plan_value, variant_value, prepared)
                return EvaluationVariantResult(
                    result.variant_id,
                    result.run_id,
                    "evals/other-group/run0.json",
                    result.result_sha256,
                )

        executor, store = BadPathExecutor(), Store()
        with self.assertRaisesRegex(ValueError, "identifiers"):
            await run_evaluation_plan(plan=make_plan(), executor=executor, store=store)
        self.assertNotIn(("checkpoint", "v0"), store.calls)
        self.assertEqual(store.calls[-1], ("fail", "execute", "v0", "ValueError"))

    async def test_cancellation_is_recorded_and_cleanup_runs(self):
        executor, store = Executor(), Store()
        executor.block = asyncio.Event()
        task = asyncio.create_task(run_evaluation_plan(plan=make_plan(), executor=executor, store=store))
        while not executor.executed:
            await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(executor.finalize_calls, 1)
        self.assertEqual(store.calls[-1], ("fail", "execute", "v0", "CancelledError"))

    async def test_finalize_error_does_not_mask_child_error_and_is_chained(self):
        executor, store = Executor(), Store()
        executor.execute_error_at = "v0"
        executor.finalize_error = OSError("cleanup")
        with self.assertRaises(RuntimeError) as raised:
            await run_evaluation_plan(plan=make_plan(), executor=executor, store=store)
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertEqual(store.calls[-1], ("fail", "execute", "v0", "RuntimeError"))

    async def test_failure_store_error_is_chained_without_masking_original(self):
        executor, store = Executor(), Store()
        executor.execute_error_at = "v0"
        store.fail_error = OSError("store unavailable")
        with self.assertRaises(RuntimeError) as raised:
            await run_evaluation_plan(plan=make_plan(), executor=executor, store=store)
        self.assertIsInstance(raised.exception.__cause__, OSError)

    async def test_succeed_store_error_is_followed_by_failure_transition(self):
        executor, store = Executor(), Store()
        store.succeed_error = OSError("persist failed")
        with self.assertRaisesRegex(OSError, "persist failed"):
            await run_evaluation_plan(plan=make_plan(), executor=executor, store=store)
        self.assertEqual(executor.finalize_calls, 1)
        self.assertEqual(store.calls[-1], ("fail", "succeed", None, "OSError"))


if __name__ == "__main__":
    unittest.main()
