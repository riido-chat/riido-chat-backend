"""Fixed-input N-variant orchestration over the existing cache-path run path."""

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from app.answering.trace import ReplayContext
from app.answering.use_case import AnswerQuestion
from app.retrieval.models import HybridSearchCall
from evaluation.cache_path_eval.execution import (
    CacheDecisionCallbacks,
    CacheDecisionExecutionAdapter,
    CacheDecisionMetrics,
    CacheDecisionObservation,
)
from evaluation.cache_path_eval.evidence import (
    CacheRuntimeEvidence,
    ModelCallEvidence,
    ModelRequestEvidenceRecord,
)
from evaluation.cache_path_eval.plan import (
    CachePathPlanRunSpec,
    PreparedCacheInputs,
    _EvidenceCheckingExecutionAdapter,
    compare_cache_path_plan,
    execute_cache_path_plan,
)
from evaluation.cache_path_eval.plan_store import CachePathPlanStore
from evaluation.cache_path_eval.store import CachePathRunMetadata, CachePathRunStore
from evaluation.core.plan import EvaluationPlan, EvaluationVariant


def _registry(root: Path) -> None:
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = git@github.com:swyg/riido-evals.git\n', encoding="utf-8"
    )
    (root / "README.md").write_text("# riido-evals\n", encoding="utf-8")
    (root / "golden").mkdir()
    (root / "golden" / "manifest.json").write_text("{}", encoding="utf-8")


def _metadata(run_id: str, judge_model: str) -> CachePathRunMetadata:
    return CachePathRunMetadata(
        run_id=run_id,
        golden_version="golden-v1",
        golden_commit="c" * 40,
        source_corpus_version="corpus-v1",
        app_build_commit="a" * 40,
        service_key="test-service",
        profile_revision_id=2,
        document_group_id=3,
        index_version_id=4,
        embedding_config_id=5,
        cache_policy_version="policy-v1",
        embedding_model_version="embed-v1",
        judge_model_version=judge_model,
        judge_prompt_version="prompt-v1",
        executed_at=f"2026-10-20T12:00:0{run_id[-1]}Z",
    )


def _manifest() -> dict:
    return {
        "group": "cache-path",
        "goldenVersion": "golden-v1",
        "sourceCorpusVersion": "corpus-v1",
        "inputs": {"draftsSha256": "d" * 64, "judgeLabelsSha256": "e" * 64},
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


class CachePathPlanTests(unittest.IsolatedAsyncioTestCase):
    def test_prepared_inputs_reject_nested_labels_and_invalid_or_duplicate_references(self):
        with self.assertRaisesRegex(ValueError, "expected-label"):
            PreparedCacheInputs(
                references=(("CACHE-1", 1),),
                payloads={
                    ("CACHE-1", 1): {"candidates": [{"group": {"expectedDecision": "CONNECT"}}]}
                },
            )
        with self.assertRaisesRegex(ValueError, "turn_no"):
            PreparedCacheInputs(references=(("CACHE-1", True),), payloads={})
        with self.assertRaisesRegex(ValueError, "unique"):
            PreparedCacheInputs(
                references=(("CACHE-1", 1), ("CACHE-1", 1)),
                payloads={("CACHE-1", 1): None},
            )
        with self.assertRaisesRegex(ValueError, "every fixed turn"):
            PreparedCacheInputs(references=(("CACHE-1", 1),), payloads={})

    async def test_evidence_checker_requires_measured_success_and_blocks_repeated_shared_cost(self):
        prepared = PreparedCacheInputs(
            references=(("CACHE-1", 1),),
            payloads={("CACHE-1", 1): None},
        )

        class Delegate:
            def __init__(self, observation):
                self.observation = observation

            async def execute(self, **_kwargs):
                return self.observation

        served_without_evidence = CacheDecisionObservation(
            "CACHE-1", 1, "SERVED", "NOT_APPLICABLE", 1, 0, 0
        )
        checker = _EvidenceCheckingExecutionAdapter(
            Delegate(served_without_evidence), prepared, "judge-1", "prompt-v1"
        )
        with self.assertRaisesRegex(ValueError, "require CacheRuntimeEvidence"):
            await checker.execute()

        exact_served = CacheDecisionObservation(
            "CACHE-1",
            1,
            "SERVED",
            "NOT_APPLICABLE",
            1,
            0,
            0,
            evidence=CacheRuntimeEvidence(exact_source_classification_id=77),
        )
        exact_checker = _EvidenceCheckingExecutionAdapter(
            Delegate(exact_served), prepared, "judge-1", "prompt-v1"
        )
        await exact_checker.execute()

        rewrite_evidence = CacheRuntimeEvidence(
            model_calls=(
                ModelCallEvidence(
                    "QUERY_EMBEDDING",
                    "test-provider",
                    "embed-v1",
                    None,
                    True,
                    1,
                    0,
                    2,
                    2,
                    None,
                    None,
                ),
            )
        )
        repeated_embedding = CacheDecisionObservation(
            "CACHE-1", 1, "SKIPPED", "NOT_APPLICABLE", 1, 1, 1, evidence=rewrite_evidence
        )
        repeat_checker = _EvidenceCheckingExecutionAdapter(
            Delegate(repeated_embedding), prepared, "judge-1", "prompt-v1"
        )
        with self.assertRaisesRegex(ValueError, "cannot repeat shared"):
            await repeat_checker.execute()

        for model_name, prompt_version, message in (
            ("other-judge", "prompt-v1", "selected judge model"),
            ("judge-1", "other-prompt", "selected judge prompt"),
        ):
            with self.subTest(model_name=model_name, prompt_version=prompt_version):
                wrong_trace = ModelCallEvidence(
                    purpose="QUESTION_CLASSIFICATION",
                    provider="test-provider",
                    model_name=model_name,
                    prompt_version=prompt_version,
                    succeeded=True,
                    latency_ms=1,
                    retry_count=0,
                    input_tokens=1,
                    output_tokens=1,
                    cached_input_tokens=None,
                    reasoning_tokens=None,
                )
                wrong_model_observation = CacheDecisionObservation(
                    "CACHE-1",
                    1,
                    "SERVED",
                    "NOT_APPLICABLE",
                    1,
                    1,
                    0,
                    evidence=CacheRuntimeEvidence(model_calls=(wrong_trace,)),
                )
                wrong_checker = _EvidenceCheckingExecutionAdapter(
                    Delegate(wrong_model_observation), prepared, "judge-1", "prompt-v1"
                )
                with self.assertRaisesRegex(ValueError, message):
                    await wrong_checker.execute()

    async def test_prepare_validation_failure_closes_returned_bundle_exactly_once(self):
        for label, references, payloads, prep_evidence, error_text in (
            ("coverage", (), {}, {}, "cover each selected turn"),
            (
                "unsafe-evidence",
                (("CACHE-1", 1),),
                {("CACHE-1", 1): None},
                {"apiKey": "not-a-real-key"},
                "credential field",
            ),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _registry(root)
                manifest = _manifest()
                metadata = _metadata("run-1", "judge-1")
                spec = CachePathPlanRunSpec(metadata, {})
                variant = EvaluationVariant("variant-1", "run-1", spec.variant_configuration())
                selection_hash = hashlib.sha256(
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                plan_id = f"plan-{label}"
                finalized = []

                async def prepare(_refs):
                    if label == "coverage":
                        _refs["turns"].clear()
                    return PreparedCacheInputs(
                        references=references,
                        payloads=payloads,
                        preparation_evidence=prep_evidence,
                        finalizer=lambda _bundle: finalized.append("closed"),
                    )

                def profile_factory(**_kwargs):
                    raise AssertionError("profile factory must not run before preparation validation")

                profile_factory.supports_fixed_prepared_inputs = True
                with self.assertRaisesRegex(ValueError, error_text):
                    await execute_cache_path_plan(
                        plan=EvaluationPlan(plan_id, "cache-path", selection_hash, (variant,)),
                        manifest=manifest,
                        run_specs={variant.variant_id: spec},
                        store=CachePathRunStore(root),
                        plan_store=CachePathPlanStore(root),
                        execution_adapter=CacheDecisionExecutionAdapter(
                            AnswerQuestion(SimpleNamespace(generate_answer=AsyncMock()))
                        ),
                        shared_input_preparer=prepare,
                        prepared_profile_factory=profile_factory,
                    )
                self.assertEqual(["closed"], finalized)
                state = json.loads(
                    (root / "evals" / "cache-path" / "plans" / "state" / f"{plan_id}.json").read_text()
                )
                self.assertEqual("FAILED", state["status"])

    async def test_successful_judge_uses_prepared_production_presentation_and_persists_hashes(self):
        from app.core.stage_settings import StageSettings
        from app.question_grouping.cache_decision import CacheDecisionService
        from app.question_grouping.judge_client import QuestionJudgeClient
        from tests.test_help_chatbot_identity import (
            ALPHA_CHUNK,
            CATALOG,
            DOCUMENT_GROUP_ID,
            EMBEDDING_CONFIG_ID,
            GATE_INPUTS,
            INDEX_VERSION_ID,
            OUTLINES,
            PROFILE_REVISION_ID,
            SCOPE,
            _judge_output,
            _unit_vector,
        )
        from app.retrieval.models import HybridRetrievalResult

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _registry(root)
            manifest = {
                "group": "cache-path",
                "goldenVersion": "golden-v1",
                "sourceCorpusVersion": "corpus-v1",
                "inputs": {"draftsSha256": "d" * 64, "judgeLabelsSha256": "e" * 64},
                "turns": [
                    {
                        "itemId": "CACHE-JUDGE",
                        "path": "P6",
                        "turnNo": 1,
                        "kind": "judge",
                        "expectedExactHit": False,
                        "expectedServed": True,
                        "expectedJudgeDecision": "CONNECT",
                        "expectedSubproblemKeysAny": ["alpha.enable"],
                    }
                ],
            }
            metadata = CachePathRunMetadata(
                run_id="judge-success-1",
                golden_version="golden-v1",
                golden_commit="c" * 40,
                source_corpus_version="corpus-v1",
                app_build_commit="a" * 40,
                service_key="test-service",
                profile_revision_id=PROFILE_REVISION_ID,
                document_group_id=DOCUMENT_GROUP_ID,
                index_version_id=INDEX_VERSION_ID,
                embedding_config_id=EMBEDDING_CONFIG_ID,
                cache_policy_version="policy-v1",
                embedding_model_version="embed-v1",
                judge_model_version="judge-fixture",
                judge_prompt_version="prompt-v1",
                executed_at="2026-10-20T12:00:00Z",
            )
            stage = StageSettings(
                model_name="judge-fixture",
                prompt_version="prompt-v1",
                instructions="Synthetic cache classification instructions.",
                params={"reasoning": {"effort": "low"}, "max_output_tokens": 128},
            )
            presentation_by_turn = {}
            candidate_service = CacheDecisionService(
                reader=SimpleNamespace(
                    load_index_scope=AsyncMock(return_value=SCOPE),
                    load_subproblem_catalog=AsyncMock(return_value=CATALOG),
                    load_gate_inputs=AsyncMock(return_value=GATE_INPUTS),
                ),
                outline_reader=SimpleNamespace(load_outlines=AsyncMock(return_value=OUTLINES)),
            )
            embedding = tuple(_unit_vector(0))
            retrieval = HybridSearchCall(
                fused_results=(
                    HybridRetrievalResult(
                        chunk=ALPHA_CHUNK,
                        rrf_score=0.9,
                        final_rank=1,
                        bm25_rank=1,
                        vector_rank=1,
                    ),
                ),
                retrieval_query="synthetic resolved query",
                query_embedding=embedding,
            )

            class FakeResponses:
                def __init__(self):
                    self.requests = []

                async def create(self, **request):
                    self.requests.append(request)
                    return SimpleNamespace(
                        status="completed",
                        output_text=_judge_output("CONNECT"),
                        model="judge-fixture",
                        usage=SimpleNamespace(input_tokens=23, output_tokens=7),
                    )

            responses = FakeResponses()
            judge_client = QuestionJudgeClient(client=SimpleNamespace(responses=responses))
            decision_service = CacheDecisionService(
                reader=SimpleNamespace(load_gate_inputs=AsyncMock(return_value=GATE_INPUTS))
            )
            generate_answer = AsyncMock()
            execution_adapter = CacheDecisionExecutionAdapter(
                AnswerQuestion(SimpleNamespace(generate_answer=generate_answer))
            )

            async def prepare(reference_manifest):
                self.assertEqual(
                    [{"itemId": "CACHE-JUDGE", "turnNo": 1}],
                    reference_manifest["turns"],
                )
                candidate_preparation = await candidate_service.prepare_candidates(
                    resolved_query="[synthetic] how do I enable alpha?",
                    embedding=embedding,
                    index_version_id=INDEX_VERSION_ID,
                    document_group_id=DOCUMENT_GROUP_ID,
                    retrieval=retrieval,
                    seed="plan-fixed-input-seed",
                )
                self.assertTrue(candidate_preparation.ready)
                presentation_by_turn[("CACHE-JUDGE", 1)] = candidate_preparation.presentation
                return PreparedCacheInputs(
                    references=(("CACHE-JUDGE", 1),),
                    payloads={
                        ("CACHE-JUDGE", 1): candidate_preparation.presentation.payload
                    },
                    opaque_context={"scope": candidate_preparation.scope},
                    preparation_evidence={"sharedCandidateBuilds": 1},
                )

            def profile_factory(*, prepared, **_kwargs):
                class Profile:
                    async def callbacks_for_turn(self, selected_turn):
                        payload = prepared.payload_for_turn(
                            selected_turn["itemId"], selected_turn["turnNo"]
                        )
                        presentation = presentation_by_turn[
                            (selected_turn["itemId"], selected_turn["turnNo"])
                        ]
                        self.assert_payload_equal(payload, presentation.payload)
                        state = {"judge": None, "result": None}

                        async def choose_source():
                            decision = await decision_service.judge(
                                client=judge_client,
                                presentation=presentation,
                                model=stage.model_name,
                                prompt_version=stage.prompt_version,
                                stage=stage,
                            )
                            state["judge"] = decision
                            state["result"] = await decision_service.decide_gate(
                                decision.judgment,
                                scope=SCOPE,
                                semantic_cache_enabled=True,
                            )
                            return state["result"]

                        def summarize(_answer):
                            result = state["result"]
                            judge_decision = state["judge"]
                            return CacheDecisionMetrics(
                                outcome=result.gate.outcome.value,
                                next_stage="NOT_APPLICABLE",
                                model_calls=1,
                                embedding_calls=0,
                                evidence=CacheRuntimeEvidence(
                                    subproblem_candidates=(),
                                    selected_subproblem_id=str(
                                        judge_decision.judgment.subproblem.subproblem_id
                                    ),
                                    model_calls=(
                                        ModelCallEvidence.from_trace(
                                            "QUESTION_CLASSIFICATION",
                                            judge_decision.call.trace,
                                        ),
                                    ),
                                ),
                            )

                        return CacheDecisionCallbacks(
                            resolve_query=AsyncMock(return_value="[synthetic] how do I enable alpha?"),
                            retrieve=AsyncMock(return_value=retrieval),
                            choose_source=choose_source,
                            should_choose_source=AsyncMock(return_value=True),
                            summarize=summarize,
                        )

                    @staticmethod
                    def assert_payload_equal(left, right):
                        if left != right:
                            raise AssertionError("prepared payload changed before judge call")

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

                return Profile()

            profile_factory.supports_fixed_prepared_inputs = True
            spec = CachePathPlanRunSpec(metadata, {"threshold": 0.7})
            variant = EvaluationVariant("variant-judge", metadata.run_id, spec.variant_configuration())
            selection_hash = hashlib.sha256(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            output = await execute_cache_path_plan(
                plan=EvaluationPlan("plan-judge-success", "cache-path", selection_hash, (variant,)),
                manifest=manifest,
                run_specs={variant.variant_id: spec},
                store=CachePathRunStore(root),
                plan_store=CachePathPlanStore(root),
                execution_adapter=execution_adapter,
                shared_input_preparer=prepare,
                prepared_profile_factory=profile_factory,
            )

            self.assertEqual(1, len(responses.requests))
            self.assertEqual(1, len(output.results))
            self.assertEqual(0, generate_answer.await_count)
            prepared_hash = hashlib.sha256(
                json.dumps(
                    presentation_by_turn[("CACHE-JUDGE", 1)].payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            child_path = root / "evals" / "cache-path" / f"{metadata.run_id}.json"
            child = json.loads(child_path.read_text(encoding="utf-8"))
            request_evidence = child["observations"][0]["evidence"]["modelCalls"][0]["requestEvidence"]
            parent = json.loads(Path(output.result_path).read_text(encoding="utf-8"))
            expected_plan_input_hash = hashlib.sha256(
                json.dumps(
                    [{"itemId": "CACHE-JUDGE", "turnNo": 1, "payloadSha256": prepared_hash}],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            self.assertEqual(prepared_hash, request_evidence["commonInputSha256"])
            self.assertEqual(expected_plan_input_hash, parent["inputSha256"])
            self.assertEqual("judge-fixture", request_evidence["responseModelName"])
            self.assertEqual("SERVED", child["observations"][0]["outcome"])
            checkpoint = json.loads(
                next((root / "evals" / "cache-path" / "checkpoints" / metadata.run_id).glob("*.json")).read_text()
            )
            self.assertEqual(
                prepared_hash,
                checkpoint["evidence"]["modelCalls"][0]["requestEvidence"]["commonInputSha256"],
            )
            self.assertEqual(
                "judge-success-1",
                compare_cache_path_plan(registry_root=root, plan_id="plan-judge-success")["variants"][0]["runId"],
            )

            plan_path = Path(output.result_path)
            plan_text = plan_path.read_text(encoding="utf-8")
            plan_record = json.loads(plan_text)
            plan_record["variants"][0]["configSha256"] = "f" * 64
            plan_path.write_text(json.dumps(plan_record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configuration digest"):
                compare_cache_path_plan(registry_root=root, plan_id="plan-judge-success")
            plan_path.write_text(plan_text, encoding="utf-8")

            state_path = root / "evals" / "cache-path" / "plans" / "state" / "plan-judge-success.json"
            state_text = state_path.read_text(encoding="utf-8")
            state = json.loads(state_text)
            state["completedVariants"][0]["resultSha256"] = "f" * 64
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "plan state"):
                compare_cache_path_plan(registry_root=root, plan_id="plan-judge-success")
            state_path.write_text(state_text, encoding="utf-8")

            child_path.write_bytes(child_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "child run result hash"):
                compare_cache_path_plan(registry_root=root, plan_id="plan-judge-success")

    async def test_one_and_three_variants_share_one_immutable_preparation(self):
        for count in (1, 3):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _registry(root)
                manifest = _manifest()
                manifest["turns"][0].update(
                    kind="judge",
                    expectedJudgeDecision="CONNECT",
                    expectedSubproblemKeysAny=["fixture.key"],
                )
                config = {"threshold": 0.75}
                specs = {}
                variants = []
                for index in range(count):
                    variant_id = f"variant-{index + 1}"
                    run_id = f"run-{index + 1}"
                    metadata = _metadata(run_id, f"judge-{index + 1}")
                    spec = CachePathPlanRunSpec(metadata, config)
                    specs[variant_id] = spec
                    variants.append(
                        EvaluationVariant(variant_id, run_id, spec.variant_configuration())
                    )
                selection_hash = hashlib.sha256(
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                plan = EvaluationPlan("plan-fixed", "cache-path", selection_hash, tuple(variants))
                preparer_calls = []
                payloads_seen = []
                generate_answer = AsyncMock()
                adapter = CacheDecisionExecutionAdapter(
                    AnswerQuestion(SimpleNamespace(generate_answer=generate_answer))
                )

                async def prepare(references):
                    preparer_calls.append(references)
                    self.assertNotIn("expectedServed", repr(references))
                    return PreparedCacheInputs(
                        references=(("CACHE-1", 1),),
                        payloads={("CACHE-1", 1): {"question": "private", "items": ["fixed"]}},
                        preparation_evidence={"embeddingCalls": 2, "latencyMs": 13},
                    )

                def profile_factory(*, variant, spec, prepared, reference_manifest):
                    class Profile:
                        async def callbacks_for_turn(self, selected_turn):
                            payload = prepared.payload_for_turn(selected_turn["itemId"], selected_turn["turnNo"])
                            payload["items"].append(variant.variant_id)
                            payloads_seen.append(tuple(payload["items"]))
                            return CacheDecisionCallbacks(
                                resolve_query=AsyncMock(return_value=payload["question"]),
                                retrieve=AsyncMock(return_value=HybridSearchCall()),
                                choose_source=AsyncMock(return_value=object()),
                                summarize=Mock(
                                    return_value=CacheDecisionMetrics(
                                        outcome="SERVED",
                                        next_stage="NOT_APPLICABLE",
                                        model_calls=0,
                                        embedding_calls=0,
                                        evidence=CacheRuntimeEvidence(
                                            exact_source_classification_id=77
                                        ),
                                    )
                                ),
                            )

                        def replay_context_for_turn(self, metadata, selected_turn):
                            return ReplayContext(
                                run_id=metadata.run_id,
                                item_id=selected_turn["itemId"],
                                turn_no=selected_turn["turnNo"],
                                golden_version=metadata.golden_version,
                                source_corpus_version=metadata.source_corpus_version,
                                service_version=metadata.service_key,
                                code_revision=metadata.app_build_commit,
                                cache_policy_version=metadata.cache_policy_version,
                                embedding_model_version=metadata.embedding_model_version,
                            )

                    return Profile()

                profile_factory.supports_fixed_prepared_inputs = True

                output = await execute_cache_path_plan(
                    plan=plan,
                    manifest=manifest,
                    run_specs=specs,
                    store=CachePathRunStore(root),
                    plan_store=CachePathPlanStore(root),
                    execution_adapter=adapter,
                    shared_input_preparer=prepare,
                    prepared_profile_factory=profile_factory,
                )

                self.assertEqual(1, len(preparer_calls))
                self.assertEqual(count, len(payloads_seen))
                self.assertEqual(
                    [("fixed", variant.variant_id) for variant in variants], payloads_seen
                )
                self.assertEqual(count, len(output.results))
                self.assertEqual(0, generate_answer.await_count)
                record = json.loads(Path(output.result_path).read_text(encoding="utf-8"))
                self.assertEqual(2, record["preparationEvidence"]["embeddingCalls"])
                self.assertEqual(count, len(record["variants"]))
                self.assertEqual(
                    config,
                    record["variants"][0]["configuration"]["profileConfig"],
                )
                first_child = json.loads(
                    (root / "evals" / "cache-path" / "run-1.json").read_text(encoding="utf-8")
                )
                self.assertEqual("SERVED", first_child["observations"][0]["outcome"])
                comparison = compare_cache_path_plan(registry_root=root, plan_id="plan-fixed")
                self.assertEqual(record["inputSha256"], comparison["inputSha256"])
                self.assertEqual(count, len(comparison["variants"]))

    async def test_bad_selection_and_incompatible_target_fail_before_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _registry(root)
            manifest = _manifest()
            metadata = _metadata("run-1", "judge-1")
            spec = CachePathPlanRunSpec(metadata, {})
            variant = EvaluationVariant("variant-1", "run-1", spec.variant_configuration())
            bad_plan = EvaluationPlan("bad-plan", "cache-path", "f" * 64, (variant,))
            prepare = AsyncMock()
            kwargs = dict(
                manifest=manifest,
                run_specs={"variant-1": spec},
                store=CachePathRunStore(root),
                plan_store=CachePathPlanStore(root),
                execution_adapter=CacheDecisionExecutionAdapter(
                    AnswerQuestion(SimpleNamespace(generate_answer=AsyncMock()))
                ),
                shared_input_preparer=prepare,
                prepared_profile_factory=Mock(),
            )
            with self.assertRaisesRegex(ValueError, "selection hash"):
                await execute_cache_path_plan(plan=bad_plan, **kwargs)
            prepare.assert_not_awaited()
            self.assertFalse((root / "evals/cache-path/plans/state/bad-plan.json").exists())

            metadata_two = replace(
                _metadata("run-2", "judge-2"), embedding_config_id=99
            )
            spec_two = CachePathPlanRunSpec(metadata_two, {})
            variant_two = EvaluationVariant("variant-2", "run-2", spec_two.variant_configuration())
            valid_selection_hash = hashlib.sha256(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            mismatched_plan = EvaluationPlan(
                "bad-target",
                "cache-path",
                valid_selection_hash,
                (variant, variant_two),
            )
            with self.assertRaisesRegex(ValueError, "share golden, target"):
                await execute_cache_path_plan(
                    plan=mismatched_plan,
                    **{
                        **kwargs,
                        "run_specs": {"variant-1": spec, "variant-2": spec_two},
                    },
                )
            prepare.assert_not_awaited()
            self.assertFalse((root / "evals/cache-path/plans/state/bad-target.json").exists())

    async def test_mismatched_question_input_evidence_fails_child_and_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _registry(root)
            manifest = _manifest()
            metadata = _metadata("run-1", "judge-1")
            spec = CachePathPlanRunSpec(metadata, {})
            variant = EvaluationVariant("variant-1", "run-1", spec.variant_configuration())
            selection_hash = hashlib.sha256(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            plan = EvaluationPlan("plan-evidence", "cache-path", selection_hash, (variant,))
            finalized = []

            class WrongEvidenceAdapter:
                async def execute(self, *, item_id, turn_no, **_kwargs):
                    return CacheDecisionObservation(
                        item_id=item_id,
                        turn_no=turn_no,
                        outcome="SERVED",
                        next_stage="NOT_APPLICABLE",
                        latency_ms=1,
                        model_calls=1,
                        embedding_calls=0,
                        evidence=CacheRuntimeEvidence(
                            model_calls=(
                                ModelCallEvidence(
                                    purpose="QUESTION_CLASSIFICATION",
                                    provider="test-provider",
                                    model_name="judge-1",
                                    prompt_version="prompt-v1",
                                    succeeded=True,
                                    latency_ms=1,
                                    retry_count=0,
                                    input_tokens=1,
                                    output_tokens=1,
                                    cached_input_tokens=None,
                                    reasoning_tokens=None,
                                    request_evidence=ModelRequestEvidenceRecord(
                                        common_input_sha256="f" * 64,
                                        request_sha256="b" * 64,
                                        input_transform_version="transform-v1",
                                        instructions_sha256="c" * 64,
                                        response_model_name="QuestionDecision",
                                    ),
                                ),
                            )
                        ),
                    )

            async def prepare(_refs):
                return PreparedCacheInputs(
                    references=(("CACHE-1", 1),),
                    payloads={("CACHE-1", 1): {"prompt": "private"}},
                    finalizer=lambda _bundle: finalized.append("closed"),
                )

            def profile_factory(**_kwargs):
                class Profile:
                    async def callbacks_for_turn(self, _turn):
                        return CacheDecisionCallbacks(
                            resolve_query=AsyncMock(return_value="private"),
                            retrieve=AsyncMock(return_value=HybridSearchCall()),
                            choose_source=AsyncMock(return_value=object()),
                            summarize=Mock(return_value=CacheDecisionMetrics("SERVED", "NOT_APPLICABLE", 1, 0)),
                        )

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

                return Profile()

            profile_factory.supports_fixed_prepared_inputs = True

            with self.assertRaisesRegex(ValueError, "does not match fixed prepared input"):
                await execute_cache_path_plan(
                    plan=plan,
                    manifest=manifest,
                    run_specs={"variant-1": spec},
                    store=CachePathRunStore(root),
                    plan_store=CachePathPlanStore(root),
                    execution_adapter=WrongEvidenceAdapter(),
                    shared_input_preparer=prepare,
                    prepared_profile_factory=profile_factory,
                )

            self.assertEqual(["closed"], finalized)
            self.assertFalse((root / "evals/cache-path/run-1.json").exists())
            child_state = json.loads((root / "evals/cache-path/state/run-1.json").read_text())
            parent_state = json.loads((root / "evals/cache-path/plans/state/plan-evidence.json").read_text())
            self.assertEqual("FAILED", child_state["status"])
            self.assertEqual("FAILED", parent_state["status"])


if __name__ == "__main__":
    unittest.main()
