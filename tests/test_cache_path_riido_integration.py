"""CLI-to-Riido profile integration with committed local inputs and fake services."""

import hashlib
import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.core.model_trace import ModelCallTrace
from app.core.model_trace import ModelRequestEvidence, canonical_request_sha256
from app.core.stage_settings import StageSettings
from app.question_grouping.judge_client import build_judge_request
from app.question_grouping.jev_judge_client import JEV_PROMPT_VERSION
from app.question_grouping.constants import EXACT_SOURCE_SUBPROBLEM_VERSION_STALE
from app.retrieval.embedding import (
    OPENAI_EMBEDDING_DIMENSIONS,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_PROVIDER,
)
from app.retrieval.embedding_config import EMBEDDING_INPUT_TEMPLATE_VERSION
from evaluation.cache_path_eval.cli import run_cache_path_cli
from evaluation.cache_path_eval.profile_registry import clear_profiles_for_tests
from evaluation.cache_path_eval.profiles.riido_chat import (
    create_riido_cache_decision_execution_adapter,
    create_riido_chat_plan_binding,
    register_riido_chat_profile,
)
from evaluation.cache_path_eval.readonly_service import EmbeddingConfigIdentity
from evaluation.cache_path_eval.store import CachePathRunMetadata
from evaluation.cache_path_eval.plan import CachePathPlanRunSpec, execute_cache_path_plan
from evaluation.cache_path_eval.plan_store import CachePathPlanStore
from evaluation.cache_path_eval.store import CachePathRunStore
from evaluation.core.plan import EvaluationPlan, EvaluationVariant
from app.retrieval.models import HybridRetrievalResult, HybridSearchCall
from tests.test_help_chatbot_identity import (
    ALPHA_CHUNK,
    CATALOG,
    DOCUMENT_GROUP_ID,
    EMBEDDING_CONFIG_ID,
    EXACT_MATCH,
    FIRST_QUESTION,
    FOLLOW_UP_QUESTION,
    INDEX_VERSION_ID,
    OUTLINES,
    PROFILE_REVISION_ID,
    SCOPE,
    GATE_INPUTS,
    _judge_output,
)


class _SnapshotContext:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.closed = False

    async def __aenter__(self):
        return self.snapshot

    async def __aexit__(self, *_args):
        self.closed = True


class _ReadOnlyData:
    def __init__(self, snapshot):
        self.contexts = []
        self.snapshot = snapshot

    def open_snapshot(self, _metadata, **_kwargs):
        context = _SnapshotContext(self.snapshot)
        self.contexts.append(context)
        return context


class RiidoCachePathCliIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        clear_profiles_for_tests()

    def tearDown(self):
        clear_profiles_for_tests()

    async def test_cli_runs_registered_real_profile_through_shared_cache_only_use_case(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = Path(temp)
            golden = registry / "golden"
            (golden / "sources").mkdir(parents=True)
            (registry / "README.md").write_text("# riido-evals\n", encoding="utf-8")
            drafts = {
                "version": "corpus-v1",
                "status": "draft",
                "items": [{
                    "id": "SYN-EXACT",
                    "historySynthetic": False,
                    "turns": [
                        {"question": FIRST_QUESTION, "historyAnswer": {"status": "COMPLETED", "content": "synthetic prior answer"}},
                        {"question": FOLLOW_UP_QUESTION},
                    ],
                }],
            }
            drafts_bytes = json.dumps(drafts, sort_keys=True).encode("utf-8")
            drafts_hash = hashlib.sha256(drafts_bytes).hexdigest()
            (golden / "sources" / "cache-drafts.json").write_bytes(drafts_bytes)
            (golden / "manifest.json").write_text(
                json.dumps({
                    "goldenVersion": "golden-v1",
                    "frozen": {"files": {"golden/sources/cache-drafts.json": {"sha256": drafts_hash}}},
                }),
                encoding="utf-8",
            )
            subprocess.run(["git", "init", str(registry)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(registry), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(registry), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(registry), "remote", "add", "origin", "https://example.invalid/riido-evals.git"], check=True)
            subprocess.run(["git", "-C", str(registry), "add", "README.md", "golden"], check=True)
            subprocess.run(["git", "-C", str(registry), "commit", "-m", "fixture"], check=True, capture_output=True)
            commit = subprocess.run(
                ["git", "-C", str(registry), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip()

            metadata = CachePathRunMetadata(
                run_id="riido-e2e-exact",
                golden_version="golden-v1",
                golden_commit=commit,
                source_corpus_version="corpus-v1",
                app_build_commit="b" * 40,
                service_key="HELP_CHATBOT",
                profile_revision_id=PROFILE_REVISION_ID,
                document_group_id=DOCUMENT_GROUP_ID,
                index_version_id=INDEX_VERSION_ID,
                embedding_config_id=EMBEDDING_CONFIG_ID,
                cache_policy_version="policy-v1",
                embedding_model_version=OPENAI_EMBEDDING_MODEL,
                judge_model_version="judge-v1",
                judge_prompt_version="prompt-v1",
                executed_at="2026-10-17T00:00:00Z",
            )
            manifest_path = registry / "selected.json"
            manifest_path.write_text(json.dumps({
                "group": "cache-path",
                "goldenVersion": "golden-v1",
                "sourceCorpusVersion": "corpus-v1",
                "inputs": {"draftsSha256": drafts_hash, "judgeLabelsSha256": "e" * 64},
                "turns": [
                    {
                        "itemId": "SYN-EXACT", "path": "P1", "turnNo": 1,
                        "kind": "exact_hit", "expectedExactHit": True, "expectedServed": True,
                    },
                    {
                        "itemId": "SYN-EXACT", "path": "P6", "turnNo": 2,
                        "kind": "judge", "expectedExactHit": False, "expectedServed": True,
                        "expectedJudgeDecision": "CONNECT", "expectedSubproblemKeysAny": ["alpha.enable"],
                    },
                ],
            }), encoding="utf-8")
            config_path = registry / "config.json"
            config_path.write_text(json.dumps({
                "metadata": {
                    "run_id": metadata.run_id,
                    "golden_version": metadata.golden_version,
                    "golden_commit": metadata.golden_commit,
                    "source_corpus_version": metadata.source_corpus_version,
                    "app_build_commit": metadata.app_build_commit,
                    "service_key": metadata.service_key,
                    "profile_revision_id": metadata.profile_revision_id,
                    "document_group_id": metadata.document_group_id,
                    "index_version_id": metadata.index_version_id,
                    "embedding_config_id": metadata.embedding_config_id,
                    "cache_policy_version": metadata.cache_policy_version,
                    "embedding_model_version": metadata.embedding_model_version,
                    "judge_model_version": metadata.judge_model_version,
                    "judge_prompt_version": metadata.judge_prompt_version,
                    "executed_at": metadata.executed_at,
                },
                "profileConfig": {},
            }), encoding="utf-8")

            class CatalogReader:
                async def load_index_scope(self, _index_version_id):
                    return SCOPE

                async def load_gate_inputs(self, _subproblem_id, _scope):
                    return GATE_INPUTS

                async def load_subproblem_catalog(self, _scope):
                    return CATALOG

            class OutlineReader:
                async def load_outlines(self, version_ids, *, chunking_config_id):
                    return {version: OUTLINES[version] for version in version_ids if version in OUTLINES}

            class ExactReader:
                async def find_exact_question_log_match(self, _group_id, _question):
                    if _question == FIRST_QUESTION:
                        return EXACT_MATCH
                    if _question == FOLLOW_UP_QUESTION:
                        return replace(
                            EXACT_MATCH,
                            normalized_question=FOLLOW_UP_QUESTION,
                            source_subproblem_version=0,
                        )
                    return None

            class SearchReader:
                async def load_active_chunks(self):
                    return [ALPHA_CHUNK]

            class Rewrite:
                def __init__(self):
                    self.calls = 0

                async def rewrite(self, _question, candidates, *, stage):
                    self.calls += 1
                    assert len(candidates) == 1
                    return SimpleNamespace(
                        trace=ModelCallTrace(
                            provider="test", model_name="rewrite-v1", succeeded=True,
                            latency_ms=1, prompt_version="rewrite-prompt-v1",
                        ),
                        resolution=SimpleNamespace(
                            should_retrieve=True, resolved_query="resolved synthetic followup"
                        ),
                    )

            class Judge:
                def __init__(self, provider="test"):
                    self.provider = provider
                    self.calls = 0
                    self.payloads = []
                    self.params = []

                async def judge(self, payload, *, before_model_call, **kwargs):
                    self.calls += 1
                    stage = kwargs["stage"]
                    self.payloads.append(payload)
                    self.params.append(stage.request_params())
                    request = build_judge_request(payload, stage)
                    request_evidence = ModelRequestEvidence(
                        common_input_sha256=canonical_request_sha256(payload),
                        request_sha256=canonical_request_sha256(request),
                        input_transform_version="openai-judge-json-v1",
                        instructions_sha256=hashlib.sha256(stage.instructions.encode()).hexdigest(),
                        response_model_name=stage.model_name,
                    )
                    return SimpleNamespace(
                        trace=ModelCallTrace(
                            provider=self.provider, model_name=stage.model_name, succeeded=True,
                            latency_ms=2, prompt_version=stage.prompt_version,
                            request_evidence=request_evidence,
                        ),
                        output_text=_judge_output("CONNECT"),
                        failure=None,
                    )

            class StrictJevJudge(Judge):
                async def judge(self, payload, *, before_model_call, **kwargs):
                    stage = kwargs["stage"]
                    from app import prompts

                    self.asserted_stage = stage
                    if stage.prompt_version != JEV_PROMPT_VERSION:
                        raise AssertionError("strict Jev client requires its registered prompt version")
                    if stage.instructions != prompts.get("judge", JEV_PROMPT_VERSION):
                        raise AssertionError("strict Jev client requires registered prompt instructions")
                    if stage.request_params():
                        raise AssertionError("strict Jev client requires empty request params")
                    return await super().judge(payload, before_model_call=before_model_call, **kwargs)

            class FakeHybridRetriever:
                calls = 0

                def __init__(self, *_args, **_kwargs):
                    type(self).calls += 1

                async def search_with_trace(self, query):
                    return HybridSearchCall(
                        fused_results=(HybridRetrievalResult(
                            chunk=ALPHA_CHUNK, rrf_score=0.9, final_rank=1,
                            bm25_rank=1, vector_rank=1,
                        ),),
                        retrieval_query=query,
                        query_embedding=(1.0,) + (0.0,) * (OPENAI_EMBEDDING_DIMENSIONS - 1),
                    )

            selection = SimpleNamespace(
                service_key="HELP_CHATBOT",
                profile_revision_id=PROFILE_REVISION_ID,
                document_group_id=DOCUMENT_GROUP_ID,
                index_version_id=INDEX_VERSION_ID,
                embedding_config_id=EMBEDDING_CONFIG_ID,
                embedding_config=EmbeddingConfigIdentity(
                    id=EMBEDDING_CONFIG_ID,
                    version="embedding-v1",
                    provider=OPENAI_EMBEDDING_PROVIDER,
                    model_name=OPENAI_EMBEDDING_MODEL,
                    dimensions=OPENAI_EMBEDDING_DIMENSIONS,
                    input_template_version=EMBEDDING_INPUT_TEMPLATE_VERSION,
                ),
                exact_cache_enabled=True,
                semantic_cache_enabled=True,
                stage_settings=SimpleNamespace(
                    rewrite=object(),
                    judge=StageSettings(
                        model_name="judge-v1",
                        prompt_version="prompt-v1",
                        instructions="synthetic test instructions",
                    ),
                ),
            )
            snapshot = SimpleNamespace(
                selection=selection,
                catalog_reader=CatalogReader(),
                outline_reader=OutlineReader(),
                search_reader=SearchReader(),
                exact_reader=ExactReader(),
            )
            service_data = _ReadOnlyData(snapshot)
            embedder, rewrite, judge = Mock(), Rewrite(), Judge()
            register_riido_chat_profile(
                service_key="HELP_CHATBOT",
                service_data=service_data,
                embedder=embedder,
                query_rewrite_service=rewrite,
                judge_client=judge,
                golden_root=registry,
            )

            with patch("evaluation.cache_path_eval.profiles.riido_chat.HybridRetriever", FakeHybridRetriever):
                output = await run_cache_path_cli(
                    registry_root=registry,
                    manifest_path=manifest_path,
                    config_path=config_path,
                    execution_adapter=create_riido_cache_decision_execution_adapter(),
                )

            self.assertEqual(["SERVED", "SERVED"], [row.outcome for row in output.observations])
            self.assertEqual([0, 2], [row.model_calls for row in output.observations])
            self.assertEqual([0, 0], [row.embedding_calls for row in output.observations])
            self.assertEqual(901, output.observations[0].evidence.exact_source_classification_id)
            self.assertEqual(
                901, output.observations[1].evidence.exact_source_classification_id
            )
            self.assertEqual(
                (EXACT_SOURCE_SUBPROBLEM_VERSION_STALE,),
                output.observations[1].evidence.exact_fallthrough_reasons,
            )
            self.assertEqual(2, len(service_data.contexts))
            self.assertTrue(all(context.closed for context in service_data.contexts))
            embedder.assert_not_called()
            self.assertEqual(1, rewrite.calls)
            self.assertEqual(1, judge.calls)
            self.assertTrue(Path(output.result_path).is_file())

            # The plan binding performs common preparation once, then reuses the
            # same presentation for each target judge through the child lifecycle.
            target_prompt = "question-grouping-v7-2"
            variants_metadata = {
                "variant-a": replace(
                    metadata,
                    run_id="riido-plan-a",
                    judge_model_version="judge-alt-a",
                    judge_prompt_version=target_prompt,
                ),
                "variant-b": replace(
                    metadata,
                    run_id="riido-plan-b",
                    judge_model_version="judge-alt-b",
                    judge_prompt_version=JEV_PROMPT_VERSION,
                ),
            }
            request_params = {
                "reasoning": {"effort": "low"},
                "max_output_tokens": 128,
            }
            specs = {
                "variant-a": CachePathPlanRunSpec(
                    variants_metadata["variant-a"],
                    {"evaluationOnlyJudgeOverride": True, "judgeParams": request_params},
                ),
                "variant-b": CachePathPlanRunSpec(
                    variants_metadata["variant-b"],
                    {"evaluationOnlyJudgeOverride": True, "judgeParams": {}},
                ),
            }
            variants = tuple(
                EvaluationVariant(variant_id, spec.metadata.run_id, spec.variant_configuration())
                for variant_id, spec in specs.items()
            )
            plan_manifest = {
                "group": "cache-path",
                "goldenVersion": "golden-v1",
                "sourceCorpusVersion": "corpus-v1",
                "inputs": {"draftsSha256": drafts_hash, "judgeLabelsSha256": "e" * 64},
                "turns": [
                    {
                        "itemId": "SYN-EXACT", "path": "P1", "turnNo": 1,
                        "kind": "exact_hit", "expectedExactHit": True, "expectedServed": True,
                    },
                    {
                        "itemId": "SYN-EXACT", "path": "P6", "turnNo": 2,
                        "kind": "judge", "expectedExactHit": False, "expectedServed": True,
                        "expectedJudgeDecision": "CONNECT", "expectedSubproblemKeysAny": ["alpha.enable"],
                    },
                ],
            }
            selection_hash = hashlib.sha256(
                json.dumps(plan_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            variant_a_client = Judge(provider="fake-provider-a")
            variant_b_client = StrictJevJudge(provider="fake-jev")
            variant_clients = {
                "variant-a": variant_a_client,
                "variant-b": variant_b_client,
            }
            client_factory_calls = []

            def select_judge_client(variant, spec):
                client_factory_calls.append((variant.variant_id, spec.metadata.judge_model_version))
                return variant_clients[variant.variant_id]

            def validate_judge_client_selection(_variant, spec):
                if spec.metadata.judge_model_version == "judge-alt-invalid-client":
                    raise ValueError("judge client selection is not configured")

            shared_preparer, prepared_profile_factory = create_riido_chat_plan_binding(
                service_key="HELP_CHATBOT",
                base_metadata=metadata,
                service_data=service_data,
                embedder=embedder,
                query_rewrite_service=rewrite,
                judge_client=judge,
                golden_root=registry,
                judge_client_factory=select_judge_client,
                judge_client_validator=validate_judge_client_selection,
                evaluation_only_judge_override=True,
            )

            # An invalid plan fails preflight before source preparation or the
            # per-variant transport factory can run.
            with self.assertRaisesRegex(ValueError, "selection hash"):
                await execute_cache_path_plan(
                    plan=EvaluationPlan("riido-invalid-plan", "cache-path", "0" * 64, variants),
                    manifest=plan_manifest,
                    run_specs=specs,
                    store=CachePathRunStore(registry),
                    plan_store=CachePathPlanStore(registry),
                    execution_adapter=create_riido_cache_decision_execution_adapter(),
                    shared_input_preparer=shared_preparer,
                    prepared_profile_factory=prepared_profile_factory,
                )
            self.assertEqual([], client_factory_calls)
            self.assertEqual(1, rewrite.calls)

            # Provider-specific prompt and params errors are rejected by plan
            # preflight, before shared rewrite/embedding preparation or client
            # construction can incur costs.
            invalid_metadata = replace(
                variants_metadata["variant-b"], run_id="riido-plan-invalid-params"
            )
            invalid_specs = {
                "variant-invalid": CachePathPlanRunSpec(
                    invalid_metadata,
                    {
                        "evaluationOnlyJudgeOverride": True,
                        "judgeParams": {"max_output_tokens": 0},
                    },
                )
            }
            invalid_variants = tuple(
                EvaluationVariant(key, spec.metadata.run_id, spec.variant_configuration())
                for key, spec in invalid_specs.items()
            )
            invalid_selection_hash = hashlib.sha256(
                json.dumps(plan_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            rewrite_before_invalid_settings = rewrite.calls
            retrievals_before_invalid_settings = FakeHybridRetriever.calls
            with self.assertRaisesRegex(ValueError, "max_output_tokens"):
                await execute_cache_path_plan(
                    plan=EvaluationPlan("riido-invalid-settings", "cache-path", invalid_selection_hash, invalid_variants),
                    manifest=plan_manifest,
                    run_specs=invalid_specs,
                    store=CachePathRunStore(registry),
                    plan_store=CachePathPlanStore(registry),
                    execution_adapter=create_riido_cache_decision_execution_adapter(),
                    shared_input_preparer=shared_preparer,
                    prepared_profile_factory=prepared_profile_factory,
                )
            self.assertEqual(rewrite_before_invalid_settings, rewrite.calls)
            self.assertEqual(retrievals_before_invalid_settings, FakeHybridRetriever.calls)
            self.assertEqual([], client_factory_calls)

            invalid_client_metadata = replace(
                variants_metadata["variant-b"],
                run_id="riido-plan-invalid-client",
                judge_model_version="judge-alt-invalid-client",
            )
            invalid_client_specs = {
                "variant-invalid-client": CachePathPlanRunSpec(
                    invalid_client_metadata,
                    {"evaluationOnlyJudgeOverride": True, "judgeParams": {}},
                )
            }
            invalid_client_variants = tuple(
                EvaluationVariant(key, spec.metadata.run_id, spec.variant_configuration())
                for key, spec in invalid_client_specs.items()
            )
            with self.assertRaisesRegex(ValueError, "client selection"):
                await execute_cache_path_plan(
                    plan=EvaluationPlan(
                        "riido-invalid-client",
                        "cache-path",
                        invalid_selection_hash,
                        invalid_client_variants,
                    ),
                    manifest=plan_manifest,
                    run_specs=invalid_client_specs,
                    store=CachePathRunStore(registry),
                    plan_store=CachePathPlanStore(registry),
                    execution_adapter=create_riido_cache_decision_execution_adapter(),
                    shared_input_preparer=shared_preparer,
                    prepared_profile_factory=prepared_profile_factory,
                )
            self.assertEqual(rewrite_before_invalid_settings, rewrite.calls)
            self.assertEqual(retrievals_before_invalid_settings, FakeHybridRetriever.calls)
            self.assertEqual([], client_factory_calls)

            invalid_prompt_metadata = replace(
                variants_metadata["variant-b"],
                run_id="riido-plan-invalid-prompt",
                judge_prompt_version="unregistered-prompt-v0",
            )
            invalid_prompt_specs = {
                "variant-invalid-prompt": CachePathPlanRunSpec(
                    invalid_prompt_metadata,
                    {"evaluationOnlyJudgeOverride": True, "judgeParams": {}},
                )
            }
            invalid_prompt_variants = tuple(
                EvaluationVariant(key, spec.metadata.run_id, spec.variant_configuration())
                for key, spec in invalid_prompt_specs.items()
            )
            with self.assertRaisesRegex(ValueError, "not registered"):
                await execute_cache_path_plan(
                    plan=EvaluationPlan(
                        "riido-invalid-prompt",
                        "cache-path",
                        invalid_selection_hash,
                        invalid_prompt_variants,
                    ),
                    manifest=plan_manifest,
                    run_specs=invalid_prompt_specs,
                    store=CachePathRunStore(registry),
                    plan_store=CachePathPlanStore(registry),
                    execution_adapter=create_riido_cache_decision_execution_adapter(),
                    shared_input_preparer=shared_preparer,
                    prepared_profile_factory=prepared_profile_factory,
                )
            self.assertEqual(rewrite_before_invalid_settings, rewrite.calls)
            self.assertEqual(retrievals_before_invalid_settings, FakeHybridRetriever.calls)
            self.assertEqual([], client_factory_calls)

            judge_calls_before_plan = judge.calls
            rewrite_calls_before_plan = rewrite.calls
            retrievals_before_plan = FakeHybridRetriever.calls

            with patch("evaluation.cache_path_eval.profiles.riido_chat.HybridRetriever", FakeHybridRetriever):
                plan_output = await execute_cache_path_plan(
                    plan=EvaluationPlan("riido-plan-variants", "cache-path", selection_hash, variants),
                    manifest=plan_manifest,
                    run_specs=specs,
                    store=CachePathRunStore(registry),
                    plan_store=CachePathPlanStore(registry),
                    execution_adapter=create_riido_cache_decision_execution_adapter(),
                    shared_input_preparer=shared_preparer,
                    prepared_profile_factory=prepared_profile_factory,
                )

            self.assertEqual(1, rewrite.calls - rewrite_calls_before_plan)
            self.assertEqual(1, FakeHybridRetriever.calls - retrievals_before_plan)
            self.assertEqual(judge_calls_before_plan, judge.calls)
            self.assertEqual(2, len(plan_output.results))
            self.assertEqual(
                [("variant-a", "judge-alt-a"), ("variant-b", "judge-alt-b")],
                client_factory_calls,
            )
            self.assertEqual(1, variant_clients["variant-a"].calls)
            self.assertEqual(1, variant_clients["variant-b"].calls)
            self.assertEqual(request_params, variant_a_client.params[0])
            self.assertEqual({}, variant_b_client.params[0])
            from app import prompts
            self.assertEqual(JEV_PROMPT_VERSION, variant_b_client.asserted_stage.prompt_version)
            self.assertEqual(
                prompts.get("judge", JEV_PROMPT_VERSION),
                variant_b_client.asserted_stage.instructions,
            )
            self.assertEqual({}, specs["variant-b"].profile_config_dict()["judgeParams"])
            plan_record = json.loads(Path(plan_output.result_path).read_text(encoding="utf-8"))
            from app import prompts

            persisted_plan_text = json.dumps(plan_record)
            self.assertEqual(1, plan_record["preparationEvidence"]["sharedModelCalls"])
            self.assertEqual(0, plan_record["preparationEvidence"]["sharedEmbeddingCalls"])
            self.assertNotIn(FIRST_QUESTION, json.dumps(plan_record))
            self.assertNotIn(FOLLOW_UP_QUESTION, json.dumps(plan_record))
            self.assertNotIn("synthetic prior answer", json.dumps(plan_record))
            self.assertNotIn("resolved synthetic followup", json.dumps(plan_record))
            self.assertNotIn(prompts.get("judge", target_prompt), persisted_plan_text)
            plan_payloads = [
                variant_clients["variant-a"].payloads[0],
                variant_clients["variant-b"].payloads[0],
            ]
            self.assertEqual(2, len(plan_payloads))
            self.assertEqual(
                {canonical_request_sha256(payload) for payload in plan_payloads},
                {canonical_request_sha256(plan_payloads[0])},
            )
            frozen_payload = plan_payloads[0]
            frozen_hash = canonical_request_sha256(frozen_payload)
            for spec in specs.values():
                child = json.loads((registry / "evals" / "cache-path" / f"{spec.metadata.run_id}.json").read_text())
                self.assertEqual(["SERVED", "SERVED"], [row["outcome"] for row in child["observations"]], json.dumps(child["observations"], ensure_ascii=False))
                exact, classified = child["observations"]
                self.assertEqual(0, exact["modelCalls"])
                self.assertEqual(0, exact["embeddingCalls"])
                request_evidence = classified["evidence"]["modelCalls"][0]["requestEvidence"]
                self.assertEqual(frozen_hash, request_evidence["commonInputSha256"])
                self.assertEqual(spec.metadata.judge_model_version, classified["evidence"]["modelCalls"][0]["modelName"])
                self.assertEqual(spec.metadata.judge_prompt_version, classified["evidence"]["modelCalls"][0]["promptVersion"])
                expected_provider = variant_clients[
                    "variant-a" if spec.metadata.judge_model_version == "judge-alt-a" else "variant-b"
                ].provider
                self.assertEqual(expected_provider, classified["evidence"]["modelCalls"][0]["provider"])
                self.assertNotIn(prompts.get("judge", target_prompt), json.dumps(child))
                self.assertNotIn(FIRST_QUESTION, json.dumps(child))
                self.assertNotIn(FOLLOW_UP_QUESTION, json.dumps(child))
                self.assertNotIn("synthetic prior answer", json.dumps(child))


if __name__ == "__main__":
    unittest.main()
