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
    register_riido_chat_profile,
)
from evaluation.cache_path_eval.readonly_service import EmbeddingConfigIdentity
from evaluation.cache_path_eval.store import CachePathRunMetadata
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

    def open_snapshot(self, _metadata):
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
                provider = "test"

                def __init__(self):
                    self.calls = 0

                async def judge(self, _payload, *, before_model_call, **_kwargs):
                    self.calls += 1
                    return SimpleNamespace(
                        trace=ModelCallTrace(
                            provider="test", model_name="judge-v1", succeeded=True,
                            latency_ms=2, prompt_version="prompt-v1",
                        ),
                        output_text=_judge_output("CONNECT"),
                        failure=None,
                    )

            class FakeHybridRetriever:
                def __init__(self, *_args, **_kwargs):
                    pass

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
                    judge=SimpleNamespace(model_name="judge-v1", prompt_version="prompt-v1"),
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


if __name__ == "__main__":
    unittest.main()
