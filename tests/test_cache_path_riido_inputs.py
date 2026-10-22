"""Private cache input projection and shared question-vector policy checks."""

import hashlib
import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.retrieval.embedding import OPENAI_EMBEDDING_MODEL
from app.chat.query_rewrite import MAX_QUERY_REWRITE_TURNS
from app.retrieval.models import HybridSearchCall
from app.question_grouping.question_vector import (
    QuestionVectorDimensionError,
    resolve_question_vector,
)
from evaluation.cache_path_eval.golden_inputs import (
    FrozenCacheCorpusLoader,
    FrozenCacheInputError,
    load_canonical_cache_path_manifest,
)
from evaluation.cache_path_eval.profile_registry import (
    clear_profiles_for_tests,
    resolve_profile_factory,
)
from evaluation.cache_path_eval.profiles.riido_chat import register_riido_chat_profile
from evaluation.cache_path_eval.run import project_reference_manifest
from evaluation.cache_path_eval.store import CachePathRunMetadata


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_ROOT = ROOT / "evaluation" / "registry"


def _metadata(golden_commit: str) -> CachePathRunMetadata:
    return CachePathRunMetadata(
        run_id="runtime-golden-input-test",
        golden_version="1.0.1",
        golden_commit=golden_commit,
        source_corpus_version="0.1.0",
        app_build_commit="a" * 40,
        service_key="HELP_CHATBOT",
        profile_revision_id=1,
        document_group_id=1,
        index_version_id=10,
        embedding_config_id=1,
        cache_policy_version="cache-policy-test",
        embedding_model_version=OPENAI_EMBEDDING_MODEL,
        judge_model_version="judge-test",
        judge_prompt_version="prompt-test",
        executed_at="2026-10-17T00:00:00Z",
    )


class FrozenCacheInputTests(unittest.TestCase):
    def test_official_cache_path_group_reads_original_p6_context_with_pinned_provenance(self):
        if not (GOLDEN_ROOT / ".git").exists():
            self.skipTest("private evaluation registry submodule is unavailable in this checkout")
        commit = subprocess.run(
            ["git", "-C", str(GOLDEN_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        repository_root = subprocess.run(
            ["git", "-C", str(GOLDEN_ROOT), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if Path(repository_root).resolve() != GOLDEN_ROOT.resolve():
            self.skipTest("private evaluation registry submodule is not initialized")
        metadata = replace(_metadata(commit), source_corpus_version=None)
        manifest = load_canonical_cache_path_manifest(
            golden_root=GOLDEN_ROOT,
            group_path=GOLDEN_ROOT / "golden" / "groups" / "cache-path" / "2.0.0.json",
            metadata=metadata,
        )
        self.assertEqual(85, len(manifest["turns"]))
        self.assertEqual(
            {"P1": 20, "P3": 32, "P4": 14, "P6": 19},
            {path: sum(turn["path"] == path for turn in manifest["turns"]) for path in ("P1", "P3", "P4", "P6")},
        )
        self.assertEqual("2.0.0", manifest["evaluationGroup"]["groupVersion"])
        self.assertEqual(
            "d198be87e7dedeb498d3646625c83f9e7887665fc25290238d64aa1f037689e1",
            manifest["inputs"]["sourceHtmlSha256"],
        )

        references = project_reference_manifest(manifest)
        self.assertTrue(all(set(turn) == {"itemId", "turnNo"} for turn in references["turns"]))
        loader = FrozenCacheCorpusLoader(
            golden_root=GOLDEN_ROOT,
            selection_manifest=references,
            metadata=metadata,
        )
        loaded_all = {
            (turn["itemId"], turn["turnNo"]): loader.load_turn(turn["itemId"], turn["turnNo"])
            for turn in manifest["turns"]
        }
        p6_turns = [turn for turn in manifest["turns"] if turn["path"] == "P6"]
        for turn in p6_turns:
            loaded = loaded_all[(turn["itemId"], turn["turnNo"])]
            candidate_by_turn = {candidate.turn_no: candidate for candidate in loaded.candidates}
            self.assertTrue(set(turn["expectedSelectedTurnNos"]).issubset(candidate_by_turn))
            for selected_turn_no in turn["expectedSelectedTurnNos"]:
                candidate = candidate_by_turn[selected_turn_no]
                self.assertTrue(
                    (candidate.answer_content and candidate.answer_content.strip())
                    or candidate.withheld_reason_code is not None
                )
            self.assertFalse(hasattr(loaded, "expected_selected_turn_nos"))

    def test_loads_only_selected_question_and_explicit_prior_context_from_git_blob(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = Path(temp)
            golden = registry / "golden"
            (golden / "sources").mkdir(parents=True)
            (registry / "README.md").write_text("# riido-evals\n", encoding="utf-8")
            drafts = {
                "version": "0.1.0",
                "status": "draft",
                "items": [{
                    "id": "CACHE-P6-001",
                    "historySynthetic": False,
                    "turns": [
                        {"question": "prior question", "historyAnswer": {"status": "COMPLETED", "content": "prior answer"}},
                        {"question": "selected question"},
                    ],
                }],
            }
            drafts_bytes = json.dumps(drafts, sort_keys=True).encode("utf-8")
            (golden / "sources" / "cache-drafts.json").write_bytes(drafts_bytes)
            drafts_hash = hashlib.sha256(drafts_bytes).hexdigest()
            golden_manifest = {
                "goldenVersion": "1.0.1",
                "frozen": {"files": {"golden/sources/cache-drafts.json": {"sha256": drafts_hash}}},
            }
            (golden / "manifest.json").write_text(json.dumps(golden_manifest), encoding="utf-8")
            subprocess.run(["git", "init", str(registry)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(registry), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(registry), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(registry), "remote", "add", "origin", "https://example.invalid/riido-evals.git"], check=True)
            subprocess.run(["git", "-C", str(registry), "add", "README.md", "golden"], check=True)
            subprocess.run(["git", "-C", str(registry), "commit", "-m", "fixture"], check=True, capture_output=True)
            commit = subprocess.run(
                ["git", "-C", str(registry), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            selection = {
                "group": "cache-path",
                "goldenVersion": "1.0.1",
                "sourceCorpusVersion": "0.1.0",
                "inputs": {"draftsSha256": drafts_hash},
                "turns": [{"itemId": "CACHE-P6-001", "turnNo": 2}],
            }

            selected = FrozenCacheCorpusLoader(
                golden_root=registry,
                selection_manifest=selection,
                metadata=_metadata(commit),
            ).load_turn("CACHE-P6-001", 2)

        self.assertEqual(1, len(selected.candidates))
        self.assertEqual(1, selected.candidates[0].turn_no)
        self.assertEqual("prior question", selected.candidates[0].user_query)
        self.assertIsNotNone(selected.candidates[0].answer_content)
        self.assertFalse(selected.history_synthetic)
        self.assertFalse(hasattr(selected, "expected_judge_decision"))

    def test_rejects_ancestor_that_is_not_the_registry_git_root(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            registry = parent / "registry"
            (registry / "golden" / "sources").mkdir(parents=True)
            subprocess.run(["git", "init", str(registry)], check=True, capture_output=True)
            loader = object.__new__(FrozenCacheCorpusLoader)
            loader._golden_root = parent.resolve()
            loader._metadata = _metadata("a" * 40)
            loader._selection_manifest = {}
            loader._payload = None
            with self.assertRaises(FrozenCacheInputError):
                loader._load_payload()

    def test_followup_without_explicit_prior_answer_fails_closed(self):
        loader = object.__new__(FrozenCacheCorpusLoader)
        loader._allowed_refs = {("private-item", 2)}
        loader._payload = {
            "items": [
                {
                    "id": "private-item",
                    "turns": [
                        {"question": "prior"},
                        {"question": "followup"},
                    ],
                }
            ]
        }

        with self.assertRaises(FrozenCacheInputError):
            loader.load_turn("private-item", 2)

    def test_followup_uses_only_production_recent_history_window(self):
        turns = [
            {
                "question": f"question {turn_no}",
                "historyResolvedQuery": f"resolved {turn_no}",
                "historyAnswer": {"status": "COMPLETED", "content": f"answer {turn_no}"},
            }
            for turn_no in range(1, 8)
        ]
        loader = object.__new__(FrozenCacheCorpusLoader)
        loader._allowed_refs = {("private-item", 8)}
        loader._payload = {
            "items": [{"id": "private-item", "turns": turns + [{"question": "current"}]}]
        }

        selected = loader.load_turn("private-item", 8)

        self.assertEqual(MAX_QUERY_REWRITE_TURNS, len(selected.candidates))
        self.assertEqual([3, 4, 5, 6, 7], [item.turn_no for item in selected.candidates])


class RiidoProfileRegistrationTests(unittest.TestCase):
    def setUp(self):
        clear_profiles_for_tests()

    def tearDown(self):
        clear_profiles_for_tests()

    def test_registers_only_real_service_keys_without_building_clients(self):
        dependencies = {
            "service_data": None,
            "embedder": None,
            "query_rewrite_service": None,
            "judge_client": None,
            "golden_root": GOLDEN_ROOT,
        }
        register_riido_chat_profile(service_key="HELP_CHATBOT", **dependencies)
        register_riido_chat_profile(service_key="HELP_CHATBOT_TEST", **dependencies)
        self.assertTrue(callable(resolve_profile_factory("HELP_CHATBOT")))
        self.assertTrue(callable(resolve_profile_factory("HELP_CHATBOT_TEST")))
        with self.assertRaises(ValueError):
            register_riido_chat_profile(service_key="riido-chat", **dependencies)


class SharedQuestionVectorPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_dimension_errors_are_typed_and_keep_the_safe_message(self):
        reused = await resolve_question_vector(
            "same query",
            HybridSearchCall(retrieval_query="same query", query_embedding=(0.25,)),
            object(),
        )
        self.assertIsInstance(reused.error, QuestionVectorDimensionError)
        self.assertEqual(
            "질문 벡터 차원이 1536이 아닙니다: 1", str(reused.error)
        )

        class WrongDimensionEmbedder:
            def embed_many_with_usage(self, *_args, **_kwargs):
                return type(
                    "Response",
                    (),
                    {"embeddings": [[0.5]], "input_tokens": 1, "retry_count": 0},
                )()

        reembedded = await resolve_question_vector(
            "resolved query",
            HybridSearchCall(retrieval_query="expanded query", query_embedding=(0.1,)),
            WrongDimensionEmbedder(),
        )
        self.assertIsInstance(reembedded.error, QuestionVectorDimensionError)
        self.assertEqual(
            "질문 벡터 차원이 1536이 아닙니다: 1", str(reembedded.error)
        )

    async def test_reuses_exact_retrieval_embedding_without_an_extra_call(self):
        class Embedder:
            def embed_many_with_usage(self, *_args, **_kwargs):
                raise AssertionError("exact retrieval vector should be reused")

        result = await resolve_question_vector(
            "same query",
            HybridSearchCall(
                retrieval_query="same query",
                query_embedding=(0.25,) * 1536,
            ),
            Embedder(),
        )

        self.assertTrue(result.reused_retrieval_embedding)
        self.assertEqual((0.25,) * 1536, result.embedding)
        self.assertIsNone(result.trace)

    async def test_reembeds_resolved_query_and_records_actual_retry_attempts(self):
        class TemporaryError(Exception):
            pass

        class Embedder:
            def __init__(self):
                self.calls = 0

            def embed_many_with_usage(self, _texts, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise TemporaryError()
                return type(
                    "Response",
                    (),
                    {"embeddings": [[0.5] * 1536], "input_tokens": 3, "retry_count": 0},
                )()

        embedder = Embedder()

        async def no_sleep(_seconds):
            return None

        with patch("app.question_grouping.question_vector.is_transient_openai_error", return_value=True), patch(
            "app.question_grouping.question_vector.asyncio.sleep", new=no_sleep
        ):
            result = await resolve_question_vector(
                "resolved query",
                HybridSearchCall(retrieval_query="expanded query", query_embedding=(0.1,) * 1536),
                embedder,
            )

        self.assertFalse(result.reused_retrieval_embedding)
        self.assertEqual(2, embedder.calls)
        self.assertEqual(1, result.trace.retry_count)
        self.assertEqual((0.5,) * 1536, result.embedding)


if __name__ == "__main__":
    unittest.main()
