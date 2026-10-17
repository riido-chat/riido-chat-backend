import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from app.answering.trace import ReplayContext
from evaluation.cache_path_eval.cli import load_run_config, run_cache_path_cli
from evaluation.cache_path_eval.execution import CacheDecisionObservation
from evaluation.cache_path_eval.profile_registry import (
    clear_profiles_for_tests,
    register_profile,
    resolve_profile_factory,
)


class CachePathCliTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        clear_profiles_for_tests()

    def tearDown(self):
        clear_profiles_for_tests()

    def write_inputs(self, root: Path):
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "group": "cache-path",
                    "goldenVersion": "g-v1",
                    "sourceCorpusVersion": "corpus-v1",
                    "inputs": {"draftsSha256": "d" * 64, "judgeLabelsSha256": "e" * 64},
                    "turns": [
                        {
                            "itemId": "ITEM-1",
                            "path": "P1",
                            "turnNo": 1,
                            "kind": "exact_hit",
                            "expectedExactHit": True,
                            "expectedServed": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        values = {
            "run_id": "cli-run",
            "golden_version": "g-v1",
            "golden_commit": "c" * 40,
            "source_corpus_version": None,
            "app_build_commit": "a" * 40,
            "service_key": "test-service",
            "profile_revision_id": 1,
            "document_group_id": 2,
            "index_version_id": 3,
            "embedding_config_id": 4,
            "cache_policy_version": "policy-1",
            "embedding_model_version": None,
            "judge_model_version": None,
            "judge_prompt_version": None,
            "executed_at": "2026-10-17T00:00:00Z",
        }
        config_path = root / "run-config.json"
        config_path.write_text(
            json.dumps({"metadata": values, "profileConfig": {"selector": "safe"}}),
            encoding="utf-8",
        )
        return manifest_path, config_path

    async def test_dispatches_registered_factory_and_invokes_shared_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path, config_path = self.write_inputs(root)
            calls = {}

            class Profile:
                async def callbacks_for_turn(self, _turn):
                    return object()

                def replay_context_for_turn(self, _metadata, _turn):
                    return ReplayContext(
                        run_id="cli-run",
                        item_id="ITEM-1",
                        turn_no=1,
                        golden_version="g-v1",
                        source_corpus_version=None,
                        service_version="test-service",
                        code_revision="a" * 40,
                        cache_policy_version="policy-1",
                        embedding_model_version=None,
                    )

            def factory(*, config, metadata, manifest):
                self.assertTrue(calls.get("availability_checked"))
                calls["factory"] = (config, metadata.service_key, manifest)
                return Profile()

            register_profile("test-service", factory)

            class FakeStore:
                def __init__(self, registry_root):
                    calls["store_root"] = registry_root

                def validate_manifest(self, _metadata, manifest):
                    return manifest["turns"]

                def check_available(self, _metadata):
                    calls["availability_checked"] = True

                def reserve(self, _metadata):
                    calls["reserved"] = True

                def checkpoint(self, _metadata, _observation):
                    calls["checkpointed"] = True

                def succeed(self, _metadata):
                    calls["succeeded"] = True

                def fail(self, _metadata, _failure):
                    calls["failed"] = True

                def begin_finalization(self, _metadata):
                    calls["finalizing"] = True

                def write_run(self, _metadata, **kwargs):
                    calls["written"] = kwargs
                    return root / "result.json"

            class FakeExecution:
                async def execute(self, **kwargs):
                    calls["turn"] = kwargs["item_id"]
                    return CacheDecisionObservation(
                        item_id=kwargs["item_id"],
                        turn_no=kwargs["turn_no"],
                        outcome="SERVED",
                        next_stage="NOT_APPLICABLE",
                        latency_ms=2,
                        model_calls=0,
                        embedding_calls=0,
                    )

            result = await run_cache_path_cli(
                registry_root=root,
                manifest_path=manifest_path,
                config_path=config_path,
                execution_adapter=FakeExecution(),
                store_factory=FakeStore,
            )
            self.assertEqual("test-service", calls["factory"][1])
            self.assertEqual({"selector": "safe"}, calls["factory"][0])
            self.assertEqual(
                {
                    "group": "cache-path",
                    "goldenVersion": "g-v1",
                    "sourceCorpusVersion": "corpus-v1",
                    "inputs": {"draftsSha256": "d" * 64, "judgeLabelsSha256": "e" * 64},
                    "turns": [{"itemId": "ITEM-1", "turnNo": 1}],
                },
                calls["factory"][2],
            )
            self.assertEqual("ITEM-1", calls["turn"])
            self.assertEqual(root, calls["store_root"])
            self.assertTrue(calls["availability_checked"])
            self.assertEqual(str(root / "result.json"), result.result_path)

    def test_requires_all_explicit_metadata_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "bad.json"
            config.write_text(json.dumps({"metadata": {}, "profileConfig": {}}))
            with self.assertRaisesRegex(ValueError, "metadata 필드가 맞지 않습니다"):
                load_run_config(config)

    def test_registry_rejects_implicit_or_duplicate_dispatch(self):
        with self.assertRaisesRegex(ValueError, "등록되지 않은"):
            resolve_profile_factory("unknown")
        register_profile("known", Mock())
        with self.assertRaisesRegex(ValueError, "이미 등록된"):
            register_profile("known", Mock())


if __name__ == "__main__":
    unittest.main()
