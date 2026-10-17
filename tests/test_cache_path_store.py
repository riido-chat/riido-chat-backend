"""Privacy and create-only behavior for cache-path run storage."""

import json
import tempfile
import unittest
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
)
from evaluation.cache_path_eval.runner import evaluate
from evaluation.cache_path_eval.store import (
    CachePathRunMetadata,
    CachePathRunStore,
    CachePathStoreError,
)


class CachePathRunStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_refuses_public_backend_repository_as_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            git_dir = root / ".git"
            git_dir.mkdir()
            (git_dir / "config").write_text(
                '[remote "origin"]\n\turl = git@github.com:riido-chat/riido-chat-backend.git\n',
                encoding="utf-8",
            )
            (root / "README.md").write_text("# riido-evals\n", encoding="utf-8")
            (root / "golden").mkdir()
            (root / "golden" / "manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(CachePathStoreError, "등록부"):
                CachePathRunStore(root)

    async def test_writes_text_free_run_once_to_registry(self):
        app_commit = "a" * 40
        context = ReplayContext(
            run_id="run-1",
            item_id="CACHE-1",
            turn_no=1,
            golden_version="1.0.1",
            source_corpus_version="cache-drafts-v1",
            service_version="service-a",
            code_revision=app_commit,
            cache_policy_version="policy-v1",
            embedding_model_version="embed-v1",
        )
        adapter = CacheDecisionExecutionAdapter(
            AnswerQuestion(SimpleNamespace(generate_answer=AsyncMock()))
        )
        observation = await adapter.execute(
            item_id="CACHE-1",
            turn_no=1,
            replay_context=context,
            callbacks=CacheDecisionCallbacks(
                resolve_query=AsyncMock(return_value="private source question"),
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
            ),
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
        summary = evaluate(manifest, [observation.to_row()], "b" * 64)["summary"]
        metadata = CachePathRunMetadata(
            run_id="run-1",
            golden_version="1.0.1",
            golden_commit="c" * 40,
            source_corpus_version="cache-drafts-v1",
            app_build_commit=app_commit,
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

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            git_dir = root / ".git"
            git_dir.mkdir()
            (git_dir / "config").write_text(
                '[remote "origin"]\n\turl = git@github.com:riido-chat/riido-evals.git\n',
                encoding="utf-8",
            )
            (root / "README.md").write_text("# riido-evals\n", encoding="utf-8")
            (root / "golden").mkdir()
            (root / "golden" / "manifest.json").write_text("{}", encoding="utf-8")
            store = CachePathRunStore(root)
            destination = store.write_run(
                metadata,
                manifest=manifest,
                observations=[observation],
                summary=summary,
                observation_sha256="b" * 64,
            )

            persisted = json.loads(destination.read_text(encoding="utf-8"))
            first_event = persisted["observations"][0]["trace"]["events"][0]
            self.assertEqual("EXACT_CACHE_LOOKUP", first_event["stage"])
            self.assertEqual("NOT_CONFIGURED", first_event["detailCode"])
            self.assertEqual(context.code_revision, persisted["target"]["appBuildCommit"])
            self.assertNotIn("private source question", destination.read_text(encoding="utf-8"))
            with self.assertRaises(CachePathStoreError):
                store.write_run(
                    metadata,
                    manifest=manifest,
                    observations=[observation],
                    summary=summary,
                    observation_sha256="b" * 64,
                )

    async def test_rejects_text_fields_in_manifest_turn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            git_dir = root / ".git"
            git_dir.mkdir()
            (git_dir / "config").write_text(
                '[remote "origin"]\n\turl = git@github.com:riido-chat/riido-evals.git\n',
                encoding="utf-8",
            )
            (root / "README.md").write_text("# riido-evals\n", encoding="utf-8")
            (root / "golden").mkdir()
            (root / "golden" / "manifest.json").write_text("{}", encoding="utf-8")
            store = CachePathRunStore(root)
            with self.assertRaisesRegex(CachePathStoreError, "텍스트"):
                store.write_run(
                    CachePathRunMetadata(
                        run_id="run-2",
                        golden_version="1.0.1",
                        golden_commit="c" * 40,
                        source_corpus_version=None,
                        app_build_commit="a" * 40,
                        service_key="service-a",
                        profile_revision_id=1,
                        document_group_id=1,
                        index_version_id=1,
                        embedding_config_id=1,
                        cache_policy_version="policy-v1",
                        embedding_model_version=None,
                        judge_model_version=None,
                        judge_prompt_version=None,
                        executed_at="2026-10-17T12:00:00Z",
                    ),
                    manifest={
                        "group": "cache-path",
                        "goldenVersion": "1.0.1",
                        "inputs": {"draftsSha256": None, "judgeLabelsSha256": None},
                        "turns": [{"itemId": "x", "turnNo": 1, "question": "secret"}],
                    },
                    observations=[],
                    summary={},
                    observation_sha256="b" * 64,
                )


if __name__ == "__main__":
    unittest.main()
