"""Tests for the injectable Riido Chat cache-path profile boundary."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from app.retrieval.embedding import OPENAI_EMBEDDING_MODEL
from evaluation.cache_path_eval.profiles.riido_chat import (
    RiidoChatCachePathEvaluationProfile,
    create_riido_chat_profile_factory,
)
from evaluation.cache_path_eval.store import CachePathRunMetadata


class RiidoChatCachePathEvaluationProfileTests(unittest.TestCase):
    def test_factory_accepts_only_reference_projection_and_stays_inert(self):
        metadata = CachePathRunMetadata(
            run_id="profile-factory",
            golden_version="golden-v1",
            golden_commit="a" * 40,
            source_corpus_version="corpus-v1",
            app_build_commit="b" * 40,
            service_key="HELP_CHATBOT",
            profile_revision_id=1,
            document_group_id=2,
            index_version_id=3,
            embedding_config_id=4,
            cache_policy_version="policy-v1",
            embedding_model_version=OPENAI_EMBEDDING_MODEL,
            judge_model_version="judge-v1",
            judge_prompt_version="prompt-v1",
            executed_at="2026-10-17T00:00:00Z",
        )
        manifest = {
            "group": "cache-path",
            "goldenVersion": "golden-v1",
            "sourceCorpusVersion": "corpus-v1",
            "inputs": {"draftsSha256": "c" * 64, "judgeLabelsSha256": "d" * 64},
            "turns": [{"itemId": "SYN-1", "turnNo": 1}],
        }
        dependencies = [Mock(name="service_data"), Mock(name="embedder"), Mock(name="rewrite"), Mock(name="judge")]
        with tempfile.TemporaryDirectory() as temp:
            factory = create_riido_chat_profile_factory(
                service_key="HELP_CHATBOT",
                service_data=dependencies[0],
                embedder=dependencies[1],
                query_rewrite_service=dependencies[2],
                judge_client=dependencies[3],
                golden_root=Path(temp),
            )
            profile = factory(config={}, metadata=metadata, manifest=manifest)

        self.assertIsInstance(profile, RiidoChatCachePathEvaluationProfile)
        for dependency in dependencies:
            dependency.assert_not_called()

    def test_factory_rejects_golden_labels_and_route_fields(self):
        metadata = CachePathRunMetadata(
            run_id="profile-labels",
            golden_version="golden-v1",
            golden_commit="a" * 40,
            source_corpus_version=None,
            app_build_commit="b" * 40,
            service_key="HELP_CHATBOT",
            profile_revision_id=1,
            document_group_id=2,
            index_version_id=3,
            embedding_config_id=4,
            cache_policy_version="policy-v1",
            embedding_model_version=OPENAI_EMBEDDING_MODEL,
            judge_model_version="judge-v1",
            judge_prompt_version="prompt-v1",
            executed_at="2026-10-17T00:00:00Z",
        )
        with tempfile.TemporaryDirectory() as temp:
            factory = create_riido_chat_profile_factory(
                service_key="HELP_CHATBOT",
                service_data=object(),
                embedder=object(),
                query_rewrite_service=object(),
                judge_client=object(),
                golden_root=Path(temp),
            )
            with self.assertRaises(ValueError):
                factory(
                    config={},
                    metadata=metadata,
                    manifest={
                        "group": "cache-path",
                        "goldenVersion": "golden-v1",
                        "turns": [{"itemId": "SYN-1", "turnNo": 1, "expectedServed": True}],
                    },
                )


if __name__ == "__main__":
    unittest.main()
