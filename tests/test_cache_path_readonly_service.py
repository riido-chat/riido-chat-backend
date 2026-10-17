from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from evaluation.cache_path_eval.readonly_service import (
    ReadOnlyRiidoServiceData,
    ReadOnlyServiceSnapshotError,
    SelectedServiceSettings,
)
from evaluation.cache_path_eval.store import CachePathRunMetadata


class _FakeSession:
    def __init__(self) -> None:
        self.autoflush = True
        self.calls: list[object] = []
        self.active = False
        self.closed = False
        self.rolled_back = False

    async def begin(self) -> None:
        self.calls.append("begin")
        self.active = True

    async def execute(self, statement):
        self.calls.append(statement)
        return SimpleNamespace(one_or_none=lambda: None)

    def in_transaction(self) -> bool:
        return self.active

    async def rollback(self) -> None:
        self.calls.append("rollback")
        self.rolled_back = True
        self.active = False

    async def close(self) -> None:
        self.calls.append("close")
        self.closed = True


class ReadOnlyRiidoServiceDataTest(unittest.IsolatedAsyncioTestCase):
    def metadata(self) -> CachePathRunMetadata:
        return CachePathRunMetadata(
            run_id="readonly-unit",
            golden_version="golden-v1",
            golden_commit="a" * 40,
            source_corpus_version=None,
            app_build_commit="b" * 40,
            service_key="HELP_CHATBOT",
            profile_revision_id=12,
            document_group_id=34,
            index_version_id=56,
            embedding_config_id=78,
            cache_policy_version="policy-v1",
            embedding_model_version=None,
            judge_model_version=None,
            judge_prompt_version=None,
            executed_at="2026-10-17T00:00:00Z",
        )

    async def test_factory_is_lazy_and_snapshot_rolls_back_and_closes(self) -> None:
        session = _FakeSession()
        factory_calls = 0

        def factory():
            nonlocal factory_calls
            factory_calls += 1
            return session

        data = ReadOnlyRiidoServiceData(factory)
        self.assertEqual(0, factory_calls)

        selection = SelectedServiceSettings(
            service_key="HELP_CHATBOT",
            profile_revision_id=12,
            document_group_id=34,
            index_version_id=56,
            embedding_config_id=78,
            embedding_config=SimpleNamespace(),
            exact_cache_enabled=True,
            semantic_cache_enabled=False,
            stage_settings=SimpleNamespace(),
        )
        with patch.object(
            data,
            "_select_settings",
            new=AsyncMock(return_value=(selection, object())),
        ):
            async with data.open_snapshot(self.metadata()) as snapshot:
                self.assertIs(selection, snapshot.selection)
                self.assertEqual(1, factory_calls)
                self.assertFalse(session.autoflush)
                self.assertEqual("begin", session.calls[0])
                self.assertIn(
                    "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY",
                    str(session.calls[1]),
                )
                self.assertEqual(34, snapshot.search_reader._document_group_id)
                self.assertEqual(56, snapshot.search_reader._index_version_id)

        self.assertTrue(session.rolled_back)
        self.assertTrue(session.closed)
        self.assertEqual("rollback", session.calls[-2])
        self.assertEqual("close", session.calls[-1])

    async def test_mismatch_fails_before_any_profile_or_reader_lookup(self) -> None:
        session = _FakeSession()
        data = ReadOnlyRiidoServiceData(lambda: session)
        with self.assertRaises(ReadOnlyServiceSnapshotError):
            async with data.open_snapshot(self.metadata()):
                self.fail("snapshot should not be yielded")
        # The profile/service/group check rejects this metadata before index,
        # embedding, or prompt configuration is loaded.
        self.assertEqual(2, len([call for call in session.calls if not isinstance(call, str)]))
        self.assertTrue(session.rolled_back)
        self.assertTrue(session.closed)

    async def test_cancellation_path_rolls_back_and_closes(self) -> None:
        session = _FakeSession()
        data = ReadOnlyRiidoServiceData(lambda: session)
        selection = SelectedServiceSettings(
            service_key="HELP_CHATBOT",
            profile_revision_id=12,
            document_group_id=34,
            index_version_id=56,
            embedding_config_id=78,
            embedding_config=SimpleNamespace(),
            exact_cache_enabled=True,
            semantic_cache_enabled=False,
            stage_settings=SimpleNamespace(),
        )
        with patch.object(
            data,
            "_select_settings",
            new=AsyncMock(return_value=(selection, object())),
        ):
            with self.assertRaises(asyncio.CancelledError):
                async with data.open_snapshot(self.metadata()):
                    raise asyncio.CancelledError()
        self.assertTrue(session.rolled_back)
        self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
