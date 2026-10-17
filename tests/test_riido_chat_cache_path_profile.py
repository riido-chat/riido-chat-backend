"""Tests for the injectable Riido Chat cache-path profile boundary."""

import unittest
from unittest.mock import AsyncMock, Mock

from app.answering.trace import ReplayContext
from evaluation.cache_path_eval.execution import CacheDecisionCallbacks
from evaluation.cache_path_eval.profiles.riido_chat import (
    RiidoChatCachePathEvaluationProfile,
)


class RiidoChatCachePathEvaluationProfileTests(unittest.IsolatedAsyncioTestCase):
    async def test_delegates_selected_read_data_and_evaluation_provenance(self):
        callbacks = CacheDecisionCallbacks(
            resolve_query=AsyncMock(),
            retrieve=AsyncMock(),
            choose_source=AsyncMock(),
            summarize=Mock(),
        )
        selected_turn = {"itemId": "SYN-1", "turnNo": 1, "path": "P1"}
        metadata = object()
        replay_context = ReplayContext(
            run_id="run-1",
            item_id="SYN-1",
            turn_no=1,
            golden_version="golden-explicit",
            source_corpus_version=None,
            service_version="service-explicit",
            code_revision="a" * 40,
            cache_policy_version="policy-explicit",
        )
        read_only_data = Mock()
        read_only_data.callbacks_for_turn = AsyncMock(return_value=callbacks)
        evaluation_context = Mock()
        evaluation_context.replay_context_for_turn.return_value = replay_context
        profile = RiidoChatCachePathEvaluationProfile(
            cache_data=read_only_data,
            evaluation_context=evaluation_context,
        )

        actual_callbacks = await profile.callbacks_for_turn(selected_turn)
        actual_context = profile.replay_context_for_turn(metadata, selected_turn)

        self.assertIs(callbacks, actual_callbacks)
        self.assertIs(replay_context, actual_context)
        read_only_data.callbacks_for_turn.assert_awaited_once_with(selected_turn)
        evaluation_context.replay_context_for_turn.assert_called_once_with(
            metadata, selected_turn
        )


if __name__ == "__main__":
    unittest.main()
