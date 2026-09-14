"""판별 공유 부품 생성과 요청별 판별 서비스 조립."""

import unittest
from unittest.mock import AsyncMock, Mock, patch

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.log_store import RagLogStore
from app.core.config import Settings
from app.question_grouping.outline_reader import DocumentOutlineCache
from app.question_grouping.runtime import (
    QuestionGroupingComponents,
    create_question_grouping_components,
)
from app.question_grouping.service import QuestionGroupingService


class QuestionGroupingRuntimeTest(unittest.TestCase):
    def test_switch_defaults_to_off(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings(_env_file=None)

        self.assertFalse(settings.question_grouping_enabled)

    def test_switch_can_be_enabled_from_environment(self) -> None:
        with patch.dict("os.environ", {"QUESTION_GROUPING_ENABLED": "true"}, clear=True):
            settings = Settings(_env_file=None)

        self.assertTrue(settings.question_grouping_enabled)

    def test_switch_off_creates_no_components(self) -> None:
        with patch("app.question_grouping.runtime.QuestionJudgeClient") as judge_client:
            components = create_question_grouping_components(
                Settings(_env_file=None, question_grouping_enabled=False)
            )

        self.assertIsNone(components)
        judge_client.assert_not_called()

    def test_switch_on_creates_shared_components(self) -> None:
        with patch("app.question_grouping.runtime.QuestionJudgeClient") as judge_client:
            components = create_question_grouping_components(
                Settings(_env_file=None, question_grouping_enabled=True)
            )

        judge_client.assert_called_once_with()
        self.assertIs(judge_client.return_value, components.judge_client)
        self.assertIsInstance(components.outline_cache, DocumentOutlineCache)

    def test_build_service_shares_session_log_store_and_cache(self) -> None:
        session = AsyncMock(spec=AsyncSession)
        log_store = RagLogStore(session)
        embedder = Mock()
        components = QuestionGroupingComponents(
            judge_client=Mock(), outline_cache=DocumentOutlineCache()
        )

        first = components.build_service(session=session, log_store=log_store, embedder=embedder)
        second = components.build_service(session=session, log_store=log_store, embedder=embedder)

        self.assertIsInstance(first, QuestionGroupingService)
        self.assertIsNot(first, second)
        self.assertIs(session, first._session)
        self.assertIs(log_store, first._log_store)
        self.assertIs(embedder, first._embedder)
        self.assertIs(components.judge_client, first._judge_client)
        self.assertIs(components.outline_cache, first._outline_reader._cache)
        self.assertIs(first._outline_reader._cache, second._outline_reader._cache)


if __name__ == "__main__":
    unittest.main()
