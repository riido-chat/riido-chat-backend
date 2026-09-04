import unittest
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Feedback, FeedbackRating
from app.database.session import get_db_session
from app.main import create_app
from app.chat.dependencies import get_rag_log_store
from app.chat.log_store import (
    FeedbackNotAllowedError,
    RagLogStore,
    RagRunNotFoundError,
)


@asynccontextmanager
async def test_lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


class FeedbackApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rag_run_id = uuid.uuid4()
        self.log_store = AsyncMock(spec=RagLogStore)
        self.session = AsyncMock(spec=AsyncSession)

        async def override_db_session() -> AsyncIterator[AsyncSession]:
            yield self.session

        with patch("app.main.lifespan", test_lifespan):
            self.app = create_app()

        self.app.dependency_overrides[get_rag_log_store] = lambda: self.log_store
        self.app.dependency_overrides[get_db_session] = override_db_session

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()

    # ------------------------------------------------------------------
    # 등록과 변경
    # ------------------------------------------------------------------

    def test_registers_rating_and_returns_stored_value(self) -> None:
        self.log_store.set_feedback.return_value = self._feedback(
            FeedbackRating.GOOD
        )

        response = self._put({"rating": "GOOD"})

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {"ragRunId": str(self.rag_run_id), "rating": "GOOD"},
            response.json(),
        )
        self.log_store.set_feedback.assert_awaited_once_with(
            self.rag_run_id,
            rating=FeedbackRating.GOOD,
        )
        self.session.commit.assert_awaited_once_with()

    def test_returns_opposite_rating_after_change(self) -> None:
        self.log_store.set_feedback.return_value = self._feedback(
            FeedbackRating.BAD
        )

        response = self._put({"rating": "BAD"})

        self.assertEqual("BAD", response.json()["rating"])

    def test_same_rating_resend_is_accepted(self) -> None:
        self.log_store.set_feedback.return_value = self._feedback(
            FeedbackRating.GOOD
        )

        first = self._put({"rating": "GOOD"})
        second = self._put({"rating": "GOOD"})

        self.assertEqual(200, first.status_code)
        self.assertEqual(200, second.status_code)
        self.assertEqual(first.json(), second.json())

    # ------------------------------------------------------------------
    # 해제
    # ------------------------------------------------------------------

    def test_clears_rating_and_returns_null(self) -> None:
        self.log_store.clear_feedback.return_value = True

        response = self._delete()

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {"ragRunId": str(self.rag_run_id), "rating": None},
            response.json(),
        )
        self.log_store.clear_feedback.assert_awaited_once_with(self.rag_run_id)
        self.session.commit.assert_awaited_once_with()

    def test_clearing_absent_rating_returns_same_response(self) -> None:
        self.log_store.clear_feedback.return_value = False

        response = self._delete()

        self.assertEqual(200, response.status_code)
        self.assertIsNone(response.json()["rating"])

    # ------------------------------------------------------------------
    # 오류
    # ------------------------------------------------------------------

    def test_unknown_rag_run_returns_404(self) -> None:
        self.log_store.set_feedback.side_effect = RagRunNotFoundError(
            f"존재하지 않는 턴입니다: {self.rag_run_id}"
        )

        response = self._put({"rating": "GOOD"})

        self.assertEqual(404, response.status_code)
        self.assertEqual(
            {"code": "NOT_FOUND", "message": "존재하지 않는 답변입니다."},
            response.json(),
        )
        self.session.commit.assert_not_awaited()

    def test_unfinished_turn_returns_409(self) -> None:
        self.log_store.set_feedback.side_effect = FeedbackNotAllowedError(
            "평가할 수 없는 상태의 턴입니다: AnswerStatus.ERROR"
        )

        response = self._put({"rating": "GOOD"})

        self.assertEqual(409, response.status_code)
        self.assertEqual(
            {
                "code": "FEEDBACK_NOT_ALLOWED",
                "message": "평가할 수 없는 답변입니다.",
            },
            response.json(),
        )
        self.session.commit.assert_not_awaited()

    def test_delete_on_unfinished_turn_returns_409(self) -> None:
        self.log_store.clear_feedback.side_effect = FeedbackNotAllowedError(
            "평가할 수 없는 상태의 턴입니다: AnswerStatus.PROCESSING"
        )

        response = self._delete()

        self.assertEqual(409, response.status_code)
        self.assertEqual("FEEDBACK_NOT_ALLOWED", response.json()["code"])

    def test_rating_outside_enum_returns_422(self) -> None:
        response = self._put({"rating": "HELPFUL"})

        self.assertEqual(422, response.status_code)
        self.log_store.set_feedback.assert_not_awaited()

    def test_extra_request_field_returns_422(self) -> None:
        response = self._put({"rating": "GOOD", "comment": "좋아요"})

        self.assertEqual(422, response.status_code)
        self.log_store.set_feedback.assert_not_awaited()

    def test_missing_rating_returns_422(self) -> None:
        response = self._put({})

        self.assertEqual(422, response.status_code)
        self.log_store.set_feedback.assert_not_awaited()

    def test_malformed_rag_run_id_returns_422(self) -> None:
        with TestClient(self.app) as client:
            response = client.put(
                "/api/chat/not-a-uuid/feedback",
                json={"rating": "GOOD"},
            )

        self.assertEqual(422, response.status_code)
        self.log_store.set_feedback.assert_not_awaited()

    # ------------------------------------------------------------------

    def _feedback(self, rating: FeedbackRating) -> Feedback:
        return Feedback(rag_run_id=self.rag_run_id, rating=rating)

    def _put(self, payload: dict):
        with TestClient(self.app) as client:
            return client.put(
                f"/api/chat/{self.rag_run_id}/feedback",
                json=payload,
            )

    def _delete(self):
        with TestClient(self.app) as client:
            return client.delete(f"/api/chat/{self.rag_run_id}/feedback")


if __name__ == "__main__":
    unittest.main()
