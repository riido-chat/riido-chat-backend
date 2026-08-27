import unittest
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    AnswerStatus,
    ExecutionStatus,
    Feedback,
    FeedbackRating,
    ModelCall,
    ModelCallPurpose,
    RagRun,
)
from app.rag.log_store import (
    FeedbackNotAllowedError,
    RagLogStore,
    RagRunNotFoundError,
)


class RagLogStoreModelCallTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.session = AsyncMock(spec=AsyncSession)
        self.store = RagLogStore(self.session)
        self.rag_run_id = uuid.uuid4()

    async def test_starts_model_call_in_processing_without_result_metrics(
        self,
    ) -> None:
        call = await self.store.start_model_call(
            rag_run_id=self.rag_run_id,
            purpose=ModelCallPurpose.EMBEDDING.value,
            provider="openai",
            model_name="text-embedding-test",
        )

        self.assertEqual(ExecutionStatus.PROCESSING, call.status)
        self.assertEqual(0, call.retry_count)
        self.assertIsNone(call.input_tokens)
        self.assertIsNone(call.output_tokens)
        self.assertIsNone(call.latency_ms)
        self.assertIsNone(call.error_message)
        self.session.add.assert_called_once_with(call)
        self.session.flush.assert_awaited_once_with()

    async def test_finishes_same_processing_model_call(self) -> None:
        call = ModelCall(
            id=7,
            rag_run_id=self.rag_run_id,
            purpose=ModelCallPurpose.GENERATION,
            provider="openai",
            model_name="gpt-test",
            status=ExecutionStatus.PROCESSING,
            retry_count=0,
        )
        self.session.get.return_value = call

        finished = await self.store.finish_model_call(
            7,
            status=ExecutionStatus.FAILED,
            input_tokens=100,
            output_tokens=20,
            latency_ms=4500,
            retry_count=1,
            error_message="upstream timeout",
        )

        self.assertIs(call, finished)
        self.assertEqual(ExecutionStatus.FAILED, finished.status)
        self.assertEqual(1, finished.retry_count)
        self.assertEqual(4500, finished.latency_ms)
        self.assertEqual("upstream timeout", finished.error_message)
        self.session.get.assert_awaited_once_with(
            ModelCall,
            7,
            with_for_update=True,
        )
        self.session.flush.assert_awaited_once_with()

    async def test_rejects_finishing_an_already_finished_call(self) -> None:
        self.session.get.return_value = ModelCall(
            id=7,
            rag_run_id=self.rag_run_id,
            purpose=ModelCallPurpose.GENERATION,
            provider="openai",
            model_name="gpt-test",
            status=ExecutionStatus.SUCCESS,
            retry_count=0,
        )

        with self.assertRaisesRegex(ValueError, "PROCESSING"):
            await self.store.finish_model_call(
                7,
                status=ExecutionStatus.FAILED,
            )

        self.session.flush.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()


class RagLogStoreFeedbackTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.session = AsyncMock(spec=AsyncSession)
        self.store = RagLogStore(self.session)
        self.rag_run_id = uuid.uuid4()
        self.session.get.return_value = self._run(AnswerStatus.COMPLETED)
        self.session.scalar.return_value = None

    # ------------------------------------------------------------------
    # 등록과 변경
    # ------------------------------------------------------------------

    async def test_registers_first_rating(self) -> None:
        feedback = await self.store.set_feedback(
            self.rag_run_id,
            rating=FeedbackRating.GOOD,
        )

        self.assertEqual(FeedbackRating.GOOD, feedback.rating)
        self.assertEqual(self.rag_run_id, feedback.rag_run_id)
        self.assertEqual(feedback.created_at, feedback.updated_at)
        self.session.add.assert_called_once_with(feedback)

    async def test_changes_rating_and_stamps_updated_at(self) -> None:
        existing = self._feedback(FeedbackRating.GOOD)
        self.session.scalar.return_value = existing

        feedback = await self.store.set_feedback(
            self.rag_run_id,
            rating=FeedbackRating.BAD,
        )

        self.assertIs(existing, feedback)
        self.assertEqual(FeedbackRating.BAD, feedback.rating)
        self.assertGreater(feedback.updated_at, feedback.created_at)
        self.session.add.assert_not_called()

    async def test_same_rating_leaves_updated_at_untouched(self) -> None:
        existing = self._feedback(FeedbackRating.GOOD)
        stamped_at = existing.updated_at
        self.session.scalar.return_value = existing

        feedback = await self.store.set_feedback(
            self.rag_run_id,
            rating=FeedbackRating.GOOD,
        )

        self.assertEqual(FeedbackRating.GOOD, feedback.rating)
        self.assertEqual(stamped_at, feedback.updated_at)
        self.session.add.assert_not_called()

    # ------------------------------------------------------------------
    # 해제
    # ------------------------------------------------------------------

    async def test_clears_existing_rating(self) -> None:
        existing = self._feedback(FeedbackRating.BAD)
        self.session.scalar.return_value = existing

        cleared = await self.store.clear_feedback(self.rag_run_id)

        self.assertTrue(cleared)
        self.session.delete.assert_awaited_once_with(existing)

    async def test_clearing_absent_rating_is_a_no_op(self) -> None:
        cleared = await self.store.clear_feedback(self.rag_run_id)

        self.assertFalse(cleared)
        self.session.delete.assert_not_awaited()
        self.session.flush.assert_not_awaited()

    # ------------------------------------------------------------------
    # 대상 턴 검증
    # ------------------------------------------------------------------

    async def test_allows_withheld_turn(self) -> None:
        self.session.get.return_value = self._run(AnswerStatus.WITHHELD)

        feedback = await self.store.set_feedback(
            self.rag_run_id,
            rating=FeedbackRating.BAD,
        )

        self.assertEqual(FeedbackRating.BAD, feedback.rating)

    async def test_rejects_turns_that_have_no_answer_to_rate(self) -> None:
        for status in (
            AnswerStatus.PROCESSING,
            AnswerStatus.ERROR,
            AnswerStatus.CANCELLED,
        ):
            with self.subTest(status=status):
                self.session.get.return_value = self._run(status)

                with self.assertRaises(FeedbackNotAllowedError):
                    await self.store.set_feedback(
                        self.rag_run_id,
                        rating=FeedbackRating.GOOD,
                    )
                with self.assertRaises(FeedbackNotAllowedError):
                    await self.store.clear_feedback(self.rag_run_id)

        self.session.add.assert_not_called()
        self.session.delete.assert_not_awaited()

    async def test_rejects_unknown_rag_run(self) -> None:
        self.session.get.return_value = None

        with self.assertRaises(RagRunNotFoundError):
            await self.store.set_feedback(
                self.rag_run_id,
                rating=FeedbackRating.GOOD,
            )
        with self.assertRaises(RagRunNotFoundError):
            await self.store.clear_feedback(self.rag_run_id)

    # ------------------------------------------------------------------

    def _run(self, status: AnswerStatus) -> RagRun:
        return RagRun(id=self.rag_run_id, status=status)

    def _feedback(self, rating: FeedbackRating) -> Feedback:
        stamped_at = datetime(2026, 8, 26, tzinfo=timezone.utc)
        return Feedback(
            rag_run_id=self.rag_run_id,
            rating=rating,
            created_at=stamped_at,
            updated_at=stamped_at,
        )
