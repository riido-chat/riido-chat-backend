import unittest
import uuid
from unittest.mock import AsyncMock

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    ExecutionStatus,
    ModelCall,
    ModelCallPurpose,
)
from app.rag.log_store import RagLogStore


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
