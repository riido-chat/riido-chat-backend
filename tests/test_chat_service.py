import unittest
import uuid
from itertools import count
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.chat_schema import (
    ChatCompletedResponse,
    ChatErrorCode,
    ChatErrorResponse,
    ChatResponseStatus,
    ChatWithheldReasonCode,
    ChatWithheldResponse,
)
from app.database.models import ConversationStatus, ExecutionStatus
from app.rag.chat_service import (
    CONTEXT_STRATEGY_NEW_TOPIC,
    ChatService,
    ConversationNotFoundError,
)
from app.rag.generation_service import WITHHELD_RESPONSES, GenerationService
from app.rag.log_store import RagLogStore
from app.rag.model_trace import ModelCallTrace
from generation.models import (
    Citation,
    FinalAnswerStatus,
    FinalGenerationResult,
    FinalWithheldReason,
)
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import (
    HybridRetrievalResult,
    HybridSearchCall,
    RetrievalChunk,
    RetrievalResult,
)


INDEX_VERSION_ID = 7


def _chunk(index: int) -> RetrievalChunk:
    return RetrievalChunk(
        document_id=f"document-{index}",
        section_id=f"section-{index}",
        document_title=f"문서 {index}",
        section_path=(f"문서 {index}", f"섹션 {index}"),
        source_url=f"https://docs.riido.io/{index}",
        category="guide",
        content=f"본문 {index}",
        chunk_id=index,
        document_version_id=100 + index,
        index_version_id=INDEX_VERSION_ID,
    )


def _embedding_trace() -> ModelCallTrace:
    return ModelCallTrace(
        provider="openai",
        model_name="text-embedding-3-large",
        succeeded=True,
        latency_ms=30,
        input_tokens=11,
    )


def _generation_trace(succeeded: bool = True) -> ModelCallTrace:
    return ModelCallTrace(
        provider="openai",
        model_name="gpt-5.4-mini",
        succeeded=succeeded,
        latency_ms=900,
        retry_count=1 if not succeeded else 0,
        input_tokens=1200,
        output_tokens=300,
        prompt_version="v1",
    )


class ChatServiceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.conversation_id = uuid.uuid4()
        self.rag_run_id = uuid.uuid4()

        self.retriever = AsyncMock(spec=HybridRetriever)
        self.generation_service = AsyncMock(spec=GenerationService)
        self.log_store = AsyncMock(spec=RagLogStore)
        self.session = AsyncMock(spec=AsyncSession)

        self.log_store.create_conversation.return_value = SimpleNamespace(
            id=self.conversation_id,
            status=ConversationStatus.ACTIVE,
        )
        self.log_store.start_rag_run.return_value = SimpleNamespace(
            id=self.rag_run_id
        )
        model_call_ids = count(1)
        self.log_store.start_model_call.side_effect = (
            lambda **_kwargs: SimpleNamespace(id=next(model_call_ids))
        )

        self._search_result = self._search_call()
        self._generation_result = None

        async def search_with_checkpoint(
            _question: str,
            *,
            before_model_call,
        ) -> HybridSearchCall:
            await before_model_call(
                "openai",
                "text-embedding-3-large",
                None,
            )
            return self._search_result

        async def generate_with_checkpoint(
            _question: str,
            _results,
            *,
            before_model_call,
        ) -> FinalGenerationResult:
            await before_model_call(
                "openai",
                "gpt-5.4-mini",
                "v1",
            )
            return self._generation_result

        self.retriever.search_with_trace.side_effect = search_with_checkpoint
        self.generation_service.generate_answer.side_effect = (
            generate_with_checkpoint
        )

        self.service = ChatService(
            retriever=self.retriever,
            generation_service=self.generation_service,
            log_store=self.log_store,
            session=self.session,
            index_version_id=INDEX_VERSION_ID,
        )

    # ------------------------------------------------------------------
    # 응답 변환
    # ------------------------------------------------------------------

    async def test_completed_response_carries_both_identifiers(self) -> None:
        self._generation_result = self._completed()

        response = await self.service.answer_question("멤버를 어떻게 초대하나요?")

        self.assertIsInstance(response, ChatCompletedResponse)
        self.assertEqual(ChatResponseStatus.COMPLETED, response.status)
        self.assertEqual(self.conversation_id, response.conversation_id)
        self.assertEqual(self.rag_run_id, response.rag_run_id)
        self.assertEqual(
            [1, 2],
            [citation.citation_number for citation in response.citations],
        )

    async def test_withheld_response_carries_both_identifiers(self) -> None:
        for reason in FinalWithheldReason:
            with self.subTest(reason=reason):
                self._generation_result = (
                    FinalGenerationResult(
                        status=FinalAnswerStatus.WITHHELD,
                        answer_markdown=WITHHELD_RESPONSES[reason],
                        citations=(),
                        withheld_reason=reason,
                        model_call=_generation_trace(),
                    )
                )

                response = await self.service.answer_question("질문")

                self.assertIsInstance(response, ChatWithheldResponse)
                self.assertEqual(
                    ChatWithheldReasonCode(reason.value),
                    response.withheld.reason_code,
                )
                self.assertEqual(self.conversation_id, response.conversation_id)
                self.assertEqual(self.rag_run_id, response.rag_run_id)

    async def test_error_response_keeps_identifiers_and_hides_internal_code(
        self,
    ) -> None:
        self._generation_result = FinalGenerationResult(
            status=FinalAnswerStatus.ERROR,
            answer_markdown=None,
            citations=(),
            error_code="UPSTREAM_ERROR",
            model_call=_generation_trace(succeeded=False),
        )

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(ChatErrorCode.INTERNAL_ERROR, response.error.code)
        self.assertEqual(self.conversation_id, response.conversation_id)
        self.assertEqual(self.rag_run_id, response.rag_run_id)
        self.assertNotIn(
            "UPSTREAM_ERROR",
            str(response.model_dump(mode="json", by_alias=True)),
        )

    # ------------------------------------------------------------------
    # 대화와 턴 시작
    # ------------------------------------------------------------------

    async def test_creates_conversation_when_request_has_no_conversation_id(
        self,
    ) -> None:
        self._generation_result = self._completed()

        await self.service.answer_question("질문")

        self.log_store.create_conversation.assert_awaited_once_with()
        self.log_store.get_conversation.assert_not_awaited()
        self.log_store.start_rag_run.assert_awaited_once_with(
            self.conversation_id,
            user_query="질문",
            index_version_id=INDEX_VERSION_ID,
            context_strategy=CONTEXT_STRATEGY_NEW_TOPIC,
        )

    async def test_reuses_active_conversation_from_request(self) -> None:
        existing_id = uuid.uuid4()
        self.log_store.get_conversation.return_value = SimpleNamespace(
            id=existing_id,
            status=ConversationStatus.ACTIVE,
        )
        self._generation_result = self._completed()

        response = await self.service.answer_question("질문", existing_id)

        self.log_store.get_conversation.assert_awaited_once_with(existing_id)
        self.log_store.create_conversation.assert_not_awaited()
        self.assertEqual(existing_id, response.conversation_id)

    async def test_rejects_unusable_conversation_before_creating_a_turn(self) -> None:
        cases = (
            None,
            SimpleNamespace(id=uuid.uuid4(), status=ConversationStatus.CLOSED),
            SimpleNamespace(id=uuid.uuid4(), status=ConversationStatus.EXPIRED),
        )

        for conversation in cases:
            with self.subTest(conversation=conversation):
                self.log_store.get_conversation.return_value = conversation

                with self.assertRaises(ConversationNotFoundError):
                    await self.service.answer_question("질문", uuid.uuid4())

        self.log_store.start_rag_run.assert_not_awaited()
        self.session.commit.assert_not_awaited()

    # ------------------------------------------------------------------
    # RagRun 2단계 + ModelCall checkpoint 커밋
    # ------------------------------------------------------------------

    async def test_commits_checkpoints_before_calling_external_models(self) -> None:
        commits_before_embedding = []
        commits_before_generation = []

        async def record_embedding_commit_count(
            _question,
            *,
            before_model_call,
        ):
            await before_model_call("openai", "text-embedding-3-large", None)
            commits_before_embedding.append(self.session.commit.await_count)
            return self._search_call()

        async def record_commit_count(
            _question,
            _results,
            *,
            before_model_call,
        ):
            await before_model_call("openai", "gpt-5.4-mini", "v1")
            commits_before_generation.append(self.session.commit.await_count)
            return self._completed()

        self.retriever.search_with_trace.side_effect = record_embedding_commit_count
        self.generation_service.generate_answer.side_effect = record_commit_count

        await self.service.answer_question("질문")

        # RagRun 시작 + Embedding 시작 + Generation 시작을 먼저 확정한다.
        self.assertEqual([2], commits_before_embedding)
        self.assertEqual([3], commits_before_generation)
        self.assertEqual(4, self.session.commit.await_count)

    async def test_returns_error_without_identifiers_when_first_commit_fails(
        self,
    ) -> None:
        self.session.commit.side_effect = RuntimeError("postgres connection detail")

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertIsNone(response.conversation_id)
        self.assertIsNone(response.rag_run_id)
        self.retriever.search_with_trace.assert_not_awaited()
        self.assertNotIn(
            "postgres connection detail",
            str(response.model_dump(mode="json", by_alias=True)),
        )

    async def test_does_not_embed_when_embedding_checkpoint_commit_fails(
        self,
    ) -> None:
        external_calls = []

        async def search_after_checkpoint(
            _question,
            *,
            before_model_call,
        ):
            await before_model_call("openai", "text-embedding-3-large", None)
            external_calls.append("embedding")
            return self._search_call()

        self.retriever.search_with_trace.side_effect = search_after_checkpoint
        self.session.commit.side_effect = [
            None,
            RuntimeError("checkpoint write failure"),
            None,
        ]

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(self.rag_run_id, response.rag_run_id)
        self.assertEqual([], external_calls)
        self.generation_service.generate_answer.assert_not_awaited()
        self.log_store.finish_model_call.assert_not_awaited()
        self.log_store.fail_rag_run.assert_awaited_once_with(
            self.rag_run_id,
            error_code="INTERNAL_ERROR",
            total_latency_ms=ANY,
        )

    async def test_does_not_generate_when_generation_checkpoint_commit_fails(
        self,
    ) -> None:
        external_calls = []

        async def generate_after_checkpoint(
            _question,
            _results,
            *,
            before_model_call,
        ):
            await before_model_call("openai", "gpt-5.4-mini", "v1")
            external_calls.append("generation")
            return self._completed()

        self.generation_service.generate_answer.side_effect = (
            generate_after_checkpoint
        )
        self.session.commit.side_effect = [
            None,
            None,
            RuntimeError("checkpoint write failure"),
            None,
        ]

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(self.rag_run_id, response.rag_run_id)
        self.assertEqual([], external_calls)
        self.assertEqual(2, self.log_store.start_model_call.await_count)
        self.assertEqual(1, self.log_store.finish_model_call.await_count)
        self.log_store.fail_rag_run.assert_awaited_once_with(
            self.rag_run_id,
            error_code="INTERNAL_ERROR",
            total_latency_ms=ANY,
        )

    async def test_returns_answer_when_final_commit_fails(self) -> None:
        self._generation_result = self._completed()
        self.session.commit.side_effect = [
            None,
            None,
            None,
            RuntimeError("write failure"),
        ]

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatCompletedResponse)
        self.assertEqual(self.rag_run_id, response.rag_run_id)

    # ------------------------------------------------------------------
    # 검색 후보 기록
    # ------------------------------------------------------------------

    async def test_records_every_candidate_of_both_retrievers(self) -> None:
        self._generation_result = self._completed()

        await self.service.answer_question("질문")

        _, candidates = self.log_store.record_retrieval_results.await_args.args
        self.assertEqual(5, len(candidates))
        self.assertEqual(
            ["BM25", "BM25", "BM25", "VECTOR", "VECTOR"],
            [candidate.retriever_type for candidate in candidates],
        )
        self.assertEqual(
            [1, 2, 3, 1, 2],
            [candidate.retriever_rank for candidate in candidates],
        )
        self.assertEqual(
            [12, 12, 12, 34, 34],
            [candidate.latency_ms for candidate in candidates],
        )

    async def test_marks_only_fused_candidates_with_rank_score_and_evidence(
        self,
    ) -> None:
        self._generation_result = self._completed()

        await self.service.answer_question("질문")

        _, candidates = self.log_store.record_retrieval_results.await_args.args
        by_key = {
            (candidate.retriever_type, candidate.chunk_id): candidate
            for candidate in candidates
        }

        fused = by_key[("BM25", 1)]
        self.assertEqual(1, fused.fused_rank)
        self.assertAlmostEqual(0.5, fused.fused_score)
        self.assertTrue(fused.selected_as_evidence)

        # 같은 청크의 Vector 행에도 같은 융합 값이 들어간다
        vector_fused = by_key[("VECTOR", 1)]
        self.assertEqual(1, vector_fused.fused_rank)
        self.assertAlmostEqual(0.5, vector_fused.fused_score)
        self.assertTrue(vector_fused.selected_as_evidence)

        dropped = by_key[("BM25", 3)]
        self.assertIsNone(dropped.fused_rank)
        self.assertIsNone(dropped.fused_score)
        self.assertFalse(dropped.selected_as_evidence)

    async def test_records_candidates_even_when_the_answer_is_withheld(self) -> None:
        self._generation_result = FinalGenerationResult(
            status=FinalAnswerStatus.WITHHELD,
            answer_markdown=WITHHELD_RESPONSES[
                FinalWithheldReason.INSUFFICIENT_EVIDENCE
            ],
            citations=(),
            withheld_reason=FinalWithheldReason.INSUFFICIENT_EVIDENCE,
            model_call=_generation_trace(),
        )

        await self.service.answer_question("질문")

        _, candidates = self.log_store.record_retrieval_results.await_args.args
        self.assertEqual(5, len(candidates))
        self.log_store.withhold_rag_run.assert_awaited_once()
        self.log_store.complete_rag_run.assert_not_awaited()

    # ------------------------------------------------------------------
    # 모델 호출 기록
    # ------------------------------------------------------------------

    async def test_starts_and_finishes_embedding_and_generation_same_rows(
        self,
    ) -> None:
        self._generation_result = self._completed()

        await self.service.answer_question("질문")

        purposes = [
            awaited.kwargs["purpose"]
            for awaited in self.log_store.start_model_call.await_args_list
        ]
        self.assertEqual(["EMBEDDING", "GENERATION"], purposes)

        embedding_start = self.log_store.start_model_call.await_args_list[0].kwargs
        self.assertEqual(self.rag_run_id, embedding_start["rag_run_id"])
        self.assertIsNone(embedding_start["prompt_version"])

        generation_start = self.log_store.start_model_call.await_args_list[1].kwargs
        self.assertEqual("v1", generation_start["prompt_version"])

        embedding_finish = self.log_store.finish_model_call.await_args_list[0]
        self.assertEqual(1, embedding_finish.args[0])
        self.assertEqual(
            ExecutionStatus.SUCCESS,
            embedding_finish.kwargs["status"],
        )
        self.assertEqual(11, embedding_finish.kwargs["input_tokens"])

        generation_finish = self.log_store.finish_model_call.await_args_list[1]
        self.assertEqual(2, generation_finish.args[0])
        self.assertEqual(
            ExecutionStatus.SUCCESS,
            generation_finish.kwargs["status"],
        )
        self.assertEqual(1200, generation_finish.kwargs["input_tokens"])
        self.assertEqual(300, generation_finish.kwargs["output_tokens"])

    async def test_records_failed_generation_call_with_retry_count(self) -> None:
        trace = ModelCallTrace(
            provider="openai",
            model_name="gpt-5.4-mini",
            succeeded=False,
            latency_ms=4200,
            retry_count=1,
            error_message="upstream timeout",
        )
        self._generation_result = FinalGenerationResult(
            status=FinalAnswerStatus.ERROR,
            answer_markdown=None,
            citations=(),
            error_code="UPSTREAM_ERROR",
            model_call=trace,
        )

        await self.service.answer_question("질문")

        generation = self.log_store.finish_model_call.await_args_list[1]
        self.assertEqual(2, generation.args[0])
        self.assertEqual(ExecutionStatus.FAILED, generation.kwargs["status"])
        self.assertEqual(1, generation.kwargs["retry_count"])
        self.assertEqual(4200, generation.kwargs["latency_ms"])
        self.assertEqual("upstream timeout", generation.kwargs["error_message"])
        self.log_store.fail_rag_run.assert_awaited_once_with(
            self.rag_run_id,
            error_code="UPSTREAM_ERROR",
            total_latency_ms=ANY,
        )

    # ------------------------------------------------------------------
    # 인용 기록
    # ------------------------------------------------------------------

    async def test_records_citations_with_chunk_and_document_version(self) -> None:
        self._generation_result = self._completed()

        await self.service.answer_question("질문")

        kwargs = self.log_store.complete_rag_run.await_args.kwargs
        citations = kwargs["citations"]
        self.assertEqual([1, 2], [item.citation_order for item in citations])
        self.assertEqual([2, 1], [item.chunk_id for item in citations])
        self.assertEqual([102, 101], [item.document_version_id for item in citations])
        self.assertEqual(
            ["문서 2 > 섹션 2", "문서 1 > 섹션 1"],
            [item.node_path_snapshot for item in citations],
        )
        self.assertEqual(
            ["https://docs.riido.io/2", "https://docs.riido.io/1"],
            [item.source_uri_snapshot for item in citations],
        )

    # ------------------------------------------------------------------
    # 실패 경로 마감
    # ------------------------------------------------------------------

    async def test_records_retrieval_failure_and_finishes_the_turn(self) -> None:
        failed_embedding = ModelCallTrace(
            provider="openai",
            model_name="text-embedding-3-large",
            succeeded=False,
            latency_ms=15,
            error_message="embedding unavailable",
        )
        self._search_result = HybridSearchCall(
            embedding_call=failed_embedding,
            error=RuntimeError("embedding unavailable"),
        )

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(self.rag_run_id, response.rag_run_id)
        self.generation_service.generate_answer.assert_not_awaited()
        self.log_store.fail_rag_run.assert_awaited_once_with(
            self.rag_run_id,
            error_code="UPSTREAM_ERROR",
            total_latency_ms=ANY,
        )
        embedding = self.log_store.finish_model_call.await_args
        self.assertEqual(1, embedding.args[0])
        self.assertEqual(ExecutionStatus.FAILED, embedding.kwargs["status"])
        self.log_store.start_model_call.assert_awaited_once()

    async def test_finishes_the_turn_when_generation_service_raises(self) -> None:
        self.generation_service.generate_answer.side_effect = RuntimeError(
            "provider secret detail"
        )

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(self.rag_run_id, response.rag_run_id)
        self.log_store.fail_rag_run.assert_awaited_once_with(
            self.rag_run_id,
            error_code="INTERNAL_ERROR",
            total_latency_ms=ANY,
        )
        self.assertNotIn(
            "provider secret detail",
            str(response.model_dump(mode="json", by_alias=True)),
        )

    async def test_finishes_the_turn_when_retriever_raises(self) -> None:
        self.retriever.search_with_trace.side_effect = RuntimeError(
            "postgresql connection detail"
        )

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.log_store.fail_rag_run.assert_awaited_once_with(
            self.rag_run_id,
            error_code="INTERNAL_ERROR",
            total_latency_ms=ANY,
        )

    async def test_swallows_failure_of_the_closing_transaction(self) -> None:
        self.generation_service.generate_answer.side_effect = RuntimeError("boom")
        self.log_store.fail_rag_run.side_effect = RuntimeError("write failure")

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(self.rag_run_id, response.rag_run_id)

    # ------------------------------------------------------------------

    @staticmethod
    def _search_call() -> HybridSearchCall:
        chunks = [_chunk(1), _chunk(2), _chunk(3)]
        bm25_results = (
            RetrievalResult(chunk=chunks[0], score=9.5, rank=1),
            RetrievalResult(chunk=chunks[1], score=4.5, rank=2),
            RetrievalResult(chunk=chunks[2], score=1.5, rank=3),
        )
        vector_results = (
            RetrievalResult(chunk=chunks[0], score=0.91, rank=1),
            RetrievalResult(chunk=chunks[1], score=0.72, rank=2),
        )
        fused_results = (
            HybridRetrievalResult(
                chunk=chunks[0],
                rrf_score=0.5,
                final_rank=1,
                bm25_rank=1,
                vector_rank=1,
            ),
            HybridRetrievalResult(
                chunk=chunks[1],
                rrf_score=0.25,
                final_rank=2,
                bm25_rank=2,
                vector_rank=2,
            ),
        )
        return HybridSearchCall(
            bm25_results=bm25_results,
            vector_results=vector_results,
            fused_results=fused_results,
            bm25_latency_ms=12,
            vector_latency_ms=34,
            embedding_call=_embedding_trace(),
        )

    @staticmethod
    def _completed() -> FinalGenerationResult:
        return FinalGenerationResult(
            status=FinalAnswerStatus.COMPLETED,
            answer_markdown="두 번째 근거입니다. [1] 첫 번째 근거입니다. [2]",
            citations=(
                Citation(
                    citation_number=1,
                    document_title="문서 2",
                    section_path=("문서 2", "섹션 2"),
                    source_url="https://docs.riido.io/2",
                    chunk_id=2,
                    document_version_id=102,
                ),
                Citation(
                    citation_number=2,
                    document_title="문서 1",
                    section_path=("문서 1", "섹션 1"),
                    source_url="https://docs.riido.io/1",
                    chunk_id=1,
                    document_version_id=101,
                ),
            ),
            model_call=_generation_trace(),
        )


if __name__ == "__main__":
    unittest.main()
