import asyncio
import unittest
import uuid
from itertools import count
from types import SimpleNamespace
from typing import Optional
from unittest.mock import ANY, AsyncMock

import httpx
from openai import APITimeoutError
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.schema import (
    ChatCompletedResponse,
    ChatErrorCode,
    ChatErrorResponse,
    ChatResponseStatus,
    ChatWithheldReasonCode,
    ChatWithheldResponse,
)
from app.database.models import ContextStrategy, ExecutionStatus
from app.chat.service import (
    ChatService,
    ConversationNotFoundError,
)
from app.answering.service import WITHHELD_RESPONSES, GenerationService
from app.chat.log_store import (
    CANCELLED_RUN_MODEL_CALL_ERROR_MESSAGE,
    ConversationBusyError,
    ConversationUnavailableError,
    RagLogStore,
)
from app.core.model_trace import ModelCallTrace
from app.document.document_key import (
    CONSOLE_URI_SCHEME,
    DEFAULT_DOCUMENT_GROUP_KEY,
    build_console_canonical_uri,
    build_upload_document_key,
)
from app.chat.progress import ProgressStage
from app.chat.query_rewrite import (
    QUERY_REWRITE_PROMPT_VERSION,
    QueryResolution,
    QueryRewriteCall,
    QueryRewriteCandidateTurn,
    QueryRewriteDecision,
    QueryRewriteService,
    QueryRewriteTurnStatus,
)
from app.answering.models import (
    Citation,
    CitationSourceKind,
    FinalAnswerStatus,
    FinalGenerationResult,
    FinalWithheldReason,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import (
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
        prompt_version="v2",
    )


class ChatServiceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.conversation_id = uuid.uuid4()
        self.rag_run_id = uuid.uuid4()

        self.retriever = AsyncMock(spec=HybridRetriever)
        self.generation_service = AsyncMock(spec=GenerationService)
        self.query_rewrite_service = AsyncMock(spec=QueryRewriteService)
        self.log_store = AsyncMock(spec=RagLogStore)
        self.session = AsyncMock(spec=AsyncSession)

        self.log_store.create_conversation.return_value = SimpleNamespace(
            id=self.conversation_id,
        )
        self.log_store.start_rag_run.return_value = SimpleNamespace(
            id=self.rag_run_id,
            turn_no=1,
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
                "v2",
            )
            return self._generation_result

        self.retriever.search_with_trace.side_effect = search_with_checkpoint
        self.generation_service.generate_answer.side_effect = (
            generate_with_checkpoint
        )

        self.service = ChatService(
            retriever=self.retriever,
            generation_service=self.generation_service,
            query_rewrite_service=self.query_rewrite_service,
            log_store=self.log_store,
            session=self.session,
            index_version_id=INDEX_VERSION_ID,
        )

    # ------------------------------------------------------------------
    # 응답 변환
    # ------------------------------------------------------------------

    async def test_cancelled_turn_closes_processing_logs(self) -> None:
        generation_started = asyncio.Event()

        async def wait_until_cancelled(
            _question,
            _results,
            *,
            before_model_call,
        ):
            await before_model_call("openai", "gpt-5.4-mini", "v2")
            generation_started.set()
            await asyncio.Event().wait()

        self.generation_service.generate_answer.side_effect = wait_until_cancelled
        task = asyncio.create_task(self.service.answer_question("질문"))

        await generation_started.wait()
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.log_store.fail_processing_model_calls.assert_awaited_once_with(
            self.rag_run_id,
            error_message=CANCELLED_RUN_MODEL_CALL_ERROR_MESSAGE,
        )
        self.log_store.cancel_rag_run.assert_awaited_once_with(self.rag_run_id)
        self.log_store.fail_rag_run.assert_not_awaited()
        self.session.commit.assert_awaited()

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

    async def test_response_section_path_excludes_document_title(self) -> None:
        self._generation_result = self._completed()

        response = await self.service.answer_question("질문")

        self.assertEqual(
            [["섹션 2"], ["섹션 1"]],
            [citation.section_path for citation in response.citations],
        )

    async def test_root_citation_returns_empty_response_section_path(self) -> None:
        self._generation_result = FinalGenerationResult(
            status=FinalAnswerStatus.COMPLETED,
            answer_markdown="문서 개요입니다. [1]",
            citations=(
                Citation(
                    citation_number=1,
                    document_title="문서 1",
                    section_path=("문서 1",),
                    source_url="https://docs.riido.io/1",
                    source_kind=CitationSourceKind.GITBOOK,
                    chunk_id=1,
                    document_version_id=101,
                ),
            ),
            model_call=_generation_trace(),
        )

        response = await self.service.answer_question("문서 개요를 알려주세요.")

        self.assertIsInstance(response, ChatCompletedResponse)
        self.assertEqual([], response.citations[0].section_path)

    async def test_console_citation_hides_internal_source_url(self) -> None:
        console_uri = build_console_canonical_uri(
            DEFAULT_DOCUMENT_GROUP_KEY,
            build_upload_document_key("업로드 문서"),
        )
        self._generation_result = FinalGenerationResult(
            status=FinalAnswerStatus.COMPLETED,
            answer_markdown="콘솔 문서 근거입니다. [1]",
            citations=(
                Citation(
                    citation_number=1,
                    document_title="업로드 문서",
                    section_path=("업로드 문서", "섹션 1"),
                    source_url=console_uri,
                    source_kind=CitationSourceKind.CONSOLE,
                    chunk_id=1,
                    document_version_id=101,
                ),
            ),
            model_call=_generation_trace(),
        )

        response = await self.service.answer_question("업로드 문서를 알려주세요.")

        self.assertIsInstance(response, ChatCompletedResponse)
        citation = response.citations[0]
        self.assertEqual(CitationSourceKind.CONSOLE, citation.source_kind)
        self.assertIsNone(citation.source_url)
        self.assertNotIn(
            CONSOLE_URI_SCHEME,
            response.model_dump_json(by_alias=True),
        )

    async def test_gitbook_citation_keeps_source_url(self) -> None:
        self._generation_result = self._completed()

        response = await self.service.answer_question("문서 개요를 알려주세요.")

        self.assertIsInstance(response, ChatCompletedResponse)
        citation = response.citations[0]
        self.assertEqual(CitationSourceKind.GITBOOK, citation.source_kind)
        self.assertEqual("https://docs.riido.io/2", citation.source_url)

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

    async def test_error_response_keeps_identifiers_and_exposes_safe_error_policy(
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
        self.assertEqual(ChatErrorCode.UPSTREAM_ERROR, response.error.code)
        self.assertTrue(response.error.retryable)
        self.assertEqual(self.conversation_id, response.conversation_id)
        self.assertEqual(self.rag_run_id, response.rag_run_id)
        self.assertNotIn(
            "provider secret",
            str(response.model_dump(mode="json", by_alias=True)),
        )

    async def test_citation_validation_error_is_not_retryable(self) -> None:
        self._generation_result = FinalGenerationResult(
            status=FinalAnswerStatus.ERROR,
            answer_markdown=None,
            citations=(),
            error_code="CITATION_VALIDATION_ERROR",
            model_call=_generation_trace(),
        )

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(
            ChatErrorCode.CITATION_VALIDATION_ERROR,
            response.error.code,
        )
        self.assertFalse(response.error.retryable)
        self.assertEqual(self.conversation_id, response.conversation_id)
        self.assertEqual(self.rag_run_id, response.rag_run_id)

    async def test_unknown_generation_error_code_fails_closed_in_response(
        self,
    ) -> None:
        self._generation_result = FinalGenerationResult(
            status=FinalAnswerStatus.ERROR,
            answer_markdown=None,
            citations=(),
            error_code="UNKNOWN_DETAIL",
            model_call=_generation_trace(succeeded=False),
        )

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(ChatErrorCode.INTERNAL_ERROR, response.error.code)
        self.assertFalse(response.error.retryable)
        self.assertNotIn(
            "UNKNOWN_DETAIL",
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
        self.log_store.start_rag_run.assert_awaited_once_with(
            self.conversation_id,
            user_query="질문",
            index_version_id=INDEX_VERSION_ID,
        )

    async def test_reuses_conversation_id_from_request(self) -> None:
        existing_id = uuid.uuid4()
        self._generation_result = self._completed()

        response = await self.service.answer_question("질문", existing_id)

        self.log_store.create_conversation.assert_not_awaited()
        self.log_store.start_rag_run.assert_awaited_once_with(
            existing_id,
            user_query="질문",
            index_version_id=INDEX_VERSION_ID,
        )
        self.assertEqual(existing_id, response.conversation_id)

    async def test_maps_unusable_conversation_to_not_found(self) -> None:
        existing_id = uuid.uuid4()
        self.log_store.start_rag_run.side_effect = ConversationUnavailableError(
            "이어갈 수 없는 대화입니다."
        )

        with self.assertRaises(ConversationNotFoundError):
            await self.service.answer_question("질문", existing_id)

        self.session.commit.assert_not_awaited()
        self.session.rollback.assert_awaited_once_with()
        self.retriever.search_with_trace.assert_not_awaited()

    async def test_propagates_busy_after_rolling_back_start_transaction(self) -> None:
        existing_id = uuid.uuid4()
        self.log_store.start_rag_run.side_effect = ConversationBusyError(existing_id)

        with self.assertRaises(ConversationBusyError) as raised:
            await self.service.answer_question("질문", existing_id)

        self.assertEqual(existing_id, raised.exception.conversation_id)
        self.session.rollback.assert_awaited_once_with()
        self.session.commit.assert_not_awaited()
        self.retriever.search_with_trace.assert_not_awaited()
        self.generation_service.generate_answer.assert_not_awaited()

    # ------------------------------------------------------------------
    # Multi-turn Query Rewrite
    # ------------------------------------------------------------------

    async def test_first_turn_skips_query_rewrite(self) -> None:
        self._generation_result = self._completed()

        await self.service.answer_question("첫 질문")

        self.query_rewrite_service.rewrite.assert_not_awaited()
        self.log_store.get_query_rewrite_candidates.assert_not_awaited()
        self.retriever.search_with_trace.assert_awaited_once_with(
            "첫 질문",
            before_model_call=ANY,
        )

    async def test_follow_up_uses_resolved_query_for_retrieval_and_generation(
        self,
    ) -> None:
        self._generation_result = self._completed()
        candidate = self._candidate_turn(1)
        self._prepare_follow_up(
            QueryRewriteCall(
                trace=self._query_rewrite_trace(),
                resolution=QueryResolution(
                    decision=QueryRewriteDecision.FOLLOW_UP_RESOLVED,
                    resolved_query="리두에서 멤버를 초대하는 방법",
                    selected_turns=(candidate,),
                ),
            ),
            [candidate],
        )

        response = await self.service.answer_question(
            "그건 어떻게 해?",
            self.conversation_id,
        )

        self.assertIsInstance(response, ChatCompletedResponse)
        self.retriever.search_with_trace.assert_awaited_once_with(
            "리두에서 멤버를 초대하는 방법",
            before_model_call=ANY,
        )
        generation_question = self.generation_service.generate_answer.await_args.args[0]
        self.assertEqual("리두에서 멤버를 초대하는 방법", generation_question)
        resolution_log = self.log_store.record_query_resolution.await_args.kwargs
        self.assertEqual(
            ContextStrategy.FOLLOW_UP_WINDOW,
            resolution_log["context_strategy"],
        )
        self.assertEqual(1, resolution_log["context_turn_count"])
        self.assertEqual(
            "v1",
            resolution_log["context_snapshot"]["schemaVersion"],
        )
        self.assertEqual(
            [1],
            [
                turn["turnNo"]
                for turn in resolution_log["context_snapshot"]["selectedTurns"]
            ],
        )
        self.assertEqual(
            ["QUERY_REWRITE", "QUERY_EMBEDDING", "ANSWER_GENERATION"],
            [
                awaited.kwargs["purpose"]
                for awaited in self.log_store.start_model_call.await_args_list
            ],
        )

    async def test_new_topic_uses_current_question_without_context_snapshot(
        self,
    ) -> None:
        self._generation_result = self._completed()
        candidate = self._candidate_turn(1)
        self._prepare_follow_up(
            QueryRewriteCall(
                trace=self._query_rewrite_trace(),
                resolution=QueryResolution(
                    decision=QueryRewriteDecision.NEW_TOPIC,
                    resolved_query="결제 수단을 변경하는 방법",
                    selected_turns=(),
                ),
            ),
            [candidate],
        )

        await self.service.answer_question(
            "결제 수단을 변경하는 방법",
            self.conversation_id,
        )

        self.retriever.search_with_trace.assert_awaited_once_with(
            "결제 수단을 변경하는 방법",
            before_model_call=ANY,
        )
        self.log_store.record_query_resolution.assert_awaited_once_with(
            self.rag_run_id,
            resolved_query="결제 수단을 변경하는 방법",
            context_strategy=ContextStrategy.NEW_TOPIC,
            context_turn_count=0,
            context_snapshot=None,
        )

    async def test_unresolved_follow_up_is_withheld_without_retrieval(self) -> None:
        self._prepare_follow_up(
            QueryRewriteCall(
                trace=self._query_rewrite_trace(),
                resolution=QueryResolution(
                    decision=QueryRewriteDecision.FOLLOW_UP_UNRESOLVED,
                    resolved_query=None,
                    selected_turns=(),
                ),
            ),
            [self._candidate_turn(1)],
        )

        on_progress_stage = AsyncMock()
        response = await self.service.answer_question(
            "그거 말고 다른 건?",
            self.conversation_id,
            on_progress_stage=on_progress_stage,
        )

        self.assertIsInstance(response, ChatWithheldResponse)
        self.assertEqual(
            ChatWithheldReasonCode.AMBIGUOUS_QUESTION,
            response.withheld.reason_code,
        )
        self.retriever.search_with_trace.assert_not_awaited()
        self.generation_service.generate_answer.assert_not_awaited()
        on_progress_stage.assert_awaited_once_with(ProgressStage.RETRIEVING)
        self.log_store.record_query_resolution.assert_awaited_once_with(
            self.rag_run_id,
            resolved_query=None,
            context_strategy=ContextStrategy.FOLLOW_UP_WINDOW,
            context_turn_count=0,
            context_snapshot=None,
        )
        self.log_store.withhold_rag_run.assert_awaited_once_with(
            self.rag_run_id,
            reason_code="AMBIGUOUS_QUESTION",
            total_latency_ms=ANY,
        )

    async def test_query_rewrite_failure_stops_before_retrieval(self) -> None:
        error = RuntimeError("invalid structured output")
        self._prepare_follow_up(
            QueryRewriteCall(
                trace=self._query_rewrite_trace(succeeded=False, error=error),
                error_code="MODEL_OUTPUT_INVALID",
                error=error,
            ),
            [self._candidate_turn(1)],
        )

        response = await self.service.answer_question(
            "그건 어떻게 해?",
            self.conversation_id,
        )

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(ChatErrorCode.MODEL_OUTPUT_INVALID, response.error.code)
        self.assertTrue(response.error.retryable)
        self.retriever.search_with_trace.assert_not_awaited()
        self.generation_service.generate_answer.assert_not_awaited()
        self.log_store.record_query_resolution.assert_not_awaited()
        self.log_store.fail_rag_run.assert_awaited_once_with(
            self.rag_run_id,
            error_code="MODEL_OUTPUT_INVALID",
            total_latency_ms=ANY,
        )
        finished = self.log_store.finish_model_call.await_args
        self.assertEqual(ExecutionStatus.FAILED, finished.kwargs["status"])

    async def test_resolution_commit_failure_stops_before_retrieval(self) -> None:
        candidate = self._candidate_turn(1)
        self._prepare_follow_up(
            QueryRewriteCall(
                trace=self._query_rewrite_trace(),
                resolution=QueryResolution(
                    decision=QueryRewriteDecision.FOLLOW_UP_RESOLVED,
                    resolved_query="독립 질문",
                    selected_turns=(candidate,),
                ),
            ),
            [candidate],
        )
        self.session.commit.side_effect = [
            None,
            None,
            RuntimeError("resolution write failure"),
            None,
        ]

        response = await self.service.answer_question(
            "그건 어떻게 해?",
            self.conversation_id,
        )

        self.assertIsInstance(response, ChatErrorResponse)
        self.retriever.search_with_trace.assert_not_awaited()
        self.generation_service.generate_answer.assert_not_awaited()
        self.log_store.fail_processing_model_calls.assert_awaited_once_with(
            self.rag_run_id
        )
        self.log_store.fail_rag_run.assert_awaited_once_with(
            self.rag_run_id,
            error_code="INTERNAL_ERROR",
            total_latency_ms=ANY,
        )

    async def test_ambiguous_final_commit_failure_cannot_return_withheld(
        self,
    ) -> None:
        self._prepare_follow_up(
            QueryRewriteCall(
                trace=self._query_rewrite_trace(),
                resolution=QueryResolution(
                    decision=QueryRewriteDecision.FOLLOW_UP_UNRESOLVED,
                    resolved_query=None,
                    selected_turns=(),
                ),
            ),
            [self._candidate_turn(1)],
        )
        self.session.commit.side_effect = [
            None,
            None,
            None,
            RuntimeError("withheld write failure"),
            None,
        ]

        response = await self.service.answer_question(
            "그거 말고 다른 건?",
            self.conversation_id,
        )

        self.assertIsInstance(response, ChatErrorResponse)
        self.retriever.search_with_trace.assert_not_awaited()
        self.generation_service.generate_answer.assert_not_awaited()
        self.log_store.fail_processing_model_calls.assert_awaited_once_with(
            self.rag_run_id
        )
        self.log_store.fail_rag_run.assert_awaited_once_with(
            self.rag_run_id,
            error_code="INTERNAL_ERROR",
            total_latency_ms=ANY,
        )

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
            await before_model_call("openai", "gpt-5.4-mini", "v2")
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
        self.assertEqual(ChatErrorCode.INTERNAL_ERROR, response.error.code)
        self.assertFalse(response.error.retryable)
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
            await before_model_call("openai", "gpt-5.4-mini", "v2")
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
        # 실패한 checkpoint transaction을 rollback한 뒤 Embedding 마감을 다시 기록한다.
        self.assertEqual(2, self.log_store.finish_model_call.await_count)
        self.assertEqual(2, self.log_store.record_retrieval_results.await_count)
        self.log_store.fail_rag_run.assert_awaited_once_with(
            self.rag_run_id,
            error_code="INTERNAL_ERROR",
            total_latency_ms=ANY,
        )

    async def test_returns_error_and_closes_run_when_final_commit_fails(self) -> None:
        self._generation_result = self._completed()
        self.session.commit.side_effect = [
            None,
            None,
            None,
            RuntimeError("write failure"),
            None,
        ]

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(self.rag_run_id, response.rag_run_id)
        self.assertEqual(ChatErrorCode.INTERNAL_ERROR, response.error.code)
        self.assertFalse(response.error.retryable)
        self.log_store.fail_processing_model_calls.assert_awaited_once_with(
            self.rag_run_id
        )
        self.log_store.fail_rag_run.assert_awaited_once_with(
            self.rag_run_id,
            error_code="INTERNAL_ERROR",
            total_latency_ms=ANY,
        )
        self.assertEqual(5, self.session.commit.await_count)

    async def test_late_worker_after_stale_recovery_cannot_return_success(
        self,
    ) -> None:
        self._generation_result = self._completed()
        self.log_store.finish_model_call.side_effect = [
            None,
            ValueError("PROCESSING 상태의 턴만 전이할 수 있습니다: ERROR"),
        ]
        self.log_store.fail_processing_model_calls.side_effect = ValueError(
            "PROCESSING 상태의 턴만 전이할 수 있습니다: ERROR"
        )

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(self.rag_run_id, response.rag_run_id)
        self.log_store.complete_rag_run.assert_not_awaited()
        self.log_store.fail_rag_run.assert_not_awaited()

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
        self.assertEqual(["QUERY_EMBEDDING", "ANSWER_GENERATION"], purposes)

        embedding_start = self.log_store.start_model_call.await_args_list[0].kwargs
        self.assertEqual(self.rag_run_id, embedding_start["rag_run_id"])
        self.assertIsNone(embedding_start["prompt_version"])

        generation_start = self.log_store.start_model_call.await_args_list[1].kwargs
        self.assertEqual("v2", generation_start["prompt_version"])

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

    async def test_finishes_trace_less_generation_error_on_checkpoint_row(
        self,
    ) -> None:
        self._generation_result = FinalGenerationResult(
            status=FinalAnswerStatus.ERROR,
            answer_markdown=None,
            citations=(),
            error_code="UPSTREAM_ERROR",
        )

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(self.rag_run_id, response.rag_run_id)
        generation = self.log_store.finish_model_call.await_args_list[1]
        self.assertEqual(2, generation.args[0])
        self.assertEqual(ExecutionStatus.FAILED, generation.kwargs["status"])
        self.assertIn("trace", generation.kwargs["error_message"])
        self.log_store.fail_rag_run.assert_awaited_once_with(
            self.rag_run_id,
            error_code="UPSTREAM_ERROR",
            total_latency_ms=ANY,
        )

    async def test_rejects_trace_less_completed_generation_result(self) -> None:
        completed = self._completed()
        self._generation_result = FinalGenerationResult(
            status=FinalAnswerStatus.COMPLETED,
            answer_markdown=completed.answer_markdown,
            citations=completed.citations,
        )

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        generation = self.log_store.finish_model_call.await_args_list[1]
        self.assertEqual(2, generation.args[0])
        self.assertEqual(ExecutionStatus.FAILED, generation.kwargs["status"])
        self.assertIn("trace", generation.kwargs["error_message"])
        self.log_store.complete_rag_run.assert_not_awaited()
        self.log_store.fail_rag_run.assert_awaited_once_with(
            self.rag_run_id,
            error_code="INTERNAL_ERROR",
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
        error = APITimeoutError(
            httpx.Request("POST", "https://api.openai.com")
        )
        failed_embedding = ModelCallTrace(
            provider="openai",
            model_name="text-embedding-3-large",
            succeeded=False,
            latency_ms=15,
            error_message="embedding unavailable",
        )
        self._search_result = HybridSearchCall(
            embedding_call=failed_embedding,
            error=error,
        )

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(ChatErrorCode.UPSTREAM_ERROR, response.error.code)
        self.assertTrue(response.error.retryable)
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

    async def test_maps_non_transient_embedding_failure_to_internal_error(
        self,
    ) -> None:
        failed_embedding = ModelCallTrace(
            provider="openai",
            model_name="text-embedding-3-large",
            succeeded=False,
            latency_ms=15,
            error_message="invalid embedding response",
        )
        self._search_result = HybridSearchCall(
            embedding_call=failed_embedding,
            error=RuntimeError("invalid embedding response"),
        )

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(ChatErrorCode.INTERNAL_ERROR, response.error.code)
        self.assertFalse(response.error.retryable)
        self.log_store.fail_rag_run.assert_awaited_once_with(
            self.rag_run_id,
            error_code="INTERNAL_ERROR",
            total_latency_ms=ANY,
        )

    async def test_retrieval_failure_commit_error_returns_internal_error(
        self,
    ) -> None:
        error = APITimeoutError(
            httpx.Request("POST", "https://api.openai.com")
        )
        self._search_result = HybridSearchCall(
            embedding_call=ModelCallTrace(
                provider="openai",
                model_name="text-embedding-3-large",
                succeeded=False,
                latency_ms=15,
                error_message="embedding unavailable",
            ),
            error=error,
        )
        self.session.commit.side_effect = [
            None,
            None,
            RuntimeError("retrieval failure write failed"),
            None,
        ]

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(ChatErrorCode.INTERNAL_ERROR, response.error.code)
        self.assertFalse(response.error.retryable)
        self.log_store.fail_rag_run.assert_awaited_with(
            self.rag_run_id,
            error_code="INTERNAL_ERROR",
            total_latency_ms=ANY,
        )

    async def test_finishes_the_turn_when_generation_service_raises(self) -> None:
        async def raise_after_checkpoint(
            _question,
            _results,
            *,
            before_model_call,
        ):
            await before_model_call("openai", "gpt-5.4-mini", "v2")
            raise RuntimeError("provider secret detail")

        self.generation_service.generate_answer.side_effect = raise_after_checkpoint

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(self.rag_run_id, response.rag_run_id)
        self.log_store.record_retrieval_results.assert_awaited_once()
        generation = self.log_store.finish_model_call.await_args_list[1]
        self.assertEqual(2, generation.args[0])
        self.assertEqual(ExecutionStatus.FAILED, generation.kwargs["status"])
        self.assertEqual(
            "provider secret detail",
            generation.kwargs["error_message"],
        )
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

    def _prepare_follow_up(
        self,
        call: QueryRewriteCall,
        candidates: list[QueryRewriteCandidateTurn],
    ) -> None:
        self.log_store.start_rag_run.return_value = SimpleNamespace(
            id=self.rag_run_id,
            turn_no=2,
        )
        self.log_store.get_query_rewrite_candidates.return_value = candidates

        async def rewrite(
            _question,
            _candidates,
            *,
            before_model_call,
        ):
            await before_model_call(
                "openai",
                "gpt-5.4-mini",
                QUERY_REWRITE_PROMPT_VERSION,
            )
            return call

        self.query_rewrite_service.rewrite.side_effect = rewrite

    @staticmethod
    def _candidate_turn(turn_no: int) -> QueryRewriteCandidateTurn:
        return QueryRewriteCandidateTurn(
            rag_run_id=uuid.UUID(int=turn_no),
            turn_no=turn_no,
            status=QueryRewriteTurnStatus.COMPLETED,
            user_query=f"이전 질문 {turn_no}",
            answer_content=f"이전 답변 {turn_no}",
            withheld_reason_code=None,
        )

    @staticmethod
    def _query_rewrite_trace(
        *,
        succeeded: bool = True,
        error: Optional[Exception] = None,
    ) -> ModelCallTrace:
        return ModelCallTrace(
            provider="openai",
            model_name="gpt-5.4-mini",
            succeeded=succeeded,
            latency_ms=120,
            input_tokens=90,
            output_tokens=20,
            prompt_version=QUERY_REWRITE_PROMPT_VERSION,
            error_message=None if error is None else str(error),
        )

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
                    source_kind=CitationSourceKind.GITBOOK,
                    chunk_id=2,
                    document_version_id=102,
                ),
                Citation(
                    citation_number=2,
                    document_title="문서 1",
                    section_path=("문서 1", "섹션 1"),
                    source_url="https://docs.riido.io/1",
                    source_kind=CitationSourceKind.GITBOOK,
                    chunk_id=1,
                    document_version_id=101,
                ),
            ),
            model_call=_generation_trace(),
        )


if __name__ == "__main__":
    unittest.main()
