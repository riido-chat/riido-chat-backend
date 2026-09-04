"""Chat HTTP DTO와 기존 RAG 파이프라인을 연결하고 턴 실행을 기록한다."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, Union

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.schema import (
    ChatAnswer,
    ChatCitation,
    ChatCompletedResponse,
    ChatError,
    ChatErrorCode,
    ChatErrorResponse,
    ChatResponse,
    ChatResponseStatus,
    ChatWithheld,
    ChatWithheldReasonCode,
    ChatWithheldResponse,
)
from app.database.models import (
    ContextStrategy,
    ExecutionStatus,
    ModelCallPurpose,
    RetrieverType,
)
from app.answering.service import (
    UPSTREAM_ERROR_CODE,
    WITHHELD_RESPONSES,
    GenerationService,
)
from app.chat.log_store import (
    CANCELLED_RUN_MODEL_CALL_ERROR_MESSAGE,
    CitationLog,
    ConversationBusyError,
    ConversationUnavailableError,
    RagLogStore,
    RetrievalCandidateLog,
)
from app.core.model_trace import ModelCallTrace
from app.core.openai_error import is_transient_openai_error
from app.chat.progress import (
    OnProgressStageHook,
    OnTurnStartedHook,
    ProgressStage,
)
from app.chat.query_rewrite import (
    QueryResolution,
    QueryRewriteCall,
    QueryRewriteDecision,
    QueryRewriteService,
    build_context_snapshot,
)
from app.answering.models import (
    Citation,
    CitationSourceKind,
    FinalAnswerStatus,
    FinalGenerationResult,
    FinalWithheldReason,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import HybridSearchCall, RetrievalResult


logger = logging.getLogger(__name__)

UPSTREAM_ERROR_MESSAGE = (
    "AI 서비스 연결이 원활하지 않습니다. 잠시 후 다시 시도해주세요."
)
MODEL_OUTPUT_INVALID_MESSAGE = (
    "AI 응답을 처리하지 못했습니다. 잠시 후 다시 시도해주세요."
)
CITATION_VALIDATION_ERROR_MESSAGE = (
    "답변 출처를 검증하는 중 오류가 발생했습니다."
)
INTERNAL_ERROR_MESSAGE = "답변을 생성하는 중 오류가 발생했습니다."
SERVICE_UNAVAILABLE_MESSAGE = "검색 데이터가 아직 준비되지 않았습니다."
CONVERSATION_NOT_FOUND_MESSAGE = (
    "이어갈 수 없는 대화입니다. 새로운 대화로 다시 질문해주세요."
)
CONVERSATION_BUSY_MESSAGE = (
    "이 대화의 이전 질문을 처리 중입니다. 잠시 후 다시 시도해주세요."
)

INTERNAL_ERROR_CODE = "INTERNAL_ERROR"

CHAT_ERROR_POLICIES: Dict[ChatErrorCode, Tuple[str, bool]] = {
    ChatErrorCode.UPSTREAM_ERROR: (UPSTREAM_ERROR_MESSAGE, True),
    ChatErrorCode.MODEL_OUTPUT_INVALID: (MODEL_OUTPUT_INVALID_MESSAGE, True),
    ChatErrorCode.CITATION_VALIDATION_ERROR: (
        CITATION_VALIDATION_ERROR_MESSAGE,
        False,
    ),
    ChatErrorCode.INTERNAL_ERROR: (INTERNAL_ERROR_MESSAGE, False),
    ChatErrorCode.SERVICE_UNAVAILABLE: (SERVICE_UNAVAILABLE_MESSAGE, False),
    ChatErrorCode.NOT_FOUND: (CONVERSATION_NOT_FOUND_MESSAGE, False),
    ChatErrorCode.CONVERSATION_BUSY: (CONVERSATION_BUSY_MESSAGE, False),
}


class ConversationNotFoundError(LookupError):
    """이어갈 수 없는 conversationId로 요청이 들어왔을 때 발생한다.

    미존재, CLOSED, EXPIRED를 구분하지 않고 하나로 다룬다. FE 입장에서 셋 다
    "새 대화를 시작하라"로 귀결되고, 존재 여부를 구분해 알려줄 이유가 없다.
    """


def chat_error_response(
    error_code: Union[str, ChatErrorCode],
    conversation_id: Optional[uuid.UUID] = None,
    rag_run_id: Optional[uuid.UUID] = None,
) -> ChatErrorResponse:
    """확정된 오류 코드에서 외부 안내와 재시도 정책을 결정한다.

    알 수 없는 코드는 내부 상세를 노출하지 않고 재시도 불가로 처리한다.
    """

    try:
        external_code = ChatErrorCode(error_code)
    except ValueError:
        external_code = ChatErrorCode.INTERNAL_ERROR
    message, retryable = CHAT_ERROR_POLICIES[external_code]
    return ChatErrorResponse(
        status=ChatResponseStatus.ERROR,
        conversation_id=conversation_id,
        rag_run_id=rag_run_id,
        answer=None,
        error=ChatError(
            code=external_code,
            message=message,
            retryable=retryable,
        ),
        citations=[],
    )


def _internal_error_response(
    conversation_id: Optional[uuid.UUID] = None,
    rag_run_id: Optional[uuid.UUID] = None,
) -> ChatErrorResponse:
    return chat_error_response(
        ChatErrorCode.INTERNAL_ERROR,
        conversation_id,
        rag_run_id,
    )


def conversation_not_found_response() -> ChatErrorResponse:
    """이어갈 수 없는 대화로 요청했을 때의 응답을 만든다."""

    return chat_error_response(ChatErrorCode.NOT_FOUND)


def conversation_busy_response(conversation_id: uuid.UUID) -> ChatErrorResponse:
    """같은 대화의 이전 턴이 처리 중일 때의 충돌 응답을 만든다."""

    return chat_error_response(
        ChatErrorCode.CONVERSATION_BUSY,
        conversation_id=conversation_id,
    )


def _to_response_section_path(citation: Citation) -> List[str]:
    """내부 전체 경로에서 API가 별도 제공하는 문서 제목을 제외한다."""

    section_path = citation.section_path
    if section_path and section_path[0] == citation.document_title:
        section_path = section_path[1:]
    return list(section_path)


def _to_response_source_url(citation: Citation) -> Optional[str]:
    """외부에 링크로 노출할 수 있는 출처만 남긴다.

    콘솔 업로드 문서의 원문 위치자는 내부 스킴이라 null로 내리고, 클라이언트는
    sourceKind로 링크 표시 여부를 판단한다.
    """

    if citation.source_kind == CitationSourceKind.CONSOLE:
        return None
    return citation.source_url


def _to_chat_response(
    result: FinalGenerationResult,
    conversation_id: uuid.UUID,
    rag_run_id: uuid.UUID,
) -> ChatResponse:
    if result.status == FinalAnswerStatus.COMPLETED:
        if result.answer_markdown is None:
            raise ValueError("COMPLETED 결과에 answer_markdown이 없습니다.")

        return ChatCompletedResponse(
            status=ChatResponseStatus.COMPLETED,
            conversation_id=conversation_id,
            rag_run_id=rag_run_id,
            answer=ChatAnswer(answer_markdown=result.answer_markdown),
            citations=[
                ChatCitation(
                    citation_number=citation.citation_number,
                    document_title=citation.document_title,
                    section_path=_to_response_section_path(citation),
                    source_url=_to_response_source_url(citation),
                    source_kind=citation.source_kind,
                )
                for citation in result.citations
            ],
        )

    if result.status == FinalAnswerStatus.WITHHELD:
        if result.withheld_reason is None or result.answer_markdown is None:
            raise ValueError("WITHHELD 결과에 보류 사유 또는 안내 문구가 없습니다.")

        return ChatWithheldResponse(
            status=ChatResponseStatus.WITHHELD,
            conversation_id=conversation_id,
            rag_run_id=rag_run_id,
            answer=None,
            withheld=ChatWithheld(
                reason_code=ChatWithheldReasonCode(result.withheld_reason.value),
                message=result.answer_markdown,
            ),
            citations=[],
        )

    if result.status == FinalAnswerStatus.ERROR:
        return chat_error_response(
            result.error_code or INTERNAL_ERROR_CODE,
            conversation_id,
            rag_run_id,
        )

    raise ValueError(f"지원하지 않는 최종 답변 상태입니다: {result.status}")


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


@dataclass(frozen=True)
class _ModelCallCheckpoint:
    id: int
    provider: str
    model_name: str
    prompt_version: Optional[str]
    started: float


@dataclass(frozen=True)
class _TurnStart:
    conversation_id: uuid.UUID
    rag_run_id: uuid.UUID
    turn_no: int


@dataclass(frozen=True)
class _QueryResolutionOutcome:
    resolved_query: Optional[str]
    terminal_response: Optional[ChatResponse] = None


def _failed_generation_trace(
    checkpoint: _ModelCallCheckpoint,
    error: Exception,
) -> ModelCallTrace:
    return ModelCallTrace(
        provider=checkpoint.provider,
        model_name=checkpoint.model_name,
        succeeded=False,
        latency_ms=_elapsed_ms(checkpoint.started),
        prompt_version=checkpoint.prompt_version,
        error_message=str(error),
    )


class ChatService:
    """Hybrid Retrieval부터 Chat HTTP 응답 변환까지 연결하고 실행 로그를 남긴다.

    RagRun 로그는 시작과 최종 마감의 2단계 커밋을 유지한다. 각 외부 모델 호출
    직전에는 ModelCall(PROCESSING)을 짧게 checkpoint commit하고, 호출 결과는
    다음 checkpoint 또는 최종 마감에서 같은 행에 반영한다.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        generation_service: GenerationService,
        query_rewrite_service: QueryRewriteService,
        log_store: RagLogStore,
        session: AsyncSession,
        index_version_id: int,
    ) -> None:
        self._retriever = retriever
        self._generation_service = generation_service
        self._query_rewrite_service = query_rewrite_service
        self._log_store = log_store
        self._session = session
        self._index_version_id = index_version_id

    async def answer_question(
        self,
        question: str,
        conversation_id: Optional[uuid.UUID] = None,
        *,
        on_turn_started: Optional[OnTurnStartedHook] = None,
        on_progress_stage: Optional[OnProgressStageHook] = None,
    ) -> ChatResponse:
        """질문을 검색·생성 파이프라인에 전달하고 실행 로그와 함께 응답한다."""

        started = time.perf_counter()

        try:
            turn = await self._start_turn(
                question,
                conversation_id,
            )
        except asyncio.CancelledError:
            await self._rollback_quietly()
            raise
        except (ConversationNotFoundError, ConversationBusyError):
            # FOR UPDATE를 포함한 시작 transaction을 닫고 HTTP 계층으로 전달한다.
            await self._rollback_quietly()
            raise
        except Exception:
            # 1차 커밋 실패다. ragRunId가 없으면 응답을 구성할 수 없어 그대로 실패시킨다.
            logger.exception("턴을 시작하지 못했습니다.")
            await self._rollback_quietly()
            return _internal_error_response()

        try:
            if on_turn_started is not None:
                await on_turn_started(turn.conversation_id, turn.rag_run_id)
            return await self._run_turn(
                question,
                turn,
                started,
                on_progress_stage=on_progress_stage,
            )
        except asyncio.CancelledError:
            await self._cancel_quietly(turn.rag_run_id)
            raise
        except Exception:
            logger.exception(
                "턴 실행 중 예상하지 못한 오류가 발생했습니다: rag_run_id=%s",
                turn.rag_run_id,
            )
            # 마감하지 않으면 PROCESSING 행이 그대로 남는다.
            await self._fail_quietly(
                turn.rag_run_id,
                INTERNAL_ERROR_CODE,
                _elapsed_ms(started),
            )
            return _internal_error_response(turn.conversation_id, turn.rag_run_id)

    async def _run_turn(
        self,
        question: str,
        turn: _TurnStart,
        started: float,
        *,
        on_progress_stage: Optional[OnProgressStageHook] = None,
    ) -> ChatResponse:
        conversation_id = turn.conversation_id
        rag_run_id = turn.rag_run_id

        # Query Rewrite도 검색 질의 확정 단계이므로 기존 RETRIEVING에 포함한다.
        if on_progress_stage is not None:
            await on_progress_stage(ProgressStage.RETRIEVING)

        if turn.turn_no == 1:
            resolved_query = question
        else:
            resolution_outcome = await self._resolve_follow_up(
                question,
                conversation_id,
                rag_run_id,
                started,
            )
            if resolution_outcome.terminal_response is not None:
                return resolution_outcome.terminal_response
            if resolution_outcome.resolved_query is None:
                raise RuntimeError("검색할 resolved_query가 확정되지 않았습니다.")
            resolved_query = resolution_outcome.resolved_query

        embedding_model_call_id: Optional[int] = None

        async def checkpoint_embedding(
            provider: str,
            model_name: str,
            prompt_version: Optional[str],
        ) -> None:
            nonlocal embedding_model_call_id
            if embedding_model_call_id is not None:
                raise RuntimeError("Query Embedding 호출이 이미 시작됐습니다.")
            embedding_model_call_id = await self._checkpoint_model_call(
                rag_run_id,
                ModelCallPurpose.QUERY_EMBEDDING,
                provider,
                model_name,
                prompt_version,
            )

        search = await self._retriever.search_with_trace(
            resolved_query,
            before_model_call=checkpoint_embedding,
        )
        # Vector read transaction을 닫고, 확보한 로그는 다음 transaction에서 마감한다.
        await self._session.rollback()

        if search.error is not None:
            search_error_code = self._search_error_code(search)
            logger.warning(
                "검색에 실패했습니다: rag_run_id=%s",
                rag_run_id,
                exc_info=search.error,
            )
            recorded = await self._record_turn(
                rag_run_id,
                search,
                None,
                None,
                _elapsed_ms(started),
                embedding_model_call_id=embedding_model_call_id,
                record_retrieval=True,
            )
            if not recorded:
                return _internal_error_response(conversation_id, rag_run_id)
            return chat_error_response(
                search_error_code,
                conversation_id,
                rag_run_id,
            )

        generation_checkpoint: Optional[_ModelCallCheckpoint] = None

        async def checkpoint_generation(
            provider: str,
            model_name: str,
            prompt_version: Optional[str],
        ) -> None:
            nonlocal generation_checkpoint
            if generation_checkpoint is not None:
                raise RuntimeError("Generation 호출이 이미 시작됐습니다.")

            # Generation 시작 commit에 앞서 확보한 Retrieval 로그도 함께 확정한다.
            await self._finish_model_call(
                embedding_model_call_id,
                search.embedding_call,
            )
            await self._record_retrieval_results(rag_run_id, search)
            model_call_id = await self._checkpoint_model_call(
                rag_run_id,
                ModelCallPurpose.ANSWER_GENERATION,
                provider,
                model_name,
                prompt_version,
            )
            generation_checkpoint = _ModelCallCheckpoint(
                id=model_call_id,
                provider=provider,
                model_name=model_name,
                prompt_version=prompt_version,
                started=time.perf_counter(),
            )
            if on_progress_stage is not None:
                await on_progress_stage(ProgressStage.GENERATING)

        generation_result: Optional[FinalGenerationResult] = None
        # 훅 미주입 시 generate_answer 호출 인자를 기존과 완전히 동일하게 둔다.
        generation_kwargs: Dict[str, OnProgressStageHook] = {}
        if on_progress_stage is not None:
            generation_kwargs["on_progress_stage"] = on_progress_stage

        try:
            generation_result = await self._generation_service.generate_answer(
                resolved_query,
                search.fused_results,
                before_model_call=checkpoint_generation,
                **generation_kwargs,
            )
            if generation_checkpoint is None:
                raise RuntimeError("Generation ModelCall checkpoint가 실행되지 않았습니다.")
            if generation_result.model_call is None:
                missing_trace_error = RuntimeError("Generation 호출 trace가 없습니다.")
                if generation_result.status != FinalAnswerStatus.ERROR:
                    raise missing_trace_error
                generation_result = replace(
                    generation_result,
                    model_call=_failed_generation_trace(
                        generation_checkpoint,
                        missing_trace_error,
                    ),
                )

            response = _to_chat_response(
                generation_result,
                conversation_id,
                rag_run_id,
            )
        except Exception as error:
            logger.exception(
                "Generation 처리 중 예상하지 못한 오류가 발생했습니다: "
                "rag_run_id=%s",
                rag_run_id,
            )
            await self._session.rollback()

            generation_trace = (
                None if generation_result is None else generation_result.model_call
            )
            if generation_checkpoint is not None and generation_trace is None:
                generation_trace = _failed_generation_trace(
                    generation_checkpoint,
                    error,
                )
            failed_result = FinalGenerationResult(
                status=FinalAnswerStatus.ERROR,
                answer_markdown=None,
                citations=(),
                error_code=INTERNAL_ERROR_CODE,
                model_call=generation_trace,
            )
            generation_model_call_id = (
                None if generation_checkpoint is None else generation_checkpoint.id
            )
            retrieval_needs_recovery = generation_checkpoint is None
            await self._record_turn(
                rag_run_id,
                search,
                failed_result,
                generation_model_call_id,
                _elapsed_ms(started),
                embedding_model_call_id=(
                    embedding_model_call_id
                    if retrieval_needs_recovery
                    else None
                ),
                record_retrieval=retrieval_needs_recovery,
            )
            return _internal_error_response(conversation_id, rag_run_id)

        recorded = await self._record_turn(
            rag_run_id,
            search,
            generation_result,
            generation_checkpoint.id,
            _elapsed_ms(started),
            embedding_model_call_id=None,
            record_retrieval=False,
        )
        if not recorded:
            return _internal_error_response(conversation_id, rag_run_id)
        return response

    # ------------------------------------------------------------------
    # 1차 트랜잭션 — 대화와 턴 생성
    # ------------------------------------------------------------------

    async def _start_turn(
        self,
        question: str,
        conversation_id: Optional[uuid.UUID],
    ) -> _TurnStart:
        if conversation_id is None:
            conversation = await self._log_store.create_conversation()
            conversation_id = conversation.id

        try:
            run = await self._log_store.start_rag_run(
                conversation_id,
                user_query=question,
                index_version_id=self._index_version_id,
            )
        except ConversationUnavailableError as error:
            raise ConversationNotFoundError(
                f"이어갈 수 없는 대화입니다: {conversation_id}"
            ) from error
        # commit 이후 객체 접근을 피하려고 식별자를 먼저 확정한다.
        turn = _TurnStart(
            conversation_id=conversation_id,
            rag_run_id=run.id,
            turn_no=run.turn_no,
        )
        await self._session.commit()
        return turn

    # ------------------------------------------------------------------
    # Query Rewrite checkpoint — 후속 질문의 검색 질의 확정
    # ------------------------------------------------------------------

    async def _resolve_follow_up(
        self,
        question: str,
        conversation_id: uuid.UUID,
        rag_run_id: uuid.UUID,
        started: float,
    ) -> _QueryResolutionOutcome:
        candidates = await self._log_store.get_query_rewrite_candidates(rag_run_id)
        model_call_id: Optional[int] = None

        async def checkpoint_query_rewrite(
            provider: str,
            model_name: str,
            prompt_version: Optional[str],
        ) -> None:
            nonlocal model_call_id
            if model_call_id is not None:
                raise RuntimeError("Query Rewrite 호출이 이미 시작됐습니다.")
            model_call_id = await self._checkpoint_model_call(
                rag_run_id,
                ModelCallPurpose.QUERY_REWRITE,
                provider,
                model_name,
                prompt_version,
            )

        call = await self._query_rewrite_service.rewrite(
            question,
            candidates,
            before_model_call=checkpoint_query_rewrite,
        )
        if model_call_id is None:
            raise RuntimeError("Query Rewrite ModelCall checkpoint가 실행되지 않았습니다.")
        if call.error is None and call.resolution is None:
            raise RuntimeError("Query Rewrite 성공 결과에 resolution이 없습니다.")
        if call.error is None and not call.trace.succeeded:
            raise RuntimeError("Query Rewrite 성공 결과의 trace가 FAILED입니다.")
        if call.error is not None and call.trace.succeeded:
            raise RuntimeError("Query Rewrite 실패 결과의 trace가 SUCCESS입니다.")

        recorded = await self._record_query_rewrite(
            rag_run_id,
            model_call_id,
            call,
            _elapsed_ms(started),
        )
        if not recorded:
            return _QueryResolutionOutcome(
                resolved_query=None,
                terminal_response=_internal_error_response(
                    conversation_id,
                    rag_run_id,
                ),
            )
        if call.error is not None:
            return _QueryResolutionOutcome(
                resolved_query=None,
                terminal_response=chat_error_response(
                    call.error_code or INTERNAL_ERROR_CODE,
                    conversation_id,
                    rag_run_id,
                ),
            )

        resolution = call.resolution
        if resolution is None:
            raise RuntimeError("기록된 Query Rewrite 결과에 resolution이 없습니다.")
        if resolution.should_retrieve:
            return _QueryResolutionOutcome(resolved_query=resolution.resolved_query)

        return _QueryResolutionOutcome(
            resolved_query=None,
            terminal_response=await self._withhold_ambiguous_follow_up(
                conversation_id,
                rag_run_id,
                started,
            ),
        )

    async def _record_query_rewrite(
        self,
        rag_run_id: uuid.UUID,
        model_call_id: int,
        call: QueryRewriteCall,
        total_latency_ms: int,
    ) -> bool:
        try:
            await self._finish_model_call(model_call_id, call.trace)
            if call.error is not None:
                await self._log_store.fail_rag_run(
                    rag_run_id,
                    error_code=call.error_code or INTERNAL_ERROR_CODE,
                    total_latency_ms=total_latency_ms,
                )
            else:
                resolution = call.resolution
                if resolution is None:
                    raise RuntimeError("Query Rewrite resolution이 없습니다.")
                await self._log_store.record_query_resolution(
                    rag_run_id,
                    resolved_query=resolution.resolved_query,
                    context_strategy=self._context_strategy(resolution),
                    context_turn_count=resolution.context_turn_count,
                    context_snapshot=build_context_snapshot(resolution),
                )
            await self._session.commit()
            return True
        except Exception:
            logger.exception(
                "Query Rewrite 결과를 저장하지 못했습니다: rag_run_id=%s",
                rag_run_id,
            )
            await self._fail_quietly(
                rag_run_id,
                INTERNAL_ERROR_CODE,
                total_latency_ms,
            )
            return False

    @staticmethod
    def _context_strategy(resolution: QueryResolution) -> ContextStrategy:
        if resolution.decision == QueryRewriteDecision.NEW_TOPIC:
            return ContextStrategy.NEW_TOPIC
        return ContextStrategy.FOLLOW_UP_WINDOW

    async def _withhold_ambiguous_follow_up(
        self,
        conversation_id: uuid.UUID,
        rag_run_id: uuid.UUID,
        started: float,
    ) -> ChatResponse:
        result = FinalGenerationResult(
            status=FinalAnswerStatus.WITHHELD,
            answer_markdown=WITHHELD_RESPONSES[
                FinalWithheldReason.AMBIGUOUS_QUESTION
            ],
            citations=(),
            withheld_reason=FinalWithheldReason.AMBIGUOUS_QUESTION,
        )
        try:
            await self._log_store.withhold_rag_run(
                rag_run_id,
                reason_code=FinalWithheldReason.AMBIGUOUS_QUESTION.value,
                total_latency_ms=_elapsed_ms(started),
            )
            await self._session.commit()
        except Exception:
            logger.exception(
                "모호한 후속 질문을 마감하지 못했습니다: rag_run_id=%s",
                rag_run_id,
            )
            await self._fail_quietly(
                rag_run_id,
                INTERNAL_ERROR_CODE,
                _elapsed_ms(started),
            )
            return _internal_error_response(conversation_id, rag_run_id)

        return _to_chat_response(result, conversation_id, rag_run_id)

    # ------------------------------------------------------------------
    # 외부 모델 호출 checkpoint
    # ------------------------------------------------------------------

    async def _checkpoint_model_call(
        self,
        rag_run_id: uuid.UUID,
        purpose: ModelCallPurpose,
        provider: str,
        model_name: str,
        prompt_version: Optional[str],
    ) -> int:
        call = await self._log_store.start_model_call(
            rag_run_id=rag_run_id,
            purpose=purpose.value,
            provider=provider,
            model_name=model_name,
            prompt_version=prompt_version,
        )
        model_call_id = call.id
        await self._session.commit()
        return model_call_id

    async def _finish_model_call(
        self,
        model_call_id: Optional[int],
        trace: Optional[ModelCallTrace],
    ) -> None:
        if model_call_id is None or trace is None:
            return

        await self._log_store.finish_model_call(
            model_call_id,
            status=(
                ExecutionStatus.SUCCESS if trace.succeeded else ExecutionStatus.FAILED
            ),
            input_tokens=trace.input_tokens,
            output_tokens=trace.output_tokens,
            latency_ms=trace.latency_ms,
            retry_count=trace.retry_count,
            error_message=trace.error_message,
        )

    # ------------------------------------------------------------------
    # 결과 트랜잭션 — ModelCall, 검색 후보, RagRun 마감
    # ------------------------------------------------------------------

    async def _record_turn(
        self,
        rag_run_id: uuid.UUID,
        search: HybridSearchCall,
        generation_result: Optional[FinalGenerationResult],
        generation_model_call_id: Optional[int],
        total_latency_ms: int,
        *,
        embedding_model_call_id: Optional[int],
        record_retrieval: bool,
    ) -> bool:
        try:
            await self._finish_model_call(
                embedding_model_call_id,
                search.embedding_call,
            )
            await self._finish_model_call(
                generation_model_call_id,
                None if generation_result is None else generation_result.model_call,
            )
            if record_retrieval:
                await self._record_retrieval_results(rag_run_id, search)
            await self._finish_rag_run(
                rag_run_id,
                search,
                generation_result,
                total_latency_ms,
            )
            await self._session.commit()
            return True
        except Exception:
            logger.exception(
                "턴 실행 로그를 저장하지 못했습니다: rag_run_id=%s",
                rag_run_id,
            )
            # 성공 응답은 최종 commit이 확정된 뒤에만 반환한다. rollback 뒤에는
            # 별도 transaction으로 RagRun과 미완료 ModelCall을 best-effort 마감한다.
            await self._fail_quietly(
                rag_run_id,
                INTERNAL_ERROR_CODE,
                total_latency_ms,
            )
            return False

    async def _record_retrieval_results(
        self,
        rag_run_id: uuid.UUID,
        search: HybridSearchCall,
    ) -> None:
        fused_by_chunk = {
            result.chunk.chunk_id: result for result in search.fused_results
        }
        candidates = [
            candidate
            for candidate in (
                *(
                    self._to_candidate(
                        result,
                        RetrieverType.BM25,
                        fused_by_chunk,
                        search.bm25_latency_ms,
                    )
                    for result in search.bm25_results
                ),
                *(
                    self._to_candidate(
                        result,
                        RetrieverType.VECTOR,
                        fused_by_chunk,
                        search.vector_latency_ms,
                    )
                    for result in search.vector_results
                ),
            )
            if candidate is not None
        ]
        if not candidates:
            return

        await self._log_store.record_retrieval_results(rag_run_id, candidates)

    @staticmethod
    def _to_candidate(
        result: RetrievalResult,
        retriever_type: RetrieverType,
        fused_by_chunk: dict,
        latency_ms: int,
    ) -> Optional[RetrievalCandidateLog]:
        chunk_id = result.chunk.chunk_id
        if chunk_id is None:
            return None

        fused = fused_by_chunk.get(chunk_id)
        return RetrievalCandidateLog(
            chunk_id=chunk_id,
            retriever_type=retriever_type.value,
            raw_score=result.score,
            retriever_rank=result.rank,
            fused_rank=None if fused is None else fused.final_rank,
            fused_score=None if fused is None else fused.rrf_score,
            # 융합 Top-5가 곧 Generation Context다. 최종 인용 여부와는 다르다.
            selected_as_evidence=fused is not None,
            latency_ms=latency_ms,
        )

    async def _finish_rag_run(
        self,
        rag_run_id: uuid.UUID,
        search: HybridSearchCall,
        generation_result: Optional[FinalGenerationResult],
        total_latency_ms: int,
    ) -> None:
        if generation_result is None:
            await self._log_store.fail_rag_run(
                rag_run_id,
                error_code=self._search_error_code(search),
                total_latency_ms=total_latency_ms,
            )
            return

        if generation_result.status == FinalAnswerStatus.COMPLETED:
            await self._log_store.complete_rag_run(
                rag_run_id,
                answer_content=generation_result.answer_markdown,
                citations=self._to_citation_logs(generation_result),
                total_latency_ms=total_latency_ms,
            )
            return

        if generation_result.status == FinalAnswerStatus.WITHHELD:
            await self._log_store.withhold_rag_run(
                rag_run_id,
                reason_code=generation_result.withheld_reason.value,
                total_latency_ms=total_latency_ms,
            )
            return

        await self._log_store.fail_rag_run(
            rag_run_id,
            error_code=generation_result.error_code or INTERNAL_ERROR_CODE,
            total_latency_ms=total_latency_ms,
        )

    @staticmethod
    def _search_error_code(search: HybridSearchCall) -> str:
        embedding_call = search.embedding_call
        error = search.error
        if (
            embedding_call is not None
            and not embedding_call.succeeded
            and error is not None
            and is_transient_openai_error(error)
        ):
            return UPSTREAM_ERROR_CODE
        return INTERNAL_ERROR_CODE

    @staticmethod
    def _to_citation_logs(
        generation_result: FinalGenerationResult,
    ) -> List[CitationLog]:
        return [
            CitationLog(
                chunk_id=citation.chunk_id,
                document_version_id=citation.document_version_id,
                citation_order=citation.citation_number,
                document_title_snapshot=citation.document_title,
                node_path_snapshot=" > ".join(citation.section_path),
                source_uri_snapshot=citation.source_url,
            )
            for citation in generation_result.citations
        ]

    # ------------------------------------------------------------------
    # 실패 경로 마감
    # ------------------------------------------------------------------

    async def _cancel_quietly(self, rag_run_id: uuid.UUID) -> None:
        try:
            # 취소 시점의 미완성 쓰기를 버리고 하나의 transaction으로 마감한다.
            await self._session.rollback()
            await self._log_store.fail_processing_model_calls(
                rag_run_id,
                error_message=CANCELLED_RUN_MODEL_CALL_ERROR_MESSAGE,
            )
            await self._log_store.cancel_rag_run(rag_run_id)
            await self._session.commit()
        except Exception:
            logger.exception(
                "턴을 CANCELLED로 마감하지 못했습니다: rag_run_id=%s",
                rag_run_id,
            )
            await self._rollback_quietly()

    async def _fail_quietly(
        self,
        rag_run_id: uuid.UUID,
        error_code: str,
        total_latency_ms: int,
    ) -> None:
        try:
            # 실패 지점까지의 미완성 쓰기를 버리고 마감만 남긴다.
            await self._session.rollback()
            await self._log_store.fail_processing_model_calls(rag_run_id)
            await self._log_store.fail_rag_run(
                rag_run_id,
                error_code=error_code,
                total_latency_ms=total_latency_ms,
            )
            await self._session.commit()
        except Exception:
            logger.exception(
                "턴을 ERROR로 마감하지 못했습니다: rag_run_id=%s",
                rag_run_id,
            )
            await self._rollback_quietly()

    async def _rollback_quietly(self) -> None:
        try:
            await self._session.rollback()
        except Exception:
            logger.exception("세션을 정리하지 못했습니다.")
