"""Chat HTTP DTO와 기존 RAG 파이프라인을 연결하고 턴 실행을 기록한다."""

import logging
import time
import uuid
from typing import List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.chat_schema import (
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
    ConversationStatus,
    ExecutionStatus,
    ModelCallPurpose,
    RetrieverType,
)
from app.rag.generation_service import UPSTREAM_ERROR_CODE, GenerationService
from app.rag.log_store import CitationLog, RagLogStore, RetrievalCandidateLog
from app.rag.model_trace import ModelCallTrace
from generation.models import FinalAnswerStatus, FinalGenerationResult
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import HybridSearchCall, RetrievalResult


logger = logging.getLogger(__name__)

INTERNAL_ERROR_MESSAGE = "답변을 생성하는 중 오류가 발생했습니다."
CONVERSATION_NOT_FOUND_MESSAGE = (
    "이어갈 수 없는 대화입니다. 새로운 대화로 다시 질문해주세요."
)

INTERNAL_ERROR_CODE = "INTERNAL_ERROR"

# 단일턴만 지원하는 동안 모든 턴은 새 주제로 기록한다
CONTEXT_STRATEGY_NEW_TOPIC = "NEW_TOPIC"


class ConversationNotFoundError(LookupError):
    """이어갈 수 없는 conversationId로 요청이 들어왔을 때 발생한다.

    미존재, CLOSED, EXPIRED를 구분하지 않고 하나로 다룬다. FE 입장에서 셋 다
    "새 대화를 시작하라"로 귀결되고, 존재 여부를 구분해 알려줄 이유가 없다.
    """


def _internal_error_response(
    conversation_id: Optional[uuid.UUID] = None,
    rag_run_id: Optional[uuid.UUID] = None,
) -> ChatErrorResponse:
    return ChatErrorResponse(
        status=ChatResponseStatus.ERROR,
        conversation_id=conversation_id,
        rag_run_id=rag_run_id,
        answer=None,
        error=ChatError(
            code=ChatErrorCode.INTERNAL_ERROR,
            message=INTERNAL_ERROR_MESSAGE,
        ),
        citations=[],
    )


def conversation_not_found_response() -> ChatErrorResponse:
    """이어갈 수 없는 대화로 요청했을 때의 응답을 만든다."""

    return ChatErrorResponse(
        status=ChatResponseStatus.ERROR,
        conversation_id=None,
        rag_run_id=None,
        answer=None,
        error=ChatError(
            code=ChatErrorCode.NOT_FOUND,
            message=CONVERSATION_NOT_FOUND_MESSAGE,
        ),
        citations=[],
    )


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
                    section_path=list(citation.section_path),
                    source_url=citation.source_url,
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
        return _internal_error_response(conversation_id, rag_run_id)

    raise ValueError(f"지원하지 않는 최종 답변 상태입니다: {result.status}")


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


class ChatService:
    """Hybrid Retrieval부터 Chat HTTP 응답 변환까지 연결하고 실행 로그를 남긴다.

    로그 저장은 2단계 커밋이다. 1차는 turn 생성까지 커밋해 ragRunId를 확정하고
    커넥션을 풀에 돌려준다. 2차는 검색 후보, 모델 호출, 마감을 하나로 묶는다.
    2차 커밋이 실패해도 답변은 이미 완성됐으므로 그대로 반환한다.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        generation_service: GenerationService,
        log_store: RagLogStore,
        session: AsyncSession,
        index_version_id: int,
    ) -> None:
        self._retriever = retriever
        self._generation_service = generation_service
        self._log_store = log_store
        self._session = session
        self._index_version_id = index_version_id

    async def answer_question(
        self,
        question: str,
        conversation_id: Optional[uuid.UUID] = None,
    ) -> ChatResponse:
        """질문을 검색·생성 파이프라인에 전달하고 실행 로그와 함께 응답한다."""

        started = time.perf_counter()

        try:
            conversation_id, rag_run_id = await self._start_turn(
                question,
                conversation_id,
            )
        except ConversationNotFoundError:
            raise
        except Exception:
            # 1차 커밋 실패다. ragRunId가 없으면 응답을 구성할 수 없어 그대로 실패시킨다.
            logger.exception("턴을 시작하지 못했습니다.")
            await self._rollback_quietly()
            return _internal_error_response()

        try:
            return await self._run_turn(
                question,
                conversation_id,
                rag_run_id,
                started,
            )
        except Exception:
            logger.exception(
                "턴 실행 중 예상하지 못한 오류가 발생했습니다: rag_run_id=%s",
                rag_run_id,
            )
            # 마감하지 않으면 PROCESSING 행이 그대로 남는다.
            await self._fail_quietly(
                rag_run_id,
                INTERNAL_ERROR_CODE,
                _elapsed_ms(started),
            )
            return _internal_error_response(conversation_id, rag_run_id)

    async def _run_turn(
        self,
        question: str,
        conversation_id: uuid.UUID,
        rag_run_id: uuid.UUID,
        started: float,
    ) -> ChatResponse:
        search = await self._retriever.search_with_trace(question)
        # LLM 호출 동안 커넥션을 붙들지 않도록 읽기 트랜잭션을 여기서 닫는다.
        await self._session.rollback()

        if search.error is not None:
            logger.warning(
                "검색에 실패했습니다: rag_run_id=%s",
                rag_run_id,
                exc_info=search.error,
            )
            await self._record_turn(
                rag_run_id,
                search,
                None,
                _elapsed_ms(started),
            )
            return _internal_error_response(conversation_id, rag_run_id)

        generation_result = await self._generation_service.generate_answer(
            question,
            search.fused_results,
        )
        response = _to_chat_response(
            generation_result,
            conversation_id,
            rag_run_id,
        )
        await self._record_turn(
            rag_run_id,
            search,
            generation_result,
            _elapsed_ms(started),
        )
        return response

    # ------------------------------------------------------------------
    # 1차 트랜잭션 — 대화와 턴 생성
    # ------------------------------------------------------------------

    async def _start_turn(
        self,
        question: str,
        conversation_id: Optional[uuid.UUID],
    ) -> Tuple[uuid.UUID, uuid.UUID]:
        if conversation_id is None:
            conversation = await self._log_store.create_conversation()
        else:
            conversation = await self._log_store.get_conversation(conversation_id)
            if (
                conversation is None
                or conversation.status != ConversationStatus.ACTIVE
            ):
                raise ConversationNotFoundError(
                    f"이어갈 수 없는 대화입니다: {conversation_id}"
                )

        run = await self._log_store.start_rag_run(
            conversation.id,
            user_query=question,
            index_version_id=self._index_version_id,
            context_strategy=CONTEXT_STRATEGY_NEW_TOPIC,
        )
        # commit 이후 객체 접근을 피하려고 식별자를 먼저 확정한다.
        identifiers = (conversation.id, run.id)
        await self._session.commit()
        return identifiers

    # ------------------------------------------------------------------
    # 2차 트랜잭션 — 검색 후보, 모델 호출, 마감
    # ------------------------------------------------------------------

    async def _record_turn(
        self,
        rag_run_id: uuid.UUID,
        search: HybridSearchCall,
        generation_result: Optional[FinalGenerationResult],
        total_latency_ms: int,
    ) -> None:
        try:
            await self._record_retrieval_results(rag_run_id, search)
            await self._record_model_calls(rag_run_id, search, generation_result)
            await self._finish_rag_run(
                rag_run_id,
                search,
                generation_result,
                total_latency_ms,
            )
            await self._session.commit()
        except Exception:
            # 답변은 이미 완성됐고 식별자도 유효하므로 로그 실패는 삼킨다.
            logger.exception(
                "턴 실행 로그를 저장하지 못했습니다: rag_run_id=%s",
                rag_run_id,
            )
            await self._rollback_quietly()

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

    async def _record_model_calls(
        self,
        rag_run_id: uuid.UUID,
        search: HybridSearchCall,
        generation_result: Optional[FinalGenerationResult],
    ) -> None:
        await self._record_model_call(
            rag_run_id,
            ModelCallPurpose.EMBEDDING,
            search.embedding_call,
        )
        if generation_result is not None:
            await self._record_model_call(
                rag_run_id,
                ModelCallPurpose.GENERATION,
                generation_result.model_call,
            )

    async def _record_model_call(
        self,
        rag_run_id: uuid.UUID,
        purpose: ModelCallPurpose,
        trace: Optional[ModelCallTrace],
    ) -> None:
        if trace is None:
            return

        await self._log_store.record_model_call(
            purpose=purpose.value,
            provider=trace.provider,
            model_name=trace.model_name,
            status=(
                ExecutionStatus.SUCCESS if trace.succeeded else ExecutionStatus.FAILED
            ),
            rag_run_id=rag_run_id,
            prompt_version=trace.prompt_version,
            input_tokens=trace.input_tokens,
            output_tokens=trace.output_tokens,
            latency_ms=trace.latency_ms,
            retry_count=trace.retry_count,
            error_message=trace.error_message,
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
        if embedding_call is not None and not embedding_call.succeeded:
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

    async def _fail_quietly(
        self,
        rag_run_id: uuid.UUID,
        error_code: str,
        total_latency_ms: int,
    ) -> None:
        try:
            # 실패 지점까지의 미완성 쓰기를 버리고 마감만 남긴다.
            await self._session.rollback()
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
