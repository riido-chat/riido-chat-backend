"""Chat HTTP DTO와 기존 RAG 파이프라인을 연결하고 턴 실행을 기록한다."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

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
    MAX_RELATED_SECTIONS,
)
from app.database.models import (
    ChatProfileRevision,
    ChatProfileRevisionStatus,
    ConversationChannel,
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
    GenerationStageTrace,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import (
    HybridRetrievalResult,
    HybridSearchCall,
    RetrievalResult,
)
from app.retrieval.corpus_state import CorpusNotLoadedError
from app.chat.profile import (
    ChatProfileConfigurationError,
    ChatProfileUnavailableError,
    ConversationProfileMismatchError,
    resolve_chat_profile_revision,
    validate_runtime_model_configuration,
)
from app.question_grouping.exact_question import exact_question_hash
from app.question_grouping.service import (
    ExactQuestionResult,
    GroupingTurn,
    QuestionGroupingService,
    RecordedJudgment,
)


logger = logging.getLogger(__name__)

OnGenerationStageTraceHook = Callable[
    [uuid.UUID, GenerationStageTrace],
    None,
]

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
CACHED_ANSWER_SUFFIX = " (캐시된 답변)"

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

    콘솔 업로드 문서의 원문 위치자는 내부 스킴이라 null로 내린다. 스냅샷이 비어
    있어 출처를 알 수 없는 경우도 링크로 쓸 수 없으므로 null로 내린다.
    클라이언트는 sourceUrl이 있을 때만 링크를 건다.
    """

    if citation.source_kind == CitationSourceKind.CONSOLE:
        return None
    return citation.source_url or None


def _to_chat_response(
    result: FinalGenerationResult,
    conversation_id: uuid.UUID,
    rag_run_id: uuid.UUID,
    retrieved_results: Sequence[HybridRetrievalResult] = (),
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
            related_sections=_related_sections(
                result.withheld_reason,
                retrieved_results,
            ),
        )

    if result.status == FinalAnswerStatus.ERROR:
        return chat_error_response(
            result.error_code or INTERNAL_ERROR_CODE,
            conversation_id,
            rag_run_id,
        )

    raise ValueError(f"지원하지 않는 최종 답변 상태입니다: {result.status}")


def _related_sections(
    reason: FinalWithheldReason,
    results: Sequence[HybridRetrievalResult],
) -> List[ChatCitation]:
    """관련 문서를 안내할 수 있는 보류 사유에만 고유 섹션을 순위대로 반환한다."""

    if reason not in {
        FinalWithheldReason.AMBIGUOUS_QUESTION,
        FinalWithheldReason.INSUFFICIENT_EVIDENCE,
    }:
        return []

    sections: List[ChatCitation] = []
    seen = set()
    for result in results:
        chunk = result.chunk
        source_kind = CitationSourceKind.from_canonical_uri(chunk.source_url)
        if source_kind != CitationSourceKind.GITBOOK:
            continue
        section_path = tuple(chunk.section_path)
        if section_path and section_path[0] == chunk.document_title:
            section_path = section_path[1:]
        identity = (chunk.source_url, section_path)
        if identity in seen:
            continue
        seen.add(identity)
        sections.append(
            ChatCitation(
                citation_number=len(sections) + 1,
                document_title=chunk.document_title,
                section_path=list(section_path),
                source_url=chunk.source_url,
                source_kind=source_kind,
            )
        )
        if len(sections) == MAX_RELATED_SECTIONS:
            break
    return sections


# answer_citations.node_path_snapshot 을 되돌리는 구분자. rag_run_view 와 같은 규칙이다.
SECTION_PATH_SEPARATOR = " > "


def _citation_log_to_chat_citation(citation: CitationLog) -> ChatCitation:
    """기록한 인용 스냅샷을 결과 조회(rag_run_view._to_chat_citation)와 같은 규칙으로 옮긴다."""

    document_title = citation.document_title_snapshot or ""
    source_url = citation.source_uri_snapshot or ""
    node_path = citation.node_path_snapshot
    restored = Citation(
        citation_number=citation.citation_order,
        document_title=document_title,
        section_path=(
            tuple(node_path.split(SECTION_PATH_SEPARATOR)) if node_path else ()
        ),
        source_url=source_url,
        source_kind=CitationSourceKind.from_canonical_uri(source_url),
    )
    return ChatCitation(
        citation_number=restored.citation_number,
        document_title=restored.document_title,
        section_path=_to_response_section_path(restored),
        source_url=_to_response_source_url(restored),
        source_kind=restored.source_kind,
    )


def _served_response(
    recorded: RecordedJudgment,
    conversation_id: uuid.UUID,
    rag_run_id: uuid.UUID,
) -> ChatCompletedResponse:
    """정본 서빙 턴의 COMPLETED 응답. 결과 조회가 같은 기록에서 같은 응답을 만든다."""

    if recorded.served_answer_markdown is None:
        raise ValueError("SERVED 기록에 정본 본문이 없습니다.")
    return ChatCompletedResponse(
        status=ChatResponseStatus.COMPLETED,
        conversation_id=conversation_id,
        rag_run_id=rag_run_id,
        answer=ChatAnswer(
            answer_markdown=_cached_answer_markdown(recorded.served_answer_markdown)
        ),
        citations=[
            _citation_log_to_chat_citation(citation)
            for citation in recorded.served_citations
        ],
    )


def _cached_answer_markdown(answer_markdown: str) -> str:
    """캐시 응답임을 표시하되 재시도·재포맷에도 접미사를 한 번만 붙인다."""

    if answer_markdown.endswith(CACHED_ANSWER_SUFFIX):
        return answer_markdown
    return f"{answer_markdown}{CACHED_ANSWER_SUFFIX}"


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
    profile_revision_id: Optional[int] = None
    document_group_id: Optional[int] = None
    index_version_id: Optional[int] = None
    retriever: Optional[HybridRetriever] = None
    # 앱 스위치 켜짐 AND 판별 서비스 주입 AND 프로필 판·문서 그룹·색인 판이 모두 있을 때만 True.
    grouping_enabled: bool = False
    semantic_cache_enabled: bool = False


@dataclass(frozen=True)
class _GroupingOutcome:
    """판별·게이트 기록 결과. response 가 있으면 정본 서빙으로 턴을 이미 마감했다."""

    recorded: RecordedJudgment
    response: Optional[ChatResponse] = None


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
        retriever: Optional[HybridRetriever],
        generation_service: GenerationService,
        query_rewrite_service: QueryRewriteService,
        log_store: RagLogStore,
        session: AsyncSession,
        index_version_id: Optional[int],
        profile_status: Optional[ChatProfileRevisionStatus] = None,
        retriever_factory: Optional[
            Callable[[int], tuple[HybridRetriever, int]]
        ] = None,
        question_grouping: Optional[QuestionGroupingService] = None,
        question_grouping_enabled: bool = False,
    ) -> None:
        self._retriever = retriever
        self._generation_service = generation_service
        self._query_rewrite_service = query_rewrite_service
        self._log_store = log_store
        self._session = session
        self._index_version_id = index_version_id
        self._profile_status = profile_status
        self._retriever_factory = retriever_factory
        self._question_grouping = question_grouping
        self._question_grouping_enabled = question_grouping_enabled

    async def answer_question(
        self,
        question: str,
        conversation_id: Optional[uuid.UUID] = None,
        *,
        on_turn_started: Optional[OnTurnStartedHook] = None,
        on_progress_stage: Optional[OnProgressStageHook] = None,
        on_generation_stage_trace: Optional[
            OnGenerationStageTraceHook
        ] = None,
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
        except (
            ConversationNotFoundError,
            ConversationBusyError,
            ConversationProfileMismatchError,
            ChatProfileUnavailableError,
            ChatProfileConfigurationError,
            CorpusNotLoadedError,
        ):
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
                on_generation_stage_trace=on_generation_stage_trace,
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
        on_generation_stage_trace: Optional[
            OnGenerationStageTraceHook
        ] = None,
    ) -> ChatResponse:
        conversation_id = turn.conversation_id
        rag_run_id = turn.rag_run_id
        retriever = turn.retriever or self._retriever

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

        grouping_turn: Optional[GroupingTurn] = None
        if turn.grouping_enabled:
            grouping_turn = await self._open_grouping_run(turn)

        # 같은 문서 그룹의 과거 첫 턴 질문과 정확히 같으면 그 로그의 현재 분류를
        # 검색·판별보다 먼저 정본 캐시 게이트에 넣는다. 게이트가 거절하면 같은 분류 행을
        # 유지한 채 일반 검색·생성으로 이어가고, 일치가 없으면 아무 행도 쓰지 않는다.
        grouping_recorded: Optional[RecordedJudgment] = None
        exact_grouping_recorded = False
        if grouping_turn is not None and turn.turn_no == 1:
            exact_result = await self._require_grouping().record_exact_question(
                grouping_turn,
                question,
                semantic_cache_enabled=turn.semantic_cache_enabled,
            )
            grouping_recorded = (
                None
                if not isinstance(exact_result, ExactQuestionResult)
                else exact_result.recorded
            )
            exact_grouping_recorded = grouping_recorded is not None
            if grouping_recorded is not None and grouping_recorded.served:
                try:
                    response = _served_response(
                        grouping_recorded, turn.conversation_id, turn.rag_run_id
                    )
                    await self._log_store.complete_rag_run(
                        turn.rag_run_id,
                        answer_content=_cached_answer_markdown(
                            grouping_recorded.served_answer_markdown
                        ),
                        citations=list(grouping_recorded.served_citations),
                        total_latency_ms=_elapsed_ms(started),
                    )
                    await self._session.commit()
                    return response
                except Exception:
                    logger.exception(
                        "정확 일치 질문 정본 답변 서빙 기록 실패로 생성으로 진행합니다: rag_run_id=%s",
                        turn.rag_run_id,
                    )
                    grouping_recorded = await self._require_grouping().record_serve_failure(
                        exact_result.prepared,
                        exact_result.judged,
                        grouping_recorded,
                    )
                    await self._session.commit()
            elif grouping_recorded is not None:
                # 정확 일치 게이트 결과는 검색 전에 확정해야 검색 후 rollback으로
                # 분류·캐시 시도 행이 사라지지 않는다.
                await self._session.commit()

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
                classification_run_id=(
                    None
                    if grouping_turn is None
                    else grouping_turn.classification_run_id
                ),
            )

        if retriever is None:
            raise RuntimeError("검색기가 구성되지 않았습니다.")
        search = await retriever.search_with_trace(
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

        # 판별 경로는 검색 임베딩 호출 마감과 검색 후보 기록을 판별 트랜잭션에서 이미 끝냈다.
        retrieval_recorded = False
        if grouping_turn is not None and not exact_grouping_recorded:
            grouping_outcome = await self._judge_and_gate(
                turn,
                grouping_turn,
                resolved_query,
                search,
                embedding_model_call_id,
                started,
            )
            retrieval_recorded = True
            if grouping_outcome.response is not None:
                return grouping_outcome.response
            grouping_recorded = grouping_outcome.recorded

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
            if not retrieval_recorded:
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

            if (
                on_generation_stage_trace is not None
                and generation_result.stage_trace is not None
            ):
                try:
                    on_generation_stage_trace(
                        rag_run_id,
                        generation_result.stage_trace,
                    )
                except Exception:
                    logger.exception(
                        "평가용 Generation 단계 trace 전달에 실패했습니다: "
                        "rag_run_id=%s",
                        rag_run_id,
                    )

            response = _to_chat_response(
                generation_result,
                conversation_id,
                rag_run_id,
                search.fused_results,
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
            retrieval_needs_recovery = (
                generation_checkpoint is None and not retrieval_recorded
            )
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
                grouping_recorded=grouping_recorded,
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
            grouping_recorded=grouping_recorded,
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
        profile_revision: Optional[ChatProfileRevision] = None
        if self._profile_status is not None:
            profile_revision = await resolve_chat_profile_revision(
                self._session,
                status=self._profile_status,
                channel=(
                    ConversationChannel.INTERNAL_TEST
                    if self._profile_status == ChatProfileRevisionStatus.TESTING
                    else ConversationChannel.PUBLIC
                ),
                conversation_id=conversation_id,
            )
            validate_runtime_model_configuration(
                profile_revision,
                generation_service=self._generation_service,
                query_rewrite_service=self._query_rewrite_service,
            )

        if conversation_id is None:
            if profile_revision is None:
                conversation = await self._log_store.create_conversation()
            else:
                conversation = await self._log_store.create_conversation(
                    chat_profile_revision_id=profile_revision.id,
                    channel=(
                        ConversationChannel.INTERNAL_TEST
                        if self._profile_status == ChatProfileRevisionStatus.TESTING
                        else ConversationChannel.PUBLIC
                    ),
                )
            conversation_id = conversation.id

        effective_index_version_id = self._index_version_id
        selected_retriever: Optional[HybridRetriever] = None
        if profile_revision is not None and self._retriever_factory is not None:
            selected_retriever, effective_index_version_id = self._retriever_factory(
                profile_revision.document_group_id
            )
        if effective_index_version_id is None and self._retriever_factory is None:
            raise RuntimeError("검색에 사용할 index version이 없습니다.")

        try:
            run = await self._log_store.start_rag_run(
                conversation_id,
                user_query=question,
                index_version_id=effective_index_version_id,
                # 모든 턴에 채워 두어야 이후 첫 턴 정확 일치 조회가 과거 로그를 찾는다.
                query_hash=exact_question_hash(question),
            )
        except ConversationUnavailableError as error:
            raise ConversationNotFoundError(
                f"이어갈 수 없는 대화입니다: {conversation_id}"
            ) from error
        document_group_id = (
            None if profile_revision is None else profile_revision.document_group_id
        )
        grouping_enabled = (
            self._question_grouping_enabled
            and self._question_grouping is not None
            and profile_revision is not None
            and document_group_id is not None
            and effective_index_version_id is not None
        )
        # commit 이후 객체 접근을 피하려고 식별자를 먼저 확정한다.
        turn = _TurnStart(
            conversation_id=conversation_id,
            rag_run_id=run.id,
            turn_no=run.turn_no,
            profile_revision_id=(
                None if profile_revision is None else profile_revision.id
            ),
            document_group_id=document_group_id,
            index_version_id=effective_index_version_id,
            retriever=selected_retriever,
            grouping_enabled=grouping_enabled,
            semantic_cache_enabled=bool(
                getattr(profile_revision, "semantic_cache_enabled", False)
            ),
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
        if (
            call.error is None
            and not call.trace.succeeded
            and call.fallback_reason is None
        ):
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
    # 질문 판별과 정본 캐시 게이트
    # ------------------------------------------------------------------

    def _require_grouping(self) -> QuestionGroupingService:
        if self._question_grouping is None:
            raise RuntimeError("질문 판별 서비스가 구성되지 않았습니다.")
        return self._question_grouping

    async def _open_grouping_run(self, turn: _TurnStart) -> GroupingTurn:
        """검색 임베딩 checkpoint 보다 먼저 열린 ONLINE 분류 실행을 확보한다(commit 포함)."""

        if turn.document_group_id is None or turn.index_version_id is None:
            raise RuntimeError("판별할 턴의 문서 그룹 또는 색인 판이 없습니다.")
        return await self._require_grouping().open_online_run(
            rag_run_id=turn.rag_run_id,
            document_group_id=turn.document_group_id,
            index_version_id=turn.index_version_id,
        )

    async def _judge_and_gate(
        self,
        turn: _TurnStart,
        grouping_turn: GroupingTurn,
        resolved_query: str,
        search: HybridSearchCall,
        embedding_model_call_id: Optional[int],
        started: float,
    ) -> _GroupingOutcome:
        """검색이 성공한 턴을 판별하고 게이트 결과를 기록한다.

        검색 임베딩 호출 마감과 검색 후보 기록은 여기서 한 번만 한다. 판별 준비가 끝났으면
        판별 checkpoint 트랜잭션에, 준비에 실패했으면 실패 행과 같은 트랜잭션에 넣는다.
        SERVED 면 같은 트랜잭션에서 턴을 완료하고 commit 한 뒤 응답을 돌려준다. 그 쓰기가
        실패하면 REJECTED[CANONICAL_SERVE_FAILED] 로 다시 기록하고 생성으로 진행한다.
        판별 외 쓰기·commit 실패는 그대로 올린다(fail-closed).
        """

        grouping = self._require_grouping()
        rag_run_id = turn.rag_run_id

        async def record_retrieval_logs() -> None:
            await self._finish_model_call(
                embedding_model_call_id,
                search.embedding_call,
            )
            await self._record_retrieval_results(rag_run_id, search)

        prepared = await grouping.prepare(grouping_turn, resolved_query, search)
        if not prepared.ready:
            await record_retrieval_logs()
            recorded = await grouping.record_preparation_failure(prepared)
            await self._session.commit()
            return _GroupingOutcome(recorded=recorded)

        judged = await grouping.judge(
            prepared,
            before_checkpoint=record_retrieval_logs,
        )
        recorded = await grouping.record_judgment_and_gate(
            prepared,
            judged,
            semantic_cache_enabled=turn.semantic_cache_enabled,
        )
        if not recorded.served:
            await self._session.commit()
            return _GroupingOutcome(recorded=recorded)

        try:
            response = _served_response(recorded, turn.conversation_id, rag_run_id)
            await self._log_store.complete_rag_run(
                rag_run_id,
                answer_content=_cached_answer_markdown(
                    recorded.served_answer_markdown
                ),
                citations=list(recorded.served_citations),
                total_latency_ms=_elapsed_ms(started),
            )
            await self._session.commit()
        except Exception:
            logger.exception(
                "정본 답변 서빙을 기록하지 못해 생성으로 진행합니다: rag_run_id=%s",
                rag_run_id,
            )
            # rollback 뒤 새 트랜잭션에서 다시 쓴다. 이 쓰기마저 실패하면 올린다.
            recorded = await grouping.record_serve_failure(prepared, judged, recorded)
            await self._session.commit()
            return _GroupingOutcome(recorded=recorded)
        return _GroupingOutcome(recorded=recorded, response=response)

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
        *,
        classification_run_id: Optional[int] = None,
    ) -> int:
        # 판별이 꺼진 턴은 기존 호출 인자를 그대로 둔다.
        owner_kwargs = (
            {}
            if classification_run_id is None
            else {"classification_run_id": classification_run_id}
        )
        call = await self._log_store.start_model_call(
            rag_run_id=rag_run_id,
            purpose=purpose.value,
            provider=provider,
            model_name=model_name,
            prompt_version=prompt_version,
            **owner_kwargs,
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
            cached_input_tokens=trace.cached_input_tokens,
            reasoning_tokens=trace.reasoning_tokens,
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
        grouping_recorded: Optional[RecordedJudgment] = None,
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
                grouping_recorded=grouping_recorded,
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
        *,
        grouping_recorded: Optional[RecordedJudgment] = None,
    ) -> None:
        if generation_result is None:
            await self._log_store.fail_rag_run(
                rag_run_id,
                error_code=self._search_error_code(search),
                total_latency_ms=total_latency_ms,
            )
            return

        if generation_result.status == FinalAnswerStatus.COMPLETED:
            citation_logs = self._to_citation_logs(generation_result)
            if grouping_recorded is not None:
                if self._question_grouping is None:
                    raise RuntimeError("판별 기록이 있는데 판별 서비스가 없습니다.")
                # 턴 끝 인용 귀속은 턴이 아직 PROCESSING 일 때, 완료와 같은 트랜잭션에서 한다.
                await self._question_grouping.finalize_attribution(
                    grouping_recorded,
                    citation_logs,
                )
            await self._log_store.complete_rag_run(
                rag_run_id,
                answer_content=generation_result.answer_markdown,
                citations=citation_logs,
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
