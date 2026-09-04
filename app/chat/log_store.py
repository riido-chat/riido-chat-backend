"""RAG 실행 로그 저장 계층.

질문 한 건(턴)의 전체 실행 과정 — 대화, 질문, 검색 후보, 모델 호출,
최종 답변·인용, 상태 전이 — 을 ERD v0.2.2 스키마에 기록하고
ragRunId 하나로 재조회한다.

설계 원칙:
- 이 계층은 commit 하지 않는다. add + flush까지만 수행하며
  트랜잭션 경계는 호출자(Chat API 등)가 관리한다.
- 만료(24시간) 판정 같은 정책은 호출자 책임으로 두고,
  여기서는 상태 전이 함수(mechanism)만 제공한다.
- 피드백 저장은 피드백 API 이슈에서 별도 구현한다.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    AnswerCitation,
    AnswerStatus,
    Conversation,
    ConversationStatus,
    ContextStrategy,
    ExecutionStatus,
    Feedback,
    FeedbackRating,
    ModelCall,
    RagRun,
    RetrievalResultRow,
)
from app.chat.query_rewrite import (
    CONTEXT_SNAPSHOT_SCHEMA_VERSION,
    MAX_QUERY_LENGTH,
    MAX_QUERY_REWRITE_TURNS,
    QueryRewriteCandidateTurn,
    QueryRewriteTurnStatus,
)

# WITHHELD 보류 사유 4종
UNVERIFIABLE_ANSWER_REASON_CODE = "UNVERIFIABLE_ANSWER"
WITHHELD_REASON_CODES = (
    "INSUFFICIENT_EVIDENCE",
    "AMBIGUOUS_QUESTION",
    "OUT_OF_SCOPE",
    UNVERIFIABLE_ANSWER_REASON_CODE,
)

# 확정 규칙: COMPLETED 답변의 유효 출처는 1~3개
MIN_CITATIONS = 1
MAX_CITATIONS = 3

# 외부 호출 timeout보다 충분히 길게 두고, 이 시간이 지난 고아 실행만 복구한다.
RAG_RUN_STALE_AFTER = timedelta(minutes=10)

# 이 시간 동안 활동이 없으면 다음 요청 시점에 대화를 EXPIRED로 마감한다.
CONVERSATION_EXPIRE_AFTER = timedelta(hours=24)
STALE_RAG_RUN_ERROR_CODE = "INTERNAL_ERROR"
STALE_MODEL_CALL_ERROR_MESSAGE = (
    "10분 이상 완료되지 않아 stale recovery로 실패 처리되었습니다."
)
FAILED_RUN_MODEL_CALL_ERROR_MESSAGE = (
    "상위 RagRun 실패 복구로 모델 호출을 마감했습니다."
)
CANCELLED_RUN_MODEL_CALL_ERROR_MESSAGE = (
    "클라이언트 연결 종료로 모델 호출을 마감했습니다."
)

# 확정 규칙: 완료와 보류 답변에만 평가를 받는다
FEEDBACK_ALLOWED_STATUSES = (AnswerStatus.COMPLETED, AnswerStatus.WITHHELD)


class RagRunNotFoundError(LookupError):
    """존재하지 않는 ragRunId를 지정했을 때 발생한다."""


class FeedbackNotAllowedError(RuntimeError):
    """평가할 수 없는 상태의 턴에 피드백을 시도했을 때 발생한다."""


class ConversationUnavailableError(LookupError):
    """존재하지 않거나 더 이어갈 수 없는 대화에 턴 생성을 시도했다."""


class ConversationBusyError(RuntimeError):
    """같은 대화에서 아직 유효한 RagRun이 처리 중일 때 발생한다."""

    def __init__(self, conversation_id: uuid.UUID) -> None:
        self.conversation_id = conversation_id
        super().__init__(f"처리 중인 턴이 있는 대화입니다: {conversation_id}")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RetrievalCandidateLog:
    """검색 후보 한 건의 기록 입력."""

    chunk_id: int
    retriever_type: str
    raw_score: Optional[float] = None
    retriever_rank: Optional[int] = None
    fused_rank: Optional[int] = None
    fused_score: Optional[float] = None
    selected_as_evidence: bool = False
    latency_ms: Optional[int] = None


@dataclass(frozen=True)
class CitationLog:
    """최종 답변 인용 한 건의 기록 입력. citation_order는 1부터."""

    chunk_id: int
    document_version_id: int
    citation_order: int
    document_title_snapshot: Optional[str] = None
    node_path_snapshot: Optional[str] = None
    source_uri_snapshot: Optional[str] = None


@dataclass(frozen=True)
class RagRunDetail:
    """ragRunId 하나로 조회한 턴 실행 전체."""

    run: RagRun
    retrieval_results: List[RetrievalResultRow] = field(default_factory=list)
    model_calls: List[ModelCall] = field(default_factory=list)
    citations: List[AnswerCitation] = field(default_factory=list)
    feedback: Optional[Feedback] = None


class RagLogStore:
    """AsyncSession을 사용해 대화와 턴 실행 로그를 기록·조회한다."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # 대화
    # ------------------------------------------------------------------

    async def create_conversation(
        self, client_key: Optional[uuid.UUID] = None
    ) -> Conversation:
        """새 대화를 ACTIVE 상태로 생성한다."""

        conversation = Conversation(
            client_key=client_key,
            status=ConversationStatus.ACTIVE,
            created_at=_utcnow(),
            last_active_at=_utcnow(),
        )
        self._session.add(conversation)
        await self._session.flush()
        return conversation

    async def get_conversation(
        self, conversation_id: uuid.UUID
    ) -> Optional[Conversation]:
        return await self._session.get(Conversation, conversation_id)

    async def close_conversation(self, conversation_id: uuid.UUID) -> Conversation:
        """새 대화 시작 등 명시적 종료 — CLOSED 처리."""

        return await self._finish_conversation(
            conversation_id, ConversationStatus.CLOSED
        )

    async def expire_conversation(self, conversation_id: uuid.UUID) -> Conversation:
        """비활성 만료 — EXPIRED 처리. 만료 판정(예: 24시간)은 호출자 책임."""

        return await self._finish_conversation(
            conversation_id, ConversationStatus.EXPIRED
        )

    async def _finish_conversation(
        self, conversation_id: uuid.UUID, status: ConversationStatus
    ) -> Conversation:
        conversation = await self._session.get(Conversation, conversation_id)
        if conversation is None:
            raise ValueError(f"존재하지 않는 대화입니다: {conversation_id}")
        conversation.status = status
        conversation.closed_at = _utcnow()
        await self._session.flush()
        return conversation

    # ------------------------------------------------------------------
    # 턴 (rag_run) 생성과 상태 전이
    # ------------------------------------------------------------------

    async def start_rag_run(
        self,
        conversation_id: uuid.UUID,
        *,
        user_query: str,
        index_version_id: int,
        sanitized_query: Optional[str] = None,
        query_hash: Optional[str] = None,
    ) -> RagRun:
        """ACTIVE 대화에 다음 턴을 PROCESSING 상태로 생성한다.

        대화 행을 잠근 뒤 PROCESSING 확인과 turn_no 채번을 한 transaction에서
        수행한다. 첫 턴은 NEW_TOPIC, 후속 턴은 UNRESOLVED로 시작한다.
        """

        if not user_query or not user_query.strip():
            raise ValueError("user_query는 비어 있을 수 없습니다.")

        conversation = await self._session.get(
            Conversation, conversation_id, with_for_update=True
        )
        if conversation is None:
            raise ConversationUnavailableError(
                f"존재하지 않는 대화입니다: {conversation_id}"
            )

        now = _utcnow()
        if (
            conversation.status == ConversationStatus.ACTIVE
            and now - conversation.last_active_at > CONVERSATION_EXPIRE_AFTER
        ):
            conversation.status = ConversationStatus.EXPIRED
            conversation.closed_at = now

        if conversation.status != ConversationStatus.ACTIVE:
            raise ConversationUnavailableError(
                f"후속 질문을 받을 수 없는 대화입니다: {conversation.status}"
            )

        processing_runs = list(
            (
                await self._session.scalars(
                    select(RagRun)
                    .where(
                        RagRun.conversation_id == conversation_id,
                        RagRun.status == AnswerStatus.PROCESSING,
                    )
                    .order_by(RagRun.created_at, RagRun.turn_no)
                    .with_for_update()
                )
            ).all()
        )
        stale_before = now - RAG_RUN_STALE_AFTER
        if any(run.created_at > stale_before for run in processing_runs):
            raise ConversationBusyError(conversation_id)

        if processing_runs:
            await self._recover_stale_runs(processing_runs, recovered_at=now)

        max_turn = await self._session.scalar(
            select(func.max(RagRun.turn_no)).where(
                RagRun.conversation_id == conversation_id
            )
        )
        next_turn_no = (max_turn or 0) + 1
        is_first_turn = next_turn_no == 1

        run = RagRun(
            conversation_id=conversation_id,
            turn_no=next_turn_no,
            index_version_id=index_version_id,
            user_query=user_query,
            sanitized_query=sanitized_query,
            resolved_query=user_query if is_first_turn else None,
            query_hash=query_hash,
            context_strategy=(
                ContextStrategy.NEW_TOPIC
                if is_first_turn
                else ContextStrategy.UNRESOLVED
            ),
            context_turn_count=0,
            context_snapshot=None,
            status=AnswerStatus.PROCESSING,
            created_at=now,
        )
        conversation.last_active_at = now
        self._session.add(run)
        await self._session.flush()
        return run

    async def _recover_stale_runs(
        self,
        runs: Sequence[RagRun],
        *,
        recovered_at: datetime,
    ) -> None:
        """Conversation lock 안에서 stale RagRun과 미완료 모델 호출을 마감한다."""

        run_ids = [run.id for run in runs]
        model_calls = list(
            (
                await self._session.scalars(
                    select(ModelCall)
                    .where(
                        ModelCall.rag_run_id.in_(run_ids),
                        ModelCall.status == ExecutionStatus.PROCESSING,
                    )
                    .with_for_update()
                )
            ).all()
        )

        for run in runs:
            run.status = AnswerStatus.ERROR
            run.error_code = STALE_RAG_RUN_ERROR_CODE
            run.completed_at = recovered_at

        for call in model_calls:
            call.status = ExecutionStatus.FAILED
            call.error_message = STALE_MODEL_CALL_ERROR_MESSAGE

        await self._session.flush()

    async def get_query_rewrite_candidates(
        self,
        rag_run_id: uuid.UUID,
    ) -> List[QueryRewriteCandidateTurn]:
        """현재 턴 이전의 유효한 최근 5턴을 시간 오름차순으로 반환한다."""

        current_run = await self._get_processing_run(rag_run_id)
        recent_runs = list(
            (
                await self._session.scalars(
                    select(RagRun)
                    .where(
                        RagRun.conversation_id == current_run.conversation_id,
                        RagRun.turn_no < current_run.turn_no,
                        RagRun.status.in_(
                            (AnswerStatus.COMPLETED, AnswerStatus.WITHHELD)
                        ),
                    )
                    .order_by(RagRun.turn_no.desc())
                    .limit(MAX_QUERY_REWRITE_TURNS)
                )
            ).all()
        )

        return [
            QueryRewriteCandidateTurn(
                rag_run_id=run.id,
                turn_no=run.turn_no,
                status=QueryRewriteTurnStatus(run.status.value),
                user_query=run.user_query,
                answer_content=(
                    run.answer_content
                    if run.status == AnswerStatus.COMPLETED
                    else None
                ),
                withheld_reason_code=(
                    run.withheld_reason_code
                    if run.status == AnswerStatus.WITHHELD
                    else None
                ),
            )
            for run in reversed(recent_runs)
        ]

    async def record_query_resolution(
        self,
        rag_run_id: uuid.UUID,
        *,
        resolved_query: Optional[str],
        context_strategy: ContextStrategy,
        context_turn_count: int,
        context_snapshot: Optional[dict[str, Any]],
    ) -> RagRun:
        """검증된 Query Rewrite 결과를 PROCESSING 턴에 checkpoint 기록한다."""

        if context_strategy not in (
            ContextStrategy.NEW_TOPIC,
            ContextStrategy.FOLLOW_UP_WINDOW,
        ):
            raise ValueError(
                f"MVP Query Rewrite에서 사용할 수 없는 문맥 전략입니다: "
                f"{context_strategy}"
            )
        if not 0 <= context_turn_count <= MAX_QUERY_REWRITE_TURNS:
            raise ValueError(
                f"context_turn_count는 0~{MAX_QUERY_REWRITE_TURNS}여야 합니다."
            )
        if resolved_query is not None:
            if not resolved_query.strip():
                raise ValueError("resolved_query는 공백일 수 없습니다.")
            if len(resolved_query) > MAX_QUERY_LENGTH:
                raise ValueError(
                    f"resolved_query는 최대 {MAX_QUERY_LENGTH}자까지 허용합니다."
                )

        selected_turns = None
        if context_snapshot is not None:
            if not isinstance(context_snapshot, dict):
                raise ValueError("context_snapshot은 객체여야 합니다.")
            if (
                context_snapshot.get("schemaVersion")
                != CONTEXT_SNAPSHOT_SCHEMA_VERSION
            ):
                raise ValueError("지원하지 않는 context_snapshot schemaVersion입니다.")
            selected_turns = context_snapshot.get("selectedTurns")
            if not isinstance(selected_turns, list):
                raise ValueError("context_snapshot.selectedTurns는 배열이어야 합니다.")
            if len(selected_turns) != context_turn_count:
                raise ValueError(
                    "context_turn_count는 snapshot의 selectedTurns 길이와 같아야 합니다."
                )

        if context_strategy == ContextStrategy.NEW_TOPIC:
            if (
                resolved_query is None
                or context_turn_count != 0
                or context_snapshot is not None
            ):
                raise ValueError(
                    "NEW_TOPIC은 resolved_query와 빈 문맥 상태가 필요합니다."
                )
        elif resolved_query is None:
            if context_turn_count != 0 or context_snapshot is not None:
                raise ValueError(
                    "해석하지 못한 FOLLOW_UP은 문맥 snapshot을 저장할 수 없습니다."
                )
        elif (
            context_turn_count == 0
            or context_snapshot is None
            or not selected_turns
        ):
            raise ValueError(
                "해석한 FOLLOW_UP은 하나 이상의 선택 문맥이 필요합니다."
            )

        run = await self._get_processing_run(rag_run_id)
        run.resolved_query = resolved_query
        run.context_strategy = context_strategy
        run.context_turn_count = context_turn_count
        run.context_snapshot = context_snapshot
        await self._session.flush()
        return run

    async def complete_rag_run(
        self,
        rag_run_id: uuid.UUID,
        *,
        answer_content: str,
        citations: Sequence[CitationLog],
        answer_schema_version: Optional[str] = None,
        total_latency_ms: Optional[int] = None,
    ) -> RagRun:
        """정상 답변 완료 — COMPLETED. 유효 인용 1~3개를 함께 기록한다."""

        if not (MIN_CITATIONS <= len(citations) <= MAX_CITATIONS):
            raise ValueError(
                f"COMPLETED 답변의 인용은 {MIN_CITATIONS}~{MAX_CITATIONS}개여야 "
                f"합니다: {len(citations)}개"
            )

        run = await self._get_processing_run(rag_run_id)
        run.status = AnswerStatus.COMPLETED
        run.answer_content = answer_content
        run.answer_schema_version = answer_schema_version
        run.citation_validated = True
        run.total_latency_ms = total_latency_ms
        run.completed_at = _utcnow()

        self._session.add_all(
            AnswerCitation(
                rag_run_id=rag_run_id,
                chunk_id=citation.chunk_id,
                document_version_id=citation.document_version_id,
                citation_order=citation.citation_order,
                document_title_snapshot=citation.document_title_snapshot,
                node_path_snapshot=citation.node_path_snapshot,
                source_uri_snapshot=citation.source_uri_snapshot,
                created_at=_utcnow(),
            )
            for citation in citations
        )
        await self._touch_conversation(run.conversation_id)
        await self._session.flush()
        return run

    async def withhold_rag_run(
        self,
        rag_run_id: uuid.UUID,
        *,
        reason_code: str,
        total_latency_ms: Optional[int] = None,
    ) -> RagRun:
        """답변 보류 — WITHHELD + 보류 사유 기록."""

        if reason_code not in WITHHELD_REASON_CODES:
            raise ValueError(f"알 수 없는 보류 사유입니다: {reason_code}")

        run = await self._get_processing_run(rag_run_id)
        run.status = AnswerStatus.WITHHELD
        run.withheld_reason_code = reason_code
        # 인용 검증까지 갔다가 유효 인용이 0개인 경우만 false다.
        # 근거 부족 등으로 생성 단계에서 보류하면 검증에 도달하지 않아 null로 둔다.
        run.citation_validated = (
            False if reason_code == UNVERIFIABLE_ANSWER_REASON_CODE else None
        )
        run.total_latency_ms = total_latency_ms
        run.completed_at = _utcnow()
        await self._touch_conversation(run.conversation_id)
        await self._session.flush()
        return run

    async def fail_rag_run(
        self,
        rag_run_id: uuid.UUID,
        *,
        error_code: Optional[str] = None,
        total_latency_ms: Optional[int] = None,
    ) -> RagRun:
        """기술적 오류 — ERROR + 내부 실패 분류 기록."""

        run = await self._get_processing_run(rag_run_id)
        run.status = AnswerStatus.ERROR
        run.error_code = error_code
        run.total_latency_ms = total_latency_ms
        run.completed_at = _utcnow()
        await self._session.flush()
        return run

    async def cancel_rag_run(self, rag_run_id: uuid.UUID) -> RagRun:
        """사용자 연결 중단 — CANCELLED."""

        run = await self._get_processing_run(rag_run_id)
        run.status = AnswerStatus.CANCELLED
        run.completed_at = _utcnow()
        await self._session.flush()
        return run

    async def _get_processing_run(self, rag_run_id: uuid.UUID) -> RagRun:
        known_run = await self._session.get(RagRun, rag_run_id)
        if known_run is None:
            raise ValueError(f"존재하지 않는 턴입니다: {rag_run_id}")

        # stale recovery와 같은 잠금 순서를 지켜 늦게 돌아온 worker의 부활과
        # Conversation ↔ ModelCall 역순 잠금 교착을 막는다.
        conversation = await self._session.scalar(
            select(Conversation)
            .where(Conversation.id == known_run.conversation_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if conversation is None:
            raise ValueError(f"턴의 대화가 존재하지 않습니다: {rag_run_id}")

        run = await self._session.scalar(
            select(RagRun)
            .where(RagRun.id == rag_run_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if run is None:
            raise ValueError(f"존재하지 않는 턴입니다: {rag_run_id}")
        if run.status != AnswerStatus.PROCESSING:
            raise ValueError(
                f"PROCESSING 상태의 턴만 전이할 수 있습니다: {run.status}"
            )
        return run

    async def _touch_conversation(self, conversation_id: uuid.UUID) -> None:
        conversation = await self._session.get(Conversation, conversation_id)
        if conversation is not None:
            conversation.last_active_at = _utcnow()

    # ------------------------------------------------------------------
    # 검색 후보·모델 호출 기록
    # ------------------------------------------------------------------

    async def record_retrieval_results(
        self,
        rag_run_id: uuid.UUID,
        candidates: Sequence[RetrievalCandidateLog],
    ) -> List[RetrievalResultRow]:
        """턴의 검색 후보를 방식·점수·순위·근거 선택 여부와 함께 기록한다."""

        rows = [
            RetrievalResultRow(
                rag_run_id=rag_run_id,
                chunk_id=candidate.chunk_id,
                retriever_type=candidate.retriever_type,
                raw_score=candidate.raw_score,
                retriever_rank=candidate.retriever_rank,
                fused_rank=candidate.fused_rank,
                fused_score=candidate.fused_score,
                selected_as_evidence=candidate.selected_as_evidence,
                latency_ms=candidate.latency_ms,
                created_at=_utcnow(),
            )
            for candidate in candidates
        ]
        self._session.add_all(rows)
        await self._session.flush()
        return rows

    async def start_model_call(
        self,
        *,
        purpose: str,
        provider: str,
        model_name: str,
        rag_run_id: Optional[uuid.UUID] = None,
        index_run_id: Optional[int] = None,
        prompt_version: Optional[str] = None,
    ) -> ModelCall:
        """외부 호출 전에 논리적 모델 호출 한 건을 PROCESSING으로 생성한다."""

        if rag_run_id is None and index_run_id is None:
            raise ValueError("rag_run_id 또는 index_run_id 중 하나는 필요합니다.")

        if rag_run_id is not None:
            await self._get_processing_run(rag_run_id)

        call = ModelCall(
            rag_run_id=rag_run_id,
            index_run_id=index_run_id,
            purpose=purpose,
            provider=provider,
            model_name=model_name,
            prompt_version=prompt_version,
            status=ExecutionStatus.PROCESSING,
            retry_count=0,
            created_at=_utcnow(),
        )
        self._session.add(call)
        await self._session.flush()
        return call

    async def finish_model_call(
        self,
        model_call_id: int,
        *,
        status: ExecutionStatus,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        estimated_cost: Optional[float] = None,
        latency_ms: Optional[int] = None,
        retry_count: int = 0,
        error_message: Optional[str] = None,
    ) -> ModelCall:
        """PROCESSING 호출을 같은 행에서 SUCCESS 또는 FAILED로 마감한다."""

        if status not in (ExecutionStatus.SUCCESS, ExecutionStatus.FAILED):
            raise ValueError(f"모델 호출의 최종 상태가 아닙니다: {status}")
        if retry_count < 0:
            raise ValueError("retry_count는 0 이상이어야 합니다.")

        known_call = await self._session.get(ModelCall, model_call_id)
        if known_call is None:
            raise ValueError(f"존재하지 않는 모델 호출입니다: {model_call_id}")

        if known_call.rag_run_id is not None:
            await self._get_processing_run(known_call.rag_run_id)

        call = await self._session.scalar(
            select(ModelCall)
            .where(ModelCall.id == model_call_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if call is None:
            raise ValueError(f"존재하지 않는 모델 호출입니다: {model_call_id}")
        if call.status != ExecutionStatus.PROCESSING:
            raise ValueError(
                f"PROCESSING 상태의 모델 호출만 마감할 수 있습니다: {call.status}"
            )

        call.status = status
        call.input_tokens = input_tokens
        call.output_tokens = output_tokens
        call.estimated_cost = estimated_cost
        call.latency_ms = latency_ms
        call.retry_count = retry_count
        call.error_message = error_message
        await self._session.flush()
        return call

    async def fail_processing_model_calls(
        self,
        rag_run_id: uuid.UUID,
        *,
        error_message: str = FAILED_RUN_MODEL_CALL_ERROR_MESSAGE,
    ) -> List[ModelCall]:
        """RagRun 실패 마감과 같은 transaction에서 미완료 모델 호출을 닫는다."""

        await self._get_processing_run(rag_run_id)
        calls = list(
            (
                await self._session.scalars(
                    select(ModelCall)
                    .where(
                        ModelCall.rag_run_id == rag_run_id,
                        ModelCall.status == ExecutionStatus.PROCESSING,
                    )
                    .with_for_update()
                )
            ).all()
        )
        for call in calls:
            call.status = ExecutionStatus.FAILED
            call.error_message = call.error_message or error_message
        await self._session.flush()
        return calls

    # ------------------------------------------------------------------
    # 피드백
    # ------------------------------------------------------------------

    async def set_feedback(
        self,
        rag_run_id: uuid.UUID,
        *,
        rating: FeedbackRating,
    ) -> Feedback:
        """답변 평가를 등록하거나 반대 값으로 변경한다.

        같은 값 재전송은 무시하므로 updated_at을 갱신하지 않는다. 그래야
        updated_at이 실제로 평가를 뒤집은 시각으로 남는다.
        """

        await self._get_feedback_target(rag_run_id)

        feedback = await self._get_feedback(rag_run_id, lock=True)
        now = _utcnow()
        if feedback is None:
            feedback = Feedback(
                rag_run_id=rag_run_id,
                rating=rating,
                created_at=now,
                updated_at=now,
            )
            self._session.add(feedback)
        elif feedback.rating != rating:
            feedback.rating = rating
            feedback.updated_at = now

        await self._session.flush()
        return feedback

    async def clear_feedback(self, rag_run_id: uuid.UUID) -> bool:
        """등록된 평가를 지운다. 이미 없으면 아무것도 하지 않고 False를 반환한다."""

        await self._get_feedback_target(rag_run_id)

        feedback = await self._get_feedback(rag_run_id, lock=True)
        if feedback is None:
            return False

        await self._session.delete(feedback)
        await self._session.flush()
        return True

    async def _get_feedback_target(self, rag_run_id: uuid.UUID) -> RagRun:
        run = await self._session.get(RagRun, rag_run_id)
        if run is None:
            raise RagRunNotFoundError(f"존재하지 않는 턴입니다: {rag_run_id}")
        if run.status not in FEEDBACK_ALLOWED_STATUSES:
            raise FeedbackNotAllowedError(
                f"평가할 수 없는 상태의 턴입니다: {run.status}"
            )
        return run

    async def _get_feedback(
        self,
        rag_run_id: uuid.UUID,
        *,
        lock: bool = False,
    ) -> Optional[Feedback]:
        statement = select(Feedback).where(Feedback.rag_run_id == rag_run_id)
        if lock:
            statement = statement.with_for_update()
        return await self._session.scalar(statement)

    # ------------------------------------------------------------------
    # 조회
    # ------------------------------------------------------------------

    async def get_rag_run_detail(
        self, rag_run_id: uuid.UUID
    ) -> Optional[RagRunDetail]:
        """ragRunId 하나로 질문→검색 후보→모델 호출→답변·인용→피드백을 조회한다."""

        run = await self._session.get(RagRun, rag_run_id)
        if run is None:
            return None

        retrieval_results = (
            (
                await self._session.execute(
                    select(RetrievalResultRow)
                    .where(RetrievalResultRow.rag_run_id == rag_run_id)
                    .order_by(
                        RetrievalResultRow.fused_rank.asc().nulls_last(),
                        RetrievalResultRow.id,
                    )
                )
            )
            .scalars()
            .all()
        )
        model_calls = (
            (
                await self._session.execute(
                    select(ModelCall)
                    .where(ModelCall.rag_run_id == rag_run_id)
                    .order_by(ModelCall.id)
                )
            )
            .scalars()
            .all()
        )
        citations = (
            (
                await self._session.execute(
                    select(AnswerCitation)
                    .where(AnswerCitation.rag_run_id == rag_run_id)
                    .order_by(AnswerCitation.citation_order)
                )
            )
            .scalars()
            .all()
        )
        feedback = await self._session.scalar(
            select(Feedback).where(Feedback.rag_run_id == rag_run_id)
        )

        return RagRunDetail(
            run=run,
            retrieval_results=list(retrieval_results),
            model_calls=list(model_calls),
            citations=list(citations),
            feedback=feedback,
        )

    async def list_conversation_runs(
        self, conversation_id: uuid.UUID
    ) -> List[RagRun]:
        """대화의 턴 목록을 순서대로 조회한다 (멀티턴 문맥 구성용)."""

        result = await self._session.execute(
            select(RagRun)
            .where(RagRun.conversation_id == conversation_id)
            .order_by(RagRun.turn_no)
        )
        return list(result.scalars().all())
