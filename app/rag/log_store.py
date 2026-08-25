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
from datetime import datetime, timezone
from typing import List, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    AnswerCitation,
    AnswerStatus,
    Conversation,
    ConversationStatus,
    ExecutionStatus,
    Feedback,
    ModelCall,
    RagRun,
    RetrievalResultRow,
)

# WITHHELD 보류 사유 4종
WITHHELD_REASON_CODES = (
    "INSUFFICIENT_EVIDENCE",
    "AMBIGUOUS_QUESTION",
    "OUT_OF_SCOPE",
    "UNVERIFIABLE_ANSWER",
)

# 확정 규칙: COMPLETED 답변의 유효 출처는 1~3개
MIN_CITATIONS = 1
MAX_CITATIONS = 3


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
        context_strategy: str,
        context_turn_count: int = 0,
        context_snapshot: Optional[dict] = None,
        sanitized_query: Optional[str] = None,
        resolved_query: Optional[str] = None,
        query_hash: Optional[str] = None,
    ) -> RagRun:
        """ACTIVE 대화에 다음 턴을 PROCESSING 상태로 생성한다.

        대화 행을 잠근 뒤 turn_no를 채번하므로 동시 요청에도 순서가 보장되고,
        (conversation_id, turn_no) unique 제약이 최종 안전망이 된다.
        """

        if not user_query or not user_query.strip():
            raise ValueError("user_query는 비어 있을 수 없습니다.")

        conversation = await self._session.get(
            Conversation, conversation_id, with_for_update=True
        )
        if conversation is None:
            raise ValueError(f"존재하지 않는 대화입니다: {conversation_id}")
        if conversation.status != ConversationStatus.ACTIVE:
            raise ValueError(
                f"후속 질문을 받을 수 없는 대화입니다: {conversation.status}"
            )

        max_turn = await self._session.scalar(
            select(func.max(RagRun.turn_no)).where(
                RagRun.conversation_id == conversation_id
            )
        )
        next_turn_no = (max_turn or 0) + 1

        run = RagRun(
            conversation_id=conversation_id,
            turn_no=next_turn_no,
            index_version_id=index_version_id,
            user_query=user_query,
            sanitized_query=sanitized_query,
            resolved_query=resolved_query,
            query_hash=query_hash,
            context_strategy=context_strategy,
            context_turn_count=context_turn_count,
            context_snapshot=context_snapshot,
            status=AnswerStatus.PROCESSING,
            created_at=_utcnow(),
        )
        conversation.last_active_at = _utcnow()
        self._session.add(run)
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
        run.citation_validated = False
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
        run = await self._session.get(RagRun, rag_run_id)
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

    async def record_model_call(
        self,
        *,
        purpose: str,
        provider: str,
        model_name: str,
        status: ExecutionStatus,
        rag_run_id: Optional[uuid.UUID] = None,
        index_run_id: Optional[int] = None,
        prompt_version: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        estimated_cost: Optional[float] = None,
        latency_ms: Optional[int] = None,
        retry_count: int = 0,
        error_message: Optional[str] = None,
    ) -> ModelCall:
        """모델 호출 한 건을 기록한다. rag_run 또는 index_run에 연결한다."""

        if rag_run_id is None and index_run_id is None:
            raise ValueError("rag_run_id 또는 index_run_id 중 하나는 필요합니다.")

        call = ModelCall(
            rag_run_id=rag_run_id,
            index_run_id=index_run_id,
            purpose=purpose,
            provider=provider,
            model_name=model_name,
            prompt_version=prompt_version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost,
            latency_ms=latency_ms,
            status=status,
            retry_count=retry_count,
            error_message=error_message,
            created_at=_utcnow(),
        )
        self._session.add(call)
        await self._session.flush()
        return call

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
