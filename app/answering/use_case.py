"""문맥 해석부터 근거 기반 답변 생성까지 정해진 순서로 실행한다."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Awaitable, Callable, Generic, Optional, TypeVar

from app.answering.models import FinalGenerationResult, GenerationStageTrace
from app.chat.progress import OnProgressStageHook
from app.core.model_trace import BeforeModelCallHook
from app.retrieval.models import HybridSearchCall

if TYPE_CHECKING:
    from app.answering.generator import GenerationStageSettings
    from app.answering.service import GenerationService


logger = logging.getLogger(__name__)

SourceDecision = TypeVar("SourceDecision")


class AnswerExecutionTarget(str, Enum):
    """마지막으로 실행할 답변 경로 단계.

    CACHE_DECISION 은 운영 답변 흐름의 앞부분(질의 확정, 검색, 캐시 판정)만
    실행한다. 캐시 결과를 앱 응답으로 처리하거나 운영 로그에 기록하지 않고,
    답변 생성 호출도 피한다.
    """

    FULL_ANSWER = "FULL_ANSWER"
    CACHE_DECISION = "CACHE_DECISION"


@dataclass(frozen=True)
class AnswerQuestionResult(Generic[SourceDecision]):
    """유스케이스 단계 결과. 기록과 외부 응답 변환은 채팅 계층이 맡는다."""

    resolved_query: Optional[str]
    retrieval: Optional[HybridSearchCall]
    flow_terminated: bool
    source_decision: Optional[SourceDecision]
    source_handled: bool
    generation_result: Optional[FinalGenerationResult]
    generation_error: Optional[Exception]
    generation_skipped: bool = False


class AnswerQuestion(Generic[SourceDecision]):
    """문맥 해석, 근거 검색, 출처 선택, 답변 생성을 정해진 순서로 실행한다."""

    def __init__(self, generation_service: GenerationService) -> None:
        self._generation_service = generation_service

    async def execute(
        self,
        *,
        resolve_query: Callable[[], Awaitable[Optional[str]]],
        retrieve: Callable[[str], Awaitable[HybridSearchCall]],
        choose_source: Optional[
            Callable[[], Awaitable[SourceDecision]]
        ] = None,
        handle_source: Optional[Callable[[SourceDecision], Awaitable[bool]]] = None,
        before_model_call: BeforeModelCallHook,
        stages: Optional[GenerationStageSettings] = None,
        on_progress_stage: Optional[OnProgressStageHook] = None,
        on_generation_stage_trace: Optional[
            Callable[[GenerationStageTrace], None]
        ] = None,
        rag_run_id: Optional[uuid.UUID] = None,
        target: AnswerExecutionTarget = AnswerExecutionTarget.FULL_ANSWER,
    ) -> AnswerQuestionResult[SourceDecision]:
        """운영 흐름을 실행하되 요청한 단계에서 종료한다.

        CACHE_DECISION 은 운영과 같은 질의·검색·캐시 출처 선택을 수행한 뒤
        판정 결과를 돌려준다. 출처 처리 콜백을 실행하지 않으며, 캐시 미적중이어도
        답변 생성에는 진입하지 않는다.
        """

        if target is AnswerExecutionTarget.CACHE_DECISION and choose_source is None:
            raise ValueError("CACHE_DECISION 대상에는 캐시 판정 함수가 필요합니다.")

        resolved_query = await resolve_query()
        if resolved_query is None:
            return AnswerQuestionResult(
                resolved_query=None,
                retrieval=None,
                flow_terminated=True,
                source_decision=None,
                source_handled=False,
                generation_result=None,
                generation_error=None,
                generation_skipped=True,
            )

        retrieval_result = await retrieve(resolved_query)
        if retrieval_result.error is not None:
            return AnswerQuestionResult(
                resolved_query=resolved_query,
                retrieval=retrieval_result,
                flow_terminated=True,
                source_decision=None,
                source_handled=False,
                generation_result=None,
                generation_error=None,
                generation_skipped=True,
            )

        source_decision: Optional[SourceDecision] = None
        if choose_source is not None:
            source_decision = await choose_source()
            if target is AnswerExecutionTarget.CACHE_DECISION:
                return AnswerQuestionResult(
                    resolved_query=resolved_query,
                    retrieval=retrieval_result,
                    flow_terminated=False,
                    source_decision=source_decision,
                    source_handled=False,
                    generation_result=None,
                    generation_error=None,
                    generation_skipped=True,
                )

            if handle_source is None:
                raise ValueError("답변 출처 선택에는 결과 처리 함수가 필요합니다.")
            if await handle_source(source_decision):
                return AnswerQuestionResult(
                    resolved_query=resolved_query,
                    retrieval=retrieval_result,
                    flow_terminated=False,
                    source_decision=source_decision,
                    source_handled=True,
                    generation_result=None,
                    generation_error=None,
                    generation_skipped=True,
                )

        generation_result: Optional[FinalGenerationResult] = None
        generation_kwargs = {}
        if on_progress_stage is not None:
            generation_kwargs["on_progress_stage"] = on_progress_stage
        if stages is not None:
            generation_kwargs["stages"] = stages

        try:
            generation_result = await self._generation_service.generate_answer(
                resolved_query,
                retrieval_result.fused_results,
                before_model_call=before_model_call,
                **generation_kwargs,
            )
            if (
                on_generation_stage_trace is not None
                and generation_result.stage_trace is not None
            ):
                try:
                    on_generation_stage_trace(generation_result.stage_trace)
                except Exception:
                    logger.exception(
                        "평가용 Generation 단계 trace 전달에 실패했습니다: "
                        "rag_run_id=%s",
                        rag_run_id,
                    )
        except Exception as error:
            return AnswerQuestionResult(
                resolved_query=resolved_query,
                retrieval=retrieval_result,
                flow_terminated=False,
                source_decision=source_decision,
                source_handled=False,
                generation_result=generation_result,
                generation_error=error,
            )

        return AnswerQuestionResult(
            resolved_query=resolved_query,
            retrieval=retrieval_result,
            flow_terminated=False,
            source_decision=source_decision,
            source_handled=False,
            generation_result=generation_result,
            generation_error=None,
        )
