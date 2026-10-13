"""문맥 해석부터 근거 기반 답변 생성까지 정해진 순서로 실행한다."""

import logging
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, Optional, TypeVar

from app.answering.generator import GenerationStageSettings
from app.answering.models import FinalGenerationResult, GenerationStageTrace
from app.answering.service import GenerationService
from app.chat.progress import OnProgressStageHook
from app.core.model_trace import BeforeModelCallHook
from app.retrieval.models import HybridSearchCall


logger = logging.getLogger(__name__)

SourceDecision = TypeVar("SourceDecision")


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
    ) -> AnswerQuestionResult[SourceDecision]:
        """문맥 해석과 검색 뒤 출처를 선택하고, 재사용으로 끝나지 않을 때만 생성한다."""

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
            )

        source_decision: Optional[SourceDecision] = None
        if choose_source is not None:
            if handle_source is None:
                raise ValueError("답변 출처 선택에는 결과 처리 함수가 필요합니다.")
            source_decision = await choose_source()
            if await handle_source(source_decision):
                return AnswerQuestionResult(
                    resolved_query=resolved_query,
                    retrieval=retrieval_result,
                    flow_terminated=False,
                    source_decision=source_decision,
                    source_handled=True,
                    generation_result=None,
                    generation_error=None,
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
