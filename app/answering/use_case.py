"""근거 검색 이후 답변 출처 선택과 생성을 고정 순서로 실행한다."""

import logging
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, Optional, Sequence, TypeVar

from app.answering.generator import GenerationStageSettings
from app.answering.models import FinalGenerationResult, GenerationStageTrace
from app.answering.service import GenerationService
from app.chat.progress import OnProgressStageHook
from app.core.model_trace import BeforeModelCallHook
from app.retrieval.models import HybridRetrievalResult


logger = logging.getLogger(__name__)

SourceDecision = TypeVar("SourceDecision")


@dataclass(frozen=True)
class AnswerQuestionResult(Generic[SourceDecision]):
    """유스케이스 단계 결과. 기록과 외부 응답 변환은 채팅 계층이 맡는다."""

    source_decision: Optional[SourceDecision]
    source_handled: bool
    generation_result: Optional[FinalGenerationResult]
    generation_error: Optional[Exception]


class AnswerQuestion(Generic[SourceDecision]):
    """출처 선택 뒤 생성하는 답변 유스케이스."""

    def __init__(self, generation_service: GenerationService) -> None:
        self._generation_service = generation_service

    async def execute(
        self,
        resolved_query: str,
        evidence: Sequence[HybridRetrievalResult],
        *,
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
        """출처를 먼저 선택하고, 재사용으로 끝나지 않을 때만 답변을 생성한다."""

        source_decision: Optional[SourceDecision] = None
        if choose_source is not None:
            if handle_source is None:
                raise ValueError("답변 출처 선택에는 결과 처리 함수가 필요합니다.")
            source_decision = await choose_source()
            if await handle_source(source_decision):
                return AnswerQuestionResult(
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
                evidence,
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
                source_decision=source_decision,
                source_handled=False,
                generation_result=generation_result,
                generation_error=error,
            )

        return AnswerQuestionResult(
            source_decision=source_decision,
            source_handled=False,
            generation_result=generation_result,
            generation_error=None,
        )
