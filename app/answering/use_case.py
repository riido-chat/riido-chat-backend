"""문맥 해석부터 근거 기반 답변 생성까지 정해진 순서로 실행한다."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Awaitable, Callable, Generic, Optional, TypeVar

from app.answering.models import FinalGenerationResult, GenerationStageTrace
from app.chat.progress import OnProgressStageHook
from app.core.model_trace import BeforeModelCallHook
from app.retrieval.models import HybridSearchCall
from app.answering.ports import (
    AnswerGeneratorPort,
    PreRetrievalCacheHandlerPort,
    PreRetrievalCacheInspectorPort,
    QueryResolverPort,
    RetrievalPort,
    SourceHandlerPort,
    SourceSelectionEnabledPort,
    SourceSelectorPort,
)
from app.answering.trace import (
    AnswerExecutionTrace,
    ExecutionStage,
    ExecutionTraceEvent,
    ExecutionTraceRecorder,
    ReplayContext,
    StageOutcome,
)

if TYPE_CHECKING:
    from app.answering.generator import GenerationStageSettings


logger = logging.getLogger(__name__)

SourceDecision = TypeVar("SourceDecision")
StageResult = TypeVar("StageResult")


class AnswerExecutionTarget(str, Enum):
    """마지막으로 실행할 답변 경로 단계.

    CACHE_DECISION 은 검색 전 정확 캐시부터 질의 확정, 검색, 의미 캐시 판정까지
    운영 순서대로 실행한 뒤 멈춘다. 캐시 결과를 앱 응답으로 처리하거나 운영 로그에
    기록하지 않고, 답변 생성 호출도 피한다.
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
    execution_trace: Optional[AnswerExecutionTrace] = None
    pre_retrieval_decision: Optional[object] = None
    pre_retrieval_handled: bool = False


class AnswerQuestion(Generic[SourceDecision]):
    """문맥 해석, 근거 검색, 출처 선택, 답변 생성을 정해진 순서로 실행한다."""

    def __init__(self, generation_service: AnswerGeneratorPort) -> None:
        self._generation_service = generation_service

    async def execute(
        self,
        *,
        resolve_query: QueryResolverPort,
        retrieve: RetrievalPort,
        inspect_pre_retrieval_cache: Optional[
            PreRetrievalCacheInspectorPort[object]
        ] = None,
        handle_pre_retrieval_cache: Optional[
            PreRetrievalCacheHandlerPort[object]
        ] = None,
        choose_source: Optional[SourceSelectorPort[SourceDecision]] = None,
        should_choose_source: Optional[SourceSelectionEnabledPort] = None,
        handle_source: Optional[SourceHandlerPort[SourceDecision]] = None,
        before_model_call: BeforeModelCallHook,
        stages: Optional[GenerationStageSettings] = None,
        on_progress_stage: Optional[OnProgressStageHook] = None,
        on_generation_stage_trace: Optional[
            Callable[[GenerationStageTrace], None]
        ] = None,
        rag_run_id: Optional[uuid.UUID] = None,
        target: AnswerExecutionTarget = AnswerExecutionTarget.FULL_ANSWER,
        replay_context: Optional[ReplayContext] = None,
        trace_recorder: Optional[ExecutionTraceRecorder] = None,
    ) -> AnswerQuestionResult[SourceDecision]:
        """운영 흐름을 실행하되 요청한 단계에서 종료한다.

        CACHE_DECISION 은 검색 전 정확 캐시, 질의·검색·검색 후 캐시 출처 선택을
        운영과 같은 순서로 수행한 뒤 판정 결과를 돌려준다. 결과 처리 콜백을
        평가에서 제공한 어댑터가 소유하며, 답변 생성에는 진입하지 않는다.
        """

        if (inspect_pre_retrieval_cache is None) != (
            handle_pre_retrieval_cache is None
        ):
            raise ValueError("검색 전 캐시 조회와 결과 처리 함수는 함께 지정해야 합니다.")
        if target is AnswerExecutionTarget.CACHE_DECISION and choose_source is None:
            raise ValueError("CACHE_DECISION 대상에는 캐시 판정 함수가 필요합니다.")

        trace_events: list[ExecutionTraceEvent] = []
        trace_id = uuid.uuid4()
        trace_recorded = False

        def build_trace() -> Optional[AnswerExecutionTrace]:
            if replay_context is None:
                return None
            return AnswerExecutionTrace(
                trace_id=trace_id,
                target=target.value,
                replay_context=replay_context,
                events=tuple(trace_events),
            )

        def record_trace() -> None:
            nonlocal trace_recorded
            trace = build_trace()
            if trace is None or trace_recorder is None or trace_recorded:
                return
            trace_recorded = True
            try:
                trace_recorder.record(trace)
            except Exception:
                logger.exception("실행 trace 저장에 실패했습니다: trace_id=%s", trace_id)

        async def run_stage(
            stage: ExecutionStage,
            operation: Callable[..., Awaitable[StageResult]],
            *args: object,
            **operation_kwargs: object,
        ) -> StageResult:
            started = time.perf_counter()
            try:
                value = await operation(*args, **operation_kwargs)
            except Exception as error:
                trace_events.append(
                    ExecutionTraceEvent(
                        stage=stage,
                        outcome=StageOutcome.FAILED,
                        duration_ms=max(
                            0, int((time.perf_counter() - started) * 1000)
                        ),
                        detail_code=type(error).__name__,
                    )
                )
                record_trace()
                raise
            trace_events.append(
                ExecutionTraceEvent(
                    stage=stage,
                    outcome=StageOutcome.SUCCEEDED,
                    duration_ms=max(
                        0, int((time.perf_counter() - started) * 1000)
                    ),
                )
            )
            return value

        def make_result(**kwargs: object) -> AnswerQuestionResult[SourceDecision]:
            trace = build_trace()
            record_trace()
            return AnswerQuestionResult(execution_trace=trace, **kwargs)  # type: ignore[arg-type]

        pre_retrieval_decision: Optional[object] = None
        pre_retrieval_handled = False
        if inspect_pre_retrieval_cache is not None:
            assert handle_pre_retrieval_cache is not None
            pre_retrieval_decision = await run_stage(
                ExecutionStage.EXACT_CACHE_LOOKUP, inspect_pre_retrieval_cache
            )
            pre_retrieval_handled = await run_stage(
                ExecutionStage.EXACT_CACHE_HANDLING,
                handle_pre_retrieval_cache,
                pre_retrieval_decision,
            )
            if pre_retrieval_handled:
                for skipped_stage in (
                    ExecutionStage.QUERY_RESOLUTION,
                    ExecutionStage.RETRIEVAL,
                    ExecutionStage.SOURCE_SELECTION,
                    ExecutionStage.ANSWER_GENERATION,
                ):
                    trace_events.append(
                        ExecutionTraceEvent(
                            skipped_stage,
                            StageOutcome.SKIPPED,
                            detail_code="EXACT_CACHE_HANDLED",
                        )
                    )
                return make_result(
                    resolved_query=None,
                    retrieval=None,
                    flow_terminated=True,
                    source_decision=None,
                    source_handled=False,
                    generation_result=None,
                    generation_error=None,
                    generation_skipped=True,
                    pre_retrieval_decision=pre_retrieval_decision,
                    pre_retrieval_handled=True,
                )
        else:
            trace_events.append(
                ExecutionTraceEvent(
                    ExecutionStage.EXACT_CACHE_LOOKUP,
                    StageOutcome.SKIPPED,
                    detail_code="NOT_CONFIGURED",
                )
            )

        resolved_query = await run_stage(ExecutionStage.QUERY_RESOLUTION, resolve_query)
        if resolved_query is None:
            trace_events.append(
                ExecutionTraceEvent(
                    ExecutionStage.RETRIEVAL, StageOutcome.SKIPPED
                )
            )
            trace_events.append(
                ExecutionTraceEvent(
                    ExecutionStage.SOURCE_SELECTION, StageOutcome.SKIPPED
                )
            )
            trace_events.append(
                ExecutionTraceEvent(
                    ExecutionStage.ANSWER_GENERATION, StageOutcome.SKIPPED
                )
            )
            return make_result(
                resolved_query=None,
                retrieval=None,
                flow_terminated=True,
                source_decision=None,
                source_handled=False,
                generation_result=None,
                generation_error=None,
                generation_skipped=True,
                pre_retrieval_decision=pre_retrieval_decision,
                pre_retrieval_handled=pre_retrieval_handled,
            )

        retrieval_result = await run_stage(ExecutionStage.RETRIEVAL, retrieve, resolved_query)
        if retrieval_result.error is not None:
            trace_events.append(
                ExecutionTraceEvent(
                    ExecutionStage.SOURCE_SELECTION, StageOutcome.SKIPPED
                )
            )
            trace_events.append(
                ExecutionTraceEvent(
                    ExecutionStage.ANSWER_GENERATION, StageOutcome.SKIPPED
                )
            )
            return make_result(
                resolved_query=resolved_query,
                retrieval=retrieval_result,
                flow_terminated=True,
                source_decision=None,
                source_handled=False,
                generation_result=None,
                generation_error=None,
                generation_skipped=True,
                pre_retrieval_decision=pre_retrieval_decision,
                pre_retrieval_handled=pre_retrieval_handled,
            )

        source_decision: Optional[SourceDecision] = None
        source_selection_enabled = choose_source is not None
        if choose_source is not None and should_choose_source is not None:
            source_selection_enabled = await run_stage(
                ExecutionStage.SOURCE_SELECTION_POLICY, should_choose_source
            )
        if choose_source is not None and source_selection_enabled:
            source_decision = await run_stage(ExecutionStage.SOURCE_SELECTION, choose_source)
            if target is AnswerExecutionTarget.CACHE_DECISION:
                trace_events.append(
                    ExecutionTraceEvent(
                        ExecutionStage.SOURCE_HANDLING,
                        StageOutcome.SKIPPED,
                        detail_code="CACHE_DECISION_TARGET",
                    )
                )
                trace_events.append(
                    ExecutionTraceEvent(
                        ExecutionStage.ANSWER_GENERATION,
                        StageOutcome.SKIPPED,
                        detail_code="CACHE_DECISION_TARGET",
                    )
                )
                return make_result(
                    resolved_query=resolved_query,
                    retrieval=retrieval_result,
                    flow_terminated=False,
                    source_decision=source_decision,
                    source_handled=False,
                    generation_result=None,
                    generation_error=None,
                    generation_skipped=True,
                    pre_retrieval_decision=pre_retrieval_decision,
                    pre_retrieval_handled=pre_retrieval_handled,
                )

            if handle_source is None:
                raise ValueError("답변 출처 선택에는 결과 처리 함수가 필요합니다.")
            source_handled = await run_stage(ExecutionStage.SOURCE_HANDLING, handle_source, source_decision)
            if source_handled:
                trace_events.append(
                    ExecutionTraceEvent(
                        ExecutionStage.ANSWER_GENERATION, StageOutcome.SKIPPED
                    )
                )
                return make_result(
                    resolved_query=resolved_query,
                    retrieval=retrieval_result,
                    flow_terminated=False,
                    source_decision=source_decision,
                    source_handled=True,
                    generation_result=None,
                    generation_error=None,
                    generation_skipped=True,
                    pre_retrieval_decision=pre_retrieval_decision,
                    pre_retrieval_handled=pre_retrieval_handled,
                )
        else:
            trace_events.append(
                ExecutionTraceEvent(
                    ExecutionStage.SOURCE_SELECTION,
                    StageOutcome.SKIPPED,
                    detail_code=(
                        "POLICY_DISABLED"
                        if choose_source is not None
                        else "NOT_CONFIGURED"
                    ),
                )
            )
            if target is AnswerExecutionTarget.CACHE_DECISION:
                trace_events.append(
                    ExecutionTraceEvent(
                        ExecutionStage.SOURCE_HANDLING,
                        StageOutcome.SKIPPED,
                        detail_code="POLICY_DISABLED",
                    )
                )
                trace_events.append(
                    ExecutionTraceEvent(
                        ExecutionStage.ANSWER_GENERATION,
                        StageOutcome.SKIPPED,
                        detail_code="CACHE_DECISION_TARGET",
                    )
                )
                return make_result(
                    resolved_query=resolved_query,
                    retrieval=retrieval_result,
                    flow_terminated=False,
                    source_decision=None,
                    source_handled=False,
                    generation_result=None,
                    generation_error=None,
                    generation_skipped=True,
                    pre_retrieval_decision=pre_retrieval_decision,
                    pre_retrieval_handled=pre_retrieval_handled,
                )

        generation_result: Optional[FinalGenerationResult] = None
        generation_kwargs = {}
        if on_progress_stage is not None:
            generation_kwargs["on_progress_stage"] = on_progress_stage
        if stages is not None:
            generation_kwargs["stages"] = stages

        try:
            generation_result = await run_stage(
                ExecutionStage.ANSWER_GENERATION,
                self._generation_service.generate_answer,
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
            return make_result(
                resolved_query=resolved_query,
                retrieval=retrieval_result,
                flow_terminated=False,
                source_decision=source_decision,
                source_handled=False,
                generation_result=generation_result,
                generation_error=error,
                pre_retrieval_decision=pre_retrieval_decision,
                pre_retrieval_handled=pre_retrieval_handled,
            )
        return make_result(
            resolved_query=resolved_query,
            retrieval=retrieval_result,
            flow_terminated=False,
            source_decision=source_decision,
            source_handled=False,
            generation_result=generation_result,
            generation_error=None,
            pre_retrieval_decision=pre_retrieval_decision,
            pre_retrieval_handled=pre_retrieval_handled,
        )
