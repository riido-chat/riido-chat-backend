"""Run one cache-only turn through the shared answer use case.

The caller supplies service-specific callbacks bound to evaluation-safe data access
and a private evaluation recorder. This module never constructs a production
database session or persists raw questions, answers, or retrieval contents.
"""

import time
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol

from app.answering.ports import (
    PreRetrievalCacheHandlerPort,
    PreRetrievalCacheInspectorPort,
    QueryResolverPort,
    RetrievalPort,
    SourceSelectionEnabledPort,
    SourceSelectorPort,
)
from app.answering.use_case import (
    AnswerExecutionTarget,
    AnswerQuestion,
    AnswerQuestionResult,
)
from app.answering.trace import (
    AnswerExecutionTrace,
    ExecutionTraceRecorder,
    ReplayContext,
)


@dataclass(frozen=True)
class CacheDecisionMetrics:
    """Text-free, schema-compatible measurements supplied by the service adapter."""

    outcome: str
    next_stage: str
    model_calls: int
    embedding_calls: int
    evidence: Any = None


class CacheDecisionSummarizerPort(Protocol):
    def __call__(
        self, result: AnswerQuestionResult[Any]
    ) -> CacheDecisionMetrics: ...


@dataclass(frozen=True)
class CacheDecisionObservation:
    """The only payload this adapter emits; it contains no query or answer text."""

    item_id: str
    turn_no: int
    outcome: str
    next_stage: str
    latency_ms: int
    model_calls: int
    embedding_calls: int
    execution_trace: Optional[AnswerExecutionTrace] = None
    evidence: Any = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.item_id, str)
            or not self.item_id
            or isinstance(self.turn_no, bool)
            or not isinstance(self.turn_no, int)
            or self.turn_no < 1
        ):
            raise ValueError("관측 문항 ID와 1 이상의 턴 번호가 필요합니다.")
        if self.outcome not in {
            "SERVED",
            "SHADOW",
            "GROUP_DISABLED",
            "REJECTED",
            "SKIPPED",
            "FAILED",
        }:
            raise ValueError("지원하지 않는 캐시 판정 결과입니다.")
        if self.next_stage not in {"PASS", "FAIL", "NOT_APPLICABLE"}:
            raise ValueError("지원하지 않는 다음 단계 결과입니다.")
        for name, value in (
            ("latency_ms", self.latency_ms),
            ("model_calls", self.model_calls),
            ("embedding_calls", self.embedding_calls),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name}은 0 이상의 정수여야 합니다.")

    def to_row(self) -> dict[str, Any]:
        return {
            "itemId": self.item_id,
            "turnNo": self.turn_no,
            "outcome": self.outcome,
            "nextStage": self.next_stage,
            "latencyMs": self.latency_ms,
            "modelCalls": self.model_calls,
            "embeddingCalls": self.embedding_calls,
        }


@dataclass(frozen=True)
class CacheDecisionCallbacks:
    """Callbacks for one evaluation profile and its isolated data/recording stores.

    Implementations must use read-only selected service data and route any run
    metadata to the private evaluation repository. Production ChatService closures
    are intentionally not accepted as a ready-made bundle: they also write online
    grouping and turn logs.
    """

    resolve_query: QueryResolverPort
    retrieve: RetrievalPort
    choose_source: SourceSelectorPort[Any]
    summarize: CacheDecisionSummarizerPort
    inspect_exact_cache: Optional[PreRetrievalCacheInspectorPort[Any]] = None
    handle_exact_cache: Optional[PreRetrievalCacheHandlerPort[Any]] = None
    should_choose_source: Optional[SourceSelectionEnabledPort] = None
    finalize: Optional[Callable[[], Awaitable[None]]] = None


class CacheOnlyGenerationAttempt(RuntimeError):
    """Raised if the shared use case unexpectedly reaches answer generation."""


async def _forbid_generation(
    _provider: str, _model_name: str, _prompt_version: Optional[str]
) -> None:
    raise CacheOnlyGenerationAttempt(
        "CACHE_DECISION 실행이 답변 생성 단계에 도달했습니다."
    )


class CacheDecisionExecutionAdapter:
    """Execute shared query/retrieval/cache logic and stop before response handling."""

    def __init__(self, answer_question: AnswerQuestion[Any]) -> None:
        self._answer_question = answer_question

    async def execute(
        self,
        *,
        item_id: str,
        turn_no: int,
        callbacks: CacheDecisionCallbacks,
        replay_context: Optional[ReplayContext] = None,
        trace_recorder: Optional[ExecutionTraceRecorder] = None,
    ) -> CacheDecisionObservation:
        if (
            not isinstance(item_id, str)
            or not item_id
            or isinstance(turn_no, bool)
            or not isinstance(turn_no, int)
            or turn_no < 1
        ):
            raise ValueError("평가 문항 ID와 1 이상의 턴 번호가 필요합니다.")

        started = time.perf_counter()
        execution_error: Optional[BaseException] = None
        try:
            result = await self._answer_question.execute(
                resolve_query=callbacks.resolve_query,
                retrieve=callbacks.retrieve,
                inspect_pre_retrieval_cache=callbacks.inspect_exact_cache,
                handle_pre_retrieval_cache=callbacks.handle_exact_cache,
                choose_source=callbacks.choose_source,
                should_choose_source=callbacks.should_choose_source,
                # This callback both prevents accidental generation and guards against
                # a future regression that ignores the requested execution target.
                before_model_call=_forbid_generation,
                target=AnswerExecutionTarget.CACHE_DECISION,
                replay_context=replay_context,
                trace_recorder=trace_recorder,
            )
            if not result.generation_skipped or result.generation_result is not None:
                raise CacheOnlyGenerationAttempt(
                    "캐시 전용 실행 결과에서 답변 생성이 생략되지 않았습니다."
                )

            metrics = callbacks.summarize(result)
            return CacheDecisionObservation(
                item_id=item_id,
                turn_no=turn_no,
                outcome=metrics.outcome,
                next_stage=metrics.next_stage,
                latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
                model_calls=metrics.model_calls,
                embedding_calls=metrics.embedding_calls,
                execution_trace=result.execution_trace,
                evidence=metrics.evidence,
            )
        except BaseException as error:
            execution_error = error
            raise
        finally:
            if callbacks.finalize is not None:
                try:
                    await callbacks.finalize()
                except BaseException as cleanup_error:
                    if execution_error is None:
                        raise
                    add_note = getattr(execution_error, "add_note", None)
                    if callable(add_note):
                        add_note(
                            "Per-turn cache snapshot cleanup also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                    logging.getLogger(__name__).error(
                        "Per-turn cache snapshot cleanup failed while preserving the execution error",
                        exc_info=cleanup_error,
                    )
