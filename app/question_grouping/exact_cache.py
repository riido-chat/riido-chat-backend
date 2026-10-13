"""정확 일치 승인 답변 캐시 경계."""

from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, Optional, TypeVar, Union

from app.question_grouping.service import (
    ExactQuestionFallthrough,
    ExactQuestionResult,
    GroupingTurn,
    QuestionGroupingService,
    RecordedJudgment,
)

T = TypeVar("T")


@dataclass(frozen=True)
class ExactCacheDecision:
    """정확 일치 조회 결과와 뒤 단계에서 필요한 인계 정보."""

    result: Optional[Union[ExactQuestionResult, ExactQuestionFallthrough]]
    recorded: Optional[RecordedJudgment]
    fallthrough: Optional[ExactQuestionFallthrough]

    @property
    def allow_similar_judgment(self) -> bool:
        """뒤에서 새 유사 질문 판정을 진행할 수 있는가."""

        return self.recorded is None


@dataclass(frozen=True)
class ExactCacheExecution(Generic[T]):
    """정확 일치가 흐름을 끝냈는지와 그 실행 결과."""

    cache_handled: bool
    result: T


class ExactAnswerCache(Generic[T]):
    """승인된 첫 턴 질문 조회로 다음 답변 유스케이스를 감싼다."""

    def __init__(self, grouping: QuestionGroupingService) -> None:
        self._grouping = grouping

    async def inspect(
        self,
        turn: GroupingTurn,
        question: str,
        *,
        semantic_cache_enabled: bool,
        exact_cache_enabled: bool,
    ) -> ExactCacheDecision:
        result = await self._grouping.record_exact_question(
            turn,
            question,
            semantic_cache_enabled=semantic_cache_enabled,
            exact_cache_enabled=exact_cache_enabled,
        )
        recorded = (
            result.recorded if isinstance(result, ExactQuestionResult) else None
        )
        fallthrough = (
            result if isinstance(result, ExactQuestionFallthrough) else None
        )
        return ExactCacheDecision(
            result=result,
            recorded=recorded,
            fallthrough=fallthrough,
        )

    async def execute(
        self,
        turn: GroupingTurn,
        question: str,
        *,
        semantic_cache_enabled: bool,
        exact_cache_enabled: bool,
        handle_decision: Callable[
            [ExactCacheDecision], Awaitable[Optional[T]]
        ],
        next_step: Callable[[ExactCacheDecision], Awaitable[T]],
    ) -> ExactCacheExecution[T]:
        """정확 일치가 끝내면 반환하고, 그 외에는 감싼 답변 흐름을 실행한다."""

        decision = await self.inspect(
            turn,
            question,
            semantic_cache_enabled=semantic_cache_enabled,
            exact_cache_enabled=exact_cache_enabled,
        )
        handled = await handle_decision(decision)
        if handled is not None:
            return ExactCacheExecution(cache_handled=True, result=handled)
        return ExactCacheExecution(
            cache_handled=False,
            result=await next_step(decision),
        )
