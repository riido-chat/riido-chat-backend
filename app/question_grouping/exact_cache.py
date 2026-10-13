"""정확 일치 승인 답변 캐시 경계."""

from dataclasses import dataclass
from typing import Optional, Union

from app.question_grouping.service import (
    ExactQuestionFallthrough,
    ExactQuestionResult,
    GroupingTurn,
    QuestionGroupingService,
    RecordedJudgment,
)


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


class ExactAnswerCache:
    """승인된 첫 턴 질문을 조회하고 결과를 인계한다."""

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
