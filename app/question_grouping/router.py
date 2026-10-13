"""검색 이후 사용할 답변 출처를 선택하는 Router."""

from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional

from app.question_grouping.service import (
    ExactQuestionFallthrough,
    GroupingTurn,
    JudgedTurn,
    PreparedJudgment,
    QuestionGroupingService,
    RecordedJudgment,
)
from app.retrieval.models import HybridSearchCall


@dataclass(frozen=True)
class AnswerSourceDecision:
    """검색 뒤 판별 결과와 Router가 고른 답변 출처."""

    recorded: RecordedJudgment
    prepared: PreparedJudgment
    judged: Optional[JudgedTurn]

    @property
    def uses_canonical_answer(self) -> bool:
        return self.recorded.served


class AnswerSourceRouter:
    """근거 검색 뒤 승인 정본을 쓸지 생성으로 넘길지 결정한다.

    Router는 생성과 턴 마감을 하지 않는다.
    채팅 계층이 검색 기록과 트랜잭션을 맡는다.
    """

    def __init__(self, grouping: QuestionGroupingService) -> None:
        self._grouping = grouping

    async def choose(
        self,
        turn: GroupingTurn,
        resolved_query: str,
        search: HybridSearchCall,
        *,
        semantic_cache_enabled: bool,
        before_judgment: Callable[[], Awaitable[None]],
        exact_fallthrough: Optional[ExactQuestionFallthrough] = None,
    ) -> AnswerSourceDecision:
        """판별 결과와 선택한 답변 출처를 돌려준다."""

        prepare_kwargs: Dict[str, ExactQuestionFallthrough] = {}
        if exact_fallthrough is not None:
            prepare_kwargs["exact_fallthrough"] = exact_fallthrough
        prepared = await self._grouping.prepare(
            turn,
            resolved_query,
            search,
            **prepare_kwargs,
        )
        if not prepared.ready:
            await before_judgment()
            recorded = await self._grouping.record_preparation_failure(prepared)
            return AnswerSourceDecision(
                recorded=recorded,
                prepared=prepared,
                judged=None,
            )

        judged = await self._grouping.judge(
            prepared,
            before_checkpoint=before_judgment,
        )
        recorded = await self._grouping.record_judgment_and_gate(
            prepared,
            judged,
            semantic_cache_enabled=semantic_cache_enabled,
        )
        return AnswerSourceDecision(
            recorded=recorded,
            prepared=prepared,
            judged=judged,
        )
