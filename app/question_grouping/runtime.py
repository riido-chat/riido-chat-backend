"""질문 판별 부품의 프로세스 공유분과 요청별 조립.

판별 모델 클라이언트와 outline 캐시는 앱 기동 시 한 번 만들어 app.state 에 둔다. 판별 서비스는
세션과 로그 저장 계층을 ChatService 와 함께 써야 하므로 요청(또는 SSE producer 세션)마다 만든다.
"""

from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.log_store import RagLogStore
from app.core.config import Settings
from app.question_grouping.judge_client import QuestionJudgeClient
from app.question_grouping.outline_reader import DocumentOutlineCache, DocumentOutlineReader
from app.question_grouping.service import QuestionGroupingService


@dataclass(frozen=True)
class QuestionGroupingComponents:
    """프로세스 전체가 공유하는 판별 부품."""

    judge_client: Any
    outline_cache: DocumentOutlineCache

    def build_service(
        self,
        *,
        session: AsyncSession,
        log_store: RagLogStore,
        embedder: Any,
    ) -> QuestionGroupingService:
        """ChatService 와 같은 session·log_store 로 요청별 판별 서비스를 만든다."""

        return QuestionGroupingService(
            session,
            log_store,
            self.judge_client,
            embedder,
            outline_reader=DocumentOutlineReader(session, cache=self.outline_cache),
        )


def create_question_grouping_components(
    settings: Settings,
) -> Optional[QuestionGroupingComponents]:
    """스위치가 켜져 있을 때만 공유 부품을 만든다. 꺼져 있으면 None(판별 미주입)."""

    if not settings.question_grouping_enabled:
        return None
    return QuestionGroupingComponents(
        judge_client=QuestionJudgeClient(),
        outline_cache=DocumentOutlineCache(),
    )
