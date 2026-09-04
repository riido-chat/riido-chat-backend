"""답변 피드백 API의 request/response HTTP DTO를 정의한다."""

import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.chat.schema import HTTP_DTO_CONFIG


class ChatFeedbackRating(str, Enum):
    """사용자가 답변에 남기는 평가."""

    GOOD = "GOOD"
    BAD = "BAD"


class ChatFeedbackErrorCode(str, Enum):
    """피드백 API가 외부에 제공하는 오류 코드."""

    NOT_FOUND = "NOT_FOUND"
    FEEDBACK_NOT_ALLOWED = "FEEDBACK_NOT_ALLOWED"


class FeedbackRequest(BaseModel):
    """PUT /api/chat/{ragRunId}/feedback 요청."""

    model_config = HTTP_DTO_CONFIG

    rating: ChatFeedbackRating


class FeedbackResponse(BaseModel):
    """저장된 최종 평가. 해제한 뒤에는 rating이 null이다.

    FE가 호출 결과만으로 버튼 상태를 맞출 수 있도록 최종 값을 그대로 돌려준다.
    """

    model_config = HTTP_DTO_CONFIG

    rag_run_id: uuid.UUID = Field(alias="ragRunId")
    rating: Optional[ChatFeedbackRating] = None


class FeedbackErrorResponse(BaseModel):
    """피드백을 저장할 수 없을 때의 응답."""

    model_config = HTTP_DTO_CONFIG

    code: ChatFeedbackErrorCode
    message: str
