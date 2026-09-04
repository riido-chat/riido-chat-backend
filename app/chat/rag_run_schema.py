"""RagRun 결과 조회(polling) API의 response HTTP DTO를 정의한다."""

import uuid
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from app.chat.schema import (
    HTTP_DTO_CONFIG,
    ChatCompletedResponse,
    ChatErrorResponse,
    ChatWithheldResponse,
)


class RagRunProcessingStatus(str, Enum):
    """결과 조회에만 존재하는 진행 중 상태.

    동기 응답에는 없는 값이라 공유 ChatResponseStatus를 넓히지 않고 따로 둔다.
    """

    PROCESSING = "PROCESSING"


class RagRunErrorCode(str, Enum):
    """결과 조회 API가 외부에 제공하는 오류 코드."""

    NOT_FOUND = "NOT_FOUND"


class RagRunProcessingResponse(BaseModel):
    """아직 마감되지 않은 턴.

    진행 단계는 어디에도 저장하지 않으므로 상태와 식별자만 제공한다.
    """

    model_config = HTTP_DTO_CONFIG

    status: Literal[RagRunProcessingStatus.PROCESSING]
    conversation_id: uuid.UUID = Field(alias="conversationId")
    rag_run_id: uuid.UUID = Field(alias="ragRunId")


class RagRunErrorResponse(BaseModel):
    """결과를 조회할 수 없을 때의 응답."""

    model_config = HTTP_DTO_CONFIG

    code: RagRunErrorCode
    message: str


# 마감된 세 상태는 동기 응답 DTO를 그대로 재사용해 두 경로의 본문을 동일하게 유지한다.
RagRunResponse = Annotated[
    Union[
        RagRunProcessingResponse,
        ChatCompletedResponse,
        ChatWithheldResponse,
        ChatErrorResponse,
    ],
    Field(discriminator="status"),
]
