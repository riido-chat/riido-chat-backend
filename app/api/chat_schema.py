"""Chat API MVP의 request/response HTTP DTO를 정의한다."""

from enum import Enum
from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


HTTP_DTO_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    populate_by_name=True,
)


class ChatRequest(BaseModel):
    """POST /api/chat 요청."""

    model_config = HTTP_DTO_CONFIG

    question: str

    @field_validator("question")
    @classmethod
    def strip_and_validate_question(cls, question: str) -> str:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question은 비어 있을 수 없습니다.")
        return normalized_question


class ChatResponseStatus(str, Enum):
    """Chat API가 외부에 제공하는 답변 상태."""

    COMPLETED = "COMPLETED"
    WITHHELD = "WITHHELD"
    ERROR = "ERROR"


class ChatWithheldReasonCode(str, Enum):
    """답변을 제공하지 않는 사용자-facing 사유 코드."""

    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AMBIGUOUS_QUESTION = "AMBIGUOUS_QUESTION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNVERIFIABLE_ANSWER = "UNVERIFIABLE_ANSWER"


class ChatErrorCode(str, Enum):
    """Chat API가 외부에 제공하는 기술 오류 코드."""

    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


class ChatAnswer(BaseModel):
    """정상 답변 본문."""

    model_config = HTTP_DTO_CONFIG

    answer_markdown: str = Field(alias="answerMarkdown")


class ChatCitation(BaseModel):
    """사용자에게 제공하는 검증된 출처."""

    model_config = HTTP_DTO_CONFIG

    citation_number: int = Field(alias="citationNumber", ge=1, le=3)
    document_title: str = Field(alias="documentTitle")
    section_path: List[str] = Field(alias="sectionPath")
    source_url: str = Field(alias="sourceUrl")


class ChatWithheld(BaseModel):
    """정상적인 답변 보류 정보."""

    model_config = HTTP_DTO_CONFIG

    reason_code: ChatWithheldReasonCode = Field(alias="reasonCode")
    message: str


class ChatError(BaseModel):
    """외부에 노출하는 기술 오류 정보."""

    model_config = HTTP_DTO_CONFIG

    code: ChatErrorCode
    message: str


class ChatCompletedResponse(BaseModel):
    """근거 검증까지 완료된 정상 답변."""

    model_config = HTTP_DTO_CONFIG

    status: Literal[ChatResponseStatus.COMPLETED]
    answer: ChatAnswer
    citations: List[ChatCitation] = Field(min_length=1, max_length=3)


class ChatWithheldResponse(BaseModel):
    """정상적으로 답변을 보류한 결과."""

    model_config = HTTP_DTO_CONFIG

    status: Literal[ChatResponseStatus.WITHHELD]
    answer: None
    withheld: ChatWithheld
    citations: List[ChatCitation] = Field(max_length=0)


class ChatErrorResponse(BaseModel):
    """기술 실패로 답변하지 못한 결과."""

    model_config = HTTP_DTO_CONFIG

    status: Literal[ChatResponseStatus.ERROR]
    answer: None
    error: ChatError
    citations: List[ChatCitation] = Field(max_length=0)


ChatResponse = Annotated[
    Union[
        ChatCompletedResponse,
        ChatWithheldResponse,
        ChatErrorResponse,
    ],
    Field(discriminator="status"),
]
