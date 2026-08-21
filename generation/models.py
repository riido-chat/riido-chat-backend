"""Generation 내부 결과와 최종 답변에 사용하는 모델."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, model_validator

from retrieval.models import RetrievalChunk


class GenerationStatus(str, Enum):
    """Generator가 판단하는 답변 가능 상태."""

    ANSWERABLE = "ANSWERABLE"
    WITHHELD = "WITHHELD"


class GenerationWithheldReason(str, Enum):
    """LLM이 판단할 수 있는 답변 보류 사유."""

    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AMBIGUOUS_QUESTION = "AMBIGUOUS_QUESTION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class GenerationResult(BaseModel):
    """LLM Structured Output으로 전달받는 내부 Generation 결과."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: GenerationStatus
    answer_markdown: Optional[str]
    withheld_reason: Optional[GenerationWithheldReason]

    @model_validator(mode="after")
    def validate_status_fields(self) -> "GenerationResult":
        """상태에 따라 답변과 보류 사유의 nullable 규칙을 검증한다."""

        if self.status == GenerationStatus.ANSWERABLE:
            if self.answer_markdown is None or not self.answer_markdown.strip():
                raise ValueError("ANSWERABLE에는 answer_markdown이 필요합니다.")
            if self.withheld_reason is not None:
                raise ValueError("ANSWERABLE에는 withheld_reason을 사용할 수 없습니다.")
            return self

        if self.answer_markdown is not None:
            raise ValueError("WITHHELD의 answer_markdown은 null이어야 합니다.")
        if self.withheld_reason is None:
            raise ValueError("WITHHELD에는 withheld_reason이 필요합니다.")
        return self


class FinalAnswerStatus(str, Enum):
    """Backend 검증 이후의 최종 답변 상태."""

    COMPLETED = "COMPLETED"
    WITHHELD = "WITHHELD"
    ERROR = "ERROR"


class FinalWithheldReason(str, Enum):
    """Backend가 유지하는 전체 답변 보류 사유."""

    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AMBIGUOUS_QUESTION = "AMBIGUOUS_QUESTION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNVERIFIABLE_ANSWER = "UNVERIFIABLE_ANSWER"


@dataclass(frozen=True)
class GenerationContextSource:
    """LLM Context 식별자와 Backend가 보존할 원본 Chunk의 연결."""

    source_id: str
    chunk: RetrievalChunk


@dataclass(frozen=True)
class Citation:
    """사용자에게 제공할 검증된 출처."""

    citation_number: int
    document_title: str
    section_path: Tuple[str, ...]
    source_url: str


@dataclass(frozen=True)
class ValidatedAnswer:
    """Citation marker 검증과 치환이 끝난 답변."""

    answer_markdown: str
    citations: Tuple[Citation, ...]


@dataclass(frozen=True)
class FinalGenerationResult:
    """Application 계층에서 결정한 최종 Generation 결과."""

    status: FinalAnswerStatus
    answer_markdown: Optional[str]
    citations: Tuple[Citation, ...]
    withheld_reason: Optional[FinalWithheldReason] = None
    error_code: Optional[str] = None
