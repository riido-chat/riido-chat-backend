"""Generation 내부 결과와 최종 답변에 사용하는 모델."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.model_trace import ModelCallTrace
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


class GenerationAnswerScope(str, Enum):
    """Source 선택 단계가 판정한 질문의 답변 범위."""

    SUMMARY = "SUMMARY"
    MULTI_DETAIL = "MULTI_DETAIL"


class GenerationEvidenceRequirement(BaseModel):
    """질문이 요구한 정보 단위와 이를 직접 뒷받침하는 Source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    information_unit: str = Field(min_length=1)
    source_ids: List[str] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def validate_source_ids(self) -> "GenerationEvidenceRequirement":
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("정보 단위의 Source ID는 중복될 수 없습니다.")
        return self


class GenerationSourcePlan(BaseModel):
    """답변 생성 전에 질문 범위와 필요한 근거를 확정하는 내부 결과."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: GenerationStatus
    answer_scope: GenerationAnswerScope
    evidence_requirements: List[GenerationEvidenceRequirement] = Field(
        max_length=8
    )
    withheld_reason: Optional[GenerationWithheldReason]

    @model_validator(mode="after")
    def validate_status_fields(self) -> "GenerationSourcePlan":
        if self.status == GenerationStatus.ANSWERABLE:
            if not self.evidence_requirements:
                raise ValueError("ANSWERABLE에는 정보 단위별 근거가 필요합니다.")
            if self.withheld_reason is not None:
                raise ValueError("ANSWERABLE에는 withheld_reason을 사용할 수 없습니다.")
            return self

        if self.evidence_requirements:
            raise ValueError("WITHHELD에는 정보 단위별 근거를 사용할 수 없습니다.")
        if self.withheld_reason is None:
            raise ValueError("WITHHELD에는 withheld_reason이 필요합니다.")
        return self


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
    """사용자에게 제공할 검증된 출처.

    사용자 응답에는 쓰이지 않지만, 인용을 answer_citations에 남기려면 근거 청크의
    DB 식별자가 필요하다. 병합된 인용은 대표 청크 하나의 식별자를 갖는다.
    """

    citation_number: int
    document_title: str
    section_path: Tuple[str, ...]
    source_url: str
    chunk_id: Optional[int] = None
    document_version_id: Optional[int] = None


@dataclass(frozen=True)
class ValidatedAnswer:
    """Citation marker 검증과 치환이 끝난 답변."""

    answer_markdown: str
    citations: Tuple[Citation, ...]


@dataclass(frozen=True)
class GenerationCall:
    """Generator 호출 한 번의 결과와 model_calls 기록용 관측값.

    실패를 곧바로 던지지 않고 error에 담는 이유는, 실패한 호출도 재시도 횟수와
    지연시간을 model_calls에 남겨야 하기 때문이다.
    """

    trace: ModelCallTrace
    result: Optional[GenerationResult] = None
    error: Optional[Exception] = None


@dataclass(frozen=True)
class FinalGenerationResult:
    """Application 계층에서 결정한 최종 Generation 결과."""

    status: FinalAnswerStatus
    answer_markdown: Optional[str]
    citations: Tuple[Citation, ...]
    withheld_reason: Optional[FinalWithheldReason] = None
    error_code: Optional[str] = None
    model_call: Optional[ModelCallTrace] = None
