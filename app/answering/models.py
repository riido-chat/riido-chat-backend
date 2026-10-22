"""Generation 내부 결과와 최종 답변에 사용하는 모델."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.model_trace import ModelCallTrace
from app.document.document_key import CONSOLE_URI_SCHEME
from app.retrieval.models import RetrievalChunk


class GenerationStatus(str, Enum):
    """Writer가 판단하는 작성 가능 상태."""

    ANSWERABLE = "ANSWERABLE"
    WITHHELD = "WITHHELD"


class GenerationPlanningStatus(str, Enum):
    """Source Planning이 판단하는 근거 활용 방식."""

    ANSWERABLE = "ANSWERABLE"
    RELATED_GUIDANCE = "RELATED_GUIDANCE"
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


class GenerationAnswerType(str, Enum):
    """질문에 맞는 Source 정책과 답변 형식을 결정하는 내부 유형."""

    DEFINITION = "DEFINITION"
    FEATURE_SUMMARY = "FEATURE_SUMMARY"
    PROCEDURE = "PROCEDURE"
    GENERAL = "GENERAL"


class GenerationEvidenceRequirement(BaseModel):
    """질문이 요구한 정보 단위와 이를 직접 뒷받침하는 Source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    information_unit: str = Field(min_length=1)
    source_ids: List[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source_ids(self) -> "GenerationEvidenceRequirement":
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("정보 단위의 Source ID는 중복될 수 없습니다.")
        return self


class GenerationSourcePlan(BaseModel):
    """답변 생성 전에 질문 범위와 필요한 근거를 확정하는 내부 결과."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: GenerationPlanningStatus
    answer_type: GenerationAnswerType
    answer_scope: GenerationAnswerScope
    evidence_requirements: List[GenerationEvidenceRequirement]
    optional_context: List[GenerationEvidenceRequirement]
    unanswered_information: List[str]
    related_guidance: List[GenerationEvidenceRequirement]
    withheld_reason: Optional[GenerationWithheldReason]

    @model_validator(mode="after")
    def validate_status_fields(self) -> "GenerationSourcePlan":
        self._validate_unanswered_information()

        if self.status == GenerationPlanningStatus.ANSWERABLE:
            if not self.evidence_requirements:
                raise ValueError("ANSWERABLE에는 정보 단위별 근거가 필요합니다.")
            if self.unanswered_information or self.related_guidance:
                raise ValueError(
                    "ANSWERABLE에는 미확인 요구와 관련 안내를 "
                    "사용할 수 없습니다."
                )
            if self.withheld_reason is not None:
                raise ValueError("ANSWERABLE에는 withheld_reason을 사용할 수 없습니다.")
            return self

        if self.status == GenerationPlanningStatus.RELATED_GUIDANCE:
            if self.evidence_requirements or self.optional_context:
                raise ValueError(
                    "RELATED_GUIDANCE에는 직접 근거와 선택적 배경을 "
                    "사용할 수 없습니다."
                )
            if not self.unanswered_information:
                raise ValueError(
                    "RELATED_GUIDANCE에는 미확인 핵심 요구가 필요합니다."
                )
            if not self.related_guidance:
                raise ValueError(
                    "RELATED_GUIDANCE에는 근거 있는 관련 안내가 필요합니다."
                )
            if self.withheld_reason is not None:
                raise ValueError(
                    "RELATED_GUIDANCE에는 withheld_reason을 사용할 수 없습니다."
                )
            return self

        if (
            self.evidence_requirements
            or self.optional_context
            or self.related_guidance
        ):
            raise ValueError("WITHHELD에는 선택 Source를 사용할 수 없습니다.")
        if self.withheld_reason is None:
            raise ValueError("WITHHELD에는 withheld_reason이 필요합니다.")
        if (
            self.withheld_reason == GenerationWithheldReason.INSUFFICIENT_EVIDENCE
            and not self.unanswered_information
        ):
            raise ValueError(
                "INSUFFICIENT_EVIDENCE 보류에는 미확인 핵심 요구가 필요합니다."
            )
        if (
            self.withheld_reason != GenerationWithheldReason.INSUFFICIENT_EVIDENCE
            and self.unanswered_information
        ):
            raise ValueError(
                "모호함·범위 밖 보류에는 미확인 요구를 기록하지 않습니다."
            )
        return self

    def _validate_unanswered_information(self) -> None:
        if any(not item.strip() for item in self.unanswered_information):
            raise ValueError("미확인 핵심 요구는 비어 있을 수 없습니다.")
        if len(self.unanswered_information) != len(
            set(self.unanswered_information)
        ):
            raise ValueError("미확인 핵심 요구는 중복될 수 없습니다.")


class GenerationResult(BaseModel):
    """LLM Structured Output으로 전달받는 내부 Generation 결과."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: GenerationStatus
    limitation_markdown: Optional[str]
    answer_markdown: Optional[str]
    withheld_reason: Optional[GenerationWithheldReason]

    @model_validator(mode="after")
    def validate_status_fields(self) -> "GenerationResult":
        """상태에 따라 답변과 보류 사유의 nullable 규칙을 검증한다."""

        if self.status == GenerationStatus.ANSWERABLE:
            if self.answer_markdown is None or not self.answer_markdown.strip():
                raise ValueError("ANSWERABLE에는 answer_markdown이 필요합니다.")
            if (
                self.limitation_markdown is not None
                and not self.limitation_markdown.strip()
            ):
                raise ValueError("limitation_markdown은 빈 문자열일 수 없습니다.")
            if self.withheld_reason is not None:
                raise ValueError("ANSWERABLE에는 withheld_reason을 사용할 수 없습니다.")
            return self

        if self.limitation_markdown is not None:
            raise ValueError("WITHHELD의 limitation_markdown은 null이어야 합니다.")
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


class CitationSourceKind(str, Enum):
    """인용 출처의 종류. 클라이언트가 외부 링크 노출 여부를 판단한다."""

    GITBOOK = "GITBOOK"
    CONSOLE = "CONSOLE"

    @classmethod
    def from_canonical_uri(cls, canonical_uri: str) -> "CitationSourceKind":
        """원문 위치자의 스킴으로 출처 종류를 판별한다.

        콘솔 업로드 문서만 내부 스킴을 쓰므로, 그 외는 GitBook 문서로 본다.
        """

        if canonical_uri.startswith(f"{CONSOLE_URI_SCHEME}://"):
            return cls.CONSOLE
        return cls.GITBOOK


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
    source_kind: CitationSourceKind
    chunk_id: Optional[int] = None
    document_version_id: Optional[int] = None


@dataclass(frozen=True)
class ValidatedAnswer:
    """Citation marker 검증과 치환이 끝난 답변."""

    answer_markdown: str
    citations: Tuple[Citation, ...]


@dataclass(frozen=True)
class GenerationStageTrace:
    """평가에서 단계별 변동을 찾기 위한 Generation 내부 관측값.

    제품 응답이나 DB 로그 모델이 아니라, 한 요청 안에서 관측 hook으로 전달할
    진단 정보다. 도달하지 못한 단계의 값은 None 또는 빈 tuple로 남긴다.
    """

    source_plan: Optional[GenerationSourcePlan] = None
    initial_source_plan: Optional[GenerationSourcePlan] = None
    selected_sources: Tuple[GenerationContextSource, ...] = ()
    pre_validation_result: Optional[GenerationResult] = None
    validation_error: Optional[str] = None
    validation_errors: Tuple[str, ...] = ()
    planning_attempt_count: int = 0
    planning_regeneration_count: int = 0
    planning_regeneration_model_call: Optional[ModelCallTrace] = None
    planning_regeneration_result: Optional[GenerationSourcePlan] = None
    answer_attempt_count: int = 0
    validation_regeneration_count: int = 0
    validation_regeneration_model_call: Optional[ModelCallTrace] = None
    validation_regeneration_result: Optional[GenerationResult] = None


@dataclass(frozen=True)
class GenerationCall:
    """Generator 호출 한 번의 결과와 model_calls 기록용 관측값.

    실패를 곧바로 던지지 않고 error에 담는 이유는, 실패한 호출도 재시도 횟수와
    지연시간을 model_calls에 남겨야 하기 때문이다.
    """

    trace: ModelCallTrace
    result: Optional[GenerationResult] = None
    error: Optional[Exception] = None
    stage_trace: Optional[GenerationStageTrace] = None


@dataclass(frozen=True)
class FinalGenerationResult:
    """Application 계층에서 결정한 최종 Generation 결과."""

    status: FinalAnswerStatus
    answer_markdown: Optional[str]
    citations: Tuple[Citation, ...]
    withheld_reason: Optional[FinalWithheldReason] = None
    error_code: Optional[str] = None
    model_call: Optional[ModelCallTrace] = None
    stage_trace: Optional[GenerationStageTrace] = None
