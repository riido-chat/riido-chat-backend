"""질문 로그 콘솔 조회 HTTP DTO.

화면 표기("-", "없음", 상대 시각)는 FE 가 정한다. 값이 없으면 null 이나 0 을 준다.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from app.chat.schema import HTTP_DTO_CONFIG


class AnswerStatus(str, Enum):
    """질문 한 건에 붙는 답변 상태 배지. 턴 상태와 캐시 시도 결과로 계산한다."""

    ANSWERED = "ANSWERED"
    CACHED_ANSWER = "CACHED_ANSWER"
    WITHHELD = "WITHHELD"
    ERROR = "ERROR"


class WithheldReason(str, Enum):
    """답변 보류 사유. rag_runs.withheld_reason_code 와 같은 값이다."""

    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AMBIGUOUS_QUESTION = "AMBIGUOUS_QUESTION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNVERIFIABLE_ANSWER = "UNVERIFIABLE_ANSWER"


class SubproblemPresence(str, Enum):
    """질문 목록의 세부 문제 유무 필터. ABSENT 는 분류 행이 없는 턴을 포함한다."""

    PRESENT = "PRESENT"
    ABSENT = "ABSENT"


class ApplyStatus(str, Enum):
    """세부 문제 적용 상태. APPROVED 정본이 있으면 APPLIED 다."""

    APPLIED = "APPLIED"
    NEEDS_CANONICAL = "NEEDS_CANONICAL"


class QuestionLogWithheldReasonCounts(BaseModel):
    """답변 불가 타일의 보류 사유별 건수."""

    model_config = HTTP_DTO_CONFIG

    insufficient_evidence: int = Field(alias="insufficientEvidence", ge=0)
    ambiguous_question: int = Field(alias="ambiguousQuestion", ge=0)
    out_of_scope: int = Field(alias="outOfScope", ge=0)
    unverifiable_answer: int = Field(alias="unverifiableAnswer", ge=0)


class QuestionLogFrequentSubproblem(BaseModel):
    """자주 묻는 세부 문제 한 줄."""

    model_config = HTTP_DTO_CONFIG

    subproblem_id: uuid.UUID = Field(alias="subproblemId")
    name: str
    # 세부 문제가 이 문서 그룹의 문서 아래에 있지 않으면 null 이다.
    document_id: Optional[int] = Field(alias="documentId")
    document_title: Optional[str] = Field(alias="documentTitle")
    question_count: int = Field(alias="questionCount", ge=1)


class QuestionLogWithheldDocument(BaseModel):
    """답변 보류가 많은 문서 한 줄. 값은 근거 부족 건수다."""

    model_config = HTTP_DTO_CONFIG

    document_id: int = Field(alias="documentId")
    document_title: str = Field(alias="documentTitle")
    insufficient_evidence_count: int = Field(alias="insufficientEvidenceCount", ge=1)


class QuestionLogDashboardResponse(BaseModel):
    """질문 분석 대시보드. 기간 없이 전체 누적이다."""

    model_config = HTTP_DTO_CONFIG

    question_count: int = Field(alias="questionCount", ge=0)
    # 답변 보류 합계. 오류는 넣지 않는다.
    unanswerable_count: int = Field(alias="unanswerableCount", ge=0)
    withheld_reason_counts: QuestionLogWithheldReasonCounts = Field(
        alias="withheldReasonCounts"
    )
    frequent_subproblems: List[QuestionLogFrequentSubproblem] = Field(
        alias="frequentSubproblems"
    )
    withheld_documents: List[QuestionLogWithheldDocument] = Field(
        alias="withheldDocuments"
    )


class QuestionLogDocumentItem(BaseModel):
    """문서 목록 한 행."""

    model_config = HTTP_DTO_CONFIG

    document_id: int = Field(alias="documentId")
    document_title: str = Field(alias="documentTitle")
    question_count: int = Field(alias="questionCount", ge=0)
    withheld_count: int = Field(alias="withheldCount", ge=0)
    subproblem_count: int = Field(alias="subproblemCount", ge=0)


class QuestionLogDocumentListResponse(BaseModel):
    """문서 목록. 보조 값은 대시보드 질문 수와 문서 합계의 차이를 설명한다."""

    model_config = HTTP_DTO_CONFIG

    items: List[QuestionLogDocumentItem]
    # 가이드 밖(문서 없음)으로 귀속된 질문 수.
    no_document_question_count: int = Field(alias="noDocumentQuestionCount", ge=0)
    # 현재 분류 행이 없는 질문 수.
    unclassified_question_count: int = Field(
        alias="unclassifiedQuestionCount", ge=0
    )


class QuestionLogDocumentRef(BaseModel):
    model_config = HTTP_DTO_CONFIG

    document_id: int = Field(alias="documentId")
    document_title: str = Field(alias="documentTitle")


class QuestionLogDocumentSummary(BaseModel):
    """문서 상세 요약 타일 4개."""

    model_config = HTTP_DTO_CONFIG

    question_count: int = Field(alias="questionCount", ge=0)
    insufficient_evidence_count: int = Field(alias="insufficientEvidenceCount", ge=0)
    cached_answer_count: int = Field(alias="cachedAnswerCount", ge=0)
    subproblem_count: int = Field(alias="subproblemCount", ge=0)


class QuestionLogSubproblemItem(BaseModel):
    """문서 상세 세부 문제 표 한 행."""

    model_config = HTTP_DTO_CONFIG

    subproblem_id: uuid.UUID = Field(alias="subproblemId")
    name: str
    question_count: int = Field(alias="questionCount", ge=0)
    # "문서명 > 절" 형태. APPROVED 정본이 없으면 null 이다.
    source_section: Optional[str] = Field(alias="sourceSection")
    apply_status: ApplyStatus = Field(alias="applyStatus")


class QuestionLogDocumentDetailResponse(BaseModel):
    model_config = HTTP_DTO_CONFIG

    document: QuestionLogDocumentRef
    summary: QuestionLogDocumentSummary
    subproblems: List[QuestionLogSubproblemItem]


class QuestionLogCanonicalAnswer(BaseModel):
    model_config = HTTP_DTO_CONFIG

    content_markdown: str = Field(alias="contentMarkdown")
    applicability_rules: List[str] = Field(alias="applicabilityRules")


class QuestionLogSubproblemDetailResponse(BaseModel):
    """세부 문제 펼침. 속한 질문은 질문 목록 조회에 subproblemId 로 받는다."""

    model_config = HTTP_DTO_CONFIG

    subproblem_id: uuid.UUID = Field(alias="subproblemId")
    name: str
    apply_status: ApplyStatus = Field(alias="applyStatus")
    canonical_answer: Optional[QuestionLogCanonicalAnswer] = Field(
        alias="canonicalAnswer"
    )


class QuestionLogSubproblemFullItem(BaseModel):
    """문서 상세 통합 조회의 세부 문제 한 행.

    문서 상세 표의 집계 필드와 세부 문제 펼침 조회의 정본 계약을 한 행에
    함께 싣는다. 포함·제외 기준도 판정 카탈로그의 원래 순서로 보낸다.
    """

    model_config = HTTP_DTO_CONFIG

    subproblem_id: uuid.UUID = Field(alias="subproblemId")
    name: str
    question_count: int = Field(alias="questionCount", ge=0)
    source_section: Optional[str] = Field(alias="sourceSection")
    apply_status: ApplyStatus = Field(alias="applyStatus")
    inclusion_criteria: List[str] = Field(alias="inclusionCriteria")
    exclusion_criteria: List[str] = Field(alias="exclusionCriteria")
    canonical_answer: Optional[QuestionLogCanonicalAnswer] = Field(
        alias="canonicalAnswer"
    )


class QuestionLogDocumentFullDetailResponse(BaseModel):
    """문서 정보·요약·세부 문제 행·정본 상세를 한 번에 주는 응답."""

    model_config = HTTP_DTO_CONFIG

    document: QuestionLogDocumentRef
    summary: QuestionLogDocumentSummary
    subproblems: List[QuestionLogSubproblemFullItem]


class QuestionLogQuestionItem(BaseModel):
    """질문 목록 한 행."""

    model_config = HTTP_DTO_CONFIG

    rag_run_id: uuid.UUID = Field(alias="ragRunId")
    # 재작성 질문이 있으면 그것을, 없으면 사용자 원문을 쓴다.
    question: str
    document_id: Optional[int] = Field(alias="documentId")
    document_title: Optional[str] = Field(alias="documentTitle")
    subproblem_id: Optional[uuid.UUID] = Field(alias="subproblemId")
    subproblem_name: Optional[str] = Field(alias="subproblemName")
    # rag_runs.created_at. 오프셋을 포함한 ISO 8601 이다.
    asked_at: datetime = Field(alias="askedAt")
    answer_status: AnswerStatus = Field(alias="answerStatus")
    # answerStatus 가 WITHHELD 일 때만 채운다.
    withheld_reason: Optional[WithheldReason] = Field(alias="withheldReason")


class QuestionLogQuestionListResponse(BaseModel):
    model_config = HTTP_DTO_CONFIG

    items: List[QuestionLogQuestionItem]
    page: int = Field(ge=1)
    size: int = Field(ge=1, le=100)
    # 필터를 적용한 뒤의 전체 건수.
    total_count: int = Field(alias="totalCount", ge=0)
