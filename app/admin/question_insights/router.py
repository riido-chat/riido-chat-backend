"""질문 로그 콘솔 조회 endpoint.

경로의 group_id 는 document_groups.id 다. 모든 수치는 그 문서 그룹 안에서 기간 없이 누적한다.

질문 목록 쿼리 값은 문자열로 받아 QuestionListFilters 가 검증한다. 형식 오류와 범위 오류가
같은 INVALID_REQUEST 문장 경로를 타게 하려는 것이다. 경로 값(group_id 등)은 기존 admin
경로처럼 FastAPI 타입으로 받고, 형식 오류는 앱의 RequestValidationError 처리기가
{code, message} 로 바꾼다.
"""

import re
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query, status

from app.admin.dependencies import get_question_insight_service
from app.admin.question_insights.schema import (
    QuestionLogCanonicalAnswer,
    QuestionLogDashboardResponse,
    QuestionLogDocumentDetailResponse,
    QuestionLogDocumentItem,
    QuestionLogDocumentListResponse,
    QuestionLogDocumentRef,
    QuestionLogDocumentSummary,
    QuestionLogFrequentSubproblem,
    QuestionLogQuestionItem,
    QuestionLogQuestionListResponse,
    QuestionLogSubproblemDetailResponse,
    QuestionLogSubproblemItem,
    QuestionLogWithheldDocument,
    QuestionLogWithheldReasonCounts,
)
from app.admin.question_insights.service import (
    DEFAULT_PAGE,
    DEFAULT_PAGE_SIZE,
    DocumentDetail,
    DocumentList,
    InvalidQuestionLogRequestError,
    QuestionDashboard,
    QuestionInsightService,
    QuestionListFilters,
    QuestionPage,
    SubproblemDetail,
)
from app.admin.schema import AdminErrorResponse
from app.document.ingestion_service import (
    DocumentGroupNotFoundError,
    DocumentNotFoundError,
)


router = APIRouter(
    prefix="/api/admin/document-groups/{group_id}/question-log",
    tags=["admin-question-log"],
)

# DB id 는 BIGINT 다. 범위를 넘는 정수는 드라이버가 인코딩하지 못해 500 이 되므로 먼저 거른다.
BIGINT_MIN = -(2**63)
BIGINT_MAX = 2**63 - 1
_INTEGER_PATTERN = re.compile(r"-?[0-9]+")

_NOT_FOUND = {
    "model": AdminErrorResponse,
    "description": "`NOT_FOUND`: 문서 그룹이 없는 경우입니다.",
}
_INVALID_PATH = {
    "model": AdminErrorResponse,
    "description": "`INVALID_REQUEST`: 경로 값의 형식이 올바르지 않은 경우입니다.",
}


@router.get(
    "/dashboard",
    response_model=QuestionLogDashboardResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: _NOT_FOUND,
        status.HTTP_422_UNPROCESSABLE_ENTITY: _INVALID_PATH,
    },
    summary="질문 분석 대시보드 조회",
)
async def get_question_log_dashboard(
    group_id: int,
    service: QuestionInsightService = Depends(get_question_insight_service),
) -> QuestionLogDashboardResponse:
    _require_group_id(group_id)
    return _to_dashboard(await service.get_dashboard(group_id))


@router.get(
    "/documents",
    response_model=QuestionLogDocumentListResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: _NOT_FOUND,
        status.HTTP_422_UNPROCESSABLE_ENTITY: _INVALID_PATH,
    },
    summary="질문 로그 문서 목록 조회",
)
async def list_question_log_documents(
    group_id: int,
    service: QuestionInsightService = Depends(get_question_insight_service),
) -> QuestionLogDocumentListResponse:
    _require_group_id(group_id)
    return _to_document_list(await service.list_documents(group_id))


@router.get(
    "/documents/{document_id}",
    response_model=QuestionLogDocumentDetailResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": AdminErrorResponse,
            "description": (
                "`NOT_FOUND`: 문서 그룹이 없거나, 문서가 없거나 그 그룹 소속이 아닌 경우입니다."
            ),
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: _INVALID_PATH,
    },
    summary="질문 로그 문서 상세 조회",
)
async def get_question_log_document(
    group_id: int,
    document_id: int,
    service: QuestionInsightService = Depends(get_question_insight_service),
) -> QuestionLogDocumentDetailResponse:
    _require_group_id(group_id)
    _require_document_id(document_id)
    return _to_document_detail(await service.get_document_detail(group_id, document_id))


@router.get(
    "/documents/{document_id}/subproblems/{subproblem_id}",
    response_model=QuestionLogSubproblemDetailResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": AdminErrorResponse,
            "description": (
                "`NOT_FOUND`: 문서 그룹이나 문서가 없거나, 세부 문제가 없거나 보관됐거나 "
                "그 문서 소속이 아닌 경우입니다."
            ),
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: _INVALID_PATH,
    },
    summary="질문 로그 세부 문제 펼침 조회",
)
async def get_question_log_subproblem(
    group_id: int,
    document_id: int,
    subproblem_id: uuid.UUID,
    service: QuestionInsightService = Depends(get_question_insight_service),
) -> QuestionLogSubproblemDetailResponse:
    _require_group_id(group_id)
    _require_document_id(document_id)
    return _to_subproblem_detail(
        await service.get_subproblem_detail(group_id, document_id, subproblem_id)
    )


@router.get(
    "/questions",
    response_model=QuestionLogQuestionListResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: _NOT_FOUND,
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "model": AdminErrorResponse,
            "description": (
                "`INVALID_REQUEST`: 경로 값이나 쿼리 값의 형식·범위가 올바르지 않거나, "
                "subproblemPresence=ABSENT 와 subproblemId 를 함께 준 경우입니다."
            ),
        },
    },
    summary="질문 목록 조회",
)
async def list_question_log_questions(
    group_id: int,
    answer_status: Optional[str] = Query(
        default=None,
        alias="answerStatus",
        description="ANSWERED | CACHED_ANSWER | WITHHELD | ERROR",
    ),
    document_id: Optional[str] = Query(
        default=None,
        alias="documentId",
        description="문서 ID(정수). 현재 분류가 그 문서로 귀속된 질문만",
    ),
    subproblem_presence: Optional[str] = Query(
        default=None,
        alias="subproblemPresence",
        description="PRESENT | ABSENT. ABSENT 는 분류되지 않은 질문을 포함",
    ),
    subproblem_id: Optional[str] = Query(
        default=None,
        alias="subproblemId",
        description="세부 문제 ID(uuid)",
    ),
    q: Optional[str] = Query(
        default=None,
        description="질문 검색어. 앞뒤 공백을 걷고 1~100자. 비면 검색하지 않음",
    ),
    page: Optional[str] = Query(default=None, description="1 이상, 기본 1"),
    size: Optional[str] = Query(default=None, description="1~100, 기본 20"),
    service: QuestionInsightService = Depends(get_question_insight_service),
) -> QuestionLogQuestionListResponse:
    filters = QuestionListFilters(
        answer_status=answer_status,
        document_id=_parse_integer(document_id, "documentId"),
        subproblem_presence=subproblem_presence,
        subproblem_id=subproblem_id,
        q=q,
        page=_parse_integer(page, "page", default=DEFAULT_PAGE),
        size=_parse_integer(size, "size", default=DEFAULT_PAGE_SIZE),
    )
    _require_group_id(group_id)
    return _to_question_page(await service.list_questions(group_id, filters))


# ---------------------------------------------------------------------------
# 입력
# ---------------------------------------------------------------------------


def _in_bigint_range(value: int) -> bool:
    return BIGINT_MIN <= value <= BIGINT_MAX


def _require_group_id(group_id: int) -> None:
    # BIGINT 를 넘는 id 의 행은 있을 수 없다.
    if not _in_bigint_range(group_id):
        raise DocumentGroupNotFoundError()


def _require_document_id(document_id: int) -> None:
    if not _in_bigint_range(document_id):
        raise DocumentNotFoundError()


def _parse_integer(
    value: Optional[str], field: str, default: Optional[int] = None
) -> Optional[int]:
    """쿼리 문자열을 정수로 바꾼다. 없으면 default 다.

    int() 가 허용하는 공백, 밑줄, 부호 + 는 받지 않는다. 범위 검사는 QuestionListFilters 가
    하지만 BIGINT 를 넘는 값은 여기서 거른다.
    """

    if value is None:
        return default
    if not _INTEGER_PATTERN.fullmatch(value):
        raise InvalidQuestionLogRequestError(f"{field} 는 정수여야 합니다.")
    number = int(value)
    if not _in_bigint_range(number):
        raise InvalidQuestionLogRequestError(f"{field} 값이 범위를 벗어났습니다.")
    return number


# ---------------------------------------------------------------------------
# 응답 변환
# ---------------------------------------------------------------------------


def _to_dashboard(dashboard: QuestionDashboard) -> QuestionLogDashboardResponse:
    reasons = dashboard.withheld_reason_counts
    return QuestionLogDashboardResponse(
        questionCount=dashboard.question_count,
        unanswerableCount=dashboard.unanswerable_count,
        withheldReasonCounts=QuestionLogWithheldReasonCounts(
            insufficientEvidence=reasons.insufficient_evidence,
            ambiguousQuestion=reasons.ambiguous_question,
            outOfScope=reasons.out_of_scope,
            unverifiableAnswer=reasons.unverifiable_answer,
        ),
        frequentSubproblems=[
            QuestionLogFrequentSubproblem(
                subproblemId=item.subproblem_id,
                name=item.name,
                documentId=item.document_id,
                documentTitle=item.document_title,
                questionCount=item.question_count,
            )
            for item in dashboard.frequent_subproblems
        ],
        withheldDocuments=[
            QuestionLogWithheldDocument(
                documentId=item.document_id,
                documentTitle=item.document_title,
                insufficientEvidenceCount=item.insufficient_evidence_count,
            )
            for item in dashboard.withheld_documents
        ],
    )


def _to_document_list(documents: DocumentList) -> QuestionLogDocumentListResponse:
    return QuestionLogDocumentListResponse(
        items=[
            QuestionLogDocumentItem(
                documentId=item.document_id,
                documentTitle=item.document_title,
                questionCount=item.question_count,
                withheldCount=item.withheld_count,
                subproblemCount=item.subproblem_count,
            )
            for item in documents.items
        ],
        noDocumentQuestionCount=documents.no_document_question_count,
        unclassifiedQuestionCount=documents.unclassified_question_count,
    )


def _to_document_detail(detail: DocumentDetail) -> QuestionLogDocumentDetailResponse:
    summary = detail.summary
    return QuestionLogDocumentDetailResponse(
        document=QuestionLogDocumentRef(
            documentId=detail.document.document_id,
            documentTitle=detail.document.document_title,
        ),
        summary=QuestionLogDocumentSummary(
            questionCount=summary.question_count,
            insufficientEvidenceCount=summary.insufficient_evidence_count,
            cachedAnswerCount=summary.cached_answer_count,
            subproblemCount=summary.subproblem_count,
        ),
        subproblems=[
            QuestionLogSubproblemItem(
                subproblemId=item.subproblem_id,
                name=item.name,
                questionCount=item.question_count,
                sourceSection=item.source_section,
                applyStatus=item.apply_status,
            )
            for item in detail.subproblems
        ],
    )


def _to_subproblem_detail(detail: SubproblemDetail) -> QuestionLogSubproblemDetailResponse:
    canonical = detail.canonical_answer
    return QuestionLogSubproblemDetailResponse(
        subproblemId=detail.subproblem_id,
        name=detail.name,
        applyStatus=detail.apply_status,
        canonicalAnswer=(
            None
            if canonical is None
            else QuestionLogCanonicalAnswer(
                contentMarkdown=canonical.content_markdown,
                applicabilityRules=list(canonical.applicability_rules),
            )
        ),
    )


def _to_question_page(page: QuestionPage) -> QuestionLogQuestionListResponse:
    return QuestionLogQuestionListResponse(
        items=[
            QuestionLogQuestionItem(
                ragRunId=item.rag_run_id,
                question=item.question,
                documentId=item.document_id,
                documentTitle=item.document_title,
                subproblemId=item.subproblem_id,
                subproblemName=item.subproblem_name,
                askedAt=item.asked_at,
                answerStatus=item.answer_status,
                withheldReason=item.withheld_reason,
            )
            for item in page.items
        ],
        page=page.page,
        size=page.size,
        totalCount=page.total_count,
    )
