"""운영자 콘솔 질문 로그의 집계와 목록을 조회한다.

모든 수치는 한 문서 그룹 안에서 기간 없이 누적한다. 규칙은 2-253 구현 계획 0절(D1~D24)이다.

- 범위: 턴의 색인 판이 속한 문서 그룹(rag_runs.index_version_id → index_versions.document_group_id).
  상태는 COMPLETED, WITHHELD, ERROR 만 센다. PROCESSING 과 CANCELLED 는 뺀다.
- 귀속: 턴마다 현재 분류 행(effective_to IS NULL) 하나만 본다. 그 행의 문제 그룹이
  이 문서 그룹 문서의 DOCUMENT 문제 그룹이면 그 문서, 아니면(가이드 밖) 문서 없음이다.
- 답변 상태: 턴 상태와 캐시 시도 결과로 계산한다. 파이썬 매핑(resolve_answer_status)과
  SQL 조건(answer_status_condition, withheld_reason_condition)을 이 모듈에 함께 둔다.

조회만 한다. 세션에 쓰거나 commit 하지 않는다.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple, Union

from sqlalchemy import Select, and_, collate, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.admin.question_insights.schema import (
    AnswerStatus,
    ApplyStatus,
    SubproblemPresence,
    WithheldReason,
)
from app.admin.schema import AdminErrorCode
from app.database.models import AnswerStatus as RunStatus
from app.database.models import (
    CacheAttemptOutcome,
    CanonicalAnswer,
    CanonicalAnswerApproval,
    CanonicalAnswerCitation,
    DocumentGroup,
    DocumentSource,
    IndexVersion,
    QuestionCacheAttempt,
    QuestionClassification,
    QuestionProblemGroup,
    QuestionProblemGroupKind,
    QuestionSubproblem,
    QuestionSubproblemStatus,
    RagRun,
)
from app.document.ingestion_service import (
    AdminApiError,
    DocumentGroupNotFoundError,
    DocumentNotFoundError,
)
from app.question_grouping.catalog_reader import (
    CatalogDataError,
    applicability_rules_from_json,
)


logger = logging.getLogger(__name__)

# 콘솔 집계 대상 턴 상태. 진행 중과 취소 턴은 질문 수에 넣지 않는다.
COUNTED_RUN_STATUSES = (RunStatus.COMPLETED, RunStatus.WITHHELD, RunStatus.ERROR)

TOP_RANK_LIMIT = 5
DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MAX_QUERY_LENGTH = 100

# content_nodes.node_path 와 인용 스냅샷이 절 제목을 잇는 구분자.
SECTION_PATH_SEPARATOR = " > "
# ILIKE 패턴의 이스케이프 문자.
LIKE_ESCAPE = "\\"
# 정렬용 이름 비교는 환경 collation 과 무관하게 코드 포인트 순서로 고정한다.
BINARY_COLLATION = "C"


# ---------------------------------------------------------------------------
# 오류
# ---------------------------------------------------------------------------


class InvalidQuestionLogRequestError(AdminApiError):
    """질문 로그 조회 조건이 올바르지 않다."""

    def __init__(self, message: str) -> None:
        super().__init__(
            AdminErrorCode.INVALID_REQUEST.value,
            message,
            HTTPStatus.UNPROCESSABLE_ENTITY,
        )


class SubproblemNotFoundError(AdminApiError):
    """세부 문제가 없거나, 보관됐거나, 그 문서 아래에 있지 않다."""

    def __init__(self) -> None:
        super().__init__(
            AdminErrorCode.NOT_FOUND.value,
            "존재하지 않는 세부 문제입니다.",
            HTTPStatus.NOT_FOUND,
        )


# ---------------------------------------------------------------------------
# 답변 상태
# ---------------------------------------------------------------------------


class ResolvedAnswerStatus(NamedTuple):
    answer_status: AnswerStatus
    # answer_status 가 WITHHELD 이고 사유 코드가 알려진 값일 때만 채운다.
    withheld_reason: Optional[WithheldReason]


def _enum_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    return value.value if hasattr(value, "value") else str(value)


def resolve_answer_status(
    status: Union[RunStatus, str],
    withheld_reason_code: Optional[str],
    cache_outcome: Optional[Union[CacheAttemptOutcome, str]],
) -> ResolvedAnswerStatus:
    """턴 상태, 보류 사유 코드, 캐시 시도 결과를 답변 상태 배지로 바꾼다.

    COMPLETED 는 캐시 시도가 SERVED 일 때만 CACHED_ANSWER 이고 나머지(시도 없음 포함)는
    ANSWERED 다. 집계 대상이 아닌 상태(PROCESSING, CANCELLED)는 ValueError 다.
    answer_status_condition 과 같은 규칙이다.
    """

    status_value = _enum_value(status)
    if status_value == RunStatus.COMPLETED.value:
        if _enum_value(cache_outcome) == CacheAttemptOutcome.SERVED.value:
            return ResolvedAnswerStatus(AnswerStatus.CACHED_ANSWER, None)
        return ResolvedAnswerStatus(AnswerStatus.ANSWERED, None)
    if status_value == RunStatus.WITHHELD.value:
        try:
            reason = (
                None
                if withheld_reason_code is None
                else WithheldReason(withheld_reason_code)
            )
        except ValueError:
            logger.warning("알 수 없는 보류 사유 코드입니다: %s", withheld_reason_code)
            reason = None
        return ResolvedAnswerStatus(AnswerStatus.WITHHELD, reason)
    if status_value == RunStatus.ERROR.value:
        return ResolvedAnswerStatus(AnswerStatus.ERROR, None)
    raise ValueError(f"질문 로그 집계 대상이 아닌 턴 상태입니다: {status_value}")


def answer_status_condition(
    answer_status: AnswerStatus,
    run_status: ColumnElement[Any],
    cache_outcome: ColumnElement[Any],
) -> ColumnElement[bool]:
    """답변 상태 하나에 해당하는 턴의 SQL 조건. resolve_answer_status 와 같은 규칙이다."""

    if answer_status is AnswerStatus.ANSWERED:
        return and_(
            run_status == RunStatus.COMPLETED,
            cache_outcome.is_distinct_from(CacheAttemptOutcome.SERVED),
        )
    if answer_status is AnswerStatus.CACHED_ANSWER:
        return and_(
            run_status == RunStatus.COMPLETED,
            cache_outcome == CacheAttemptOutcome.SERVED,
        )
    if answer_status is AnswerStatus.WITHHELD:
        return run_status == RunStatus.WITHHELD
    if answer_status is AnswerStatus.ERROR:
        return run_status == RunStatus.ERROR
    raise ValueError(f"알 수 없는 답변 상태입니다: {answer_status}")


def withheld_reason_condition(
    reason: WithheldReason,
    run_status: ColumnElement[Any],
    withheld_reason_code: ColumnElement[Any],
) -> ColumnElement[bool]:
    """보류 사유 하나에 해당하는 턴의 SQL 조건."""

    return and_(
        run_status == RunStatus.WITHHELD,
        withheld_reason_code == reason.value,
    )


# ---------------------------------------------------------------------------
# 입력
# ---------------------------------------------------------------------------


def _coerce_enum(enum_cls: Any, value: Any, field: str) -> Any:
    if value is None or isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError:
        raise InvalidQuestionLogRequestError(f"{field} 값이 올바르지 않습니다.") from None


def _coerce_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidQuestionLogRequestError(f"{field} 는 정수여야 합니다.")
    return value


@dataclass(frozen=True)
class QuestionListFilters:
    """질문 목록 조회 조건. 생성할 때 정규화하고 검증한다.

    - q 는 앞뒤 공백을 걷는다. 비면 검색하지 않고, 100자를 넘으면 오류다.
    - page 는 1 이상, size 는 1~100 이다.
    - subproblem_presence ABSENT 와 subproblem_id 는 함께 줄 수 없다.
    """

    answer_status: Optional[AnswerStatus] = None
    document_id: Optional[int] = None
    subproblem_presence: Optional[SubproblemPresence] = None
    subproblem_id: Optional[uuid.UUID] = None
    q: Optional[str] = None
    page: int = DEFAULT_PAGE
    size: int = DEFAULT_PAGE_SIZE

    def __post_init__(self) -> None:
        answer_status = _coerce_enum(AnswerStatus, self.answer_status, "answerStatus")
        presence = _coerce_enum(
            SubproblemPresence, self.subproblem_presence, "subproblemPresence"
        )
        document_id = (
            None
            if self.document_id is None
            else _coerce_int(self.document_id, "documentId")
        )
        subproblem_id = self.subproblem_id
        if subproblem_id is not None and not isinstance(subproblem_id, uuid.UUID):
            try:
                subproblem_id = uuid.UUID(str(subproblem_id))
            except ValueError:
                raise InvalidQuestionLogRequestError(
                    "subproblemId 값이 올바르지 않습니다."
                ) from None

        q = self.q
        if q is not None:
            if not isinstance(q, str):
                raise InvalidQuestionLogRequestError("q 는 문자열이어야 합니다.")
            q = q.strip() or None
            if q is not None and len(q) > MAX_QUERY_LENGTH:
                raise InvalidQuestionLogRequestError(
                    f"검색어는 {MAX_QUERY_LENGTH}자 이하로 입력해 주세요."
                )

        page = _coerce_int(self.page, "page")
        if page < 1:
            raise InvalidQuestionLogRequestError("page 는 1 이상이어야 합니다.")
        size = _coerce_int(self.size, "size")
        if not 1 <= size <= MAX_PAGE_SIZE:
            raise InvalidQuestionLogRequestError(
                f"size 는 1 이상 {MAX_PAGE_SIZE} 이하여야 합니다."
            )

        if presence is SubproblemPresence.ABSENT and subproblem_id is not None:
            raise InvalidQuestionLogRequestError(
                "세부 문제 없음과 세부 문제를 함께 지정할 수 없습니다."
            )

        object.__setattr__(self, "answer_status", answer_status)
        object.__setattr__(self, "subproblem_presence", presence)
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "subproblem_id", subproblem_id)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "page", page)
        object.__setattr__(self, "size", size)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


def escape_like(value: str) -> str:
    """ILIKE 패턴에서 %, _, \\ 를 글자 그대로 찾도록 이스케이프한다."""

    return (
        value.replace(LIKE_ESCAPE, LIKE_ESCAPE * 2)
        .replace("%", LIKE_ESCAPE + "%")
        .replace("_", LIKE_ESCAPE + "_")
    )


def compose_source_section(
    document_title: Optional[str], node_path: Optional[str]
) -> Optional[str]:
    """인용 스냅샷으로 "문서명 > 절" 을 만든다.

    node_path 는 보통 문서 제목부터 시작하므로 첫 칸이 문서 제목과 같으면 한 번만 쓴다.
    """

    title = (document_title or "").strip()
    parts = [
        part.strip()
        for part in (node_path or "").split(SECTION_PATH_SEPARATOR)
        if part.strip()
    ]
    if title and parts and parts[0] == title:
        parts = parts[1:]
    if title:
        parts = [title] + parts
    return SECTION_PATH_SEPARATOR.join(parts) if parts else None


def _canonical_rules(value: Any, canonical_answer_id: Any) -> List[str]:
    try:
        return list(applicability_rules_from_json(value))
    except CatalogDataError:
        # 게이트는 같은 데이터를 CANONICAL_DATA_INVALID 로 거부한다. 조회 화면은 막지 않는다.
        logger.warning(
            "정본 적용 범위 규칙의 모양이 올바르지 않아 비워 보냅니다: canonical_answer_id=%s",
            canonical_answer_id,
        )
        return []


# ---------------------------------------------------------------------------
# 결과
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WithheldReasonCounts:
    insufficient_evidence: int
    ambiguous_question: int
    out_of_scope: int
    unverifiable_answer: int


@dataclass(frozen=True)
class FrequentSubproblem:
    subproblem_id: uuid.UUID
    name: str
    document_id: Optional[int]
    document_title: Optional[str]
    question_count: int


@dataclass(frozen=True)
class WithheldDocument:
    document_id: int
    document_title: str
    insufficient_evidence_count: int


@dataclass(frozen=True)
class QuestionDashboard:
    question_count: int
    unanswerable_count: int
    withheld_reason_counts: WithheldReasonCounts
    frequent_subproblems: List[FrequentSubproblem]
    withheld_documents: List[WithheldDocument]


@dataclass(frozen=True)
class DocumentRow:
    document_id: int
    document_title: str
    question_count: int
    withheld_count: int
    subproblem_count: int


@dataclass(frozen=True)
class DocumentList:
    items: List[DocumentRow]
    no_document_question_count: int
    unclassified_question_count: int


@dataclass(frozen=True)
class DocumentRef:
    document_id: int
    document_title: str


@dataclass(frozen=True)
class DocumentSummary:
    question_count: int
    insufficient_evidence_count: int
    cached_answer_count: int
    subproblem_count: int


@dataclass(frozen=True)
class SubproblemRow:
    subproblem_id: uuid.UUID
    name: str
    question_count: int
    source_section: Optional[str]
    apply_status: ApplyStatus


@dataclass(frozen=True)
class DocumentDetail:
    document: DocumentRef
    summary: DocumentSummary
    subproblems: List[SubproblemRow]


@dataclass(frozen=True)
class CanonicalAnswerView:
    content_markdown: str
    applicability_rules: List[str]


@dataclass(frozen=True)
class SubproblemDetail:
    subproblem_id: uuid.UUID
    name: str
    apply_status: ApplyStatus
    canonical_answer: Optional[CanonicalAnswerView]


@dataclass(frozen=True)
class QuestionRow:
    rag_run_id: uuid.UUID
    question: str
    document_id: Optional[int]
    document_title: Optional[str]
    subproblem_id: Optional[uuid.UUID]
    subproblem_name: Optional[str]
    asked_at: datetime
    answer_status: AnswerStatus
    withheld_reason: Optional[WithheldReason]


@dataclass(frozen=True)
class QuestionPage:
    items: List[QuestionRow]
    page: int
    size: int
    total_count: int


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


def document_title_expression(document: Any) -> ColumnElement[Any]:
    """문서 제목. 비어 있으면 document_key 를 쓴다."""

    return func.coalesce(func.nullif(func.btrim(document.title), ""), document.document_key)


def question_text_expression() -> ColumnElement[Any]:
    """표시 질문. 재작성 질문이 비어 있지 않으면 그것을, 아니면 사용자 원문을 쓴다."""

    return func.coalesce(
        func.nullif(func.btrim(RagRun.resolved_query), ""), RagRun.user_query
    )


def scoped_turns_query(group_id: int) -> Select:
    """문서 그룹의 집계 대상 턴 한 건당 한 행.

    캐시 시도(턴당 많아야 하나)와 현재 분류 행(턴당 많아야 하나)을 LEFT JOIN 하므로 행이
    늘지 않는다. document_id 는 현재 분류 행의 문제 그룹이 이 문서 그룹 문서의 DOCUMENT
    문제 그룹일 때만 채운다.
    """

    return (
        select(
            RagRun.id.label("rag_run_id"),
            RagRun.created_at.label("created_at"),
            RagRun.status.label("status"),
            RagRun.withheld_reason_code.label("withheld_reason_code"),
            RagRun.user_query.label("user_query"),
            question_text_expression().label("question"),
            QuestionCacheAttempt.outcome.label("cache_outcome"),
            QuestionClassification.id.label("classification_id"),
            QuestionClassification.subproblem_id.label("subproblem_id"),
            QuestionSubproblem.name.label("subproblem_name"),
            DocumentSource.id.label("document_id"),
            document_title_expression(DocumentSource).label("document_title"),
        )
        .select_from(RagRun)
        .join(IndexVersion, IndexVersion.id == RagRun.index_version_id)
        .outerjoin(QuestionCacheAttempt, QuestionCacheAttempt.rag_run_id == RagRun.id)
        .outerjoin(
            QuestionClassification,
            and_(
                QuestionClassification.rag_run_id == RagRun.id,
                QuestionClassification.effective_to.is_(None),
            ),
        )
        .outerjoin(
            QuestionSubproblem,
            QuestionSubproblem.id == QuestionClassification.subproblem_id,
        )
        .outerjoin(
            QuestionProblemGroup,
            and_(
                QuestionProblemGroup.id == QuestionClassification.problem_group_id,
                QuestionProblemGroup.kind == QuestionProblemGroupKind.DOCUMENT,
            ),
        )
        .outerjoin(
            DocumentSource,
            and_(
                DocumentSource.id == QuestionProblemGroup.document_source_id,
                DocumentSource.document_group_id == group_id,
            ),
        )
        .where(
            IndexVersion.document_group_id == group_id,
            RagRun.status.in_(COUNTED_RUN_STATUSES),
        )
    )


def _question_conditions(turns: Any, filters: QuestionListFilters) -> List[ColumnElement[bool]]:
    conditions: List[ColumnElement[bool]] = []
    if filters.answer_status is not None:
        conditions.append(
            answer_status_condition(
                filters.answer_status, turns.c.status, turns.c.cache_outcome
            )
        )
    if filters.document_id is not None:
        conditions.append(turns.c.document_id == filters.document_id)
    if filters.subproblem_presence is SubproblemPresence.PRESENT:
        conditions.append(turns.c.subproblem_id.is_not(None))
    elif filters.subproblem_presence is SubproblemPresence.ABSENT:
        conditions.append(turns.c.subproblem_id.is_(None))
    if filters.subproblem_id is not None:
        conditions.append(turns.c.subproblem_id == filters.subproblem_id)
    if filters.q is not None:
        pattern = f"%{escape_like(filters.q)}%"
        conditions.append(
            or_(
                turns.c.question.ilike(pattern, escape=LIKE_ESCAPE),
                turns.c.user_query.ilike(pattern, escape=LIKE_ESCAPE),
            )
        )
    return conditions


def question_list_statements(group_id: int, filters: QuestionListFilters) -> Tuple[Select, Select]:
    """질문 목록의 건수 조회와 페이지 조회. EXPLAIN 에도 쓴다."""

    turns = scoped_turns_query(group_id).cte("scoped_turns")
    conditions = _question_conditions(turns, filters)
    count_statement = select(func.count()).select_from(turns).where(*conditions)
    page_statement = (
        select(
            turns.c.rag_run_id,
            turns.c.question,
            turns.c.document_id,
            turns.c.document_title,
            turns.c.subproblem_id,
            turns.c.subproblem_name,
            turns.c.created_at,
            turns.c.status,
            turns.c.withheld_reason_code,
            turns.c.cache_outcome,
        )
        .where(*conditions)
        .order_by(turns.c.created_at.desc(), turns.c.rag_run_id.desc())
        .limit(filters.size)
        .offset(filters.offset)
    )
    return count_statement, page_statement


# ---------------------------------------------------------------------------
# 서비스
# ---------------------------------------------------------------------------


class QuestionInsightService:
    """질문 로그 콘솔 조회. 쓰지 않는다."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_dashboard(self, group_id: int) -> QuestionDashboard:
        await self._require_group(group_id)
        turns = scoped_turns_query(group_id).cte("scoped_turns")

        def reason_count(reason: WithheldReason) -> Any:
            return func.count().filter(
                withheld_reason_condition(
                    reason, turns.c.status, turns.c.withheld_reason_code
                )
            )

        totals = (
            await self._session.execute(
                select(
                    func.count().label("question_count"),
                    func.count()
                    .filter(
                        answer_status_condition(
                            AnswerStatus.WITHHELD, turns.c.status, turns.c.cache_outcome
                        )
                    )
                    .label("unanswerable_count"),
                    reason_count(WithheldReason.INSUFFICIENT_EVIDENCE).label("ie"),
                    reason_count(WithheldReason.AMBIGUOUS_QUESTION).label("aq"),
                    reason_count(WithheldReason.OUT_OF_SCOPE).label("oos"),
                    reason_count(WithheldReason.UNVERIFIABLE_ANSWER).label("ua"),
                ).select_from(turns)
            )
        ).one()

        subproblem_turns = scoped_turns_query(group_id).cte("scoped_turns")
        subproblem_count = func.count()
        frequent_rows = (
            await self._session.execute(
                select(
                    QuestionSubproblem.id,
                    QuestionSubproblem.name,
                    DocumentSource.id.label("document_id"),
                    document_title_expression(DocumentSource).label("document_title"),
                    subproblem_count.label("question_count"),
                )
                .select_from(subproblem_turns)
                .join(
                    QuestionSubproblem,
                    QuestionSubproblem.id == subproblem_turns.c.subproblem_id,
                )
                .join(
                    QuestionProblemGroup,
                    QuestionProblemGroup.id == QuestionSubproblem.problem_group_id,
                )
                .outerjoin(
                    DocumentSource,
                    and_(
                        QuestionProblemGroup.kind == QuestionProblemGroupKind.DOCUMENT,
                        DocumentSource.id == QuestionProblemGroup.document_source_id,
                        DocumentSource.document_group_id == group_id,
                    ),
                )
                .where(QuestionSubproblem.status != QuestionSubproblemStatus.ARCHIVED)
                .group_by(
                    QuestionSubproblem.id,
                    QuestionSubproblem.name,
                    DocumentSource.id,
                    DocumentSource.title,
                    DocumentSource.document_key,
                )
                .order_by(
                    subproblem_count.desc(),
                    collate(QuestionSubproblem.name, BINARY_COLLATION),
                    QuestionSubproblem.id,
                )
                .limit(TOP_RANK_LIMIT)
            )
        ).all()

        document_turns = scoped_turns_query(group_id).cte("scoped_turns")
        insufficient = func.count().filter(
            withheld_reason_condition(
                WithheldReason.INSUFFICIENT_EVIDENCE,
                document_turns.c.status,
                document_turns.c.withheld_reason_code,
            )
        )
        document_questions = func.count()
        withheld_rows = (
            await self._session.execute(
                select(
                    document_turns.c.document_id,
                    document_turns.c.document_title,
                    insufficient.label("insufficient_evidence_count"),
                )
                .where(document_turns.c.document_id.is_not(None))
                .group_by(document_turns.c.document_id, document_turns.c.document_title)
                .having(insufficient > 0)
                .order_by(
                    insufficient.desc(),
                    document_questions.desc(),
                    collate(document_turns.c.document_title, BINARY_COLLATION),
                    document_turns.c.document_id,
                )
                .limit(TOP_RANK_LIMIT)
            )
        ).all()

        return QuestionDashboard(
            question_count=totals.question_count,
            unanswerable_count=totals.unanswerable_count,
            withheld_reason_counts=WithheldReasonCounts(
                insufficient_evidence=totals.ie,
                ambiguous_question=totals.aq,
                out_of_scope=totals.oos,
                unverifiable_answer=totals.ua,
            ),
            frequent_subproblems=[
                FrequentSubproblem(
                    subproblem_id=row.id,
                    name=row.name,
                    document_id=row.document_id,
                    document_title=row.document_title,
                    question_count=row.question_count,
                )
                for row in frequent_rows
            ],
            withheld_documents=[
                WithheldDocument(
                    document_id=row.document_id,
                    document_title=row.document_title,
                    insufficient_evidence_count=row.insufficient_evidence_count,
                )
                for row in withheld_rows
            ],
        )

    async def list_documents(self, group_id: int) -> DocumentList:
        await self._require_group(group_id)
        turns = scoped_turns_query(group_id).cte("scoped_turns")

        question_counts = (
            select(
                turns.c.document_id,
                func.count().label("question_count"),
                func.count()
                .filter(
                    answer_status_condition(
                        AnswerStatus.WITHHELD, turns.c.status, turns.c.cache_outcome
                    )
                )
                .label("withheld_count"),
            )
            .where(turns.c.document_id.is_not(None))
            .group_by(turns.c.document_id)
            .subquery("question_counts")
        )
        subproblem_counts = _active_subproblem_counts()
        question_count = func.coalesce(question_counts.c.question_count, 0)
        withheld_count = func.coalesce(question_counts.c.withheld_count, 0)
        subproblem_count = func.coalesce(subproblem_counts.c.subproblem_count, 0)
        title = document_title_expression(DocumentSource)

        rows = (
            await self._session.execute(
                select(
                    DocumentSource.id.label("document_id"),
                    title.label("document_title"),
                    question_count.label("question_count"),
                    withheld_count.label("withheld_count"),
                    subproblem_count.label("subproblem_count"),
                )
                .select_from(QuestionProblemGroup)
                .join(
                    DocumentSource,
                    and_(
                        DocumentSource.id == QuestionProblemGroup.document_source_id,
                        DocumentSource.document_group_id == group_id,
                    ),
                )
                .outerjoin(
                    question_counts,
                    question_counts.c.document_id == DocumentSource.id,
                )
                .outerjoin(
                    subproblem_counts,
                    subproblem_counts.c.problem_group_id == QuestionProblemGroup.id,
                )
                .where(
                    QuestionProblemGroup.kind == QuestionProblemGroupKind.DOCUMENT,
                    or_(question_count > 0, subproblem_count > 0),
                )
                .order_by(
                    question_count.desc(),
                    withheld_count.desc(),
                    collate(title, BINARY_COLLATION),
                    DocumentSource.id,
                )
            )
        ).all()

        aux_turns = scoped_turns_query(group_id).cte("scoped_turns")
        aux = (
            await self._session.execute(
                select(
                    func.count()
                    .filter(
                        aux_turns.c.classification_id.is_not(None),
                        aux_turns.c.document_id.is_(None),
                    )
                    .label("no_document"),
                    func.count()
                    .filter(aux_turns.c.classification_id.is_(None))
                    .label("unclassified"),
                ).select_from(aux_turns)
            )
        ).one()

        return DocumentList(
            items=[
                DocumentRow(
                    document_id=row.document_id,
                    document_title=row.document_title,
                    question_count=row.question_count,
                    withheld_count=row.withheld_count,
                    subproblem_count=row.subproblem_count,
                )
                for row in rows
            ],
            no_document_question_count=aux.no_document,
            unclassified_question_count=aux.unclassified,
        )

    async def get_document_detail(self, group_id: int, document_id: int) -> DocumentDetail:
        await self._require_group(group_id)
        document = await self._require_document(group_id, document_id)

        turns = scoped_turns_query(group_id).cte("scoped_turns")
        summary = (
            await self._session.execute(
                select(
                    func.count().label("question_count"),
                    func.count()
                    .filter(
                        withheld_reason_condition(
                            WithheldReason.INSUFFICIENT_EVIDENCE,
                            turns.c.status,
                            turns.c.withheld_reason_code,
                        )
                    )
                    .label("insufficient_evidence_count"),
                    func.count()
                    .filter(
                        answer_status_condition(
                            AnswerStatus.CACHED_ANSWER,
                            turns.c.status,
                            turns.c.cache_outcome,
                        )
                    )
                    .label("cached_answer_count"),
                )
                .select_from(turns)
                .where(turns.c.document_id == document_id)
            )
        ).one()

        problem_group_id = await self._document_problem_group_id(document_id)
        subproblems: List[SubproblemRow] = []
        if problem_group_id is not None:
            subproblems = await self._document_subproblems(group_id, problem_group_id)

        return DocumentDetail(
            document=document,
            summary=DocumentSummary(
                question_count=summary.question_count,
                insufficient_evidence_count=summary.insufficient_evidence_count,
                cached_answer_count=summary.cached_answer_count,
                subproblem_count=len(subproblems),
            ),
            subproblems=subproblems,
        )

    async def get_subproblem_detail(
        self, group_id: int, document_id: int, subproblem_id: uuid.UUID
    ) -> SubproblemDetail:
        await self._require_group(group_id)
        await self._require_document(group_id, document_id)

        row = (
            await self._session.execute(
                select(
                    QuestionSubproblem.id,
                    QuestionSubproblem.name,
                    CanonicalAnswer.id.label("canonical_answer_id"),
                    CanonicalAnswer.content_markdown,
                    CanonicalAnswer.applicability_rules,
                )
                .select_from(QuestionSubproblem)
                .join(
                    QuestionProblemGroup,
                    QuestionProblemGroup.id == QuestionSubproblem.problem_group_id,
                )
                .outerjoin(
                    CanonicalAnswer,
                    and_(
                        CanonicalAnswer.subproblem_id == QuestionSubproblem.id,
                        CanonicalAnswer.approval == CanonicalAnswerApproval.APPROVED,
                    ),
                )
                .where(
                    QuestionSubproblem.id == subproblem_id,
                    QuestionSubproblem.status != QuestionSubproblemStatus.ARCHIVED,
                    QuestionProblemGroup.kind == QuestionProblemGroupKind.DOCUMENT,
                    QuestionProblemGroup.document_source_id == document_id,
                )
            )
        ).one_or_none()
        if row is None:
            raise SubproblemNotFoundError()

        canonical = None
        if row.canonical_answer_id is not None:
            canonical = CanonicalAnswerView(
                content_markdown=row.content_markdown,
                applicability_rules=_canonical_rules(
                    row.applicability_rules, row.canonical_answer_id
                ),
            )
        return SubproblemDetail(
            subproblem_id=row.id,
            name=row.name,
            apply_status=(
                ApplyStatus.NEEDS_CANONICAL if canonical is None else ApplyStatus.APPLIED
            ),
            canonical_answer=canonical,
        )

    async def list_questions(
        self, group_id: int, filters: Optional[QuestionListFilters] = None
    ) -> QuestionPage:
        filters = filters or QuestionListFilters()
        await self._require_group(group_id)

        count_statement, page_statement = question_list_statements(group_id, filters)
        total_count = int(await self._session.scalar(count_statement) or 0)
        rows: Sequence[Any] = []
        if filters.offset < total_count:
            rows = (await self._session.execute(page_statement)).all()

        items = []
        for row in rows:
            resolved = resolve_answer_status(
                row.status, row.withheld_reason_code, row.cache_outcome
            )
            items.append(
                QuestionRow(
                    rag_run_id=row.rag_run_id,
                    question=row.question,
                    document_id=row.document_id,
                    document_title=row.document_title,
                    subproblem_id=row.subproblem_id,
                    subproblem_name=row.subproblem_name,
                    asked_at=row.created_at,
                    answer_status=resolved.answer_status,
                    withheld_reason=resolved.withheld_reason,
                )
            )
        return QuestionPage(
            items=items,
            page=filters.page,
            size=filters.size,
            total_count=total_count,
        )

    async def _require_group(self, group_id: int) -> None:
        found = await self._session.scalar(
            select(DocumentGroup.id).where(DocumentGroup.id == group_id)
        )
        if found is None:
            raise DocumentGroupNotFoundError()

    async def _require_document(self, group_id: int, document_id: int) -> DocumentRef:
        row = (
            await self._session.execute(
                select(
                    DocumentSource.id,
                    document_title_expression(DocumentSource).label("document_title"),
                ).where(
                    DocumentSource.id == document_id,
                    DocumentSource.document_group_id == group_id,
                )
            )
        ).one_or_none()
        if row is None:
            raise DocumentNotFoundError()
        return DocumentRef(document_id=row.id, document_title=row.document_title)

    async def _document_problem_group_id(self, document_id: int) -> Optional[uuid.UUID]:
        return await self._session.scalar(
            select(QuestionProblemGroup.id).where(
                QuestionProblemGroup.kind == QuestionProblemGroupKind.DOCUMENT,
                QuestionProblemGroup.document_source_id == document_id,
            )
        )

    async def _document_subproblems(
        self, group_id: int, problem_group_id: uuid.UUID
    ) -> List[SubproblemRow]:
        turns = scoped_turns_query(group_id).cte("scoped_turns")
        counts = (
            select(turns.c.subproblem_id, func.count().label("question_count"))
            .where(turns.c.subproblem_id.is_not(None))
            .group_by(turns.c.subproblem_id)
            .subquery("subproblem_question_counts")
        )
        question_count = func.coalesce(counts.c.question_count, 0)
        rows = (
            await self._session.execute(
                select(
                    QuestionSubproblem.id,
                    QuestionSubproblem.name,
                    question_count.label("question_count"),
                    CanonicalAnswer.id.label("canonical_answer_id"),
                    CanonicalAnswerCitation.document_title_snapshot,
                    CanonicalAnswerCitation.node_path_snapshot,
                )
                .select_from(QuestionSubproblem)
                .outerjoin(counts, counts.c.subproblem_id == QuestionSubproblem.id)
                .outerjoin(
                    CanonicalAnswer,
                    and_(
                        CanonicalAnswer.subproblem_id == QuestionSubproblem.id,
                        CanonicalAnswer.approval == CanonicalAnswerApproval.APPROVED,
                    ),
                )
                .outerjoin(
                    CanonicalAnswerCitation,
                    and_(
                        CanonicalAnswerCitation.canonical_answer_id == CanonicalAnswer.id,
                        CanonicalAnswerCitation.citation_order == 1,
                    ),
                )
                .where(
                    QuestionSubproblem.problem_group_id == problem_group_id,
                    QuestionSubproblem.status != QuestionSubproblemStatus.ARCHIVED,
                )
                .order_by(
                    question_count.desc(),
                    collate(QuestionSubproblem.name, BINARY_COLLATION),
                    QuestionSubproblem.id,
                )
            )
        ).all()
        return [
            SubproblemRow(
                subproblem_id=row.id,
                name=row.name,
                question_count=row.question_count,
                source_section=(
                    None
                    if row.canonical_answer_id is None
                    else compose_source_section(
                        row.document_title_snapshot, row.node_path_snapshot
                    )
                ),
                apply_status=(
                    ApplyStatus.NEEDS_CANONICAL
                    if row.canonical_answer_id is None
                    else ApplyStatus.APPLIED
                ),
            )
            for row in rows
        ]


def _active_subproblem_counts() -> Any:
    """문제 그룹마다 보관하지 않은 세부 문제 수."""

    return (
        select(
            QuestionSubproblem.problem_group_id,
            func.count().label("subproblem_count"),
        )
        .where(QuestionSubproblem.status != QuestionSubproblemStatus.ARCHIVED)
        .group_by(QuestionSubproblem.problem_group_id)
        .subquery("active_subproblem_counts")
    )
