"""세부 문제와 정본을 문서 그룹에 적재하는 시드 스크립트.

실행(기본 dry-run, 쓰기 없음·임베딩 API 호출 없음):

    python -m app.ops.seed_question_grouping --group-key HELP_CHATBOT_TEST \\
        --input "<subproblems.json 또는 폴더>" --actor <이름> \\
        [--apply] [--include-draft] [--serving-state UNUSED|SHADOW|SERVING|STOPPED]

입력은 세부 문제 그룹핑 결과의 subproblems.json(문서 하나) 형식이다. 폴더를 주면 아래의
subproblems.json 을 모두 읽는다.

규칙 요약:
- reviewStatus approved 만 적재한다. --include-draft 면 draft 도 적재한다(테스트 환경용).
  rejected 는 적재하지 않는다.
- 세부 문제는 파일의 document.path 문서의 DOCUMENT 문제 그룹에 둔다. 인용 [1] 의 문서는
  그 문서여야 한다. key 는 대상 문서 그룹 전체에서 유일해야 한다.
- 문서 경로("폴더/문서.md")는 GitBook 문서 키("폴더/문서", build_gitbook_document_key 와
  같은 규칙)로 대상 문서 그룹의 document_sources 를 찾는다. 절 경로는 ACTIVE 색인 판에 든
  문서 판의 청크 중 청킹 설정이 색인 판과 같고 문서 제목을 뺀 절 경로가 같은 청크로 찾는다.
- 세부 문제는 (문서 문제 그룹, key) 로 식별한다. 이름·포함·제외 기준이 같으면 개정을 만들지
  않는다. 다르면 current_version + 1 개정을 만들고 포함 기준 임베딩을 계산한다.
- 정본은 (본문, 적용 범위, 인용(순서, 문서 키, 절 경로), 세부 문제 판) 해시가 같으면 두고,
  다르면 기존 APPROVED 를 REVOKED 로 내린 뒤 새 APPROVED 정본을 넣는다. 해석한 청크 id·문서 판
  id 는 해시에 넣지 않는다. 재색인으로 청크 id 만 바뀐 경우 정본을 교체하지 않고, 서빙 게이트가
  R17 로 현재 색인의 절을 찾는다. 새로 넣는 정본의 canonical_answer_citations 에는 이번에 해석한
  청크 id·문서 판 id 를 쓴다.
- 입력에 없는 기존 세부 문제는 건드리지 않고 보고만 한다.
- 모든 검증을 통과해야 쓴다. --apply 는 한 트랜잭션으로 쓰고 commit 한다.
- 임베딩 호출은 model_calls 에 남기지 않는다(소유 조합 제약상 턴·실행에 속하지 않는 호출).
  토큰 사용량은 로그로만 남긴다.

종료 코드: 0 성공(dry-run 검증 통과 또는 적용 완료), 1 검증 실패(쓰기 없음),
2 인자·입력 경로 오류, 3 실행 중 예외(롤백).
"""

import argparse
import asyncio
import enum
import json
import logging
import re
import sys
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from sqlalchemy import and_, delete, func, or_, select, text, update
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.error_message import sanitize_error_message
from app.core.hashing import sha256_hex
from app.database.models import (
    EMBEDDING_DIMENSIONS,
    CanonicalAnswer,
    CanonicalAnswerApproval,
    CanonicalAnswerCitation,
    CanonicalAnswerOrigin,
    ContentNode,
    DocumentChunk,
    DocumentGroup,
    DocumentSource,
    DocumentVersion,
    EmbeddingConfig,
    IndexDocument,
    IndexVersion,
    IndexVersionStatus,
    QuestionProblemGroup,
    QuestionProblemGroupKind,
    QuestionSubproblem,
    QuestionSubproblemRevision,
    QuestionSubproblemServingState,
    QuestionSubproblemStatus,
    ExactQuestionMatchSource,
    ExactQuestionMatchState,
    RecommendedQuestion,
)
from app.question_grouping.canonical_validation import (
    check_canonical_citations,
    check_canonical_content,
)
from app.question_grouping.catalog_reader import CatalogDataError, applicability_rules_from_json
from app.question_grouping.constants import HEADING_SEPARATOR, SUBPROBLEM_EMBEDDING_TEXT_VERSION
from app.question_grouping.payload import split_criteria
from app.question_grouping.recommended import normalize_recommended_question
from app.question_grouping.store import QuestionGroupingStore
from app.question_grouping.subproblem_search import build_subproblem_embedding_text
from app.retrieval.embedding import (
    OPENAI_EMBEDDING_DIMENSIONS,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_PROVIDER,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
INPUT_FILE_NAME = "subproblems.json"
DOCUMENT_PATH_SUFFIX = ".md"

REVIEW_APPROVED = "approved"
REVIEW_DRAFT = "draft"
REVIEW_REJECTED = "rejected"
REVIEW_STATUSES = (REVIEW_APPROVED, REVIEW_DRAFT, REVIEW_REJECTED)

KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
MAX_KEY_LENGTH = 200
MAX_NAME_LENGTH = 200
MAX_ACTOR_LENGTH = 100
CRITERIA_SEPARATOR = "\n"
APPLICABILITY_RULES_FIELD = "rules"
CANONICAL_HASH_VERSION = 2
EMBEDDING_BATCH_SIZE = 64
ADVISORY_LOCK_PREFIX = "seed_question_grouping:"

EXIT_OK = 0
EXIT_VALIDATION_FAILED = 1
EXIT_USAGE = 2
EXIT_RUNTIME_ERROR = 3

# 검증 오류 코드(쓰기를 막는다)
ERROR_INPUT_READ = "INPUT_READ_FAILED"
ERROR_SCHEMA = "SCHEMA_INVALID"
ERROR_REVIEW_STATUS = "REVIEW_STATUS_INVALID"
ERROR_DUPLICATE_DOCUMENT = "DUPLICATE_DOCUMENT"
ERROR_DOCUMENT_PATH = "DOCUMENT_PATH_INVALID"
ERROR_KEY = "KEY_INVALID"
ERROR_DUPLICATE_KEY = "DUPLICATE_KEY"
ERROR_NAME = "NAME_INVALID"
ERROR_CRITERIA = "CRITERIA_INVALID"
ERROR_CITATION_ORDER = "CITATION_ORDER_INVALID"
ERROR_CANONICAL_CITATIONS = "CANONICAL_CITATION_MISMATCH"
ERROR_ANSWER_CONTENT = "ANSWER_CONTENT_INVALID"
ERROR_PLACEMENT = "PLACEMENT_MISMATCH"
ERROR_GROUP_NOT_FOUND = "DOCUMENT_GROUP_NOT_FOUND"
ERROR_ACTIVE_INDEX_NOT_FOUND = "ACTIVE_INDEX_NOT_FOUND"
ERROR_EMBEDDING_CONFIG = "EMBEDDING_CONFIG_UNSUPPORTED"
ERROR_DOCUMENT_NOT_IN_GROUP = "DOCUMENT_NOT_IN_GROUP"
ERROR_DOCUMENT_AMBIGUOUS = "DOCUMENT_AMBIGUOUS"
ERROR_DOCUMENT_DISABLED = "DOCUMENT_DISABLED"
ERROR_DOCUMENT_NOT_INDEXED = "DOCUMENT_NOT_INDEXED"
ERROR_SECTION_NOT_FOUND = "SECTION_NOT_FOUND"
ERROR_SECTION_AMBIGUOUS = "SECTION_AMBIGUOUS"
ERROR_KEY_IN_OTHER_PROBLEM_GROUP = "KEY_IN_OTHER_PROBLEM_GROUP"
ERROR_SUBPROBLEM_STATUS = "SUBPROBLEM_NOT_APPROVED_IN_DB"
ERROR_CURRENT_REVISION_MISSING = "CURRENT_REVISION_MISSING"
ERROR_RECOMMENDED_QUESTION = "RECOMMENDED_QUESTION_INVALID"
ERROR_RECOMMENDED_QUESTION_DUPLICATE = "RECOMMENDED_QUESTION_DUPLICATE"
ERROR_RECOMMENDED_QUESTION_OWNERSHIP = "RECOMMENDED_QUESTION_OWNERSHIP_INVALID"

# 경고 코드(쓰기를 막지 않는다)
WARNING_CONTENT_HASH = "DOCUMENT_CONTENT_HASH_MISMATCH"
WARNING_TITLE = "DOCUMENT_TITLE_MISMATCH"
WARNING_EVIDENCE = "EVIDENCE_NOT_FOUND"
WARNING_INCLUDE_DRAFT = "DRAFT_INCLUDED"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# 입력 모델과 보고 항목
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedIssue:
    """검증 오류나 경고 한 건."""

    code: str
    message: str
    source_file: Optional[str] = None
    document_path: Optional[str] = None
    key: Optional[str] = None

    def render(self) -> str:
        where = [part for part in (self.document_path, self.key and f"key={self.key}") if part]
        if not where and self.source_file:
            where.append(self.source_file)
        location = f" {' '.join(where)}" if where else ""
        return f"[{self.code}]{location}: {self.message}"


@dataclass(frozen=True)
class SeedCitation:
    order: int
    document_path: str
    section_path: Tuple[str, ...]
    evidence: Optional[str] = None


@dataclass(frozen=True)
class SeedSubproblem:
    key: str
    name: str
    inclusion_criteria: Tuple[str, ...]
    exclusion_criteria: Tuple[str, ...]
    content_markdown: str
    applicability_rules: Tuple[str, ...]
    citations: Tuple[SeedCitation, ...]
    review_status: str
    legacy_id: Optional[str] = None
    # None means this input does not manage existing mappings; () explicitly removes them.
    recommended_questions: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class SeedDocument:
    """subproblems.json 한 파일. subproblems 는 이번 실행에 적재할 항목만 담는다."""

    source_file: str
    path: str
    title: Optional[str]
    parent_path: Optional[str]
    content_sha256: Optional[str]
    subproblems: Tuple[SeedSubproblem, ...]
    skipped_draft_keys: Tuple[str, ...] = ()
    skipped_rejected_keys: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedInput:
    documents: Tuple[SeedDocument, ...]
    errors: Tuple[SeedIssue, ...] = ()

    @property
    def selected_count(self) -> int:
        return sum(len(document.subproblems) for document in self.documents)


# ---------------------------------------------------------------------------
# 입력 읽기와 파싱
# ---------------------------------------------------------------------------


def find_input_files(input_path: Path) -> List[Path]:
    """파일이면 그 파일, 폴더면 아래의 subproblems.json 전부(경로 순)."""

    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob(INPUT_FILE_NAME))
    raise FileNotFoundError(f"입력 경로가 없습니다: {input_path}")


def load_input(input_path: Path, *, include_draft: bool) -> ParsedInput:
    files = find_input_files(input_path)
    if not files:
        return ParsedInput(
            documents=(),
            errors=(SeedIssue(ERROR_INPUT_READ, f"{INPUT_FILE_NAME} 파일이 없습니다: {input_path}"),),
        )
    raw: List[Tuple[str, Any]] = []
    errors: List[SeedIssue] = []
    for path in files:
        try:
            raw.append((str(path), json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            errors.append(SeedIssue(ERROR_INPUT_READ, f"JSON 을 읽지 못했습니다: {error}", source_file=str(path)))
    parsed = parse_input(raw, include_draft=include_draft)
    return ParsedInput(documents=parsed.documents, errors=tuple(errors) + parsed.errors)


def parse_input(raw_documents: Sequence[Tuple[str, Any]], *, include_draft: bool) -> ParsedInput:
    """(파일 이름, JSON 값) 목록을 파싱하고 DB 없이 할 수 있는 검증을 모두 한다."""

    documents: List[SeedDocument] = []
    errors: List[SeedIssue] = []
    for source_file, data in raw_documents:
        document, document_errors = parse_document(data, source_file, include_draft=include_draft)
        errors.extend(document_errors)
        if document is not None:
            documents.append(document)
    errors.extend(validate_documents(documents))
    return ParsedInput(documents=tuple(documents), errors=tuple(errors))


class _SchemaError(ValueError):
    pass


def _require(mapping: Mapping[str, Any], name: str, kind: type, where: str, *, optional: bool = False) -> Any:
    if name not in mapping or mapping[name] is None:
        if optional:
            return None
        raise _SchemaError(f"{where}.{name} 이 없습니다.")
    value = mapping[name]
    if kind is int and isinstance(value, bool):
        raise _SchemaError(f"{where}.{name} 은 정수여야 합니다.")
    if not isinstance(value, kind):
        raise _SchemaError(f"{where}.{name} 의 형식이 {kind.__name__} 이 아닙니다.")
    return value


def _string_list(mapping: Mapping[str, Any], name: str, where: str, *, optional: bool = False) -> Tuple[str, ...]:
    value = _require(mapping, name, list, where, optional=optional)
    if value is None:
        return ()
    if not all(isinstance(item, str) for item in value):
        raise _SchemaError(f"{where}.{name} 은 문자열 목록이어야 합니다.")
    return tuple(value)


def parse_document(
    data: Any,
    source_file: str,
    *,
    include_draft: bool,
) -> Tuple[Optional[SeedDocument], List[SeedIssue]]:
    """subproblems.json 한 파일. 적재 대상이 아닌 항목(draft·rejected)은 형식 검사도 하지 않는다."""

    errors: List[SeedIssue] = []
    try:
        if not isinstance(data, dict):
            raise _SchemaError("최상위 값은 객체여야 합니다.")
        version = _require(data, "schemaVersion", int, "$")
        if version != SCHEMA_VERSION:
            raise _SchemaError(f"schemaVersion {version} 은 지원하지 않습니다(지원: {SCHEMA_VERSION}).")
        document = _require(data, "document", dict, "$")
        path = _require(document, "path", str, "document")
        title = _require(document, "title", str, "document", optional=True)
        parent_path = _require(document, "parentPath", str, "document", optional=True)
        content_sha256 = _require(document, "contentSha256", str, "document", optional=True)
        items = _require(data, "subproblems", list, "$")
    except _SchemaError as error:
        return None, [SeedIssue(ERROR_SCHEMA, str(error), source_file=source_file)]

    selected: List[SeedSubproblem] = []
    skipped_draft: List[str] = []
    skipped_rejected: List[str] = []
    for index, item in enumerate(items):
        where = f"subproblems[{index}]"
        key = item.get("key") if isinstance(item, dict) else None
        key_label = key if isinstance(key, str) else None
        status = item.get("reviewStatus") if isinstance(item, dict) else None
        if status not in REVIEW_STATUSES:
            errors.append(
                SeedIssue(
                    ERROR_REVIEW_STATUS,
                    f"{where}.reviewStatus 는 {'/'.join(REVIEW_STATUSES)} 중 하나여야 합니다: {status!r}",
                    source_file=source_file,
                    document_path=path,
                    key=key_label,
                )
            )
            continue
        if status == REVIEW_REJECTED:
            skipped_rejected.append(key_label or where)
            continue
        if status == REVIEW_DRAFT and not include_draft:
            skipped_draft.append(key_label or where)
            continue
        try:
            selected.append(_parse_subproblem(item, where))
        except _SchemaError as error:
            errors.append(
                SeedIssue(ERROR_SCHEMA, str(error), source_file=source_file, document_path=path, key=key_label)
            )
    return (
        SeedDocument(
            source_file=source_file,
            path=path,
            title=title,
            parent_path=parent_path,
            content_sha256=content_sha256,
            subproblems=tuple(selected),
            skipped_draft_keys=tuple(skipped_draft),
            skipped_rejected_keys=tuple(skipped_rejected),
        ),
        errors,
    )


def _parse_subproblem(item: Mapping[str, Any], where: str) -> SeedSubproblem:
    canonical = _require(item, "canonical", dict, where)
    citations = []
    for index, citation in enumerate(_require(canonical, "citations", list, f"{where}.canonical")):
        citation_where = f"{where}.canonical.citations[{index}]"
        if not isinstance(citation, dict):
            raise _SchemaError(f"{citation_where} 는 객체여야 합니다.")
        citations.append(
            SeedCitation(
                order=_require(citation, "order", int, citation_where),
                document_path=_require(citation, "documentPath", str, citation_where),
                section_path=_string_list(citation, "sectionPath", citation_where),
                evidence=_require(citation, "evidence", str, citation_where, optional=True),
            )
        )
    rules = _string_list(canonical, "applicabilityRules", f"{where}.canonical", optional=True)
    return SeedSubproblem(
        key=_require(item, "key", str, where),
        name=_require(item, "name", str, where).strip(),
        inclusion_criteria=tuple(criterion.strip() for criterion in _string_list(item, "inclusionCriteria", where)),
        exclusion_criteria=tuple(
            criterion.strip() for criterion in _string_list(item, "exclusionCriteria", where, optional=True)
        ),
        content_markdown=_require(canonical, "contentMarkdown", str, f"{where}.canonical").strip(),
        applicability_rules=tuple(rule.strip() for rule in rules if rule.strip()),
        citations=tuple(sorted(citations, key=lambda citation: citation.order)),
        review_status=item["reviewStatus"],
        legacy_id=_require(item, "legacyId", str, where, optional=True),
        recommended_questions=(
            None
            if "recommendedQuestions" not in item
            else _string_list(item, "recommendedQuestions", where)
        ),
    )


# ---------------------------------------------------------------------------
# DB 없이 하는 검증
# ---------------------------------------------------------------------------


def join_criteria(criteria: Sequence[str]) -> Optional[str]:
    """기준 목록을 DB text 칸(한 줄에 하나)으로 합친다. 빈 목록은 널."""

    return CRITERIA_SEPARATOR.join(criteria) if criteria else None


def document_key_from_path(document_path: str) -> str:
    """data/clean 기준 문서 경로("폴더/문서.md")를 GitBook 문서 키("폴더/문서")로 바꾼다."""

    return document_path[: -len(DOCUMENT_PATH_SUFFIX)] if document_path.endswith(DOCUMENT_PATH_SUFFIX) else document_path


def document_path_error(document_path: str) -> Optional[str]:
    if not document_path.endswith(DOCUMENT_PATH_SUFFIX) or len(document_path) <= len(DOCUMENT_PATH_SUFFIX):
        return f"문서 경로는 '.md' 로 끝나는 상대 경로여야 합니다: {document_path!r}"
    if document_path.startswith("/") or "\\" in document_path:
        return f"문서 경로는 '/' 로 시작하지 않는 상대 경로여야 합니다: {document_path!r}"
    if any(part in ("", ".", "..") for part in document_path.split("/")):
        return f"문서 경로에 빈 조각이나 '.', '..' 가 있습니다: {document_path!r}"
    return None


def key_error(key: str) -> Optional[str]:
    if len(key) > MAX_KEY_LENGTH:
        return f"key 는 {MAX_KEY_LENGTH}자 이하여야 합니다: {len(key)}자"
    if not KEY_PATTERN.match(key):
        return "key 는 소문자 영문·숫자로 시작하고 소문자 영문·숫자·점·하이픈만 써야 합니다."
    return None


def criteria_errors(label: str, criteria: Sequence[str], *, required: bool) -> List[str]:
    """한 줄에 하나로 저장하고 split_criteria 로 되읽어 같아야 한다."""

    problems = []
    if required and not criteria:
        problems.append(f"{label} 이 비어 있습니다.")
    for index, criterion in enumerate(criteria, 1):
        if not criterion:
            problems.append(f"{label} {index}번이 비어 있습니다.")
        elif "\n" in criterion or "\r" in criterion:
            problems.append(f"{label} {index}번에 줄바꿈이 있습니다.")
    if not problems and criteria and split_criteria(join_criteria(criteria)) != tuple(criteria):
        problems.append(f"{label} 을 한 줄에 하나로 저장하면 되읽은 값이 달라집니다(앞의 '-', '*' 기호 등).")
    return problems


def validate_subproblem(document_path: str, subproblem: SeedSubproblem, source_file: Optional[str] = None) -> List[SeedIssue]:
    issues: List[SeedIssue] = []

    def issue(code: str, message: str) -> None:
        issues.append(SeedIssue(code, message, source_file=source_file, document_path=document_path, key=subproblem.key))

    problem = key_error(subproblem.key)
    if problem:
        issue(ERROR_KEY, problem)
    if not subproblem.name or len(subproblem.name) > MAX_NAME_LENGTH:
        issue(ERROR_NAME, f"이름은 1~{MAX_NAME_LENGTH}자여야 합니다.")
    for message in criteria_errors("포함 기준", subproblem.inclusion_criteria, required=True):
        issue(ERROR_CRITERIA, message)
    for message in criteria_errors("제외 기준", subproblem.exclusion_criteria, required=False):
        issue(ERROR_CRITERIA, message)

    orders = [citation.order for citation in subproblem.citations]
    check = check_canonical_citations(subproblem.content_markdown, orders)
    if not check.valid:
        issue(
            ERROR_CANONICAL_CITATIONS,
            f"본문 [n] {sorted(check.body_numbers)} 과 인용 번호 {sorted(orders)} 가 맞지 않습니다: "
            f"{', '.join(check.errors)}",
        )
    if orders and sorted(orders) != list(range(1, len(orders) + 1)):
        issue(ERROR_CITATION_ORDER, f"인용 번호는 1..{len(orders)} 여야 합니다: {sorted(orders)}")
    content_problem = check_canonical_content(subproblem.content_markdown)
    if content_problem:
        issue(ERROR_ANSWER_CONTENT, content_problem)
    for citation in subproblem.citations:
        path_problem = document_path_error(citation.document_path)
        if path_problem:
            issue(ERROR_DOCUMENT_PATH, f"인용 [{citation.order}] {path_problem}")
        if any(not part.strip() for part in citation.section_path):
            issue(ERROR_SCHEMA, f"인용 [{citation.order}] 절 경로에 빈 제목이 있습니다.")
    if subproblem.recommended_questions is not None:
        for index, question in enumerate(subproblem.recommended_questions):
            if not question.strip():
                issue(ERROR_RECOMMENDED_QUESTION, f"추천 질문 {index + 1}번이 비어 있습니다.")
    first = next((citation for citation in subproblem.citations if citation.order == 1), None)
    if first is not None and first.document_path != document_path:
        issue(
            ERROR_PLACEMENT,
            f"인용 [1] 의 문서({first.document_path})가 세부 문제 문서({document_path})와 다릅니다.",
        )
    return issues


def validate_documents(documents: Sequence[SeedDocument]) -> List[SeedIssue]:
    issues: List[SeedIssue] = []
    files_by_path: Dict[str, List[str]] = {}
    documents_by_key: Dict[str, List[str]] = {}
    recommended_by_normalized: Dict[str, List[Tuple[str, str, str]]] = {}
    for document in documents:
        files_by_path.setdefault(document.path, []).append(document.source_file)
        path_problem = document_path_error(document.path)
        if path_problem:
            issues.append(SeedIssue(ERROR_DOCUMENT_PATH, path_problem, source_file=document.source_file))
        for subproblem in document.subproblems:
            documents_by_key.setdefault(subproblem.key, []).append(document.path)
            issues.extend(validate_subproblem(document.path, subproblem, document.source_file))
            if subproblem.recommended_questions is not None:
                for question in subproblem.recommended_questions:
                    if not question.strip():
                        continue
                    normalized = normalize_recommended_question(question)
                    recommended_by_normalized.setdefault(normalized, []).append(
                        (document.path, subproblem.key, question)
                    )
    for path, files in sorted(files_by_path.items()):
        if len(files) > 1:
            issues.append(SeedIssue(ERROR_DUPLICATE_DOCUMENT, f"같은 문서가 여러 파일에 있습니다: {files}", document_path=path))
    for key, paths in sorted(documents_by_key.items()):
        if len(paths) > 1:
            issues.append(
                SeedIssue(ERROR_DUPLICATE_KEY, f"입력 안에서 key 가 겹칩니다: {paths}", key=key)
            )
    for normalized, entries in sorted(recommended_by_normalized.items()):
        if len(entries) > 1:
            labels = [f"{path}:{key}" for path, key, _ in entries]
            issues.append(
                SeedIssue(
                    ERROR_RECOMMENDED_QUESTION_DUPLICATE,
                    f"정규화한 추천 질문이 입력에서 겹칩니다: {normalized!r} ({labels})",
                    document_path=entries[0][0],
                    key=entries[0][1],
                )
            )
    return issues


# ---------------------------------------------------------------------------
# 계획(순수 함수)
# ---------------------------------------------------------------------------


class SubproblemAction(str, enum.Enum):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    UNCHANGED = "UNCHANGED"


class EmbeddingAction(str, enum.Enum):
    NONE = "NONE"
    NEW_REVISION = "NEW_REVISION"
    REFRESH_CURRENT_REVISION = "REFRESH_CURRENT_REVISION"


class CanonicalAction(str, enum.Enum):
    CREATE = "CREATE"
    REPLACE = "REPLACE"
    UNCHANGED = "UNCHANGED"


@dataclass(frozen=True)
class ExistingRevision:
    revision_id: int
    version: int
    has_embedding: bool
    embedding_config_id: Optional[int]
    embedding_text_version: Optional[str]


# 정본 해시의 인용 한 건: (순서, 문서 키, 문서 제목을 뺀 절 경로)
CanonicalCitationKey = Tuple[int, str, Tuple[str, ...]]


@dataclass(frozen=True)
class ExistingCanonical:
    canonical_answer_id: uuid.UUID
    content_markdown: str
    # 저장된 적용 범위를 읽지 못하면 None(해시가 달라져 교체된다).
    applicability_rules: Optional[Tuple[str, ...]]
    subproblem_version: int
    citations: Tuple[CanonicalCitationKey, ...]  # (citation_order, document_key, section_path)


@dataclass(frozen=True)
class ExistingSubproblem:
    subproblem_id: uuid.UUID
    problem_group_id: uuid.UUID
    problem_group_kind: QuestionProblemGroupKind
    document_source_id: Optional[int]
    document_key: Optional[str]
    key: str
    name: str
    inclusion_criteria: str
    exclusion_criteria: Optional[str]
    current_version: int
    status: QuestionSubproblemStatus
    serving_state: QuestionSubproblemServingState
    current_revision: Optional[ExistingRevision] = None
    approved_canonical: Optional[ExistingCanonical] = None


@dataclass(frozen=True)
class ExistingRecommendedQuestion:
    mapping_id: int
    subproblem_id: uuid.UUID
    question: str
    normalized_question: str
    source: ExactQuestionMatchSource = ExactQuestionMatchSource.RECOMMENDED
    state: ExactQuestionMatchState = ExactQuestionMatchState.ACTIVE


class RecommendedQuestionAction(str, enum.Enum):
    NONE = "NONE"
    UPSERT = "UPSERT"
    DELETE = "DELETE"


@dataclass(frozen=True)
class RecommendedQuestionPlan:
    document_group_id: int
    subproblem_key: str
    subproblem_id: Optional[uuid.UUID]
    questions: Optional[Tuple[str, ...]]
    existing: Tuple[ExistingRecommendedQuestion, ...]
    action: RecommendedQuestionAction

    @property
    def normalized_questions(self) -> Tuple[str, ...]:
        if self.questions is None:
            return ()
        return tuple(normalize_recommended_question(q) for q in self.questions)

    @property
    def writes(self) -> bool:
        return self.action != RecommendedQuestionAction.NONE


@dataclass(frozen=True)
class ResolvedCitation:
    order: int
    chunk_id: int
    document_version_id: int
    document_source_id: int
    document_title: Optional[str]
    node_path: Optional[str]
    source_uri: Optional[str]
    document_key: str
    section_path: Tuple[str, ...]

    @property
    def hash_key(self) -> CanonicalCitationKey:
        return (self.order, self.document_key, self.section_path)


@dataclass(frozen=True)
class SubproblemPlan:
    document_path: str
    document_source_id: Optional[int]
    seed: SeedSubproblem
    existing: Optional[ExistingSubproblem]
    action: SubproblemAction
    definition_changes: Tuple[str, ...]
    target_version: int
    embedding_action: EmbeddingAction
    embedding_text: str
    serving_state: QuestionSubproblemServingState
    canonical_action: CanonicalAction
    canonical_hash: str
    citations: Tuple[ResolvedCitation, ...]

    @property
    def serving_state_changed(self) -> bool:
        return self.existing is not None and self.existing.serving_state != self.serving_state

    @property
    def writes(self) -> bool:
        return (
            self.action != SubproblemAction.UNCHANGED
            or self.embedding_action != EmbeddingAction.NONE
            or self.canonical_action != CanonicalAction.UNCHANGED
            or self.serving_state_changed
        )


def canonical_content_hash(
    content_markdown: str,
    applicability_rules: Optional[Sequence[str]],
    citations: Iterable[Tuple[int, str, Sequence[str]]],
    subproblem_version: int,
) -> str:
    """정본 교체 판단 해시.

    인용은 (순서, 문서 키, 절 경로) 를 순서로 정렬해 쓴다. 문서 키는 "폴더/문서.md" 경로를 줘도
    "폴더/문서" 로 맞춘다. 청크 id·문서 판 id 는 넣지 않는다(재색인에 흔들리지 않게).
    """

    normalized = sorted(
        (int(order), document_key_from_path(document), list(section_path))
        for order, document, section_path in citations
    )
    payload = {
        "hashVersion": CANONICAL_HASH_VERSION,
        "contentMarkdown": content_markdown.strip(),
        "applicabilityRules": None if applicability_rules is None else list(applicability_rules),
        "citations": [list(citation) for citation in normalized],
        "subproblemVersion": subproblem_version,
    }
    return sha256_hex(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def existing_canonical_hash(canonical: ExistingCanonical) -> str:
    return canonical_content_hash(
        canonical.content_markdown,
        canonical.applicability_rules,
        canonical.citations,
        canonical.subproblem_version,
    )


def definition_changes(seed: SeedSubproblem, existing: ExistingSubproblem) -> Tuple[str, ...]:
    changes = []
    if existing.name.strip() != seed.name:
        changes.append("name")
    if split_criteria(existing.inclusion_criteria) != seed.inclusion_criteria:
        changes.append("inclusion")
    if split_criteria(existing.exclusion_criteria) != seed.exclusion_criteria:
        changes.append("exclusion")
    return tuple(changes)


def plan_subproblem(
    document_path: str,
    seed: SeedSubproblem,
    existing: Optional[ExistingSubproblem],
    citations: Sequence[ResolvedCitation],
    *,
    embedding_config_id: int,
    serving_state: Optional[QuestionSubproblemServingState],
    document_source_id: Optional[int] = None,
    embedding_text_version: str = SUBPROBLEM_EMBEDDING_TEXT_VERSION,
) -> SubproblemPlan:
    """입력 세부 문제 하나와 기존 상태로 할 일을 정한다.

    - 새 세부 문제: 판 1, 개정 임베딩 계산, serving_state 는 옵션값 또는 UNUSED.
    - 정의(이름·포함·제외 기준)가 다르면 판 + 1 과 새 개정. 같으면 개정을 만들지 않되,
      현재 개정 임베딩이 없거나 설정·문장 판이 다르면 그 개정의 임베딩만 다시 계산한다.
    - 기존 serving_state 는 옵션을 명시했을 때만 바꾼다.
    - 정본 해시는 목표 판(target_version)으로 계산하므로 판이 오르면 정본도 교체된다.
    """

    embedding_text = build_subproblem_embedding_text(
        seed.name, seed.inclusion_criteria, text_version=embedding_text_version
    )
    if existing is None:
        action = SubproblemAction.CREATE
        changes: Tuple[str, ...] = ()
        target_version = 1
        embedding_action = EmbeddingAction.NEW_REVISION
        state = serving_state or QuestionSubproblemServingState.UNUSED
    else:
        changes = definition_changes(seed, existing)
        state = serving_state or existing.serving_state
        if changes:
            action = SubproblemAction.UPDATE
            target_version = existing.current_version + 1
            embedding_action = EmbeddingAction.NEW_REVISION
        else:
            action = SubproblemAction.UNCHANGED
            target_version = existing.current_version
            revision = existing.current_revision
            if revision is None:
                raise ValueError(f"현재 개정이 없는 세부 문제입니다: {existing.key}")
            stale = (
                not revision.has_embedding
                or revision.embedding_config_id != embedding_config_id
                or revision.embedding_text_version != embedding_text_version
            )
            embedding_action = EmbeddingAction.REFRESH_CURRENT_REVISION if stale else EmbeddingAction.NONE

    ordered = tuple(sorted(citations, key=lambda citation: citation.order))
    desired_hash = canonical_content_hash(
        seed.content_markdown,
        seed.applicability_rules,
        [citation.hash_key for citation in ordered],
        target_version,
    )
    current = None if existing is None else existing.approved_canonical
    if current is None:
        canonical_action = CanonicalAction.CREATE
    elif existing_canonical_hash(current) == desired_hash:
        canonical_action = CanonicalAction.UNCHANGED
    else:
        canonical_action = CanonicalAction.REPLACE

    return SubproblemPlan(
        document_path=document_path,
        document_source_id=document_source_id,
        seed=seed,
        existing=existing,
        action=action,
        definition_changes=changes,
        target_version=target_version,
        embedding_action=embedding_action,
        embedding_text=embedding_text,
        serving_state=state,
        canonical_action=canonical_action,
        canonical_hash=desired_hash,
        citations=ordered,
    )


# ---------------------------------------------------------------------------
# DB 읽기
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupContext:
    document_group_id: int
    group_key: str
    index_version_id: int
    chunking_config_id: int
    embedding_config_id: int


@dataclass(frozen=True)
class SourceRow:
    document_source_id: int
    document_key: str
    title: Optional[str]
    canonical_uri: str
    enabled: bool


@dataclass(frozen=True)
class IndexedVersionRow:
    document_version_id: int
    version_no: int
    normalized_content_hash: str


@dataclass(frozen=True)
class SectionRow:
    chunk_id: int
    document_version_id: int
    local_path: Tuple[str, ...]
    node_path: Optional[str]
    content: str


def local_section_path(node_path: Optional[str], metadata: Any) -> Tuple[str, ...]:
    """청크의 문서 제목을 뺀 절 경로. metadata.section_path 가 있으면 그것을, 없으면 node_path 를 쓴다."""

    if isinstance(metadata, dict):
        section_path = metadata.get("section_path")
        if isinstance(section_path, list) and section_path and all(isinstance(p, str) for p in section_path):
            return tuple(section_path[1:])
    if not node_path:
        return ()
    return tuple(node_path.split(HEADING_SEPARATOR)[1:])


_EVIDENCE_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_NON_WORD = re.compile(r"[\W_]+")


def _comparable(value: str) -> str:
    """문자·숫자만 남긴다. 목록 기호, 따옴표 모양, 굵게 표시, 공백 차이를 무시하려는 것이다."""

    return _NON_WORD.sub("", value)


def evidence_found(evidence: str, content: str) -> bool:
    """evidence 의 문장마다 문자·숫자만 남겨 절 본문에 들어 있는지 본다(검토용 경고 판단)."""

    haystack = _comparable(content)
    sentences = [_comparable(sentence) for sentence in _EVIDENCE_SENTENCE_BOUNDARY.split(evidence)]
    sentences = [sentence for sentence in sentences if sentence]
    return bool(sentences) and all(sentence in haystack for sentence in sentences)


@dataclass
class SeedPlan:
    """dry-run 과 apply 가 함께 쓰는 계획. errors 가 있으면 쓰지 않는다."""

    group_key: str
    context: Optional[GroupContext] = None
    documents: List[Tuple[SeedDocument, Optional[int], List[SubproblemPlan]]] = field(default_factory=list)
    errors: List[SeedIssue] = field(default_factory=list)
    warnings: List[SeedIssue] = field(default_factory=list)
    untouched: Dict[str, List[str]] = field(default_factory=dict)
    no_document_group_id: Optional[uuid.UUID] = None
    document_problem_groups: Dict[int, uuid.UUID] = field(default_factory=dict)
    missing_document_problem_groups: List[int] = field(default_factory=list)
    recommended_question_plans: List[RecommendedQuestionPlan] = field(default_factory=list)

    @property
    def subproblem_plans(self) -> List[SubproblemPlan]:
        return [plan for _, _, plans in self.documents for plan in plans]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def embedding_plans(self) -> List[SubproblemPlan]:
        return [plan for plan in self.subproblem_plans if plan.embedding_action != EmbeddingAction.NONE]

    @property
    def recommended_writes(self) -> List[RecommendedQuestionPlan]:
        return [item for item in self.recommended_question_plans if item.writes]


class SeedReader:
    """시드 계획에 필요한 DB 상태를 칼럼 단위로 읽는다. 쓰지 않는다."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def group_context(self, group_key: str, plan: SeedPlan) -> Optional[GroupContext]:
        group_id = await self._session.scalar(select(DocumentGroup.id).where(DocumentGroup.group_key == group_key))
        if group_id is None:
            plan.errors.append(SeedIssue(ERROR_GROUP_NOT_FOUND, f"문서 그룹이 없습니다: {group_key}"))
            return None
        row = (
            await self._session.execute(
                select(
                    IndexVersion.id,
                    IndexVersion.chunking_config_id,
                    IndexVersion.embedding_config_id,
                    EmbeddingConfig.provider,
                    EmbeddingConfig.model_name,
                    EmbeddingConfig.dimensions,
                )
                .join(EmbeddingConfig, EmbeddingConfig.id == IndexVersion.embedding_config_id)
                .where(
                    IndexVersion.document_group_id == group_id,
                    IndexVersion.status == IndexVersionStatus.ACTIVE,
                )
            )
        ).one_or_none()
        if row is None:
            plan.errors.append(SeedIssue(ERROR_ACTIVE_INDEX_NOT_FOUND, f"ACTIVE 색인 판이 없습니다: {group_key}"))
            return None
        if (
            row.provider != OPENAI_EMBEDDING_PROVIDER
            or row.model_name != OPENAI_EMBEDDING_MODEL
            or row.dimensions != OPENAI_EMBEDDING_DIMENSIONS
            or row.dimensions != EMBEDDING_DIMENSIONS
        ):
            plan.errors.append(
                SeedIssue(
                    ERROR_EMBEDDING_CONFIG,
                    "ACTIVE 색인의 임베딩 설정이 시드 임베딩 클라이언트와 다릅니다: "
                    f"{row.provider}/{row.model_name}/{row.dimensions} "
                    f"(클라이언트 {OPENAI_EMBEDDING_PROVIDER}/{OPENAI_EMBEDDING_MODEL}/{OPENAI_EMBEDDING_DIMENSIONS})",
                )
            )
        return GroupContext(
            document_group_id=group_id,
            group_key=group_key,
            index_version_id=row.id,
            chunking_config_id=row.chunking_config_id,
            embedding_config_id=row.embedding_config_id,
        )

    async def sources(self, context: GroupContext) -> List[SourceRow]:
        rows = (
            await self._session.execute(
                select(
                    DocumentSource.id,
                    DocumentSource.document_key,
                    DocumentSource.title,
                    DocumentSource.canonical_uri,
                    DocumentSource.enabled,
                )
                .where(DocumentSource.document_group_id == context.document_group_id)
                .order_by(DocumentSource.document_key, DocumentSource.id)
            )
        ).all()
        return [
            SourceRow(
                document_source_id=row.id,
                document_key=row.document_key,
                title=row.title,
                canonical_uri=row.canonical_uri,
                enabled=row.enabled,
            )
            for row in rows
        ]

    async def indexed_versions(self, context: GroupContext, source_ids: Sequence[int]) -> Dict[int, IndexedVersionRow]:
        if not source_ids:
            return {}
        rows = (
            await self._session.execute(
                select(
                    DocumentVersion.id,
                    DocumentVersion.document_source_id,
                    DocumentVersion.version_no,
                    DocumentVersion.normalized_content_hash,
                )
                .join(
                    IndexDocument,
                    and_(
                        IndexDocument.document_version_id == DocumentVersion.id,
                        IndexDocument.index_version_id == context.index_version_id,
                    ),
                )
                .where(DocumentVersion.document_source_id.in_(sorted(set(source_ids))))
                .order_by(DocumentVersion.document_source_id, DocumentVersion.version_no.desc())
            )
        ).all()
        chosen: Dict[int, IndexedVersionRow] = {}
        for row in rows:
            chosen.setdefault(
                row.document_source_id,
                IndexedVersionRow(row.id, row.version_no, row.normalized_content_hash),
            )
        return chosen

    async def sections(self, context: GroupContext, version_ids: Sequence[int]) -> Dict[int, List[SectionRow]]:
        if not version_ids:
            return {}
        rows = (
            await self._session.execute(
                select(
                    ContentNode.id,
                    ContentNode.document_version_id,
                    ContentNode.node_path,
                    ContentNode.metadata_,
                    ContentNode.normalized_content,
                )
                .join(DocumentChunk, DocumentChunk.id == ContentNode.id)
                .where(
                    ContentNode.document_version_id.in_(sorted(set(version_ids))),
                    DocumentChunk.chunking_config_id == context.chunking_config_id,
                )
                .order_by(ContentNode.document_version_id, ContentNode.node_order, ContentNode.id)
            )
        ).all()
        result: Dict[int, List[SectionRow]] = {}
        for row in rows:
            result.setdefault(row.document_version_id, []).append(
                SectionRow(
                    chunk_id=row.id,
                    document_version_id=row.document_version_id,
                    local_path=local_section_path(row.node_path, row.metadata_),
                    node_path=row.node_path,
                    content=row.normalized_content,
                )
            )
        return result

    async def problem_groups(
        self, context: GroupContext, source_ids: Sequence[int]
    ) -> Tuple[Optional[uuid.UUID], Dict[int, uuid.UUID]]:
        conditions = [QuestionProblemGroup.document_group_id == context.document_group_id]
        if source_ids:
            conditions.append(QuestionProblemGroup.document_source_id.in_(sorted(set(source_ids))))
        rows = (
            await self._session.execute(
                select(
                    QuestionProblemGroup.id,
                    QuestionProblemGroup.kind,
                    QuestionProblemGroup.document_source_id,
                ).where(or_(*conditions))
            )
        ).all()
        no_document = None
        by_source: Dict[int, uuid.UUID] = {}
        for row in rows:
            if row.kind == QuestionProblemGroupKind.NO_DOCUMENT:
                no_document = row.id
            elif row.document_source_id is not None:
                by_source[row.document_source_id] = row.id
        return no_document, by_source

    async def existing_subproblems(self, context: GroupContext) -> List[ExistingSubproblem]:
        subproblem = QuestionSubproblem
        group = QuestionProblemGroup
        rows = (
            await self._session.execute(
                select(
                    subproblem.id,
                    subproblem.problem_group_id,
                    subproblem.key,
                    subproblem.name,
                    subproblem.inclusion_criteria,
                    subproblem.exclusion_criteria,
                    subproblem.current_version,
                    subproblem.status,
                    subproblem.serving_state,
                    group.kind,
                    group.document_source_id,
                    DocumentSource.document_key,
                )
                .join(group, group.id == subproblem.problem_group_id)
                .outerjoin(DocumentSource, DocumentSource.id == group.document_source_id)
                .where(
                    or_(
                        and_(
                            group.kind == QuestionProblemGroupKind.DOCUMENT,
                            DocumentSource.document_group_id == context.document_group_id,
                        ),
                        and_(
                            group.kind == QuestionProblemGroupKind.NO_DOCUMENT,
                            group.document_group_id == context.document_group_id,
                        ),
                    )
                )
                .order_by(subproblem.key, subproblem.id)
            )
        ).all()
        if not rows:
            return []
        ids = [row.id for row in rows]
        revisions = await self._current_revisions(ids)
        canonicals = await self._approved_canonicals(ids)
        return [
            ExistingSubproblem(
                subproblem_id=row.id,
                problem_group_id=row.problem_group_id,
                problem_group_kind=row.kind,
                document_source_id=row.document_source_id,
                document_key=row.document_key,
                key=row.key,
                name=row.name,
                inclusion_criteria=row.inclusion_criteria,
                exclusion_criteria=row.exclusion_criteria,
                current_version=row.current_version,
                status=row.status,
                serving_state=row.serving_state,
                current_revision=revisions.get(row.id),
                approved_canonical=canonicals.get(row.id),
            )
            for row in rows
        ]

    async def recommended_questions(
        self, context: GroupContext
    ) -> List[ExistingRecommendedQuestion]:
        """그룹에 기록된 추천 질문을 읽고 세부 문제 소속을 함께 검증한다."""

        mapping = RecommendedQuestion
        group = QuestionProblemGroup
        source = DocumentSource
        rows = (
            await self._session.execute(
                select(
                    mapping.id,
                    mapping.subproblem_id,
                    mapping.question,
                    mapping.normalized_question,
                    mapping.source,
                    mapping.state,
                    source.document_group_id,
                )
                .join(QuestionSubproblem, QuestionSubproblem.id == mapping.subproblem_id)
                .join(group, group.id == QuestionSubproblem.problem_group_id)
                .outerjoin(source, source.id == group.document_source_id)
                .where(mapping.document_group_id == context.document_group_id)
                .order_by(mapping.normalized_question, mapping.id)
            )
        ).all()
        invalid = [
            row.id
            for row in rows
            if row.document_group_id != context.document_group_id
        ]
        if invalid:
            raise CatalogDataError(
                "추천 질문이 문서 그룹 밖의 세부 문제를 가리킵니다: "
                f"document_group_id={context.document_group_id}, mapping_ids={invalid}"
            )
        return [
            ExistingRecommendedQuestion(
                mapping_id=row.id,
                subproblem_id=row.subproblem_id,
                question=row.question,
                normalized_question=row.normalized_question,
                source=row.source,
                state=row.state,
            )
            for row in rows
        ]

    async def _current_revisions(self, subproblem_ids: Sequence[uuid.UUID]) -> Dict[uuid.UUID, ExistingRevision]:
        revision = QuestionSubproblemRevision
        rows = (
            await self._session.execute(
                select(
                    revision.subproblem_id,
                    revision.id,
                    revision.version,
                    revision.inclusion_embedding.is_not(None).label("has_embedding"),
                    revision.embedding_config_id,
                    revision.embedding_text_version,
                )
                .join(
                    QuestionSubproblem,
                    and_(
                        QuestionSubproblem.id == revision.subproblem_id,
                        QuestionSubproblem.current_version == revision.version,
                    ),
                )
                .where(revision.subproblem_id.in_(subproblem_ids))
            )
        ).all()
        return {
            row.subproblem_id: ExistingRevision(
                revision_id=row.id,
                version=row.version,
                has_embedding=bool(row.has_embedding),
                embedding_config_id=row.embedding_config_id,
                embedding_text_version=row.embedding_text_version,
            )
            for row in rows
        }

    async def _approved_canonicals(self, subproblem_ids: Sequence[uuid.UUID]) -> Dict[uuid.UUID, ExistingCanonical]:
        """승인 정본과 인용의 (순서, 문서 키, 절 경로). 절 경로는 인용 청크(옛 판이어도)에서 읽는다."""

        rows = (
            await self._session.execute(
                select(
                    CanonicalAnswer.id,
                    CanonicalAnswer.subproblem_id,
                    CanonicalAnswer.content_markdown,
                    CanonicalAnswer.applicability_rules,
                    CanonicalAnswer.subproblem_version,
                ).where(
                    CanonicalAnswer.subproblem_id.in_(subproblem_ids),
                    CanonicalAnswer.approval == CanonicalAnswerApproval.APPROVED,
                )
            )
        ).all()
        if not rows:
            return {}
        citation_rows = (
            await self._session.execute(
                select(
                    CanonicalAnswerCitation.canonical_answer_id,
                    CanonicalAnswerCitation.citation_order,
                    ContentNode.node_path,
                    ContentNode.metadata_,
                    DocumentSource.document_key,
                )
                .join(ContentNode, ContentNode.id == CanonicalAnswerCitation.chunk_id)
                .join(DocumentVersion, DocumentVersion.id == CanonicalAnswerCitation.document_version_id)
                .join(DocumentSource, DocumentSource.id == DocumentVersion.document_source_id)
                .where(CanonicalAnswerCitation.canonical_answer_id.in_([row.id for row in rows]))
            )
        ).all()
        citations: Dict[uuid.UUID, List[CanonicalCitationKey]] = {}
        for row in citation_rows:
            citations.setdefault(row.canonical_answer_id, []).append(
                (row.citation_order, row.document_key, local_section_path(row.node_path, row.metadata_))
            )
        result = {}
        for row in rows:
            try:
                rules: Optional[Tuple[str, ...]] = applicability_rules_from_json(row.applicability_rules)
            except CatalogDataError:
                rules = None
            result[row.subproblem_id] = ExistingCanonical(
                canonical_answer_id=row.id,
                content_markdown=row.content_markdown,
                applicability_rules=rules,
                subproblem_version=row.subproblem_version,
                citations=tuple(sorted(citations.get(row.id, []))),
            )
        return result


async def build_seed_plan(
    session: AsyncSession,
    *,
    group_key: str,
    parsed: ParsedInput,
    serving_state: Optional[QuestionSubproblemServingState],
    include_draft: bool = False,
) -> SeedPlan:
    """입력 파싱 결과와 DB 상태로 계획을 세운다. 쓰지 않는다."""

    plan = SeedPlan(group_key=group_key, errors=list(parsed.errors))
    if include_draft:
        plan.warnings.append(
            SeedIssue(WARNING_INCLUDE_DRAFT, "draft 세부 문제도 APPROVED 로 적재합니다. 테스트 환경에서만 쓰세요.")
        )
    reader = SeedReader(session)
    context = await reader.group_context(group_key, plan)
    if context is None:
        return plan
    plan.context = context

    sources = await reader.sources(context)
    sources_by_key: Dict[str, List[SourceRow]] = {}
    for source in sources:
        sources_by_key.setdefault(source.document_key, []).append(source)
    enabled_ids = [source.document_source_id for source in sources if source.enabled]

    indexed = await reader.indexed_versions(context, [source.document_source_id for source in sources])
    cited_versions = set()
    for document in parsed.documents:
        for subproblem in document.subproblems:
            for citation in subproblem.citations:
                for source in sources_by_key.get(document_key_from_path(citation.document_path), []):
                    if source.document_source_id in indexed:
                        cited_versions.add(indexed[source.document_source_id].document_version_id)
    sections = await reader.sections(context, sorted(cited_versions))

    plan.no_document_group_id, plan.document_problem_groups = await reader.problem_groups(
        context, [source.document_source_id for source in sources]
    )
    plan.missing_document_problem_groups = [
        source_id for source_id in enabled_ids if source_id not in plan.document_problem_groups
    ]

    existing = await reader.existing_subproblems(context)
    existing_by_key: Dict[str, List[ExistingSubproblem]] = {}
    for item in existing:
        existing_by_key.setdefault(item.key, []).append(item)

    try:
        existing_recommended = await reader.recommended_questions(context)
    except CatalogDataError as error:
        plan.errors.append(SeedIssue(ERROR_RECOMMENDED_QUESTION_OWNERSHIP, str(error)))
        existing_recommended = []
    recommended_by_subproblem: Dict[uuid.UUID, List[ExistingRecommendedQuestion]] = {}
    for item in existing_recommended:
        recommended_by_subproblem.setdefault(item.subproblem_id, []).append(item)
    existing_recommended_by_normalized = {
        item.normalized_question: item for item in existing_recommended
    }

    def resolve_source(document_path: str, where: SeedIssue) -> Optional[SourceRow]:
        matches = sources_by_key.get(document_key_from_path(document_path), [])
        if not matches:
            plan.errors.append(replace(where, code=ERROR_DOCUMENT_NOT_IN_GROUP, message=f"{where.message}문서가 문서 그룹에 없습니다: {document_path}"))
            return None
        if len(matches) > 1:
            plan.errors.append(
                replace(
                    where,
                    code=ERROR_DOCUMENT_AMBIGUOUS,
                    message=f"{where.message}같은 문서 키의 문서가 여러 개입니다: {document_path} "
                    f"(document_source_id={[m.document_source_id for m in matches]})",
                )
            )
            return None
        source = matches[0]
        if not source.enabled:
            plan.errors.append(replace(where, code=ERROR_DOCUMENT_DISABLED, message=f"{where.message}꺼진 문서입니다: {document_path}"))
            return None
        if source.document_source_id not in indexed:
            plan.errors.append(
                replace(where, code=ERROR_DOCUMENT_NOT_INDEXED, message=f"{where.message}ACTIVE 색인에 문서 판이 없습니다: {document_path}")
            )
            return None
        return source

    input_keys: Set[str] = set()
    for document in parsed.documents:
        if not document.subproblems:
            plan.documents.append((document, None, []))
            continue
        base = SeedIssue("", "", source_file=document.source_file, document_path=document.path)
        document_source = resolve_source(document.path, base)
        if document_source is not None:
            version = indexed[document_source.document_source_id]
            if document.content_sha256 and document.content_sha256 != version.normalized_content_hash:
                plan.warnings.append(
                    replace(
                        base,
                        code=WARNING_CONTENT_HASH,
                        message="입력을 만든 원문 해시가 ACTIVE 색인 문서 판의 정규화 해시와 다릅니다 "
                        f"(document_version_id={version.document_version_id}). 인용 절과 정본 내용을 다시 확인하세요.",
                    )
                )
            if document.title and document_source.title and document.title != document_source.title:
                plan.warnings.append(
                    replace(base, code=WARNING_TITLE, message=f"문서 제목이 다릅니다: 입력 {document.title!r}, DB {document_source.title!r}")
                )

        subproblem_plans: List[SubproblemPlan] = []
        for subproblem in document.subproblems:
            input_keys.add(subproblem.key)
            where = replace(base, key=subproblem.key)
            resolved: List[ResolvedCitation] = []
            for citation in subproblem.citations:
                if document_source is None and citation.document_path == document.path:
                    continue  # 문서 오류를 이미 보고했다.
                citation_where = replace(where, message=f"인용 [{citation.order}] ")
                source = resolve_source(citation.document_path, citation_where)
                if source is None:
                    continue
                version = indexed[source.document_source_id]
                candidates = [
                    section
                    for section in sections.get(version.document_version_id, [])
                    if section.local_path == citation.section_path
                ]
                label = HEADING_SEPARATOR.join(citation.section_path) or "<문서 머리말>"
                if not candidates:
                    available = [
                        HEADING_SEPARATOR.join(section.local_path) or "<문서 머리말>"
                        for section in sections.get(version.document_version_id, [])
                    ]
                    plan.errors.append(
                        replace(
                            where,
                            code=ERROR_SECTION_NOT_FOUND,
                            message=f"인용 [{citation.order}] {citation.document_path} 의 절 '{label}' 을 "
                            f"ACTIVE 색인 청크에서 찾지 못했습니다. 있는 절: {available}",
                        )
                    )
                    continue
                if len(candidates) > 1:
                    plan.errors.append(
                        replace(
                            where,
                            code=ERROR_SECTION_AMBIGUOUS,
                            message=f"인용 [{citation.order}] {citation.document_path} 의 절 '{label}' 청크가 "
                            f"여러 개입니다: {[section.chunk_id for section in candidates]}",
                        )
                    )
                    continue
                section = candidates[0]
                if citation.evidence and not evidence_found(citation.evidence, section.content):
                    plan.warnings.append(
                        replace(
                            where,
                            code=WARNING_EVIDENCE,
                            message=f"인용 [{citation.order}] evidence 문장이 절 '{label}' 본문에 그대로 없습니다.",
                        )
                    )
                resolved.append(
                    ResolvedCitation(
                        order=citation.order,
                        chunk_id=section.chunk_id,
                        document_version_id=section.document_version_id,
                        document_source_id=source.document_source_id,
                        document_title=source.title,
                        node_path=section.node_path,
                        source_uri=source.canonical_uri,
                        document_key=source.document_key,
                        section_path=section.local_path,
                    )
                )

            current = None
            for item in [] if document_source is None else existing_by_key.get(subproblem.key, []):
                if (
                    item.problem_group_kind == QuestionProblemGroupKind.DOCUMENT
                    and item.document_source_id == document_source.document_source_id
                ):
                    current = item
                else:
                    owner = item.document_key if item.problem_group_kind == QuestionProblemGroupKind.DOCUMENT else "NO_DOCUMENT"
                    plan.errors.append(
                        replace(
                            where,
                            code=ERROR_KEY_IN_OTHER_PROBLEM_GROUP,
                            message=f"같은 key 의 세부 문제가 문서 그룹의 다른 문제 그룹({owner})에 이미 있습니다. "
                            "시드는 세부 문제를 옮기지 않습니다.",
                        )
                    )
            if current is not None and current.status != QuestionSubproblemStatus.APPROVED:
                plan.errors.append(
                    replace(
                        where,
                        code=ERROR_SUBPROBLEM_STATUS,
                        message=f"기존 세부 문제 상태가 {current.status.value} 입니다. 시드는 상태를 되돌리지 않습니다.",
                    )
                )
                continue
            if current is not None and current.current_revision is None:
                plan.errors.append(
                    replace(where, code=ERROR_CURRENT_REVISION_MISSING, message=f"현재 판 {current.current_version} 개정 행이 없습니다.")
                )
                continue
            if document_source is None or len(resolved) != len(subproblem.citations):
                continue
            try:
                subproblem_plans.append(
                    plan_subproblem(
                        document.path,
                        subproblem,
                        current,
                        resolved,
                        embedding_config_id=context.embedding_config_id,
                        serving_state=serving_state,
                        document_source_id=document_source.document_source_id,
                    )
                )
            except ValueError as error:
                # 이름·포함 기준 형식 오류는 입력 검증이 이미 보고했다. 그 밖의 경우만 더한다.
                if not any(issue.key == subproblem.key and issue.code in (ERROR_NAME, ERROR_CRITERIA) for issue in plan.errors):
                    plan.errors.append(replace(where, code=ERROR_CRITERIA, message=str(error)))
        plan.documents.append(
            (document, None if document_source is None else document_source.document_source_id, subproblem_plans)
        )

        # 추천 질문은 정본/개정과 독립적으로 관리한다. 필드가 빠진 세부 문제는
        # 기존 매핑을 그대로 두고, 빈 배열만 명시적 삭제로 해석한다.
        plans_by_key = {item.seed.key: item for item in subproblem_plans}
        for subproblem in document.subproblems:
            if subproblem.recommended_questions is None:
                continue
            target = plans_by_key.get(subproblem.key)
            if target is None:
                continue
            target_id = None if target.existing is None else target.existing.subproblem_id
            current_rows = () if target_id is None else tuple(recommended_by_subproblem.get(target_id, ()))
            desired = tuple(normalize_recommended_question(q) for q in subproblem.recommended_questions)
            for normalized in desired:
                owner = existing_recommended_by_normalized.get(normalized)
                if (
                    owner is not None
                    and owner.source == ExactQuestionMatchSource.RECOMMENDED
                    and owner.subproblem_id != target_id
                ):
                    plan.errors.append(
                        SeedIssue(
                            ERROR_RECOMMENDED_QUESTION_DUPLICATE,
                            f"추천 질문이 이미 다른 세부 문제에 매핑되어 있습니다: {normalized!r}",
                            source_file=document.source_file,
                            document_path=document.path,
                            key=subproblem.key,
                        )
                    )
            existing_by_normalized = {
                item.normalized_question: (item.question, item.source)
                for item in current_rows
            }
            desired_by_normalized = {
                normalize_recommended_question(question): question
                for question in subproblem.recommended_questions
            }
            if not desired_by_normalized and current_rows:
                action = RecommendedQuestionAction.DELETE
            elif desired_by_normalized != {
                normalized: question
                for normalized, (question, _source) in existing_by_normalized.items()
            } or any(
                source != ExactQuestionMatchSource.RECOMMENDED
                for _normalized, (_question, source) in existing_by_normalized.items()
                if _normalized in desired_by_normalized
            ):
                action = RecommendedQuestionAction.UPSERT
            else:
                action = RecommendedQuestionAction.NONE
            plan.recommended_question_plans.append(
                RecommendedQuestionPlan(
                    document_group_id=context.document_group_id,
                    subproblem_key=subproblem.key,
                    subproblem_id=target_id,
                    questions=subproblem.recommended_questions,
                    existing=current_rows,
                    action=action,
                )
            )

    for item in existing:
        if item.key in input_keys:
            continue
        owner = item.document_key if item.problem_group_kind == QuestionProblemGroupKind.DOCUMENT else "NO_DOCUMENT"
        plan.untouched.setdefault(owner or "?", []).append(item.key)
    return plan


# ---------------------------------------------------------------------------
# 쓰기
# ---------------------------------------------------------------------------


@dataclass
class ApplyResult:
    problem_groups_created: int = 0
    subproblems_created: int = 0
    subproblems_updated: int = 0
    revisions_created: int = 0
    revisions_embedding_refreshed: int = 0
    serving_state_changed: int = 0
    canonicals_created: int = 0
    canonicals_replaced: int = 0
    embedding_texts: int = 0
    embedding_input_tokens: Optional[int] = None
    recommended_questions_created: int = 0
    recommended_questions_updated: int = 0
    recommended_questions_deleted: int = 0


async def compute_embeddings(embedder: Any, texts: Sequence[str]) -> Tuple[List[List[float]], Optional[int], int]:
    """포함 기준 임베딩을 묶음으로 계산한다. model_calls 에 남기지 않고 사용량만 로그로 남긴다."""

    vectors: List[List[float]] = []
    input_tokens: Optional[int] = None
    retries = 0
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = list(texts[start : start + EMBEDDING_BATCH_SIZE])
        started = time.monotonic()
        response = await asyncio.to_thread(embedder.embed_many_with_usage, batch)
        latency_ms = int((time.monotonic() - started) * 1000)
        if len(response.embeddings) != len(batch):
            raise RuntimeError(f"임베딩 응답 개수가 입력과 다릅니다: 입력 {len(batch)}, 응답 {len(response.embeddings)}")
        for vector in response.embeddings:
            if len(vector) != EMBEDDING_DIMENSIONS:
                raise ValueError(f"세부 문제 임베딩은 {EMBEDDING_DIMENSIONS}차원이어야 합니다: {len(vector)}")
            vectors.append([float(value) for value in vector])
        if response.input_tokens is not None:
            input_tokens = (input_tokens or 0) + response.input_tokens
        retries += response.retry_count
        logger.info(
            "세부 문제 포함 기준 임베딩: texts=%d input_tokens=%s retries=%d latency_ms=%d (model_calls 미기록)",
            len(batch),
            response.input_tokens,
            response.retry_count,
            latency_ms,
        )
    return vectors, input_tokens, retries


async def apply_seed_plan(
    session: AsyncSession,
    plan: SeedPlan,
    *,
    actor: str,
    embedder: Any,
) -> ApplyResult:
    """검증을 통과한 계획을 현재 트랜잭션에 쓴다. flush 까지만 하고 commit 하지 않는다.

    임베딩을 먼저 모두 계산한다. API 가 실패하면 아무것도 쓰지 않은 채 예외가 올라간다.
    """

    if not plan.ok or plan.context is None:
        raise ValueError("검증 오류가 있는 계획은 적용하지 않습니다.")
    context = plan.context
    result = ApplyResult()

    embedding_plans = plan.embedding_plans
    vectors_by_key: Dict[str, List[float]] = {}
    if embedding_plans:
        vectors, input_tokens, _ = await compute_embeddings(embedder, [item.embedding_text for item in embedding_plans])
        vectors_by_key = {item.seed.key: vector for item, vector in zip(embedding_plans, vectors)}
        result.embedding_texts = len(embedding_plans)
        result.embedding_input_tokens = input_tokens

    store = QuestionGroupingStore(session)
    if plan.no_document_group_id is None:
        plan.no_document_group_id = await store.ensure_no_document_problem_group(context.document_group_id)
        result.problem_groups_created += 1
    for source_id in plan.missing_document_problem_groups:
        plan.document_problem_groups[source_id] = await store.ensure_document_problem_group(source_id)
        result.problem_groups_created += 1

    now = _utcnow()
    created_subproblem_ids: Dict[str, uuid.UUID] = {}
    for item in plan.subproblem_plans:
        seed = item.seed
        vector = vectors_by_key.get(seed.key)
        if item.action == SubproblemAction.CREATE:
            assert item.document_source_id is not None
            subproblem_id = uuid.uuid4()
            session.add(
                QuestionSubproblem(
                    id=subproblem_id,
                    problem_group_id=plan.document_problem_groups[item.document_source_id],
                    key=seed.key,
                    name=seed.name,
                    inclusion_criteria=join_criteria(seed.inclusion_criteria),
                    exclusion_criteria=join_criteria(seed.exclusion_criteria),
                    current_version=1,
                    status=QuestionSubproblemStatus.APPROVED,
                    serving_state=item.serving_state,
                    created_by=actor,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.flush()
            result.subproblems_created += 1
        else:
            assert item.existing is not None
            subproblem_id = item.existing.subproblem_id
            values: Dict[str, Any] = {}
            if item.action == SubproblemAction.UPDATE:
                values.update(
                    name=seed.name,
                    inclusion_criteria=join_criteria(seed.inclusion_criteria),
                    exclusion_criteria=join_criteria(seed.exclusion_criteria),
                    current_version=item.target_version,
                )
                result.subproblems_updated += 1
            if item.serving_state_changed:
                values["serving_state"] = item.serving_state
                result.serving_state_changed += 1
            if values:
                values["updated_at"] = now
                updated = await session.execute(
                    update(QuestionSubproblem)
                    .where(
                        QuestionSubproblem.id == subproblem_id,
                        QuestionSubproblem.current_version == item.existing.current_version,
                    )
                    .values(**values)
                    .execution_options(synchronize_session=False)
                )
                if updated.rowcount != 1:
                    raise RuntimeError(f"세부 문제가 계획 이후 바뀌었습니다: key={seed.key}")
        created_subproblem_ids[seed.key] = subproblem_id

        if item.embedding_action == EmbeddingAction.NEW_REVISION:
            assert vector is not None
            session.add(
                QuestionSubproblemRevision(
                    subproblem_id=subproblem_id,
                    version=item.target_version,
                    name_snapshot=seed.name,
                    inclusion_snapshot=join_criteria(seed.inclusion_criteria),
                    exclusion_snapshot=join_criteria(seed.exclusion_criteria),
                    change_reason=(
                        "seed: created"
                        if item.action == SubproblemAction.CREATE
                        else f"seed: {', '.join(item.definition_changes)} changed"
                    ),
                    inclusion_embedding=vector,
                    embedding_config_id=context.embedding_config_id,
                    embedding_text_version=SUBPROBLEM_EMBEDDING_TEXT_VERSION,
                    approved_by=actor,
                    created_at=now,
                )
            )
            await session.flush()
            result.revisions_created += 1
        elif item.embedding_action == EmbeddingAction.REFRESH_CURRENT_REVISION:
            assert vector is not None and item.existing is not None and item.existing.current_revision is not None
            await session.execute(
                update(QuestionSubproblemRevision)
                .where(QuestionSubproblemRevision.id == item.existing.current_revision.revision_id)
                .values(
                    inclusion_embedding=vector,
                    embedding_config_id=context.embedding_config_id,
                    embedding_text_version=SUBPROBLEM_EMBEDDING_TEXT_VERSION,
                )
                .execution_options(synchronize_session=False)
            )
            result.revisions_embedding_refreshed += 1

        if item.canonical_action == CanonicalAction.UNCHANGED:
            continue
        if item.canonical_action == CanonicalAction.REPLACE:
            assert item.existing is not None and item.existing.approved_canonical is not None
            await session.execute(
                update(CanonicalAnswer)
                .where(
                    CanonicalAnswer.id == item.existing.approved_canonical.canonical_answer_id,
                    CanonicalAnswer.approval == CanonicalAnswerApproval.APPROVED,
                )
                .values(approval=CanonicalAnswerApproval.REVOKED, valid_to=now)
                .execution_options(synchronize_session=False)
            )
            await session.flush()
            result.canonicals_replaced += 1
        else:
            result.canonicals_created += 1
        canonical_id = uuid.uuid4()
        session.add(
            CanonicalAnswer(
                id=canonical_id,
                subproblem_id=subproblem_id,
                origin=CanonicalAnswerOrigin.AUTHORED,
                content_markdown=seed.content_markdown,
                applicability_rules={APPLICABILITY_RULES_FIELD: list(seed.applicability_rules)},
                subproblem_version=item.target_version,
                approval=CanonicalAnswerApproval.APPROVED,
                approved_by=actor,
                valid_from=now,
                created_at=now,
            )
        )
        await session.flush()
        session.add_all(
            [
                CanonicalAnswerCitation(
                    canonical_answer_id=canonical_id,
                    citation_order=citation.order,
                    chunk_id=citation.chunk_id,
                    document_version_id=citation.document_version_id,
                    document_title_snapshot=citation.document_title,
                    node_path_snapshot=citation.node_path,
                    source_uri_snapshot=citation.source_uri,
                    created_at=now,
                )
                for citation in item.citations
            ]
        )
        await session.flush()

    for item in plan.recommended_question_plans:
        if item.action == RecommendedQuestionAction.NONE:
            continue
        subproblem_id = item.subproblem_id or created_subproblem_ids.get(item.subproblem_key)
        if subproblem_id is None:
            raise RuntimeError(
                f"추천 질문 대상 세부 문제를 찾지 못했습니다: key={item.subproblem_key}"
            )
        desired_by_normalized = {
            normalize_recommended_question(question): question
            for question in (item.questions or ())
        }
        existing_by_normalized = {
            row.normalized_question: row for row in item.existing
        }
        for normalized, row in existing_by_normalized.items():
            if normalized not in desired_by_normalized:
                if row.source != ExactQuestionMatchSource.RECOMMENDED:
                    continue
                await session.execute(
                    delete(RecommendedQuestion).where(
                        RecommendedQuestion.id == row.mapping_id,
                        RecommendedQuestion.document_group_id == item.document_group_id,
                    )
                )
                result.recommended_questions_deleted += 1
        for normalized, question in desired_by_normalized.items():
            current = existing_by_normalized.get(normalized)
            if current is not None:
                if (
                    current.question != question
                    or current.source != ExactQuestionMatchSource.RECOMMENDED
                    or current.state != ExactQuestionMatchState.ACTIVE
                ):
                    await session.execute(
                        update(RecommendedQuestion)
                        .where(
                            RecommendedQuestion.id == current.mapping_id,
                            RecommendedQuestion.document_group_id == item.document_group_id,
                        )
                        .values(
                            question=question,
                            normalized_question=normalized,
                            source=ExactQuestionMatchSource.RECOMMENDED,
                            state=ExactQuestionMatchState.ACTIVE,
                            canonical_answer_id=None,
                            source_rag_run_id=None,
                        )
                        .execution_options(synchronize_session=False)
                    )
                    result.recommended_questions_updated += 1
                continue
            session.add(
                RecommendedQuestion(
                    document_group_id=item.document_group_id,
                    subproblem_id=subproblem_id,
                    question=question,
                    normalized_question=normalized,
                    created_at=now,
                )
            )
            await session.flush()
            result.recommended_questions_created += 1
    return result


@dataclass
class SeedReport:
    plan: SeedPlan
    parsed: ParsedInput
    apply: bool
    applied: Optional[ApplyResult] = None

    @property
    def ok(self) -> bool:
        return self.plan.ok


async def run_seed(
    session: AsyncSession,
    *,
    group_key: str,
    parsed: ParsedInput,
    actor: str,
    apply: bool,
    serving_state: Optional[QuestionSubproblemServingState] = None,
    include_draft: bool = False,
    embedder_factory: Optional[Callable[[], Any]] = None,
) -> SeedReport:
    """계획을 세우고 apply 면 쓴다. commit·rollback 은 호출자가 한다.

    apply 면 문서 그룹 단위 advisory 트랜잭션 잠금을 먼저 잡아 동시 시드가 계획과 쓰기 사이에
    끼어들지 못하게 한다. dry-run 은 임베딩 클라이언트를 만들지 않는다.
    """

    if apply:
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:name))"),
            {"name": f"{ADVISORY_LOCK_PREFIX}{group_key}"},
        )
    plan = await build_seed_plan(
        session,
        group_key=group_key,
        parsed=parsed,
        serving_state=serving_state,
        include_draft=include_draft,
    )
    report = SeedReport(plan=plan, parsed=parsed, apply=apply)
    if not apply or not plan.ok:
        return report
    embedder = None
    if plan.embedding_plans:
        if embedder_factory is None:
            raise ValueError("임베딩이 필요한데 임베딩 클라이언트가 없습니다.")
        embedder = embedder_factory()
    report.applied = await apply_seed_plan(session, plan, actor=actor, embedder=embedder)
    return report


# ---------------------------------------------------------------------------
# 보고
# ---------------------------------------------------------------------------

_ACTION_MARK = {
    SubproblemAction.CREATE: "+",
    SubproblemAction.UPDATE: "~",
    SubproblemAction.UNCHANGED: "=",
}


def render_report(report: SeedReport, *, database: Optional[str] = None) -> str:
    plan = report.plan
    lines = [f"[세부 문제 시드] {'APPLY' if report.apply else 'DRY-RUN (쓰기·임베딩 호출 없음)'}"]
    if database:
        lines.append(f"DB: {database}")
    context = plan.context
    if context is not None:
        lines.append(
            f"문서 그룹: {context.group_key} (id={context.document_group_id}), ACTIVE 색인 id={context.index_version_id}, "
            f"chunking_config={context.chunking_config_id}, embedding_config={context.embedding_config_id}"
        )
    parsed = report.parsed
    skipped_draft = sum(len(document.skipped_draft_keys) for document in parsed.documents)
    skipped_rejected = sum(len(document.skipped_rejected_keys) for document in parsed.documents)
    lines.append(
        f"입력: 문서 {len(parsed.documents)}개, 적재 대상 세부 문제 {parsed.selected_count}개 "
        f"(제외: draft {skipped_draft}, rejected {skipped_rejected})"
    )
    if context is not None:
        lines.append(
            "문제 그룹: "
            f"NO_DOCUMENT {'생성 예정' if plan.no_document_group_id is None else '있음'}, "
            f"DOCUMENT 생성 예정 {len(plan.missing_document_problem_groups)} / 있음 {len(plan.document_problem_groups)}"
        )
    plans = plan.subproblem_plans
    new_revisions = sum(1 for item in plans if item.embedding_action == EmbeddingAction.NEW_REVISION)
    refresh = sum(1 for item in plans if item.embedding_action == EmbeddingAction.REFRESH_CURRENT_REVISION)
    lines.append(f"임베딩: {new_revisions + refresh}건 (새 개정 {new_revisions}, 기존 개정 갱신 {refresh})")
    lines.append(
        f"추천 질문 매핑: {len(plan.recommended_question_plans)}건 관리, "
        f"변경 {len(plan.recommended_writes)}건"
    )

    for document, source_id, subproblem_plans in plan.documents:
        lines.append("")
        lines.append(f"## {document.path}" + ("" if source_id is None else f" (document_source_id={source_id})"))
        if not document.subproblems:
            lines.append("  적재 대상 없음")
        counts = {action: 0 for action in SubproblemAction}
        canonical_counts = {action: 0 for action in CanonicalAction}
        for item in subproblem_plans:
            counts[item.action] += 1
            canonical_counts[item.canonical_action] += 1
        if subproblem_plans:
            lines.append(
                f"  세부 문제: 생성 {counts[SubproblemAction.CREATE]}, 갱신 {counts[SubproblemAction.UPDATE]}, "
                f"변경 없음 {counts[SubproblemAction.UNCHANGED]} | 정본: 생성 {canonical_counts[CanonicalAction.CREATE]}, "
                f"교체 {canonical_counts[CanonicalAction.REPLACE]}, 변경 없음 {canonical_counts[CanonicalAction.UNCHANGED]}"
            )
        for item in subproblem_plans:
            version = (
                f"v{item.target_version}"
                if item.existing is None or item.existing.current_version == item.target_version
                else f"v{item.existing.current_version}→v{item.target_version}"
            )
            details = []
            if item.definition_changes:
                details.append(f"변경: {', '.join(item.definition_changes)}")
            if item.embedding_action == EmbeddingAction.REFRESH_CURRENT_REVISION:
                details.append("임베딩 갱신")
            state = item.serving_state.value
            if item.serving_state_changed:
                assert item.existing is not None
                state = f"{item.existing.serving_state.value}→{state}"
            lines.append(
                f"  {_ACTION_MARK[item.action]} {item.action.value:<9} {item.seed.key} {version} [{state}] "
                f"정본 {item.canonical_action.value} (인용 {len(item.citations)})"
                + (f" {'; '.join(details)}" if details else "")
            )
        not_loaded = [*document.skipped_draft_keys, *document.skipped_rejected_keys]
        if not_loaded:
            lines.append(f"  적재 제외(검토 상태): {', '.join(not_loaded)}")
        untouched = plan.untouched.get(document_key_from_path(document.path))
        if untouched:
            lines.append(f"  입력에 없는 기존 세부 문제(변경 안 함): {', '.join(untouched)}")

    input_keys = {document_key_from_path(document.path) for document in parsed.documents}
    other_untouched = {owner: keys for owner, keys in plan.untouched.items() if owner not in input_keys}
    if other_untouched:
        lines.append("")
        lines.append("입력 파일이 없는 문서의 기존 세부 문제(변경 안 함):")
        for owner, keys in sorted(other_untouched.items()):
            lines.append(f"  {owner}: {', '.join(keys)}")

    if plan.warnings:
        lines.append("")
        lines.append(f"경고 ({len(plan.warnings)}):")
        lines.extend(f"  - {issue.render()}" for issue in plan.warnings)
    if plan.errors:
        lines.append("")
        lines.append(f"오류 ({len(plan.errors)}):")
        lines.extend(f"  - {issue.render()}" for issue in plan.errors)

    lines.append("")
    if not plan.ok:
        lines.append("결과: 검증 실패. 아무것도 쓰지 않았습니다.")
    elif report.applied is not None:
        applied = report.applied
        lines.append(
            "결과: 적용 완료. "
            f"문제 그룹 생성 {applied.problem_groups_created}, 세부 문제 생성 {applied.subproblems_created}, "
            f"갱신 {applied.subproblems_updated}, 개정 생성 {applied.revisions_created}, "
            f"임베딩 갱신 {applied.revisions_embedding_refreshed}, 서빙 상태 변경 {applied.serving_state_changed}, "
            f"정본 생성 {applied.canonicals_created}, 정본 교체 {applied.canonicals_replaced}, "
            f"임베딩 입력 {applied.embedding_texts}건 (input_tokens={applied.embedding_input_tokens}), "
            f"추천 질문 생성 {applied.recommended_questions_created}, "
            f"갱신 {applied.recommended_questions_updated}, 삭제 {applied.recommended_questions_deleted}"
        )
    else:
        writes = sum(1 for item in plans if item.writes)
        lines.append(f"결과: 검증 통과. 쓰기 대상 세부 문제 {writes}개. 적용하려면 --apply 를 붙이세요.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _actor(value: str) -> str:
    actor = value.strip()
    if not actor or len(actor) > MAX_ACTOR_LENGTH:
        raise argparse.ArgumentTypeError(f"--actor 는 1~{MAX_ACTOR_LENGTH}자여야 합니다.")
    return actor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.ops.seed_question_grouping",
        description="세부 문제 그룹핑 결과(subproblems.json)를 문서 그룹에 적재합니다. 기본은 dry-run 입니다.",
    )
    parser.add_argument("--group-key", required=True, help="대상 문서 그룹 키 (예: HELP_CHATBOT_TEST)")
    parser.add_argument("--input", required=True, type=Path, help="subproblems.json 파일 또는 그 파일들이 있는 폴더")
    parser.add_argument("--actor", required=True, type=_actor, help="created_by / approved_by 에 남길 이름")
    parser.add_argument("--apply", action="store_true", help="검증 통과 시 한 트랜잭션으로 쓰고 commit 합니다.")
    parser.add_argument(
        "--include-draft",
        action="store_true",
        help="reviewStatus draft 도 적재합니다(테스트 환경용). rejected 는 항상 제외합니다.",
    )
    parser.add_argument(
        "--serving-state",
        choices=[state.value for state in QuestionSubproblemServingState],
        default=None,
        help="이번 실행이 쓰는 세부 문제의 서빙 상태. 생략하면 새 세부 문제는 UNUSED, 기존 세부 문제는 유지합니다.",
    )
    return parser


def _safe_database_url(url: str) -> str:
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        return "<DATABASE_URL 해석 불가>"


async def _main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        parsed = load_input(args.input, include_draft=args.include_draft)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return EXIT_USAGE

    from app.core.config import get_settings
    from app.database.session import dispose_engine, get_session_factory

    serving_state = None if args.serving_state is None else QuestionSubproblemServingState(args.serving_state)

    def embedder_factory() -> Any:
        from app.retrieval.embedding import OpenAIEmbedder

        return OpenAIEmbedder()

    database = _safe_database_url(get_settings().database_url)
    try:
        async with get_session_factory()() as session:
            try:
                report = await run_seed(
                    session,
                    group_key=args.group_key,
                    parsed=parsed,
                    actor=args.actor,
                    apply=args.apply,
                    serving_state=serving_state,
                    include_draft=args.include_draft,
                    embedder_factory=embedder_factory,
                )
                if report.applied is not None:
                    await session.commit()
                else:
                    await session.rollback()
            except Exception:
                await session.rollback()
                raise
    except Exception as error:
        print(f"시드 실행 중 오류로 롤백했습니다: {sanitize_error_message(error)}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    finally:
        await dispose_engine()

    print(render_report(report, database=database))
    return EXIT_OK if report.ok else EXIT_VALIDATION_FAILED


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return asyncio.run(_main(argv))


if __name__ == "__main__":
    sys.exit(main())
