"""질문 판별과 캐시 게이트가 주고받는 순수 데이터 모델.

DB 조회와 쓰기는 이 모델을 채우거나 읽기만 한다. 저장 값과 같은 집합을 쓰는
판정, 귀속, 서빙 상태, 캐시 시도 결과는 app.database.models 의 str Enum 을 그대로
쓴다. 판별 응답에만 있는 문서 결정은 여기서 정의한다.
"""

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from app.core.model_trace import ModelCallTrace
from app.database.models import (
    AttributionSource,
    CacheAttemptOutcome,
    ClassificationDecision,
    QuestionProblemGroupKind,
    QuestionSubproblemServingState,
    QuestionSubproblemStatus,
    ExactQuestionMatchSource,
    ExactQuestionMatchState,
)


class DocumentDecision(str, enum.Enum):
    """v7.2 판별 응답의 문서 결정."""

    MATCHED = "MATCHED"
    NONE = "NONE"
    SUBPROBLEM_DOCUMENT = "SUBPROBLEM_DOCUMENT"


DOCUMENT_RATIONALE_CODES: Dict[DocumentDecision, Tuple[str, ...]] = {
    DocumentDecision.MATCHED: ("ASK_COVERED", "TOPIC_ONLY"),
    DocumentDecision.NONE: (
        "TANGENTIAL_OVERLAP",
        "NO_CANDIDATE_COVERS",
        "TOO_VAGUE_TO_PLACE",
    ),
    DocumentDecision.SUBPROBLEM_DOCUMENT: (),
}


class JudgeFailureKind(str, enum.Enum):
    """판별 호출 실패 종류. judgment_input.failure.kind 에 남긴다."""

    API_ERROR = "API_ERROR"
    INCOMPLETE_RESPONSE = "INCOMPLETE_RESPONSE"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# 판별 입력: 카탈로그와 후보
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogCanonicalAnswer:
    """payload 에 싣는 승인 정본. 인용은 게이트 입력으로 따로 읽는다."""

    canonical_answer_id: uuid.UUID
    content_markdown: str
    applicability_rules: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SubproblemCatalogItem:
    """문서 그룹의 APPROVED 세부 문제 한 건과 최신 개정 기준.

    기준 문장은 DB text 칸을 줄 단위로 나눈 목록이다(payload.split_criteria).
    document_title 은 payload 의 group.name 으로만 쓴다.
    """

    subproblem_id: uuid.UUID
    key: str
    name: str
    inclusion_criteria: Tuple[str, ...]
    exclusion_criteria: Tuple[str, ...]
    current_version: int
    problem_group_id: uuid.UUID
    # 카탈로그 리더는 DOCUMENT 문제 그룹의 세부 문제만 읽어 항상 채운다(결정 C).
    # 가이드 밖(NO_DOCUMENT) 문제 그룹에는 세부 문제를 두지 않는다. 직접 만든 항목이
    # None 이면 document_key 는 constants.NO_DOCUMENT_GROUP_KEY 를 쓴다.
    document_source_id: Optional[int]
    document_key: str
    serving_state: QuestionSubproblemServingState
    document_title: Optional[str] = None
    canonical_answer: Optional[CatalogCanonicalAnswer] = None
    inclusion_embedding: Tuple[float, ...] = ()


@dataclass(frozen=True)
class SubproblemCandidate:
    """질문 벡터와 포함 기준 임베딩의 코사인으로 뽑은 세부 문제 후보."""

    item: SubproblemCatalogItem
    similarity: float
    retrieval_rank: int


@dataclass(frozen=True)
class RankedDocument:
    """검색 청크 RRF 점수를 문서 판별 최고값으로 모은 문서 순위."""

    document_version_id: int
    score: float
    retrieval_rank: int
    best_chunk_id: int


@dataclass(frozen=True)
class DocumentOutline:
    """판별에 보여주는 문서 모습. 본문과 점수는 담지 않는다."""

    document_source_id: int
    document_version_id: int
    document_key: str
    title: str
    parent_path: str
    headings: Tuple[str, ...]


@dataclass(frozen=True)
class DocumentCandidate:
    """outline 을 붙인 문서 후보."""

    outline: DocumentOutline
    score: float
    retrieval_rank: int


# ---------------------------------------------------------------------------
# 제시 목록과 payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PresentedSubproblem:
    """판별에 제시한 세부 문제. LLM 이 적은 key 를 UUID 로 되돌리는 대응표다."""

    key: str
    subproblem_id: uuid.UUID
    subproblem_version: int
    problem_group_id: uuid.UUID
    document_source_id: Optional[int]
    document_key: str
    canonical_answer_id: Optional[uuid.UUID]
    similarity: float
    retrieval_rank: int
    presented_order: int
    inclusion_count: int
    exclusion_count: int

    def to_judgment_input(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "subproblemId": str(self.subproblem_id),
            "subproblemVersion": self.subproblem_version,
            "problemGroupId": str(self.problem_group_id),
            "documentKey": self.document_key,
            "canonicalAnswerId": (
                None
                if self.canonical_answer_id is None
                else str(self.canonical_answer_id)
            ),
            "similarity": self.similarity,
            "retrievalRank": self.retrieval_rank,
            "presentedOrder": self.presented_order,
        }


@dataclass(frozen=True)
class RecommendedQuestionMatch:
    """문서 그룹 소속 검증까지 끝난 정확 일치 매핑.

    ``source``와 historical provenance는 추천 질문과 과거 SERVED 질문을 같은
    조회 경로에서 구분하기 위해 보존한다.
    """

    mapping_id: int
    document_group_id: int
    subproblem_id: uuid.UUID
    key: str
    name: str
    problem_group_id: uuid.UUID
    current_version: int
    document_source_id: int
    document_key: str
    question: str
    normalized_question: str
    source: ExactQuestionMatchSource = ExactQuestionMatchSource.RECOMMENDED
    state: ExactQuestionMatchState = ExactQuestionMatchState.ACTIVE
    subproblem_version: int = 1
    canonical_answer_id: Optional[uuid.UUID] = None
    source_rag_run_id: Optional[uuid.UUID] = None


@dataclass(frozen=True)
class PresentedDocument:
    """판별에 D 번호로 제시한 문서. D 번호는 제시 순서다."""

    id: str
    document_source_id: int
    document_version_id: int
    document_key: str
    title: str
    parent_path: str
    score: float
    retrieval_rank: int
    presented_order: int

    def to_judgment_input(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "documentSourceId": self.document_source_id,
            "documentVersionId": self.document_version_id,
            "documentKey": self.document_key,
            "title": self.title,
            "parentPath": self.parent_path,
            "score": self.score,
            "retrievalRank": self.retrieval_rank,
            "presentedOrder": self.presented_order,
        }


@dataclass(frozen=True)
class JudgePresentation:
    """판별 호출 한 번의 payload 와 제시 목록."""

    payload: Mapping[str, Any]
    subproblems: Tuple[PresentedSubproblem, ...]
    documents: Tuple[PresentedDocument, ...]
    subproblem_seed: str
    document_seed: str

    def subproblem_by_key(self) -> Dict[str, PresentedSubproblem]:
        return {item.key: item for item in self.subproblems}

    def document_by_id(self) -> Dict[str, PresentedDocument]:
        return {item.id: item for item in self.documents}


# ---------------------------------------------------------------------------
# 판별 호출과 정규화
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeFailure:
    """판별 실패. safe_message 에는 질문, payload, 모델 출력을 담지 않는다."""

    kind: JudgeFailureKind
    safe_message: str


@dataclass(frozen=True)
class JudgeCall:
    """논리적 판별 호출 한 건. 예외 대신 실패를 담아 올린다.

    failure 가 없으면 응답이 완료된 것이고 output_text 를 decision 이 정규화한다.
    output_text 가 비었거나 JSON 이 아니어도 호출 실패가 아니라 정규화 무효다.
    """

    trace: ModelCallTrace
    output_text: Optional[str] = None
    failure: Optional[JudgeFailure] = None

    @property
    def succeeded(self) -> bool:
        return self.failure is None


@dataclass(frozen=True)
class NormalizedJudgment:
    """R7 규칙으로 보정한 판별 결과.

    invalid_reason 이 있으면 판정이 없는 것이고 decision 은 UNCLASSIFIED 다.
    raw_output 은 LLM 원 출력(JSON 객체)을 형식 위반까지 그대로 담는다.
    """

    decision: ClassificationDecision
    subproblem: Optional[PresentedSubproblem] = None
    document_decision: Optional[DocumentDecision] = None
    document: Optional[PresentedDocument] = None
    confidence: Optional[float] = None
    rationale_code: Optional[str] = None
    ambiguity_reason: Optional[str] = None
    raw_output: Optional[Mapping[str, Any]] = None
    invalid_reason: Optional[str] = None
    document_fields_ignored: bool = False
    document_field_violations: Tuple[str, ...] = ()
    subproblem_field_violations: Tuple[str, ...] = ()
    group_id_mismatch: bool = False
    unknown_document_id: bool = False
    criteria_report: Optional[Mapping[str, Any]] = None

    @property
    def valid(self) -> bool:
        return self.invalid_reason is None

    def normalization_judgment_input(self) -> Dict[str, Any]:
        return {
            "invalidReason": self.invalid_reason,
            "documentFieldsIgnored": self.document_fields_ignored,
            "documentFieldViolations": list(self.document_field_violations),
            "subproblemFieldViolations": list(self.subproblem_field_violations),
            "groupIdMismatch": self.group_id_mismatch,
            "unknownDocumentId": self.unknown_document_id,
            "criteriaReport": (
                None if self.criteria_report is None else dict(self.criteria_report)
            ),
        }


@dataclass(frozen=True)
class AttributionTarget:
    """질문 연결 행의 문제 그룹을 정할 대상.

    SUBPROBLEM 은 세부 문제의 problem_group_id 를 바로 쓴다. DOCUMENT 와 CITATION 은
    문서의 DOCUMENT 문제 그룹을, NONE 은 문서 그룹의 NO_DOCUMENT 문제 그룹을 저장
    계층이 찾거나 만든다.
    """

    attribution_source: AttributionSource
    group_kind: QuestionProblemGroupKind
    problem_group_id: Optional[uuid.UUID] = None
    document_source_id: Optional[int] = None


@dataclass(frozen=True)
class CitedDocument:
    """턴 인용 한 건이 가리키는 문서. answer_citations 행 하나에 대응한다."""

    citation_order: int
    document_source_id: int


@dataclass(frozen=True)
class TurnJudgment:
    """턴 하나의 판별 결과와 초기 귀속. 게이트와 판별 행 쓰기의 입력이다.

    failure 가 있으면(호출 실패 또는 R7 무효) decision 은 UNCLASSIFIED,
    귀속은 NO_DOCUMENT + NONE 이고 캐시 시도는 FAILED 다.
    """

    decision: ClassificationDecision
    attribution: AttributionTarget
    subproblem: Optional[PresentedSubproblem] = None
    normalized: Optional[NormalizedJudgment] = None
    failure: Optional[JudgeFailure] = None

    @property
    def failed(self) -> bool:
        return self.failure is not None

    @property
    def subproblem_version(self) -> Optional[int]:
        return None if self.subproblem is None else self.subproblem.subproblem_version


# ---------------------------------------------------------------------------
# 캐시 게이트
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateSubproblemState:
    """게이트 직전에 다시 읽은 세부 문제 상태."""

    subproblem_id: uuid.UUID
    status: QuestionSubproblemStatus
    serving_state: QuestionSubproblemServingState
    current_version: int


@dataclass(frozen=True)
class GateCanonicalAnswer:
    """게이트 직전에 다시 읽은 세부 문제의 APPROVED 정본."""

    canonical_answer_id: uuid.UUID
    subproblem_version: int
    content_markdown: str
    applicability_rules: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CanonicalCitationSnapshot:
    """정본 인용 한 건과 인용 청크의 옛 절 해시.

    canonical_answer_citations.chunk_id → content_nodes(공유 PK) 조인으로 읽는다.
    """

    citation_order: int
    chunk_id: int
    document_version_id: int
    document_source_id: int
    content_hash: str
    node_order: int
    node_identity_hash: Optional[str] = None


@dataclass(frozen=True)
class IndexedSection:
    """이번 턴 색인 판의 청킹 설정으로 만든 절 한 건과 현재 스냅샷."""

    chunk_id: int
    document_version_id: int
    content_hash: str
    node_order: int
    node_identity_hash: Optional[str] = None
    document_title: Optional[str] = None
    node_path: Optional[str] = None
    source_uri: Optional[str] = None


@dataclass(frozen=True)
class CitationIndexContext:
    """인용 문서가 이번 턴 색인에 들어 있는 모습.

    indexed_document_version_id 는 같은 document_source 중 턴 색인 판의
    index_documents 에 있는 판이고, 없으면 None 이다. sections 는 그 판의 절 중
    턴 색인 판의 chunking_config 로 만든 것만 담는다.
    """

    indexed_document_version_id: Optional[int]
    sections: Tuple[IndexedSection, ...] = ()


@dataclass(frozen=True)
class CitationResolution:
    """R17 알고리즘으로 인용 한 건을 현재 색인 절에 대응시킨 결과."""

    citation_order: int
    section: Optional[IndexedSection] = None
    step: Optional[int] = None
    rejection_reason: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.section is not None

    def to_judgment_input(self) -> Dict[str, Any]:
        return {
            "citationOrder": self.citation_order,
            "step": self.step,
            "rejectionReason": self.rejection_reason,
            "chunkId": None if self.section is None else self.section.chunk_id,
            "documentVersionId": (
                None if self.section is None else self.section.document_version_id
            ),
        }


@dataclass(frozen=True)
class GateResult:
    """캐시 시도 한 건의 결과. question_cache_attempts 행과 SERVED 인용의 입력이다.

    CHECK 제약대로 SERVED/SHADOW/GROUP_DISABLED 만 정본 id 를 갖고,
    REJECTED 만 거부 사유를 갖는다.
    """

    outcome: CacheAttemptOutcome
    canonical_answer_id: Optional[uuid.UUID] = None
    rejection_reasons: Tuple[str, ...] = ()
    served_citations: Tuple[CitationResolution, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# DB 조회 결과 묶음
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexScope:
    """턴 색인 판의 범위. 카탈로그, 게이트 입력, outline 조회가 같은 값을 쓴다."""

    index_version_id: int
    document_group_id: int
    chunking_config_id: int
    embedding_config_id: int


@dataclass(frozen=True)
class SubproblemCatalog:
    """문서 그룹의 세부 문제 카탈로그와 임베딩 문제로 뺀 건수.

    skipped_missing_embedding 은 현재 개정이 없거나 포함 기준 임베딩이 빈 건수,
    skipped_embedding_config_mismatch 는 턴 색인 판과 다른 임베딩 설정으로 계산한 건수다.
    embedding_text_versions 는 실은 항목들의 임베딩 문장 구성 판(중복 제거, 정렬)이다.
    """

    items: Tuple[SubproblemCatalogItem, ...]
    skipped_missing_embedding: int = 0
    skipped_embedding_config_mismatch: int = 0
    embedding_text_versions: Tuple[str, ...] = ()

    @property
    def skipped_count(self) -> int:
        return self.skipped_missing_embedding + self.skipped_embedding_config_mismatch


@dataclass(frozen=True)
class GateInputs:
    """게이트 직전에 다시 읽은 입력. gate.resolve_citations 와 evaluate_cache_gate 에 넘긴다.

    세부 문제가 없으면 나머지는 비고, 정본이 없으면 인용과 문서 맥락이 빈다.
    contexts_by_source_id 는 인용 문서마다 턴 색인 판에 든 모습이다.
    """

    subproblem: Optional[GateSubproblemState] = None
    canonical_answer: Optional[GateCanonicalAnswer] = None
    citations: Tuple[CanonicalCitationSnapshot, ...] = ()
    contexts_by_source_id: Mapping[int, CitationIndexContext] = field(
        default_factory=dict
    )
