"""ERD v0.2.2 기준 애플리케이션 ORM model을 정의한다.

- 문서 영역: document_sources → document_versions → content_nodes ↔ document_chunks(공유 PK 1:1)
- 검색·색인 영역: embedding_configs, chunk_embeddings, index_versions, index_documents, index_runs
- 대화·RAG 영역: conversations → rag_runs → retrieval_results / model_calls / answer_citations / feedbacks
- 질문 그룹핑 영역: question_problem_groups → question_subproblems → canonical_answers,
  classification_runs → question_classifications → question_cache_attempts
"""

import enum
import uuid
from typing import Any, List, Optional

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    TIMESTAMP,
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base

# 확정된 임베딩 차원. OpenAI text embedding 1536차원으로 고정한다
EMBEDDING_DIMENSIONS = 1536

# 명명 규칙(uq_%(table_name)s_%(column_0_N_name)s)을 따르지만 부분 unique index라
# 선언 시점에 이름을 직접 지정한다.
ACTIVE_INDEX_VERSION_CONSTRAINT = "uq_index_versions_document_group_id"
INDEX_VERSION_NO_CONSTRAINT = "uq_index_versions_document_group_id_version_no"
# 명명 규칙대로 referred table까지 붙이면 66자가 되어 식별자 63자 제한을 넘는다.
DUPLICATE_DOCUMENT_SOURCE_CONSTRAINT = (
    "fk_ingestion_runs_duplicate_of_document_source_id"
)
# The normal naming convention would exceed PostgreSQL's 63-character
# identifier limit for this relationship, so keep the historical name
# explicit in both the ORM and its migration.
CHAT_PROFILE_REVISION_FK_CONSTRAINT = (
    "fk_conversations_profile_revision_id_chat_profile_revisions"
)
# 복합 외래키는 명명 규칙대로 붙이면 63자를 넘어 이름을 직접 지정한다.
DOCUMENT_SOURCE_GROUP_SOURCE_FK_CONSTRAINT = (
    "fk_document_sources_group_source_id_document_group_id"
)
GROUP_SOURCE_DOCUMENT_GROUP_UNIQUE_CONSTRAINT = (
    "uq_document_group_sources_id_document_group_id"
)
CURRENT_CLASSIFICATION_CONSTRAINT = "uq_question_classifications_rag_run_id_current"
OPEN_ONLINE_CLASSIFICATION_RUN_CONSTRAINT = "uq_classification_runs_open_online"
APPROVED_CANONICAL_ANSWER_CONSTRAINT = "uq_canonical_answers_subproblem_id_approved"

# 턴, 분류 실행, 색인, 수집 중 어느 실행 칸을 함께 채울 수 있는지 정한다.
# 즉시 판별은 턴과 분류 실행을 모두 채우고, 백필 판정은 분류 실행만 채운다.
MODEL_CALL_OWNER_COMBINATION = (
    "(classification_run_id IS NULL"
    " AND purpose <> 'QUESTION_CLASSIFICATION'"
    " AND num_nonnulls(rag_run_id, index_run_id, ingestion_run_id) = 1)"
    " OR (classification_run_id IS NOT NULL"
    " AND index_run_id IS NULL AND ingestion_run_id IS NULL"
    " AND (purpose = 'QUESTION_CLASSIFICATION'"
    " OR (purpose = 'QUERY_EMBEDDING' AND rag_run_id IS NOT NULL)))"
)


# ---------------------------------------------------------------------------
# 상태 Enum
# 저장은 VARCHAR + CHECK 제약을 사용해 값 추가·변경 시 마이그레이션을 단순화한다.
# ---------------------------------------------------------------------------


class DocumentVersionStatus(str, enum.Enum):
    """문서 버전을 색인에 사용할 수 있는지 나타낸다."""

    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class IndexVersionStatus(str, enum.Enum):
    """색인 버전을 사용자 검색에 사용할 수 있는지 나타낸다."""

    BUILDING = "BUILDING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    INACTIVE = "INACTIVE"


class IndexRunStage(str, enum.Enum):
    """색인 실행 한 건이 현재 수행 중인 단계를 나타낸다."""

    BUILDING = "BUILDING"
    VALIDATING = "VALIDATING"
    APPLYING = "APPLYING"


class IndexOperationType(str, enum.Enum):
    """색인 실행이 요청받은 작업 범위를 나타낸다.

    1차에서 실제로 사용하는 값은 BUILD_AND_APPLY와 APPLY다.
    BUILD는 후보 생성만 수행하는 2차 확장을 위해 값만 미리 둔다.
    """

    BUILD_AND_APPLY = "BUILD_AND_APPLY"
    BUILD = "BUILD"
    APPLY = "APPLY"


class IngestionResultCode(str, enum.Enum):
    """수집 실행 한 건이 문서에 만든 결과를 나타낸다."""

    CREATED = "CREATED"
    UPDATED = "UPDATED"
    NO_CHANGE = "NO_CHANGE"
    DUPLICATE_CONTENT = "DUPLICATE_CONTENT"


class IngestionStage(str, enum.Enum):
    """수집 실행 한 건이 현재 수행 중인 단계를 나타낸다."""

    RECEIVING = "RECEIVING"
    VALIDATING = "VALIDATING"
    NORMALIZING = "NORMALIZING"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    PERSISTING = "PERSISTING"


class ExecutionStatus(str, enum.Enum):
    """내부 작업(수집·색인·모델 호출) 한 건의 처리 결과를 나타낸다."""

    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ConversationStatus(str, enum.Enum):
    """대화에 후속 질문을 이어갈 수 있는지 나타낸다."""

    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"


class ConversationChannel(str, enum.Enum):
    """Endpoint channel that owns a conversation's pinned revision."""

    PUBLIC = "PUBLIC"
    INTERNAL_TEST = "INTERNAL_TEST"


class ChatProfileRevisionStatus(str, enum.Enum):
    """프로필 설정 판의 lifecycle 상태."""

    DRAFT = "DRAFT"
    TESTING = "TESTING"
    PUBLISHED = "PUBLISHED"
    RETIRED = "RETIRED"


class AnswerStatus(str, enum.Enum):
    """사용자 질문 한 건의 답변 결과를 나타낸다. API 응답의 status와 동일한 값을 사용한다."""

    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    WITHHELD = "WITHHELD"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class ContextStrategy(str, enum.Enum):
    """현재 질문을 해석할 때 이전 대화 문맥을 사용한 방식을 나타낸다."""

    NEW_TOPIC = "NEW_TOPIC"
    FULL = "FULL"
    WINDOW = "WINDOW"
    SUMMARY = "SUMMARY"
    UNRESOLVED = "UNRESOLVED"
    FOLLOW_UP_FULL = "FOLLOW_UP_FULL"
    FOLLOW_UP_WINDOW = "FOLLOW_UP_WINDOW"
    FOLLOW_UP_SUMMARY = "FOLLOW_UP_SUMMARY"


class RetrieverType(str, enum.Enum):
    """검색 후보를 만들어낸 검색기를 나타낸다. 같은 청크도 검색기별로 1행이다."""

    BM25 = "BM25"
    VECTOR = "VECTOR"


class ModelCallPurpose(str, enum.Enum):
    """모델 호출의 용도. 기존 값은 contract migration 전까지 함께 읽는다."""

    EMBEDDING = "EMBEDDING"
    GENERATION = "GENERATION"
    QUERY_EMBEDDING = "QUERY_EMBEDDING"
    CHUNK_EMBEDDING = "CHUNK_EMBEDDING"
    ANSWER_GENERATION = "ANSWER_GENERATION"
    QUERY_REWRITE = "QUERY_REWRITE"
    CONVERSATION_SUMMARY = "CONVERSATION_SUMMARY"
    QUESTION_CLASSIFICATION = "QUESTION_CLASSIFICATION"


class FeedbackRating(str, enum.Enum):
    """답변에 대한 사용자 평가. 취소는 없고 반대 값으로 변경만 가능하다."""

    GOOD = "GOOD"
    BAD = "BAD"


class QuestionProblemGroupKind(str, enum.Enum):
    """문제 그룹이 문서 한 건을 가리키는지, 가이드 밖 질문을 받는지 나타낸다."""

    DOCUMENT = "DOCUMENT"
    NO_DOCUMENT = "NO_DOCUMENT"


class QuestionSubproblemStatus(str, enum.Enum):
    """세부 문제 정의의 lifecycle 상태."""

    DRAFT = "DRAFT"
    APPROVED = "APPROVED"
    ARCHIVED = "ARCHIVED"


class QuestionSubproblemServingState(str, enum.Enum):
    """캐시 게이트가 매 턴 읽는 세부 문제의 서빙 상태. 긴급 정지는 STOPPED 다."""

    UNUSED = "UNUSED"
    SHADOW = "SHADOW"
    SERVING = "SERVING"
    STOPPED = "STOPPED"


class ClassificationRunKind(str, enum.Enum):
    """질문 연결 행을 만든 분류 실행의 종류. MVP 는 ONLINE 만 쓴다."""

    ONLINE = "ONLINE"
    BACKFILL = "BACKFILL"
    OPERATOR = "OPERATOR"
    REJUDGE = "REJUDGE"


class ClassificationDecision(str, enum.Enum):
    """질문을 세부 문제에 붙였는지 나타낸다. 판별 호출이 실패하면 UNCLASSIFIED 다."""

    CONNECT = "CONNECT"
    SEPARATE = "SEPARATE"
    UNCLASSIFIED = "UNCLASSIFIED"


class AttributionSource(str, enum.Enum):
    """질문 연결 행의 문제 그룹을 어디서 얻었는지 나타낸다."""

    SUBPROBLEM = "SUBPROBLEM"
    CITATION = "CITATION"
    DOCUMENT = "DOCUMENT"
    NONE = "NONE"


class CanonicalAnswerOrigin(str, enum.Enum):
    """정본을 최근 답변에서 선정했는지 운영자가 직접 썼는지 나타낸다."""

    SELECTED = "SELECTED"
    AUTHORED = "AUTHORED"


class CanonicalAnswerApproval(str, enum.Enum):
    """정본 승인 상태. 세부 문제마다 APPROVED 는 하나뿐이다."""

    DRAFT = "DRAFT"
    APPROVED = "APPROVED"
    REVOKED = "REVOKED"


class CacheAttemptOutcome(str, enum.Enum):
    """턴 하나의 캐시 시도가 무엇을 반환했거나 왜 반환하지 않았는지 나타낸다."""

    SERVED = "SERVED"
    SHADOW = "SHADOW"
    GROUP_DISABLED = "GROUP_DISABLED"
    REJECTED = "REJECTED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


def _status_enum(
    enum_cls: type[enum.Enum],
    name: str,
    length: int = 20,
) -> SAEnum:
    """VARCHAR + CHECK 제약으로 저장되는 상태 Enum 컬럼 타입을 만든다."""

    return SAEnum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=length,
        values_callable=lambda cls: [member.value for member in cls],
    )


# ---------------------------------------------------------------------------
# 문서 영역
# ---------------------------------------------------------------------------


class DocumentGroup(Base):
    """문서와 검색 버전을 독립적으로 관리하는 확장 단위."""

    __tablename__ = "document_groups"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    group_key: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    consumer_key: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class ChatProfile(Base):
    """어떤 소비자에게 제공할 챗봇 프로필의 안정적인 식별자."""

    __tablename__ = "chat_profiles"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    profile_key: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class ChatProfileRevision(Base):
    """불변으로 취급하는 챗봇 프로필 설정 판."""

    __tablename__ = "chat_profile_revisions"
    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "version",
            name="uq_chat_profile_revisions_profile_id_version",
        ),
        Index(
            "uq_chat_profile_revisions_profile_published",
            "profile_id",
            unique=True,
            postgresql_where=text("status = 'PUBLISHED'"),
        ),
        Index(
            "uq_chat_profile_revisions_profile_testing",
            "profile_id",
            unique=True,
            postgresql_where=text("status = 'TESTING'"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    profile_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("chat_profiles.id", ondelete="RESTRICT"),
        nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    document_group_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_groups.id", ondelete="RESTRICT"),
        nullable=False,
    )
    status: Mapped[ChatProfileRevisionStatus] = mapped_column(
        _status_enum(ChatProfileRevisionStatus, "chat_profile_revision_status"),
        nullable=False,
    )
    generation_model_name: Mapped[str] = mapped_column(String(150), nullable=False)
    generation_prompt_version: Mapped[str] = mapped_column(String(50), nullable=False)
    query_rewrite_model_name: Mapped[str] = mapped_column(String(150), nullable=False)
    query_rewrite_prompt_version: Mapped[str] = mapped_column(String(50), nullable=False)
    semantic_cache_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    verifier_model_name: Mapped[Optional[str]] = mapped_column(String(150))
    verifier_prompt_version: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class SourceProvider(str, enum.Enum):
    """수집 원천의 제공자."""

    GITBOOK = "GITBOOK"


class DocumentGroupSource(Base):
    """문서 그룹이 문서를 끌어오는 외부 원천.

    콘솔 업로드처럼 밀어 넣는 문서에는 원천이 없다.
    """

    __tablename__ = "document_group_sources"
    __table_args__ = (
        UniqueConstraint("document_group_id", "root_url"),
        # 문서가 원천과 문서 그룹을 함께 가리키는 복합 외래키의 대상이다.
        UniqueConstraint(
            "id",
            "document_group_id",
            name=GROUP_SOURCE_DOCUMENT_GROUP_UNIQUE_CONSTRAINT,
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    document_group_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_groups.id", ondelete="RESTRICT"),
        nullable=False,
    )
    provider: Mapped[SourceProvider] = mapped_column(
        _status_enum(SourceProvider, "source_provider", length=20),
        nullable=False,
    )
    root_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class DocumentSource(Base):
    """문서 원본의 고정 식별자와 수집 위치."""

    __tablename__ = "document_sources"
    __table_args__ = (
        # 끌어오는 문서는 원천 안에서, 밀어 넣는 문서는 그룹 안에서 유일하다.
        # 두 GitBook 이 같은 경로 키를 가져도 서로 다른 문서로 남는다.
        Index(
            "uq_document_sources_group_source_id_document_key",
            "group_source_id",
            "document_key",
            unique=True,
            postgresql_where=text("group_source_id IS NOT NULL"),
        ),
        Index(
            "uq_document_sources_document_group_id_document_key",
            "document_group_id",
            "document_key",
            unique=True,
            postgresql_where=text("group_source_id IS NULL"),
        ),
        UniqueConstraint("document_group_id", "canonical_uri"),
        Index(None, "document_group_id"),
        Index(None, "group_source_id"),
        # 원천을 거친 문서 그룹과 직접 적은 문서 그룹이 같아야 한다.
        # 원천이 없는 업로드 문서는 복합 외래키 검사를 받지 않는다.
        ForeignKeyConstraint(
            ["group_source_id", "document_group_id"],
            [
                "document_group_sources.id",
                "document_group_sources.document_group_id",
            ],
            name=DOCUMENT_SOURCE_GROUP_SOURCE_FK_CONSTRAINT,
            ondelete="RESTRICT",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    document_group_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_groups.id", ondelete="RESTRICT"),
        nullable=False,
    )
    group_source_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    document_key: Mapped[str] = mapped_column(String(300), nullable=False)
    source_type: Mapped[str] = mapped_column(String(30), nullable=False)
    canonical_uri: Mapped[str] = mapped_column(String(1000), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(300))
    metadata_: Mapped[Optional[dict[str, Any]]] = mapped_column("metadata", JSONB)
    enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class DocumentVersion(Base):
    """문서가 변경될 때마다 생성하는 불변 버전."""

    __tablename__ = "document_versions"
    __table_args__ = (
        CheckConstraint(
            "raw_content_uri IS NOT NULL OR raw_content IS NOT NULL",
            name="raw_content_storage",
        ),
        UniqueConstraint("document_source_id", "version_no"),
        Index(None, "document_source_id", "normalized_content_hash"),
        Index(None, "normalized_content_hash"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    document_source_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_sources.id", ondelete="RESTRICT"),
        nullable=False,
    )
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_content_uri: Mapped[Optional[str]] = mapped_column(String(1000))
    raw_content: Mapped[Optional[str]] = mapped_column(Text)
    mime_type: Mapped[str] = mapped_column(String(150), nullable=False)
    raw_content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    normalized_content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    parser_name: Mapped[str] = mapped_column(String(100), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[DocumentVersionStatus] = mapped_column(
        _status_enum(DocumentVersionStatus, "document_version_status"),
        nullable=False,
    )
    source_updated_at: Mapped[Optional[Any]] = mapped_column(TIMESTAMP(timezone=True))
    collected_at: Mapped[Any] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class IngestionRun(Base):
    """문서 수집과 파싱 실행 이력."""

    __tablename__ = "ingestion_runs"
    __table_args__ = (Index(None, "batch_id"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    document_source_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_sources.id", ondelete="RESTRICT"),
        nullable=False,
    )
    produced_version_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("document_versions.id", ondelete="SET NULL")
    )
    # 명명 규칙대로 referred table까지 붙이면 식별자 63자 제한을 넘어 이름만 줄인다.
    duplicate_of_document_source_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey(
            "document_sources.id",
            ondelete="SET NULL",
            name=DUPLICATE_DOCUMENT_SOURCE_CONSTRAINT,
        ),
    )
    trigger_type: Mapped[str] = mapped_column(String(30), nullable=False)
    parser_name: Mapped[str] = mapped_column(String(100), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[ExecutionStatus] = mapped_column(
        _status_enum(ExecutionStatus, "ingestion_execution_status"), nullable=False
    )
    result_code: Mapped[Optional[IngestionResultCode]] = mapped_column(
        _status_enum(IngestionResultCode, "ingestion_result_code")
    )
    stage: Mapped[Optional[IngestionStage]] = mapped_column(
        _status_enum(IngestionStage, "ingestion_stage")
    )
    error_code: Mapped[Optional[str]] = mapped_column(String(50))
    batch_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    summary: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[Any] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    finished_at: Mapped[Optional[Any]] = mapped_column(TIMESTAMP(timezone=True))


class ContentNode(Base):
    """파싱과 청킹을 거쳐 저장이 확정된 검색 가능한 최소 논리 단위."""

    __tablename__ = "content_nodes"
    __table_args__ = (
        Index(None, "document_version_id", "node_order"),
        Index(None, "document_version_id", "content_hash"),
        Index(None, "document_version_id", "node_identity_hash"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    document_version_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    parent_node_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("content_nodes.id", ondelete="SET NULL")
    )
    node_type: Mapped[str] = mapped_column(String(40), nullable=False)
    node_path: Mapped[Optional[str]] = mapped_column(String(1000))
    node_order: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(500))
    normalized_content: Mapped[str] = mapped_column(Text, nullable=False)
    source_locator: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    # 재색인 간 동일 노드 추적용 신원 해시 — content_hash와 달리 내용이 바뀌어도 불변.
    # MVP는 nullable로 시작하고 적재 로직 안정 후 제약을 조인다
    node_identity_hash: Mapped[Optional[str]] = mapped_column(String(128))
    node_identity_kind: Mapped[Optional[str]] = mapped_column(String(30))
    metadata_: Mapped[Optional[dict[str, Any]]] = mapped_column("metadata", JSONB)
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class ChunkingConfig(Base):
    """청크 크기와 overlap 등 청킹 정책 버전."""

    __tablename__ = "chunking_configs"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    strategy: Mapped[str] = mapped_column(String(50), nullable=False)
    max_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    overlap_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    parameters: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class DocumentChunk(Base):
    """content_nodes의 검색·토큰·임베딩 입력 속성을 담는 공유 PK 1:1 확장 객체."""

    __tablename__ = "document_chunks"
    __table_args__ = (Index(None, "chunking_config_id", "chunk_index"),)

    id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("content_nodes.id", ondelete="CASCADE"),
        primary_key=True,
        comment="content_nodes.id와 동일한 공유 PK",
    )
    chunking_config_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("chunking_configs.id", ondelete="RESTRICT"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    embedding_input_hash: Mapped[Optional[str]] = mapped_column(String(128))
    keyword_search_text: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


# ---------------------------------------------------------------------------
# 검색·색인 영역
# ---------------------------------------------------------------------------


class EmbeddingConfig(Base):
    """임베딩 제공자와 모델 설정 버전."""

    __tablename__ = "embedding_configs"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    model_name: Mapped[str] = mapped_column(String(150), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    input_template_version: Mapped[str] = mapped_column(String(50), nullable=False)
    parameters: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class ChunkEmbedding(Base):
    """청크에서 생성한 임베딩 벡터. 재생성 가능한 파생 데이터."""

    __tablename__ = "chunk_embeddings"
    __table_args__ = (UniqueConstraint("chunk_id", "embedding_config_id"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    chunk_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=False,
    )
    embedding_config_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("embedding_configs.id", ondelete="RESTRICT"),
        nullable=False,
    )
    embedding: Mapped[List[float]] = mapped_column(
        VECTOR(EMBEDDING_DIMENSIONS), nullable=False
    )
    embedding_input_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class IndexVersion(Base):
    """검색에 사용할 문서와 검색 설정의 버전."""

    __tablename__ = "index_versions"
    __table_args__ = (
        Index(None, "document_group_id"),
        # 그룹마다 ACTIVE 색인은 최대 하나다.
        Index(
            ACTIVE_INDEX_VERSION_CONSTRAINT,
            "document_group_id",
            unique=True,
            postgresql_where=text("status = 'ACTIVE'"),
        ),
        # 번호는 READY 시점에 부여하므로 그 전에는 NULL이다.
        Index(
            INDEX_VERSION_NO_CONSTRAINT,
            "document_group_id",
            "version_no",
            unique=True,
            postgresql_where=text("version_no IS NOT NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    document_group_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_groups.id", ondelete="RESTRICT"),
        nullable=False,
    )
    version: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    version_no: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[IndexVersionStatus] = mapped_column(
        _status_enum(IndexVersionStatus, "index_version_status"), nullable=False
    )
    chunking_config_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("chunking_configs.id", ondelete="RESTRICT"),
        nullable=False,
    )
    embedding_config_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("embedding_configs.id", ondelete="RESTRICT"),
        nullable=False,
    )
    keyword_config: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    fusion_config: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    activated_at: Mapped[Optional[Any]] = mapped_column(TIMESTAMP(timezone=True))


class IndexDocument(Base):
    """하나의 색인 버전에 포함된 문서 버전 목록."""

    __tablename__ = "index_documents"

    index_version_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("index_versions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    document_version_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_versions.id", ondelete="RESTRICT"),
        primary_key=True,
    )


class IndexRun(Base):
    """색인 생성, 검증, 활성화 실행 이력."""

    __tablename__ = "index_runs"
    __table_args__ = (Index(None, "index_version_id", "started_at"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    index_version_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("index_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    trigger_type: Mapped[str] = mapped_column(String(30), nullable=False)
    operation_type: Mapped[IndexOperationType] = mapped_column(
        _status_enum(IndexOperationType, "index_operation_type"), nullable=False
    )
    stage: Mapped[IndexRunStage] = mapped_column(
        _status_enum(IndexRunStage, "index_run_stage"), nullable=False
    )
    actor_id: Mapped[Optional[str]] = mapped_column(String(100))
    status: Mapped[ExecutionStatus] = mapped_column(
        _status_enum(ExecutionStatus, "index_execution_status"), nullable=False
    )
    error_code: Mapped[Optional[str]] = mapped_column(String(50))
    summary: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[Any] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    finished_at: Mapped[Optional[Any]] = mapped_column(TIMESTAMP(timezone=True))


# ---------------------------------------------------------------------------
# 대화·RAG 영역
# ---------------------------------------------------------------------------


class Conversation(Base):
    """다중 턴 대화의 상위 객체. 질문과 답변은 rag_runs에 턴 단위로 저장한다."""

    __tablename__ = "conversations"
    __table_args__ = (Index(None, "client_key", "last_active_at"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    chat_profile_revision_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "chat_profile_revisions.id",
            name=CHAT_PROFILE_REVISION_FK_CONSTRAINT,
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    channel: Mapped[ConversationChannel] = mapped_column(
        _status_enum(ConversationChannel, "conversation_channel"),
        nullable=False,
        default=ConversationChannel.PUBLIC,
        server_default=ConversationChannel.PUBLIC.value,
    )
    client_key: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        comment="익명/Mock 사용자 식별값. MVP에서는 기록하지 않고 로그인 확장 시 사용",
    )
    status: Mapped[ConversationStatus] = mapped_column(
        _status_enum(ConversationStatus, "conversation_status"),
        nullable=False,
        default=ConversationStatus.ACTIVE,
        server_default=ConversationStatus.ACTIVE.value,
    )
    title: Mapped[Optional[str]] = mapped_column(String(300))
    summary_text: Mapped[Optional[str]] = mapped_column(Text)
    summary_version: Mapped[Optional[str]] = mapped_column(String(50))
    summary_updated_turn_no: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    last_active_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    closed_at: Mapped[Optional[Any]] = mapped_column(TIMESTAMP(timezone=True))


class RagRun(Base):
    """대화 안의 사용자 질문 한 번과 답변 한 번을 처리한 RAG 턴 실행."""

    __tablename__ = "rag_runs"
    __table_args__ = (
        UniqueConstraint("conversation_id", "turn_no"),
        Index(None, "conversation_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    trace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    turn_no: Mapped[int] = mapped_column(Integer, nullable=False)
    index_version_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("index_versions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    resolved_query: Mapped[Optional[str]] = mapped_column(Text)
    query_hash: Mapped[Optional[str]] = mapped_column(String(128))
    context_strategy: Mapped[ContextStrategy] = mapped_column(
        _status_enum(ContextStrategy, "context_strategy", length=30),
        nullable=False,
    )
    context_turn_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    context_snapshot: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    status: Mapped[AnswerStatus] = mapped_column(
        _status_enum(AnswerStatus, "answer_status"),
        nullable=False,
        comment="API 응답의 status와 동일한 값을 사용한다(매핑 계층 없음)",
    )
    withheld_reason_code: Mapped[Optional[str]] = mapped_column(
        String(50),
        comment=(
            "WITHHELD일 때만 기록: INSUFFICIENT_EVIDENCE, AMBIGUOUS_QUESTION, "
            "OUT_OF_SCOPE, UNVERIFIABLE_ANSWER"
        ),
    )
    error_code: Mapped[Optional[str]] = mapped_column(
        String(50),
        comment="ERROR일 때만 기록: UPSTREAM_ERROR, CITATION_VALIDATION_ERROR 등",
    )
    answer_content: Mapped[Optional[str]] = mapped_column(Text)
    answer_schema_version: Mapped[Optional[str]] = mapped_column(String(50))
    citation_validated: Mapped[Optional[bool]] = mapped_column(Boolean)
    total_latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[Optional[Any]] = mapped_column(TIMESTAMP(timezone=True))


class RetrievalResultRow(Base):
    """턴별 검색 후보, 순위와 최종 근거 선택 결과."""

    __tablename__ = "retrieval_results"
    __table_args__ = (
        UniqueConstraint("rag_run_id", "chunk_id", "retriever_type"),
        Index(None, "rag_run_id", "fused_rank"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    rag_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rag_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_chunks.id", ondelete="RESTRICT"),
        nullable=False,
    )
    retriever_type: Mapped[RetrieverType] = mapped_column(
        _status_enum(RetrieverType, "retriever_type", length=30), nullable=False
    )
    raw_score: Mapped[Optional[float]] = mapped_column(Numeric)
    retriever_rank: Mapped[Optional[int]] = mapped_column(Integer)
    fused_rank: Mapped[Optional[int]] = mapped_column(Integer)
    fused_score: Mapped[Optional[float]] = mapped_column(
        Numeric,
        comment="융합 결과에 든 청크의 RRF 점수. 검색기별 행에 같은 값을 기록한다",
    )
    selected_as_evidence: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class ModelCall(Base):
    """질문 재작성, 임베딩, 답변 생성, 대화 요약, 질문 판별 등 모델 호출 이력."""

    __tablename__ = "model_calls"
    __table_args__ = (
        CheckConstraint(MODEL_CALL_OWNER_COMBINATION, name="owner_combination"),
        # 캐시 입력과 추론 출력은 각 총량에 포함된 부분값이다.
        CheckConstraint(
            "cached_input_tokens IS NULL OR (cached_input_tokens >= 0"
            " AND (input_tokens IS NULL OR cached_input_tokens <= input_tokens))",
            name="cached_input_tokens",
        ),
        CheckConstraint(
            "reasoning_tokens IS NULL OR (reasoning_tokens >= 0"
            " AND (output_tokens IS NULL OR reasoning_tokens <= output_tokens))",
            name="reasoning_tokens",
        ),
        Index(None, "classification_run_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    rag_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rag_runs.id", ondelete="CASCADE")
    )
    index_run_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("index_runs.id", ondelete="CASCADE")
    )
    ingestion_run_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("ingestion_runs.id", ondelete="CASCADE")
    )
    classification_run_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("classification_runs.id", ondelete="CASCADE")
    )
    purpose: Mapped[ModelCallPurpose] = mapped_column(
        _status_enum(ModelCallPurpose, "model_call_purpose", length=40),
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    model_name: Mapped[str] = mapped_column(String(150), nullable=False)
    prompt_version: Mapped[Optional[str]] = mapped_column(String(50))
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    cached_input_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    reasoning_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    estimated_cost: Mapped[Optional[float]] = mapped_column(Numeric)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[ExecutionStatus] = mapped_column(
        _status_enum(ExecutionStatus, "model_call_execution_status"), nullable=False
    )
    retry_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class AnswerCitation(Base):
    """턴의 최종 답변과 실제 근거 청크 연결. 메타데이터 스냅샷 보존."""

    __tablename__ = "answer_citations"
    __table_args__ = (UniqueConstraint("rag_run_id", "citation_order"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    rag_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rag_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_chunks.id", ondelete="RESTRICT"),
        nullable=False,
    )
    document_version_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_versions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    citation_order: Mapped[int] = mapped_column(Integer, nullable=False)
    document_title_snapshot: Mapped[Optional[str]] = mapped_column(String(500))
    node_path_snapshot: Mapped[Optional[str]] = mapped_column(String(1000))
    source_uri_snapshot: Mapped[Optional[str]] = mapped_column(String(1000))
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class Feedback(Base):
    """턴별 최종 답변에 대한 사용자 평가. 답변당 1건(반대 값으로 변경 가능)."""

    __tablename__ = "feedbacks"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    rag_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rag_runs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    rating: Mapped[FeedbackRating] = mapped_column(
        _status_enum(FeedbackRating, "feedback_rating", length=30), nullable=False
    )
    reason_code: Mapped[Optional[str]] = mapped_column(String(50))
    comment: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="평가를 반대 값으로 변경한 시각. 신규 등록 시에는 created_at과 같다",
    )


# ---------------------------------------------------------------------------
# 질문 그룹핑 영역
# 명명 규칙대로면 식별자 63자를 넘는 외래키와 유니크는 참조 테이블을 빼고 이름을 직접 지정한다.
# ---------------------------------------------------------------------------


class QuestionProblemGroup(Base):
    """세부 문제의 상위. 문서 한 건이나 문서 그룹의 가이드 밖 질문을 가리킨다.

    DOCUMENT 는 문서만, NO_DOCUMENT 는 문서 그룹만 채운다. 문서 그룹을 두 번
    적지 않아 어긋날 수 없고, 제목은 복사하지 않는다.
    """

    __tablename__ = "question_problem_groups"
    __table_args__ = (
        CheckConstraint(
            "(kind = 'DOCUMENT'"
            " AND document_source_id IS NOT NULL AND document_group_id IS NULL)"
            " OR (kind = 'NO_DOCUMENT'"
            " AND document_source_id IS NULL AND document_group_id IS NOT NULL)",
            name="kind_target",
        ),
        # 널은 서로 다르므로 채운 행끼리만 유일하다.
        UniqueConstraint("document_source_id"),
        # NO_DOCUMENT 행만 문서 그룹을 채우므로 문서 그룹당 가이드 밖 그룹은 하나다.
        UniqueConstraint("document_group_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    kind: Mapped[QuestionProblemGroupKind] = mapped_column(
        _status_enum(QuestionProblemGroupKind, "problem_group_kind"),
        nullable=False,
    )
    document_source_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("document_sources.id", ondelete="RESTRICT")
    )
    document_group_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("document_groups.id", ondelete="RESTRICT")
    )
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class QuestionSubproblem(Base):
    """분류의 단위이자 캐시의 단위인 세부 문제."""

    __tablename__ = "question_subproblems"
    __table_args__ = (
        # 같은 분류 체계를 다른 문서 그룹에 넣을 수 있어 문제 그룹 안에서만 유일하다.
        UniqueConstraint("problem_group_id", "key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    problem_group_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "question_problem_groups.id",
            ondelete="RESTRICT",
            name="fk_question_subproblems_problem_group_id",
        ),
        nullable=False,
    )
    # 시드가 재실행과 환경을 넘어 같은 세부 문제를 알아보는 외부 식별자. 바꾸지 않는다.
    key: Mapped[str] = mapped_column(String(200), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    # 임베딩과 판정에 쓴다. 제외 기준은 판정에만 쓴다.
    inclusion_criteria: Mapped[str] = mapped_column(Text, nullable=False)
    exclusion_criteria: Mapped[Optional[str]] = mapped_column(Text)
    current_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[QuestionSubproblemStatus] = mapped_column(
        _status_enum(QuestionSubproblemStatus, "subproblem_status"),
        nullable=False,
    )
    serving_state: Mapped[QuestionSubproblemServingState] = mapped_column(
        _status_enum(QuestionSubproblemServingState, "subproblem_serving_state"),
        nullable=False,
        default=QuestionSubproblemServingState.UNUSED,
        server_default=QuestionSubproblemServingState.UNUSED.value,
    )
    created_by: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class QuestionSubproblemRevision(Base):
    """판정이 기록한 세부 문제 개정 번호가 가리키는 정의 스냅샷."""

    __tablename__ = "question_subproblem_revisions"
    __table_args__ = (
        UniqueConstraint("subproblem_id", "version"),
        # 벡터, 임베딩 설정, 입력 문장 구성 판은 함께 채우거나 함께 비운다.
        CheckConstraint(
            "num_nulls(inclusion_embedding, embedding_config_id,"
            " embedding_text_version) IN (0, 3)",
            name="inclusion_embedding",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    subproblem_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "question_subproblems.id",
            ondelete="RESTRICT",
            name="fk_question_subproblem_revisions_subproblem_id",
        ),
        nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    name_snapshot: Mapped[str] = mapped_column(String(200), nullable=False)
    inclusion_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    exclusion_snapshot: Mapped[Optional[str]] = mapped_column(Text)
    change_reason: Mapped[Optional[str]] = mapped_column(Text)
    # 온라인 세부 문제 후보 검색에 쓰는 포함 기준 벡터. 시드 스크립트가 계산한다.
    inclusion_embedding: Mapped[Optional[List[float]]] = mapped_column(
        VECTOR(EMBEDDING_DIMENSIONS)
    )
    embedding_config_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey(
            "embedding_configs.id",
            ondelete="RESTRICT",
            name="fk_question_subproblem_revisions_embedding_config_id",
        ),
    )
    embedding_text_version: Mapped[Optional[str]] = mapped_column(String(50))
    approved_by: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class ClassificationRun(Base):
    """질문 연결 행을 어떤 문서 그룹, 색인 판, 모델, 프롬프트로 만들었는지 묶는 실행."""

    __tablename__ = "classification_runs"
    __table_args__ = (
        Index(None, "document_group_id", "started_at"),
        # 동시 턴이 같은 설정의 ONLINE 실행을 둘 열지 못하게 한다.
        Index(
            OPEN_ONLINE_CLASSIFICATION_RUN_CONSTRAINT,
            "document_group_id",
            "index_version_id",
            "model",
            "prompt_version",
            unique=True,
            postgresql_where=text("kind = 'ONLINE' AND finished_at IS NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    document_group_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_groups.id", ondelete="RESTRICT"),
        nullable=False,
    )
    index_version_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("index_versions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    kind: Mapped[ClassificationRunKind] = mapped_column(
        _status_enum(ClassificationRunKind, "classification_run_kind"),
        nullable=False,
    )
    model: Mapped[str] = mapped_column(String(150), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(50), nullable=False)
    row_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    started_at: Mapped[Any] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    finished_at: Mapped[Optional[Any]] = mapped_column(TIMESTAMP(timezone=True))
    actor: Mapped[Optional[str]] = mapped_column(String(100))


class QuestionClassification(Base):
    """질문이 어느 세부 문제와 문제 그룹에 붙었는지 기록하는 연결 행.

    재분류는 이전 행에 effective_to 를 찍고 새 행을 추가한다. effective_to 가
    널인 행이 현재 값이다. 턴 끝 인용 귀속만 같은 행의 problem_group_id 와
    attribution_source 를 덮어쓴다.
    """

    __tablename__ = "question_classifications"
    __table_args__ = (
        CheckConstraint(
            "(decision = 'CONNECT') = (subproblem_id IS NOT NULL)",
            name="connect_subproblem",
        ),
        CheckConstraint(
            "(decision = 'CONNECT') = (attribution_source = 'SUBPROBLEM')",
            name="connect_attribution",
        ),
        CheckConstraint(
            "(subproblem_id IS NULL) = (subproblem_version IS NULL)",
            name="subproblem_version",
        ),
        CheckConstraint(
            "effective_to IS NULL OR effective_to >= effective_from",
            name="effective_period",
        ),
        Index(
            CURRENT_CLASSIFICATION_CONSTRAINT,
            "rag_run_id",
            unique=True,
            postgresql_where=text("effective_to IS NULL"),
        ),
        Index(None, "problem_group_id"),
        Index(None, "subproblem_id"),
        Index(None, "run_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    rag_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rag_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    subproblem_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("question_subproblems.id", ondelete="RESTRICT"),
    )
    # 세부 문제가 이미 그룹을 갖지만 판정 시점의 그룹을 기록으로 남긴다.
    problem_group_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "question_problem_groups.id",
            ondelete="RESTRICT",
            name="fk_question_classifications_problem_group_id",
        ),
        nullable=False,
    )
    run_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("classification_runs.id", ondelete="RESTRICT"),
        nullable=False,
    )
    decision: Mapped[ClassificationDecision] = mapped_column(
        _status_enum(ClassificationDecision, "classification_decision"),
        nullable=False,
    )
    subproblem_version: Mapped[Optional[int]] = mapped_column(Integer)
    attribution_source: Mapped[AttributionSource] = mapped_column(
        _status_enum(AttributionSource, "attribution_source"),
        nullable=False,
    )
    is_composite: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    confidence: Mapped[Optional[float]] = mapped_column(Numeric)
    judgment_input: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    effective_from: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    effective_to: Mapped[Optional[Any]] = mapped_column(TIMESTAMP(timezone=True))


class QuestionEmbedding(Base):
    """판별과 같은 resolved_query 를 임베딩한 질문 벡터."""

    __tablename__ = "question_embeddings"

    rag_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rag_runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding: Mapped[List[float]] = mapped_column(
        VECTOR(EMBEDDING_DIMENSIONS), nullable=False
    )
    embedding_config_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("embedding_configs.id", ondelete="RESTRICT"),
        nullable=False,
    )
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class CanonicalAnswer(Base):
    """캐시가 반환할 정본. 행을 고치지 않고 REVOKED 로 내린 뒤 새 행을 넣는다."""

    __tablename__ = "canonical_answers"
    __table_args__ = (
        Index(
            APPROVED_CANONICAL_ANSWER_CONSTRAINT,
            "subproblem_id",
            unique=True,
            postgresql_where=text("approval = 'APPROVED'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    subproblem_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("question_subproblems.id", ondelete="RESTRICT"),
        nullable=False,
    )
    origin: Mapped[CanonicalAnswerOrigin] = mapped_column(
        _status_enum(CanonicalAnswerOrigin, "canonical_answer_origin"),
        nullable=False,
    )
    # 출처로만 남긴다. 질문 행이 지워져도 정본은 본문과 인용을 직접 소유한다.
    source_rag_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rag_runs.id", ondelete="SET NULL")
    )
    content_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    applicability_rules: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    subproblem_version: Mapped[int] = mapped_column(Integer, nullable=False)
    filter_results: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    approval: Mapped[CanonicalAnswerApproval] = mapped_column(
        _status_enum(CanonicalAnswerApproval, "canonical_answer_approval"),
        nullable=False,
    )
    approved_by: Mapped[Optional[str]] = mapped_column(String(100))
    valid_from: Mapped[Optional[Any]] = mapped_column(TIMESTAMP(timezone=True))
    valid_to: Mapped[Optional[Any]] = mapped_column(TIMESTAMP(timezone=True))
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class CanonicalAnswerCitation(Base):
    """정본 인용. 캐시 적중 턴의 answer_citations 로 그대로 복사한다."""

    __tablename__ = "canonical_answer_citations"
    __table_args__ = (
        UniqueConstraint(
            "canonical_answer_id",
            "citation_order",
            name="uq_canonical_answer_citations_answer_id_citation_order",
        ),
        Index(None, "chunk_id"),
        Index(None, "document_version_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    canonical_answer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "canonical_answers.id",
            ondelete="CASCADE",
            name="fk_canonical_answer_citations_canonical_answer_id",
        ),
        nullable=False,
    )
    citation_order: Mapped[int] = mapped_column(Integer, nullable=False)
    # 정본이 쓰는 청크와 문서 판이 지워지지 않게 막는다.
    chunk_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_chunks.id", ondelete="RESTRICT"),
        nullable=False,
    )
    document_version_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "document_versions.id",
            ondelete="RESTRICT",
            name="fk_canonical_answer_citations_document_version_id",
        ),
        nullable=False,
    )
    document_title_snapshot: Mapped[Optional[str]] = mapped_column(String(500))
    node_path_snapshot: Mapped[Optional[str]] = mapped_column(String(1000))
    source_uri_snapshot: Mapped[Optional[str]] = mapped_column(String(1000))
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class QuestionCacheAttempt(Base):
    """재작성을 통과한 턴의 캐시 시도와 게이트 결과. 턴마다 많아야 하나다."""

    __tablename__ = "question_cache_attempts"
    __table_args__ = (
        CheckConstraint(
            "(outcome IN ('SERVED', 'SHADOW', 'GROUP_DISABLED'))"
            " = (canonical_answer_id IS NOT NULL)",
            name="outcome_canonical_answer",
        ),
        CheckConstraint(
            "(outcome = 'REJECTED')"
            " = (COALESCE(cardinality(rejection_reasons), 0) > 0)",
            name="outcome_rejection_reasons",
        ),
        Index(None, "canonical_answer_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    rag_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rag_runs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    # 판정 없이 건너뛰면 비운다. 널은 서로 다르므로 채운 행끼리만 유일하다.
    # 턴 삭제가 연결 행과 시도를 함께 지우므로 문장 끝에 검사하는 NO ACTION 을 쓴다.
    classification_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey(
            "question_classifications.id",
            ondelete="NO ACTION",
            name="fk_question_cache_attempts_classification_id",
        ),
        unique=True,
    )
    outcome: Mapped[CacheAttemptOutcome] = mapped_column(
        _status_enum(CacheAttemptOutcome, "cache_attempt_outcome"),
        nullable=False,
    )
    canonical_answer_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "canonical_answers.id",
            ondelete="RESTRICT",
            name="fk_question_cache_attempts_canonical_answer_id",
        ),
    )
    rejection_reasons: Mapped[Optional[List[str]]] = mapped_column(
        ARRAY(String(50))
    )
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[Any] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
