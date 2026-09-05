"""ERD v0.2.2 기준 애플리케이션 ORM model을 정의한다.

- 문서 영역: document_sources → document_versions → content_nodes ↔ document_chunks(공유 PK 1:1)
- 검색·색인 영역: embedding_configs, chunk_embeddings, index_versions, index_documents, index_runs
- 대화·RAG 영역: conversations → rag_runs → retrieval_results / model_calls / answer_citations / feedbacks
- Legacy: ERD 도입 전 Vector Retrieval 최소 테이블(legacy_*).
  파이프라인 → DB 적재 매핑 확정 후 ERD 테이블로 흡수하고 제거한다.
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


class FeedbackRating(str, enum.Enum):
    """답변에 대한 사용자 평가. 취소는 없고 반대 값으로 변경만 가능하다."""

    GOOD = "GOOD"
    BAD = "BAD"


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


class SourceProvider(str, enum.Enum):
    """수집 원천의 제공자."""

    GITBOOK = "GITBOOK"


class DocumentGroupSource(Base):
    """문서 그룹이 문서를 끌어오는 외부 원천.

    콘솔 업로드처럼 밀어 넣는 문서에는 원천이 없다.
    """

    __tablename__ = "document_group_sources"
    __table_args__ = (UniqueConstraint("document_group_id", "root_url"),)

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
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    document_group_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_groups.id", ondelete="RESTRICT"),
        nullable=False,
    )
    group_source_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("document_group_sources.id", ondelete="RESTRICT"),
    )
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
    sanitized_query: Mapped[Optional[str]] = mapped_column(Text)
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
    """질문 재작성, 임베딩, 답변 생성, 대화 요약 등 모델 호출 이력."""

    __tablename__ = "model_calls"

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
    purpose: Mapped[ModelCallPurpose] = mapped_column(
        _status_enum(ModelCallPurpose, "model_call_purpose", length=40),
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    model_name: Mapped[str] = mapped_column(String(150), nullable=False)
    prompt_version: Mapped[Optional[str]] = mapped_column(String(50))
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer)
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
# Legacy — ERD 도입 전 Vector Retrieval 최소 테이블
# 파이프라인 → ERD 적재 매핑 확정 시 ERD 테이블로 흡수하고 제거한다.
# ---------------------------------------------------------------------------


class LegacyDocumentChunk(Base):
    """(legacy) 검색 결과를 RetrievalChunk로 복원하기 위한 Chunk와 metadata."""

    __tablename__ = "legacy_document_chunks"

    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(Text, nullable=False)
    section_id: Mapped[str] = mapped_column(Text, nullable=False)
    document_title: Mapped[str] = mapped_column(Text, nullable=False)
    section_path: Mapped[List[str]] = mapped_column(ARRAY(Text), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)


class LegacyChunkEmbedding(Base):
    """(legacy) Chunk에 1:1로 종속되는 OpenAI embedding."""

    __tablename__ = "legacy_chunk_embeddings"
    __table_args__ = (
        UniqueConstraint("chunk_id", name="uq_chunk_embeddings_chunk_id"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
    )
    chunk_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey(
            "legacy_document_chunks.chunk_id",
            name="fk_chunk_embeddings_chunk_id_document_chunks",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    embedding: Mapped[List[float]] = mapped_column(
        VECTOR(EMBEDDING_DIMENSIONS),
        nullable=False,
    )
