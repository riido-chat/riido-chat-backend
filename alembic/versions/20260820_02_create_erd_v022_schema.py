"""ERD v0.2.2 전체 스키마를 생성하고 기존 최소 테이블을 legacy로 보존한다.

- 기존 Vector Retrieval 최소 테이블(document_chunks, chunk_embeddings)을
  legacy_document_chunks, legacy_chunk_embeddings로 개명한다 (데이터·임베딩 보존).
- ERD v0.2.2의 17개 테이블을 생성한다 (docs/04-통합ERD.md 기준).
- node_identity_hash / node_identity_kind는 MVP 동안 nullable로 두고,
  적재 로직 안정 후 별도 migration으로 제약을 조인다 (docs/92-현행ID체계참고.md 3.3).

Revision ID: 20260820_02
Revises: 20260818_01
Create Date: 2026-08-20
"""

from typing import Optional, Sequence, Union

from alembic import op
from pgvector.sqlalchemy import VECTOR
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260820_02"
down_revision: Optional[str] = "20260818_01"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


EMBEDDING_DIMENSIONS = 1536

ERD_TABLES_IN_DROP_ORDER = (
    "feedbacks",
    "answer_citations",
    "model_calls",
    "retrieval_results",
    "rag_runs",
    "conversations",
    "index_runs",
    "index_documents",
    "index_versions",
    "chunk_embeddings",
    "document_chunks",
    "content_nodes",
    "ingestion_runs",
    "document_versions",
    "embedding_configs",
    "chunking_configs",
    "document_sources",
)


def _enum(name: str, *values: str) -> sa.Enum:
    """VARCHAR + CHECK 제약으로 저장되는 상태 Enum 타입을 만든다."""

    return sa.Enum(
        *values,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=20,
    )


def _document_version_status() -> sa.Enum:
    return _enum(
        "document_version_status", "PROCESSING", "READY", "FAILED", "ARCHIVED"
    )


def _index_version_status() -> sa.Enum:
    return _enum(
        "index_version_status",
        "BUILDING",
        "VALIDATING",
        "ACTIVE",
        "FAILED",
        "INACTIVE",
    )


def _execution_status(name: str) -> sa.Enum:
    return _enum(name, "PROCESSING", "SUCCESS", "FAILED")


def _conversation_status() -> sa.Enum:
    return _enum("conversation_status", "ACTIVE", "CLOSED", "EXPIRED")


def _answer_status() -> sa.Enum:
    return _enum(
        "answer_status",
        "PROCESSING",
        "COMPLETED",
        "WITHHELD",
        "ERROR",
        "CANCELLED",
    )


def upgrade() -> None:
    # 1) 기존 최소 테이블을 legacy로 보존
    op.rename_table("document_chunks", "legacy_document_chunks")
    op.rename_table("chunk_embeddings", "legacy_chunk_embeddings")
    # PK 제약은 인덱스로 구현되어 테이블과 relation 네임스페이스를 공유하므로,
    # ERD 테이블이 같은 pk_* 이름을 쓸 수 있도록 legacy PK 이름을 함께 개명한다.
    op.execute(
        "ALTER TABLE legacy_document_chunks "
        "RENAME CONSTRAINT pk_document_chunks TO pk_legacy_document_chunks"
    )
    op.execute(
        "ALTER TABLE legacy_chunk_embeddings "
        "RENAME CONSTRAINT pk_chunk_embeddings TO pk_legacy_chunk_embeddings"
    )

    # 2) 문서 영역
    op.create_table(
        "document_sources",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("source_type", sa.String(30), nullable=False),
        sa.Column("canonical_uri", sa.String(1000), nullable=False),
        sa.Column("title", sa.String(300), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_document_sources"),
        sa.UniqueConstraint(
            "canonical_uri", name="uq_document_sources_canonical_uri"
        ),
    )

    op.create_table(
        "document_versions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("document_source_id", sa.BigInteger(), nullable=False),
        sa.Column("version_no", sa.Integer(), nullable=False),
        sa.Column("raw_content_uri", sa.String(1000), nullable=True),
        sa.Column("mime_type", sa.String(150), nullable=False),
        sa.Column("raw_content_hash", sa.String(128), nullable=False),
        sa.Column("normalized_content_hash", sa.String(128), nullable=False),
        sa.Column("parser_name", sa.String(100), nullable=False),
        sa.Column("parser_version", sa.String(50), nullable=False),
        sa.Column("status", _document_version_status(), nullable=False),
        sa.Column("source_updated_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("collected_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_document_versions"),
        sa.ForeignKeyConstraint(
            ["document_source_id"],
            ["document_sources.id"],
            name="fk_document_versions_document_source_id_document_sources",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "document_source_id",
            "version_no",
            name="uq_document_versions_document_source_id_version_no",
        ),
    )
    op.create_index(
        "ix_document_versions_document_source_id_normalized_content_hash"[:63],
        "document_versions",
        ["document_source_id", "normalized_content_hash"],
    )

    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("document_source_id", sa.BigInteger(), nullable=False),
        sa.Column("produced_version_id", sa.BigInteger(), nullable=True),
        sa.Column("trigger_type", sa.String(30), nullable=False),
        sa.Column("parser_name", sa.String(100), nullable=False),
        sa.Column("parser_version", sa.String(50), nullable=False),
        sa.Column(
            "status", _execution_status("ingestion_execution_status"), nullable=False
        ),
        sa.Column("summary", postgresql.JSONB(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_ingestion_runs"),
        sa.ForeignKeyConstraint(
            ["document_source_id"],
            ["document_sources.id"],
            name="fk_ingestion_runs_document_source_id_document_sources",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["produced_version_id"],
            ["document_versions.id"],
            name="fk_ingestion_runs_produced_version_id_document_versions",
            ondelete="SET NULL",
        ),
    )

    op.create_table(
        "content_nodes",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("document_version_id", sa.BigInteger(), nullable=False),
        sa.Column("parent_node_id", sa.BigInteger(), nullable=True),
        sa.Column("node_type", sa.String(40), nullable=False),
        sa.Column("node_path", sa.String(1000), nullable=True),
        sa.Column("node_order", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("normalized_content", sa.Text(), nullable=False),
        sa.Column("source_locator", postgresql.JSONB(), nullable=True),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("node_identity_hash", sa.String(128), nullable=True),
        sa.Column("node_identity_kind", sa.String(30), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_content_nodes"),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
            name="fk_content_nodes_document_version_id_document_versions",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_node_id"],
            ["content_nodes.id"],
            name="fk_content_nodes_parent_node_id_content_nodes",
            ondelete="SET NULL",
        ),
    )
    op.create_index(
        "ix_content_nodes_document_version_id_node_order",
        "content_nodes",
        ["document_version_id", "node_order"],
    )
    op.create_index(
        "ix_content_nodes_document_version_id_content_hash",
        "content_nodes",
        ["document_version_id", "content_hash"],
    )
    op.create_index(
        "ix_content_nodes_document_version_id_node_identity_hash",
        "content_nodes",
        ["document_version_id", "node_identity_hash"],
    )

    op.create_table(
        "chunking_configs",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("strategy", sa.String(50), nullable=False),
        sa.Column("max_tokens", sa.Integer(), nullable=False),
        sa.Column(
            "overlap_tokens",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("parameters", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chunking_configs"),
        sa.UniqueConstraint("version", name="uq_chunking_configs_version"),
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("chunking_config_id", sa.BigInteger(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("embedding_input_hash", sa.String(128), nullable=True),
        sa.Column("keyword_search_text", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_document_chunks"),
        sa.ForeignKeyConstraint(
            ["id"],
            ["content_nodes.id"],
            name="fk_document_chunks_id_content_nodes",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunking_config_id"],
            ["chunking_configs.id"],
            name="fk_document_chunks_chunking_config_id_chunking_configs",
            ondelete="RESTRICT",
        ),
    )
    op.create_index(
        "ix_document_chunks_chunking_config_id_chunk_index",
        "document_chunks",
        ["chunking_config_id", "chunk_index"],
    )

    # 3) 검색·색인 영역
    op.create_table(
        "embedding_configs",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("provider", sa.String(80), nullable=False),
        sa.Column("model_name", sa.String(150), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("input_template_version", sa.String(50), nullable=False),
        sa.Column("parameters", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_embedding_configs"),
        sa.UniqueConstraint("version", name="uq_embedding_configs_version"),
    )

    op.create_table(
        "chunk_embeddings",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("chunk_id", sa.BigInteger(), nullable=False),
        sa.Column("embedding_config_id", sa.BigInteger(), nullable=False),
        sa.Column("embedding", VECTOR(EMBEDDING_DIMENSIONS), nullable=False),
        sa.Column("embedding_input_hash", sa.String(128), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chunk_embeddings"),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["document_chunks.id"],
            name="fk_chunk_embeddings_chunk_id_document_chunks_erd",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["embedding_config_id"],
            ["embedding_configs.id"],
            name="fk_chunk_embeddings_embedding_config_id_embedding_configs",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "chunk_id",
            "embedding_config_id",
            name="uq_chunk_embeddings_chunk_id_embedding_config_id",
        ),
    )

    op.create_table(
        "index_versions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("status", _index_version_status(), nullable=False),
        sa.Column("chunking_config_id", sa.BigInteger(), nullable=False),
        sa.Column("embedding_config_id", sa.BigInteger(), nullable=False),
        sa.Column("keyword_config", postgresql.JSONB(), nullable=True),
        sa.Column("fusion_config", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("activated_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_index_versions"),
        sa.UniqueConstraint("version", name="uq_index_versions_version"),
        sa.ForeignKeyConstraint(
            ["chunking_config_id"],
            ["chunking_configs.id"],
            name="fk_index_versions_chunking_config_id_chunking_configs",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["embedding_config_id"],
            ["embedding_configs.id"],
            name="fk_index_versions_embedding_config_id_embedding_configs",
            ondelete="RESTRICT",
        ),
    )

    op.create_table(
        "index_documents",
        sa.Column("index_version_id", sa.BigInteger(), nullable=False),
        sa.Column("document_version_id", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint(
            "index_version_id", "document_version_id", name="pk_index_documents"
        ),
        sa.ForeignKeyConstraint(
            ["index_version_id"],
            ["index_versions.id"],
            name="fk_index_documents_index_version_id_index_versions",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
            name="fk_index_documents_document_version_id_document_versions",
            ondelete="RESTRICT",
        ),
    )

    op.create_table(
        "index_runs",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("index_version_id", sa.BigInteger(), nullable=False),
        sa.Column("trigger_type", sa.String(30), nullable=False),
        sa.Column("actor_id", sa.String(100), nullable=True),
        sa.Column(
            "status", _execution_status("index_execution_status"), nullable=False
        ),
        sa.Column("summary", postgresql.JSONB(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_index_runs"),
        sa.ForeignKeyConstraint(
            ["index_version_id"],
            ["index_versions.id"],
            name="fk_index_runs_index_version_id_index_versions",
            ondelete="CASCADE",
        ),
    )

    # 4) 대화·RAG 영역
    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("client_key", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "status",
            _conversation_status(),
            nullable=False,
            server_default="ACTIVE",
        ),
        sa.Column("title", sa.String(300), nullable=True),
        sa.Column("summary_text", sa.Text(), nullable=True),
        sa.Column("summary_version", sa.String(50), nullable=True),
        sa.Column("summary_updated_turn_no", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "last_active_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("closed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_conversations"),
    )
    op.create_index(
        "ix_conversations_client_key_last_active_at",
        "conversations",
        ["client_key", "last_active_at"],
    )

    op.create_table(
        "rag_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("trace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("turn_no", sa.Integer(), nullable=False),
        sa.Column("index_version_id", sa.BigInteger(), nullable=False),
        sa.Column("user_query", sa.Text(), nullable=False),
        sa.Column("sanitized_query", sa.Text(), nullable=True),
        sa.Column("resolved_query", sa.Text(), nullable=True),
        sa.Column("query_hash", sa.String(128), nullable=True),
        sa.Column("context_strategy", sa.String(30), nullable=False),
        sa.Column(
            "context_turn_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("context_snapshot", postgresql.JSONB(), nullable=True),
        sa.Column("status", _answer_status(), nullable=False),
        sa.Column("withheld_reason_code", sa.String(50), nullable=True),
        sa.Column("error_code", sa.String(50), nullable=True),
        sa.Column("answer_content", sa.Text(), nullable=True),
        sa.Column("answer_schema_version", sa.String(50), nullable=True),
        sa.Column("citation_validated", sa.Boolean(), nullable=True),
        sa.Column("total_latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_rag_runs"),
        sa.UniqueConstraint("trace_id", name="uq_rag_runs_trace_id"),
        sa.UniqueConstraint(
            "conversation_id", "turn_no", name="uq_rag_runs_conversation_id_turn_no"
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_rag_runs_conversation_id_conversations",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["index_version_id"],
            ["index_versions.id"],
            name="fk_rag_runs_index_version_id_index_versions",
            ondelete="RESTRICT",
        ),
    )
    op.create_index(
        "ix_rag_runs_conversation_id_created_at",
        "rag_runs",
        ["conversation_id", "created_at"],
    )

    op.create_table(
        "retrieval_results",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("rag_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", sa.BigInteger(), nullable=False),
        sa.Column("retriever_type", sa.String(30), nullable=False),
        sa.Column("raw_score", sa.Numeric(), nullable=True),
        sa.Column("retriever_rank", sa.Integer(), nullable=True),
        sa.Column("fused_rank", sa.Integer(), nullable=True),
        sa.Column(
            "selected_as_evidence",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_retrieval_results"),
        sa.UniqueConstraint(
            "rag_run_id",
            "chunk_id",
            "retriever_type",
            name="uq_retrieval_results_rag_run_id_chunk_id_retriever_type",
        ),
        sa.ForeignKeyConstraint(
            ["rag_run_id"],
            ["rag_runs.id"],
            name="fk_retrieval_results_rag_run_id_rag_runs",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["document_chunks.id"],
            name="fk_retrieval_results_chunk_id_document_chunks",
            ondelete="RESTRICT",
        ),
    )
    op.create_index(
        "ix_retrieval_results_rag_run_id_fused_rank",
        "retrieval_results",
        ["rag_run_id", "fused_rank"],
    )

    op.create_table(
        "model_calls",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("rag_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("index_run_id", sa.BigInteger(), nullable=True),
        sa.Column("purpose", sa.String(40), nullable=False),
        sa.Column("provider", sa.String(80), nullable=False),
        sa.Column("model_name", sa.String(150), nullable=False),
        sa.Column("prompt_version", sa.String(50), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("estimated_cost", sa.Numeric(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "status",
            _execution_status("model_call_execution_status"),
            nullable=False,
        ),
        sa.Column(
            "retry_count", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_model_calls"),
        sa.ForeignKeyConstraint(
            ["rag_run_id"],
            ["rag_runs.id"],
            name="fk_model_calls_rag_run_id_rag_runs",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["index_run_id"],
            ["index_runs.id"],
            name="fk_model_calls_index_run_id_index_runs",
            ondelete="CASCADE",
        ),
    )

    op.create_table(
        "answer_citations",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("rag_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", sa.BigInteger(), nullable=False),
        sa.Column("document_version_id", sa.BigInteger(), nullable=False),
        sa.Column("citation_order", sa.Integer(), nullable=False),
        sa.Column("document_title_snapshot", sa.String(500), nullable=True),
        sa.Column("node_path_snapshot", sa.String(1000), nullable=True),
        sa.Column("source_uri_snapshot", sa.String(1000), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_answer_citations"),
        sa.UniqueConstraint(
            "rag_run_id",
            "citation_order",
            name="uq_answer_citations_rag_run_id_citation_order",
        ),
        sa.ForeignKeyConstraint(
            ["rag_run_id"],
            ["rag_runs.id"],
            name="fk_answer_citations_rag_run_id_rag_runs",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["document_chunks.id"],
            name="fk_answer_citations_chunk_id_document_chunks",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
            name="fk_answer_citations_document_version_id_document_versions",
            ondelete="RESTRICT",
        ),
    )

    op.create_table(
        "feedbacks",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("rag_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("rating", sa.String(30), nullable=False),
        sa.Column("reason_code", sa.String(50), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_feedbacks"),
        sa.UniqueConstraint("rag_run_id", name="uq_feedbacks_rag_run_id"),
        sa.ForeignKeyConstraint(
            ["rag_run_id"],
            ["rag_runs.id"],
            name="fk_feedbacks_rag_run_id_rag_runs",
            ondelete="CASCADE",
        ),
    )


def downgrade() -> None:
    for table_name in ERD_TABLES_IN_DROP_ORDER:
        op.drop_table(table_name)

    op.execute(
        "ALTER TABLE legacy_chunk_embeddings "
        "RENAME CONSTRAINT pk_legacy_chunk_embeddings TO pk_chunk_embeddings"
    )
    op.execute(
        "ALTER TABLE legacy_document_chunks "
        "RENAME CONSTRAINT pk_legacy_document_chunks TO pk_document_chunks"
    )
    op.rename_table("legacy_chunk_embeddings", "chunk_embeddings")
    op.rename_table("legacy_document_chunks", "document_chunks")
