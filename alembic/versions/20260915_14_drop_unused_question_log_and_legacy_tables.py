"""쓰지 않는 콘솔 검토 테이블, legacy 청크 테이블, rag_runs.sanitized_query를 지운다.

- question_group_revisions, question_reviews, question_review_intents를 지운다.
  question_access_audits는 추후 원문 열람 감사에 쓰도록 남긴다.
- ERD 도입 전 legacy_document_chunks, legacy_chunk_embeddings를 지운다.
- 한 번도 채워진 적이 없는 rag_runs.sanitized_query를 지운다. query_hash는 남긴다.

downgrade는 스키마만 되돌리고 지운 행은 복원하지 않는다.

Revision ID: 20260915_14
Revises: 20260915_13
Create Date: 2026-09-15
"""

from typing import Optional, Sequence, Union

from alembic import op
from pgvector.sqlalchemy import VECTOR
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260915_14"
down_revision: Optional[str] = "20260915_13"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


EMBEDDING_DIMENSIONS = 1536

CONSOLE_REVIEW_TABLES_IN_DROP_ORDER = (
    "question_review_intents",
    "question_reviews",
    "question_group_revisions",
)
LEGACY_TABLES_IN_DROP_ORDER = (
    "legacy_chunk_embeddings",
    "legacy_document_chunks",
)


def _check_unused_before_drop(connection) -> None:
    """쓰는 코드가 없어 비어 있어야 할 대상에 값이 있으면 멈춘다."""

    filled = []
    for table_name in CONSOLE_REVIEW_TABLES_IN_DROP_ORDER:
        count = connection.execute(
            sa.text(f"SELECT COUNT(*) FROM {table_name}")
        ).scalar_one()
        if count:
            filled.append(f"{table_name}={count}")
    sanitized = connection.execute(
        sa.text("SELECT COUNT(*) FROM rag_runs WHERE sanitized_query IS NOT NULL")
    ).scalar_one()
    if sanitized:
        filled.append(f"rag_runs.sanitized_query={sanitized}")
    if filled:
        raise RuntimeError(
            "비어 있어야 할 삭제 대상에 데이터가 있어 멈춥니다: "
            + ", ".join(filled)
        )


def upgrade() -> None:
    _check_unused_before_drop(op.get_bind())

    for table_name in CONSOLE_REVIEW_TABLES_IN_DROP_ORDER:
        op.drop_table(table_name)
    for table_name in LEGACY_TABLES_IN_DROP_ORDER:
        op.drop_table(table_name)
    op.drop_column("rag_runs", "sanitized_query")


def _stamp() -> sa.Column:
    return sa.Column(
        "created_at",
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.func.clock_timestamp(),
    )


def downgrade() -> None:
    op.add_column(
        "rag_runs",
        sa.Column("sanitized_query", sa.Text(), nullable=True),
    )

    # 20260818_01에서 만들고 20260820_02에서 개명한 모양 그대로 되돌린다.
    op.create_table(
        "legacy_document_chunks",
        sa.Column("chunk_id", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("section_id", sa.Text(), nullable=False),
        sa.Column("document_title", sa.Text(), nullable=False),
        sa.Column("section_path", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("category", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("chunk_id", name="pk_legacy_document_chunks"),
    )
    op.create_table(
        "legacy_chunk_embeddings",
        sa.Column(
            "id",
            sa.BigInteger(),
            sa.Identity(always=False),
            nullable=False,
        ),
        sa.Column("chunk_id", sa.Text(), nullable=False),
        sa.Column("embedding", VECTOR(EMBEDDING_DIMENSIONS), nullable=False),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["legacy_document_chunks.chunk_id"],
            name="fk_chunk_embeddings_chunk_id_document_chunks",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_legacy_chunk_embeddings"),
        sa.UniqueConstraint("chunk_id", name="uq_chunk_embeddings_chunk_id"),
    )

    # 20260908_10에서 만든 모양 그대로 되돌린다.
    op.create_table(
        "question_group_revisions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("group_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("inclusion", sa.Text(), nullable=False),
        sa.Column("exclusion", sa.Text(), nullable=False),
        sa.Column("document_urls", postgresql.JSONB(), nullable=False),
        sa.Column("change_reason", sa.Text(), nullable=False),
        sa.Column(
            "representative_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("rag_runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("approved_by", sa.String(100), nullable=False),
        _stamp(),
        sa.UniqueConstraint("group_id", "version"),
    )
    op.create_table(
        "question_reviews",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column(
            "rag_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("rag_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("display_question", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("reviewed_by", sa.String(100), nullable=False),
        _stamp(),
    )
    op.create_table(
        "question_review_intents",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column(
            "review_id",
            sa.BigInteger(),
            sa.ForeignKey("question_reviews.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "group_revision_id",
            sa.BigInteger(),
            sa.ForeignKey("question_group_revisions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("intent", sa.Text(), nullable=False),
    )
    for table, column in (
        ("question_reviews", "rag_run_id"),
        ("question_review_intents", "review_id"),
        ("question_review_intents", "group_revision_id"),
    ):
        op.create_index(f"ix_{table}_{column}", table, [column])
