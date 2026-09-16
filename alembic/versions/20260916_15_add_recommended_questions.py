"""추천 질문과 과거 정본 서빙 질문의 정확 일치 매핑을 추가한다.

Revision ID: 20260916_15
Revises: 20260915_14
Create Date: 2026-09-16
"""

from typing import Optional, Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260916_15"
down_revision: Optional[str] = "20260915_14"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


def upgrade() -> None:
    op.create_table(
        "exact_question_matches",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("document_group_id", sa.BigInteger(), nullable=False),
        sa.Column("subproblem_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("normalized_question", sa.Text(), nullable=False),
        sa.Column("source", sa.String(30), nullable=False, server_default="RECOMMENDED"),
        sa.Column("state", sa.String(20), nullable=False, server_default="ACTIVE"),
        sa.Column("subproblem_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("canonical_answer_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_rag_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["document_group_id"], ["document_groups.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["subproblem_id"], ["question_subproblems.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["canonical_answer_id"], ["canonical_answers.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["source_rag_run_id"], ["rag_runs.id"], ondelete="RESTRICT"
        ),
        sa.UniqueConstraint(
            "document_group_id",
            "normalized_question",
            name="uq_exact_question_matches_document_group_normalized",
        ),
        sa.CheckConstraint(
            "length(btrim(question)) > 0", name="question_nonempty"
        ),
        sa.CheckConstraint(
            "length(btrim(normalized_question)) > 0",
            name="normalized_question_nonempty",
        ),
        sa.CheckConstraint(
            "source IN ('RECOMMENDED', 'HISTORICAL_SERVED')",
            name="exact_question_match_source",
        ),
        sa.CheckConstraint(
            "state IN ('ACTIVE', 'CONFLICT')",
            name="exact_question_match_state",
        ),
        sa.CheckConstraint(
            "subproblem_version > 0",
            name="ck_exact_question_matches_subproblem_version_positive",
        ),
        sa.CheckConstraint(
            "(source = 'RECOMMENDED' AND source_rag_run_id IS NULL) OR "
            "(source = 'HISTORICAL_SERVED' AND source_rag_run_id IS NOT NULL)",
            name="ck_exact_question_matches_source_provenance",
        ),
    )
    op.create_index(
        "ix_exact_question_matches_subproblem_id",
        "exact_question_matches",
        ["subproblem_id"],
    )
    op.create_index(
        "ix_exact_question_matches_source_rag_run_id",
        "exact_question_matches",
        ["source_rag_run_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_exact_question_matches_source_rag_run_id", table_name="exact_question_matches"
    )
    op.drop_index(
        "ix_exact_question_matches_subproblem_id", table_name="exact_question_matches"
    )
    op.drop_table("exact_question_matches")
