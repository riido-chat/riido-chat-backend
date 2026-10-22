"""Versioned question grouping, reviewed display text and access audits.

Revision ID: 20260908_10
Revises: 20260905_09
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

revision = "20260908_10"
down_revision = "20260905_09"
branch_labels = None
depends_on = None


def stamp():
    return sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.clock_timestamp())


def upgrade():
    op.create_table(
        "question_group_revisions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("group_id", pg.UUID(as_uuid=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("inclusion", sa.Text(), nullable=False),
        sa.Column("exclusion", sa.Text(), nullable=False),
        sa.Column("document_urls", pg.JSONB(), nullable=False),
        sa.Column("change_reason", sa.Text(), nullable=False),
        sa.Column("representative_run_id", pg.UUID(as_uuid=True), sa.ForeignKey("rag_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("approved_by", sa.String(100), nullable=False), stamp(),
        sa.UniqueConstraint("group_id", "version"),
    )
    op.create_table(
        "question_reviews",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("rag_run_id", pg.UUID(as_uuid=True), sa.ForeignKey("rag_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("display_question", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("reviewed_by", sa.String(100), nullable=False), stamp(),
    )
    op.create_table(
        "question_review_intents",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("review_id", sa.BigInteger(), sa.ForeignKey("question_reviews.id", ondelete="CASCADE"), nullable=False),
        sa.Column("group_revision_id", sa.BigInteger(), sa.ForeignKey("question_group_revisions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("intent", sa.Text(), nullable=False),
    )
    op.create_table(
        "question_access_audits",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("rag_run_id", pg.UUID(as_uuid=True), sa.ForeignKey("rag_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("actor", sa.String(100), nullable=False),
        sa.Column("action", sa.String(40), nullable=False), stamp(),
    )
    for table, column in (
        ("question_reviews", "rag_run_id"), ("question_review_intents", "review_id"),
        ("question_review_intents", "group_revision_id"), ("question_access_audits", "rag_run_id"),
    ):
        op.create_index(f"ix_{table}_{column}", table, [column])


def downgrade():
    for table in ("question_access_audits", "question_review_intents", "question_reviews", "question_group_revisions"):
        op.drop_table(table)
