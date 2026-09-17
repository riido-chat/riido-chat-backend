"""Require explicit question-log approval for exact-match cache serving.

Revision ID: 20260918_16
Revises: 20260918_15
Create Date: 2026-09-18
"""

from typing import Optional, Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260918_16"
down_revision: Optional[str] = "20260918_15"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


def upgrade() -> None:
    # 기존 질문 로그는 추천 질문으로 승인한 근거가 없으므로 모두 fail-closed 한다.
    op.add_column(
        "question_classifications",
        sa.Column(
            "exact_cache_approved",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    # 정확 일치 요청마다 과거 전체 질문 로그를 훑지 않도록 해시 후보를 먼저 찾는다.
    op.create_index("ix_rag_runs_query_hash", "rag_runs", ["query_hash"])


def downgrade() -> None:
    op.drop_index("ix_rag_runs_query_hash", table_name="rag_runs")
    op.drop_column("question_classifications", "exact_cache_approved")
