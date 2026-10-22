"""Add provenance columns for turns, judge classifications and canonical answers.

모두 널 허용 칸이고 백필하지 않는다. 이 마이그레이션 뒤에 쓰는 행부터 채운다.

- rag_runs.profile_revision_id, build_version: 턴이 실제로 쓴 프로필 판과 서버 빌드.
  대화가 판을 고정해도 이후 턴마다 판을 고르게 되므로 턴 행에 남긴다.
- question_classifications.presented_canonical_answer_id: 판별에 보여 준 고른 세부 문제의 정본.
  judgment_input 제시 목록과 같은 값이다. 운영자 연결과 정확 일치 재사용 행은 비운다.
- canonical_answers.generation_*: 정본 본문을 만든 생성 프롬프트와 모델. 시드 입력이 줄 때만 채운다.

Revision ID: 20261001_18
Revises: 20261001_17
Create Date: 2026-10-01
"""

from typing import Optional, Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20261001_18"
down_revision: Optional[str] = "20261001_17"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


RAG_RUN_PROFILE_REVISION_FK = "fk_rag_runs_profile_revision_id_chat_profile_revisions"
# 명명 규칙대로면 referred table 까지 붙어 63자를 넘는다.
PRESENTED_CANONICAL_FK = "fk_question_classifications_presented_canonical_answer_id"
CANONICAL_SHA256_CHECK = (
    "generation_prompt_sha256 IS NULL"
    " OR generation_prompt_sha256 ~ '^[0-9a-f]{64}$'"
)


def upgrade() -> None:
    op.add_column(
        "rag_runs",
        sa.Column("profile_revision_id", sa.BigInteger(), nullable=True),
    )
    op.create_foreign_key(
        RAG_RUN_PROFILE_REVISION_FK,
        "rag_runs",
        "chat_profile_revisions",
        ["profile_revision_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.add_column(
        "rag_runs",
        sa.Column("build_version", sa.String(length=100), nullable=True),
    )

    op.add_column(
        "question_classifications",
        sa.Column(
            "presented_canonical_answer_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        PRESENTED_CANONICAL_FK,
        "question_classifications",
        "canonical_answers",
        ["presented_canonical_answer_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    op.add_column(
        "canonical_answers",
        sa.Column("generation_prompt_version", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "canonical_answers",
        sa.Column("generation_prompt_sha256", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "canonical_answers",
        sa.Column("generation_model_name", sa.String(length=150), nullable=True),
    )
    op.create_check_constraint(
        "generation_prompt_sha256",
        "canonical_answers",
        CANONICAL_SHA256_CHECK,
    )


def downgrade() -> None:
    # 이름은 명명 규칙이 ck_canonical_answers_ 를 붙여 만든다.
    op.drop_constraint(
        "generation_prompt_sha256",
        "canonical_answers",
        type_="check",
    )
    op.drop_column("canonical_answers", "generation_model_name")
    op.drop_column("canonical_answers", "generation_prompt_sha256")
    op.drop_column("canonical_answers", "generation_prompt_version")
    op.drop_constraint(
        PRESENTED_CANONICAL_FK, "question_classifications", type_="foreignkey"
    )
    op.drop_column("question_classifications", "presented_canonical_answer_id")
    op.drop_column("rag_runs", "build_version")
    op.drop_constraint(RAG_RUN_PROFILE_REVISION_FK, "rag_runs", type_="foreignkey")
    op.drop_column("rag_runs", "profile_revision_id")
