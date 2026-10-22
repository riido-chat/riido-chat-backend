"""RAG 실행 로그 통합에 필요한 컬럼과 Enum CHECK 제약을 추가한다.

- retrieval_results.fused_score: 융합 결과에 든 청크의 RRF 점수
- feedbacks.updated_at: 평가를 반대 값으로 변경한 시각
- retriever_type, purpose, rating을 Enum으로 확정하고 CHECK 제약을 건다

Revision ID: 20260825_03
Revises: 20260820_02
Create Date: 2026-08-25
"""

from typing import Optional, Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260825_03"
down_revision: Optional[str] = "20260820_02"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


# (제약 이름, 테이블, 컬럼, 허용 값) — app/database/models.py의 Enum 선언과 짝을 이룬다
ENUM_CONSTRAINTS = (
    (
        "retriever_type",
        "retrieval_results",
        "retriever_type",
        ("BM25", "VECTOR"),
    ),
    (
        "model_call_purpose",
        "model_calls",
        "purpose",
        ("EMBEDDING", "GENERATION"),
    ),
    (
        "feedback_rating",
        "feedbacks",
        "rating",
        ("GOOD", "BAD"),
    ),
)


def _allowed_values_condition(column: str, values: Sequence[str]) -> str:
    literals = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({literals})"


def upgrade() -> None:
    op.add_column(
        "retrieval_results",
        sa.Column(
            "fused_score",
            sa.Numeric(),
            nullable=True,
            comment=(
                "융합 결과에 든 청크의 RRF 점수. 검색기별 행에 같은 값을 기록한다"
            ),
        ),
    )
    op.add_column(
        "feedbacks",
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
            comment=(
                "평가를 반대 값으로 변경한 시각. 신규 등록 시에는 created_at과 같다"
            ),
        ),
    )

    for constraint_name, table_name, column_name, values in ENUM_CONSTRAINTS:
        op.create_check_constraint(
            constraint_name,
            table_name,
            _allowed_values_condition(column_name, values),
        )


def downgrade() -> None:
    for constraint_name, table_name, _, _ in reversed(ENUM_CONSTRAINTS):
        # 이름 규칙(ck_%(table_name)s_%(constraint_name)s)은 alembic이 붙인다
        op.drop_constraint(constraint_name, table_name, type_="check")

    op.drop_column("feedbacks", "updated_at")
    op.drop_column("retrieval_results", "fused_score")
