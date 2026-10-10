"""Drop unused turn log columns and store missing context snapshots as SQL NULL.

턴 기록의 미사용 칸과 정리 대상 칸을 지운다(#215).

- model_calls.estimated_cost: 한 번도 쓰지 않은 칸. 비용은 분석 시 토큰과 날짜별 단가표로 계산한다.
- conversations.summary_text, summary_version, summary_updated_turn_no: 대화 요약 기능이 없어 쓰지 않는다.
- model_calls.purpose 의 CONVERSATION_SUMMARY 값: 쓰는 호출이 없다. 허용값 CHECK 를 다시 만든다.
- chat_profile_revisions.verifier_model_name, verifier_prompt_version: 검증 단계가 없어 쓰지 않는다.
- rag_runs.context_snapshot: 문맥이 없는 행의 JSON null 을 SQL NULL 로 바꾼다.

지울 칸에 값이 있거나 CONVERSATION_SUMMARY 호출 행이 있으면 지우지 않고 실패한다.

downgrade 는 칸을 널 허용으로 되살리고 허용값에 CONVERSATION_SUMMARY 를 다시 넣는다. 값은 되살리지
않는다(올릴 때 모두 비어 있었다). context_snapshot 의 JSON null 과 SQL NULL 구분은 되살릴 수 없어
SQL NULL 로 남는다. 애플리케이션은 두 표현을 모두 None 으로 읽는다.

Revision ID: 20261001_19
Revises: 20261001_18
Create Date: 2026-10-01
"""

from typing import Optional, Sequence, Tuple, Union

from alembic import context, op
import sqlalchemy as sa


revision: str = "20261001_19"
down_revision: Optional[str] = "20261001_18"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


# 이름은 명명 규칙이 ck_model_calls_ 를 붙여 만든다.
MODEL_CALL_PURPOSE_CONSTRAINT = "model_call_purpose"
REMOVED_MODEL_CALL_PURPOSE = "CONVERSATION_SUMMARY"
MODEL_CALL_PURPOSES = (
    "EMBEDDING",
    "GENERATION",
    "QUERY_EMBEDDING",
    "CHUNK_EMBEDDING",
    "ANSWER_GENERATION",
    "QUERY_REWRITE",
    "QUESTION_CLASSIFICATION",
)
PREVIOUS_MODEL_CALL_PURPOSES = (
    "EMBEDDING",
    "GENERATION",
    "QUERY_EMBEDDING",
    "CHUNK_EMBEDDING",
    "ANSWER_GENERATION",
    "QUERY_REWRITE",
    REMOVED_MODEL_CALL_PURPOSE,
    "QUESTION_CLASSIFICATION",
)


def _dropped_columns() -> Sequence[Tuple[str, sa.Column]]:
    """(테이블, 칸) 순서. downgrade 는 같은 정의로 되살린다."""

    return (
        ("model_calls", sa.Column("estimated_cost", sa.Numeric(), nullable=True)),
        ("conversations", sa.Column("summary_text", sa.Text(), nullable=True)),
        (
            "conversations",
            sa.Column("summary_version", sa.String(length=50), nullable=True),
        ),
        (
            "conversations",
            sa.Column("summary_updated_turn_no", sa.Integer(), nullable=True),
        ),
        (
            "chat_profile_revisions",
            sa.Column("verifier_model_name", sa.String(length=150), nullable=True),
        ),
        (
            "chat_profile_revisions",
            sa.Column("verifier_prompt_version", sa.String(length=50), nullable=True),
        ),
    )


def _allowed_values_condition(column: str, values: Sequence[str]) -> str:
    literals = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({literals})"


def _check_nothing_to_lose(connection) -> None:
    """지울 칸과 지울 용도 값이 정말 비어 있는지 확인한다."""

    used = []
    for table, column in _dropped_columns():
        count = connection.execute(
            sa.text(f"SELECT count(*) FROM {table} WHERE {column.name} IS NOT NULL")
        ).scalar_one()
        if count:
            used.append(f"{table}.{column.name}={count}")
    purpose_count = connection.execute(
        sa.text("SELECT count(*) FROM model_calls WHERE purpose = :purpose"),
        {"purpose": REMOVED_MODEL_CALL_PURPOSE},
    ).scalar_one()
    if purpose_count:
        used.append(f"model_calls.purpose={REMOVED_MODEL_CALL_PURPOSE}={purpose_count}")
    if used:
        raise RuntimeError(
            "값이 있는 칸은 지우지 않습니다. 먼저 값을 옮기거나 확인하세요: "
            + ", ".join(used)
        )


def upgrade() -> None:
    # --sql(오프라인) 렌더링에서는 행을 읽을 수 없어 확인을 건너뛴다.
    if not context.is_offline_mode():
        _check_nothing_to_lose(op.get_bind())

    op.execute(
        "UPDATE rag_runs SET context_snapshot = NULL"
        " WHERE jsonb_typeof(context_snapshot) = 'null'"
    )

    for table, column in _dropped_columns():
        op.drop_column(table, column.name)

    op.drop_constraint(
        MODEL_CALL_PURPOSE_CONSTRAINT,
        "model_calls",
        type_="check",
    )
    op.create_check_constraint(
        MODEL_CALL_PURPOSE_CONSTRAINT,
        "model_calls",
        _allowed_values_condition("purpose", MODEL_CALL_PURPOSES),
    )


def downgrade() -> None:
    op.drop_constraint(
        MODEL_CALL_PURPOSE_CONSTRAINT,
        "model_calls",
        type_="check",
    )
    op.create_check_constraint(
        MODEL_CALL_PURPOSE_CONSTRAINT,
        "model_calls",
        _allowed_values_condition("purpose", PREVIOUS_MODEL_CALL_PURPOSES),
    )

    for table, column in _dropped_columns():
        op.add_column(table, column)
    # context_snapshot 의 JSON null 은 되살리지 않는다. SQL NULL 로 남아도 읽는 쪽은 같다.
