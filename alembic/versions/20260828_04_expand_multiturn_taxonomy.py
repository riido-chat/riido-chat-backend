"""Multi-turn 도입을 위해 모델 호출 용도와 문맥 전략을 확장한다.

- model_calls.purpose는 기존 2종과 신규 5종을 함께 허용한다.
- rag_runs.context_strategy는 기존 4종과 신규 4종을 함께 허용한다.
- 데이터 변환과 기존 값 제거는 후속 contract migration에서 수행한다.

Revision ID: 20260828_04
Revises: 20260825_03
Create Date: 2026-08-28
"""

from typing import Optional, Sequence, Union

from alembic import op


revision: str = "20260828_04"
down_revision: Optional[str] = "20260825_03"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


MODEL_CALL_PURPOSE_CONSTRAINT = "model_call_purpose"
CONTEXT_STRATEGY_CONSTRAINT = "context_strategy"

LEGACY_MODEL_CALL_PURPOSES = (
    "EMBEDDING",
    "GENERATION",
)

EXPANDED_MODEL_CALL_PURPOSES = (
    *LEGACY_MODEL_CALL_PURPOSES,
    "QUERY_EMBEDDING",
    "CHUNK_EMBEDDING",
    "ANSWER_GENERATION",
    "QUERY_REWRITE",
    "CONVERSATION_SUMMARY",
)

EXPANDED_CONTEXT_STRATEGIES = (
    "NEW_TOPIC",
    "FULL",
    "WINDOW",
    "SUMMARY",
    "UNRESOLVED",
    "FOLLOW_UP_FULL",
    "FOLLOW_UP_WINDOW",
    "FOLLOW_UP_SUMMARY",
)


def _allowed_values_condition(column: str, values: Sequence[str]) -> str:
    literals = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({literals})"


def _create_model_call_purpose_constraint(values: Sequence[str]) -> None:
    op.create_check_constraint(
        MODEL_CALL_PURPOSE_CONSTRAINT,
        "model_calls",
        _allowed_values_condition("purpose", values),
    )


def upgrade() -> None:
    op.drop_constraint(
        MODEL_CALL_PURPOSE_CONSTRAINT,
        "model_calls",
        type_="check",
    )
    _create_model_call_purpose_constraint(EXPANDED_MODEL_CALL_PURPOSES)

    op.create_check_constraint(
        CONTEXT_STRATEGY_CONSTRAINT,
        "rag_runs",
        _allowed_values_condition(
            "context_strategy",
            EXPANDED_CONTEXT_STRATEGIES,
        ),
    )


def downgrade() -> None:
    op.drop_constraint(
        CONTEXT_STRATEGY_CONSTRAINT,
        "rag_runs",
        type_="check",
    )
    op.drop_constraint(
        MODEL_CALL_PURPOSE_CONSTRAINT,
        "model_calls",
        type_="check",
    )
    # 신규 purpose 행은 임의로 변환하지 않고 기존 CHECK 복원 단계에서 실패시킨다.
    _create_model_call_purpose_constraint(LEGACY_MODEL_CALL_PURPOSES)
