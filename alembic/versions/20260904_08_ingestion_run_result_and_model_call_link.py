"""수집 실행에 결과 코드와 단계를 두고 모델 호출을 수집 실행에 연결한다.

- ingestion_runs에 result_code, stage, error_code, batch_id를 추가한다.
- 동일 콘텐츠 판정 근거로 duplicate_of_document_source_id를 둔다.
- model_calls를 수집 실행에도 연결해 업로드 단계 호출 비용을 남긴다.

Revision ID: 20260904_08
Revises: 20260904_07
Create Date: 2026-09-04
"""

from typing import Optional, Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260904_08"
down_revision: Optional[str] = "20260904_07"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


INGESTION_RESULT_CODE_CONSTRAINT = "ingestion_result_code"
INGESTION_STAGE_CONSTRAINT = "ingestion_stage"

INGESTION_BATCH_INDEX = "ix_ingestion_runs_batch_id"
# 명명 규칙대로 referred table까지 붙이면 66자가 되어 식별자 63자 제한을 넘는다.
DUPLICATE_DOCUMENT_SOURCE_FK = "fk_ingestion_runs_duplicate_of_document_source_id"
MODEL_CALL_INGESTION_RUN_FK = "fk_model_calls_ingestion_run_id_ingestion_runs"

INGESTION_RESULT_CODES = ("CREATED", "UPDATED", "NO_CHANGE", "DUPLICATE_CONTENT")
INGESTION_STAGES = (
    "RECEIVING",
    "VALIDATING",
    "NORMALIZING",
    "PARSING",
    "CHUNKING",
    "EMBEDDING",
    "PERSISTING",
)


def _allowed_values_condition(column: str, values: Sequence[str]) -> str:
    literals = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({literals})"


def upgrade() -> None:
    connection = op.get_bind()

    op.add_column(
        "ingestion_runs",
        sa.Column("result_code", sa.String(length=20), nullable=True),
    )
    op.add_column(
        "ingestion_runs",
        sa.Column("stage", sa.String(length=20), nullable=True),
    )
    op.add_column(
        "ingestion_runs",
        sa.Column("error_code", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "ingestion_runs",
        sa.Column("batch_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "ingestion_runs",
        sa.Column(
            "duplicate_of_document_source_id",
            sa.BigInteger(),
            nullable=True,
        ),
    )

    # 성공한 실행은 만들어낸 판이 첫 판인지로 신규와 수정본을 구분한다.
    connection.execute(
        sa.text(
            "UPDATE ingestion_runs r SET"
            " result_code = CASE WHEN v.version_no = 1 THEN 'CREATED'"
            " ELSE 'UPDATED' END,"
            " stage = 'PERSISTING'"
            " FROM document_versions v"
            " WHERE r.produced_version_id = v.id AND r.status = 'SUCCESS'"
        )
    )
    # 실패한 실행은 result_code를 남기지 않고 summary의 호환용 값만 옮긴다.
    connection.execute(
        sa.text(
            "UPDATE ingestion_runs SET"
            " stage = CASE summary->>'failed_stage'"
            " WHEN 'LOADING' THEN 'RECEIVING'"
            " WHEN 'NORMALIZING' THEN 'NORMALIZING'"
            " WHEN 'PERSISTING' THEN 'PERSISTING' END,"
            " error_code = summary->>'error_code'"
            " WHERE status = 'FAILED'"
        )
    )

    op.create_check_constraint(
        INGESTION_RESULT_CODE_CONSTRAINT,
        "ingestion_runs",
        _allowed_values_condition("result_code", INGESTION_RESULT_CODES),
    )
    op.create_check_constraint(
        INGESTION_STAGE_CONSTRAINT,
        "ingestion_runs",
        _allowed_values_condition("stage", INGESTION_STAGES),
    )
    op.create_index(
        INGESTION_BATCH_INDEX,
        "ingestion_runs",
        ["batch_id"],
    )
    op.create_foreign_key(
        DUPLICATE_DOCUMENT_SOURCE_FK,
        "ingestion_runs",
        "document_sources",
        ["duplicate_of_document_source_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.add_column(
        "model_calls",
        sa.Column("ingestion_run_id", sa.BigInteger(), nullable=True),
    )
    op.create_foreign_key(
        MODEL_CALL_INGESTION_RUN_FK,
        "model_calls",
        "ingestion_runs",
        ["ingestion_run_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint(
        MODEL_CALL_INGESTION_RUN_FK,
        "model_calls",
        type_="foreignkey",
    )
    op.drop_column("model_calls", "ingestion_run_id")

    op.drop_constraint(
        DUPLICATE_DOCUMENT_SOURCE_FK,
        "ingestion_runs",
        type_="foreignkey",
    )
    op.drop_index(INGESTION_BATCH_INDEX, table_name="ingestion_runs")
    op.drop_constraint(
        INGESTION_STAGE_CONSTRAINT,
        "ingestion_runs",
        type_="check",
    )
    op.drop_constraint(
        INGESTION_RESULT_CODE_CONSTRAINT,
        "ingestion_runs",
        type_="check",
    )
    op.drop_column("ingestion_runs", "duplicate_of_document_source_id")
    op.drop_column("ingestion_runs", "batch_id")
    op.drop_column("ingestion_runs", "error_code")
    op.drop_column("ingestion_runs", "stage")
    op.drop_column("ingestion_runs", "result_code")
