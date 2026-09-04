"""검색 버전에 READY 상태와 번호를, 색인 실행에 단계와 작업 범위를 추가한다.

- index_versions.status에 READY를 허용하고 그룹 단위 version_no를 둔다.
- index_runs에 stage, operation_type, error_code를 추가하고 기존 행을 채운다.

Revision ID: 20260904_07
Revises: 20260904_06
Create Date: 2026-09-04
"""

from typing import Optional, Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260904_07"
down_revision: Optional[str] = "20260904_06"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


INDEX_VERSION_STATUS_CONSTRAINT = "index_version_status"
INDEX_RUN_STAGE_CONSTRAINT = "index_run_stage"
INDEX_OPERATION_TYPE_CONSTRAINT = "index_operation_type"

INDEX_VERSION_NO_CONSTRAINT = "uq_index_versions_document_group_id_version_no"
INDEX_RUN_STARTED_AT_INDEX = "ix_index_runs_index_version_id_started_at"

LEGACY_INDEX_VERSION_STATUSES = (
    "BUILDING",
    "VALIDATING",
    "ACTIVE",
    "FAILED",
    "INACTIVE",
)
EXPANDED_INDEX_VERSION_STATUSES = (
    "BUILDING",
    "VALIDATING",
    "READY",
    "ACTIVE",
    "FAILED",
    "INACTIVE",
)
INDEX_RUN_STAGES = ("BUILDING", "VALIDATING", "APPLYING")
# BUILD는 후보 생성만 수행하는 2차 확장을 위해 허용 목록에만 둔다.
INDEX_OPERATION_TYPES = ("BUILD_AND_APPLY", "BUILD", "APPLY")


def _allowed_values_condition(column: str, values: Sequence[str]) -> str:
    literals = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({literals})"


def upgrade() -> None:
    connection = op.get_bind()

    op.drop_constraint(
        INDEX_VERSION_STATUS_CONSTRAINT,
        "index_versions",
        type_="check",
    )
    op.create_check_constraint(
        INDEX_VERSION_STATUS_CONSTRAINT,
        "index_versions",
        _allowed_values_condition("status", EXPANDED_INDEX_VERSION_STATUSES),
    )

    op.add_column(
        "index_versions",
        sa.Column("version_no", sa.Integer(), nullable=True),
    )
    # 이미 적용된 적이 있는 버전에만 생성 순서대로 번호를 부여한다.
    connection.execute(
        sa.text(
            "WITH ranked AS ("
            " SELECT id, row_number() OVER ("
            " PARTITION BY document_group_id ORDER BY created_at, id) AS rn"
            " FROM index_versions WHERE status IN ('ACTIVE', 'INACTIVE'))"
            " UPDATE index_versions v SET version_no = r.rn"
            " FROM ranked r WHERE v.id = r.id"
        )
    )
    op.create_index(
        INDEX_VERSION_NO_CONSTRAINT,
        "index_versions",
        ["document_group_id", "version_no"],
        unique=True,
        postgresql_where=sa.text("version_no IS NOT NULL"),
    )

    op.add_column(
        "index_runs",
        sa.Column("stage", sa.String(length=20), nullable=True),
    )
    op.add_column(
        "index_runs",
        sa.Column("operation_type", sa.String(length=20), nullable=True),
    )
    op.add_column(
        "index_runs",
        sa.Column("error_code", sa.String(length=50), nullable=True),
    )

    # 기존 실행은 모두 후보 생성과 적용을 한 번에 수행했다.
    connection.execute(
        sa.text("UPDATE index_runs SET operation_type = 'BUILD_AND_APPLY'")
    )
    # summary의 호환용 단계 값을 stage 컬럼으로 옮긴다.
    connection.execute(
        sa.text(
            "UPDATE index_runs SET stage = CASE"
            " WHEN summary->>'failed_stage' IN ('STARTING', 'EMBEDDING', 'PERSISTING')"
            " THEN 'BUILDING'"
            " WHEN summary->>'failed_stage' = 'VALIDATING' THEN 'VALIDATING'"
            " WHEN summary->>'failed_stage' = 'ACTIVATING' THEN 'APPLYING'"
            " WHEN summary->>'failed_stage' IS NULL"
            " AND summary->>'stage' = 'ACTIVE' THEN 'APPLYING'"
            " ELSE 'BUILDING' END"
        )
    )

    op.alter_column("index_runs", "stage", nullable=False)
    op.alter_column("index_runs", "operation_type", nullable=False)
    op.create_check_constraint(
        INDEX_RUN_STAGE_CONSTRAINT,
        "index_runs",
        _allowed_values_condition("stage", INDEX_RUN_STAGES),
    )
    op.create_check_constraint(
        INDEX_OPERATION_TYPE_CONSTRAINT,
        "index_runs",
        _allowed_values_condition("operation_type", INDEX_OPERATION_TYPES),
    )
    op.create_index(
        INDEX_RUN_STARTED_AT_INDEX,
        "index_runs",
        ["index_version_id", "started_at"],
    )


def downgrade() -> None:
    op.drop_index(INDEX_RUN_STARTED_AT_INDEX, table_name="index_runs")
    op.drop_constraint(
        INDEX_OPERATION_TYPE_CONSTRAINT,
        "index_runs",
        type_="check",
    )
    op.drop_constraint(
        INDEX_RUN_STAGE_CONSTRAINT,
        "index_runs",
        type_="check",
    )
    op.drop_column("index_runs", "error_code")
    op.drop_column("index_runs", "operation_type")
    op.drop_column("index_runs", "stage")

    op.drop_index(INDEX_VERSION_NO_CONSTRAINT, table_name="index_versions")
    op.drop_column("index_versions", "version_no")

    # READY 행을 임의 상태로 바꾸지 않고 CHECK 복원 단계에서 실패시킨다.
    op.drop_constraint(
        INDEX_VERSION_STATUS_CONSTRAINT,
        "index_versions",
        type_="check",
    )
    op.create_check_constraint(
        INDEX_VERSION_STATUS_CONSTRAINT,
        "index_versions",
        _allowed_values_condition("status", LEGACY_INDEX_VERSION_STATUSES),
    )
