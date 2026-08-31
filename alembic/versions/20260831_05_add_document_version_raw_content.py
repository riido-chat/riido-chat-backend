"""관리자 업로드 원문을 document_versions에 저장할 수 있게 한다.

- 기존 수집 문서는 raw_content_uri를 계속 사용한다.
- 관리자 업로드 문서는 raw_content에 Markdown 원문을 저장할 수 있다.
- 두 저장 위치 중 하나 이상은 반드시 존재해야 한다.

Revision ID: 20260831_05
Revises: 20260828_04
Create Date: 2026-08-31
"""

from typing import Optional, Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260831_05"
down_revision: Optional[str] = "20260828_04"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


RAW_CONTENT_STORAGE_CONSTRAINT = "raw_content_storage"


def upgrade() -> None:
    op.add_column(
        "document_versions",
        sa.Column("raw_content", sa.Text(), nullable=True),
    )
    op.create_check_constraint(
        RAW_CONTENT_STORAGE_CONSTRAINT,
        "document_versions",
        "raw_content_uri IS NOT NULL OR raw_content IS NOT NULL",
    )


def downgrade() -> None:
    # inline 원문이 유일한 저장소인 행을 남긴 채 컬럼을 제거하지 않는다.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM document_versions
                WHERE raw_content_uri IS NULL
            ) THEN
                RAISE EXCEPTION
                    'raw_content가 유일한 원문인 document_versions가 있어 downgrade할 수 없습니다.';
            END IF;
        END
        $$
        """
    )
    op.drop_constraint(
        RAW_CONTENT_STORAGE_CONSTRAINT,
        "document_versions",
        type_="check",
    )
    op.drop_column("document_versions", "raw_content")
