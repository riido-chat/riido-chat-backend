"""문서 그룹이 문서를 끌어오는 수집 원천을 별도 행으로 둔다.

- document_group_sources를 만들고 그룹의 GitBook 루트를 한 행으로 남긴다.
- document_sources.group_source_id로 문서가 어느 원천에서 왔는지 가리킨다.
- 문서 키 유일성을 끌어오는 문서는 원천 안에서, 밀어 넣는 문서는 그룹
  안에서 보장하도록 부분 유니크 둘로 나눈다.

지금까지는 그룹이 어느 GitBook을 수집하는지를 문서의 canonical_uri에서
역산했다. 그룹 단위 사실을 자식 행에서 추측하는 구조라 불변식의 주인이
없었고, 두 GitBook을 한 그룹에 넣으면 경로 키가 겹쳐 서로 다른 문서가
조용히 병합됐다.

Revision ID: 20260905_09
Revises: 20260904_08
Create Date: 2026-09-05
"""

from typing import Optional, Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260905_09"
down_revision: Optional[str] = "20260904_08"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


SOURCE_PROVIDER_CONSTRAINT = "source_provider"
GROUP_SOURCE_FK = "fk_document_sources_group_source_id_document_group_sources"
GROUP_SOURCE_INDEX = "ix_document_sources_group_source_id"

LEGACY_DOCUMENT_KEY_CONSTRAINT = "uq_document_sources_document_group_id_document_key"
PULLED_DOCUMENT_KEY_INDEX = "uq_document_sources_group_source_id_document_key"
PUSHED_DOCUMENT_KEY_INDEX = "uq_document_sources_document_group_id_document_key"

GITBOOK_URL_PREFIX = "https://docs.riido.io/"
GITBOOK_ROOT_URL = "https://docs.riido.io"


def _check_single_gitbook_root(connection) -> None:
    """그룹마다 GitBook 문서의 루트가 하나인지 확인한다.

    백필은 그룹당 원천 한 행을 전제한다. 루트가 섞여 있으면 어느 행에
    연결할지 정할 수 없으므로 멈춘다.
    """

    rows = connection.execute(
        sa.text(
            """
            SELECT document_group_id,
                   COUNT(*) FILTER (
                       WHERE canonical_uri NOT LIKE :prefix || '%'
                   ) AS outside
            FROM document_sources
            WHERE source_type = 'GITBOOK'
            GROUP BY document_group_id
            """
        ),
        {"prefix": GITBOOK_URL_PREFIX},
    ).all()
    mismatched = [row.document_group_id for row in rows if row.outside]
    if mismatched:
        raise RuntimeError(
            "GitBook 문서의 루트가 예상과 다른 문서 그룹이 있습니다: "
            f"{mismatched}"
        )


def upgrade() -> None:
    connection = op.get_bind()
    _check_single_gitbook_root(connection)

    op.create_table(
        "document_group_sources",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("document_group_id", sa.BigInteger(), nullable=False),
        sa.Column("provider", sa.String(length=20), nullable=False),
        sa.Column("root_url", sa.String(length=1000), nullable=False),
        sa.Column(
            "enabled",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_group_id"],
            ["document_groups.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_group_id", "root_url"),
        sa.CheckConstraint(
            "provider IN ('GITBOOK')",
            name=SOURCE_PROVIDER_CONSTRAINT,
        ),
        comment="문서 그룹이 문서를 끌어오는 외부 원천",
    )

    op.add_column(
        "document_sources",
        sa.Column("group_source_id", sa.BigInteger(), nullable=True),
    )
    op.create_foreign_key(
        GROUP_SOURCE_FK,
        "document_sources",
        "document_group_sources",
        ["group_source_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        GROUP_SOURCE_INDEX,
        "document_sources",
        ["group_source_id"],
    )

    # GitBook 문서가 있는 그룹마다 원천 한 행을 만들고 연결한다.
    connection.execute(
        sa.text(
            """
            INSERT INTO document_group_sources
                (document_group_id, provider, root_url, enabled)
            SELECT DISTINCT document_group_id, 'GITBOOK', :root_url, true
            FROM document_sources
            WHERE source_type = 'GITBOOK'
            """
        ),
        {"root_url": GITBOOK_ROOT_URL},
    )
    connection.execute(
        sa.text(
            """
            UPDATE document_sources AS ds
            SET group_source_id = gs.id
            FROM document_group_sources AS gs
            WHERE ds.source_type = 'GITBOOK'
              AND gs.document_group_id = ds.document_group_id
              AND gs.root_url = :root_url
            """
        ),
        {"root_url": GITBOOK_ROOT_URL},
    )

    remaining = connection.execute(
        sa.text(
            """
            SELECT COUNT(*) FROM document_sources
            WHERE source_type = 'GITBOOK' AND group_source_id IS NULL
            """
        )
    ).scalar_one()
    if remaining:
        raise RuntimeError(
            f"원천에 연결하지 못한 GitBook 문서가 {remaining}건 남았습니다."
        )

    # 문서 키 유일성을 두 갈래로 나눈다.
    op.drop_constraint(
        LEGACY_DOCUMENT_KEY_CONSTRAINT,
        "document_sources",
        type_="unique",
    )
    op.create_index(
        PULLED_DOCUMENT_KEY_INDEX,
        "document_sources",
        ["group_source_id", "document_key"],
        unique=True,
        postgresql_where=sa.text("group_source_id IS NOT NULL"),
    )
    op.create_index(
        PUSHED_DOCUMENT_KEY_INDEX,
        "document_sources",
        ["document_group_id", "document_key"],
        unique=True,
        postgresql_where=sa.text("group_source_id IS NULL"),
    )


def downgrade() -> None:
    connection = op.get_bind()

    # 그룹 안에서 문서 키가 겹치면 옛 제약을 되돌릴 수 없다.
    duplicated = connection.execute(
        sa.text(
            """
            SELECT COUNT(*) FROM (
                SELECT document_group_id, document_key
                FROM document_sources
                GROUP BY document_group_id, document_key
                HAVING COUNT(*) > 1
            ) AS duplicated
            """
        )
    ).scalar_one()
    if duplicated:
        raise RuntimeError(
            "문서 키가 그룹 안에서 겹쳐 이전 제약으로 되돌릴 수 없습니다: "
            f"{duplicated}건"
        )

    op.drop_index(PUSHED_DOCUMENT_KEY_INDEX, table_name="document_sources")
    op.drop_index(PULLED_DOCUMENT_KEY_INDEX, table_name="document_sources")
    op.create_unique_constraint(
        LEGACY_DOCUMENT_KEY_CONSTRAINT,
        "document_sources",
        ["document_group_id", "document_key"],
    )

    op.drop_index(GROUP_SOURCE_INDEX, table_name="document_sources")
    op.drop_constraint(GROUP_SOURCE_FK, "document_sources", type_="foreignkey")
    op.drop_column("document_sources", "group_source_id")
    op.drop_table("document_group_sources")
