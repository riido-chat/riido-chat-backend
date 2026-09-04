"""문서 그룹을 도입하고 문서와 색인 버전을 그룹에 소속시킨다.

- document_groups를 만들고 1차 그룹 HELP_CHATBOT 한 행을 넣는다.
- document_sources를 (document_group_id, document_key)로 식별하도록 바꾼다.
- source_type 값을 출처만 나타내는 GITBOOK, UPLOAD로 정리한다.
- index_versions를 그룹에 소속시키고 그룹마다 ACTIVE를 하나로 제한한다.

원문(raw_content, raw_content_uri)은 이 migration의 대상이 아니다.

Revision ID: 20260904_06
Revises: 20260831_05
Create Date: 2026-09-04
"""

from typing import Optional, Sequence, Union

from alembic import op
import sqlalchemy as sa

from app.admin.document_key import (
    DEFAULT_DOCUMENT_GROUP_CONSUMER_KEY,
    DEFAULT_DOCUMENT_GROUP_KEY,
    DEFAULT_DOCUMENT_GROUP_NAME,
    SOURCE_TYPE_GITBOOK,
    SOURCE_TYPE_UPLOAD,
    build_console_canonical_uri,
    build_upload_document_key,
)


revision: str = "20260904_06"
down_revision: Optional[str] = "20260831_05"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


LEGACY_GITBOOK_SOURCE_TYPE = "GITBOOK_MARKDOWN"
LEGACY_UPLOAD_SOURCE_TYPE = "ADMIN_MARKDOWN"

DOCUMENT_SOURCE_URI_CONSTRAINT = "uq_document_sources_canonical_uri"
DOCUMENT_SOURCE_KEY_CONSTRAINT = "uq_document_sources_document_group_id_document_key"
DOCUMENT_SOURCE_GROUP_URI_CONSTRAINT = (
    "uq_document_sources_document_group_id_canonical_uri"
)
DOCUMENT_SOURCE_GROUP_INDEX = "ix_document_sources_document_group_id"
DOCUMENT_SOURCE_GROUP_FK = "fk_document_sources_document_group_id_document_groups"

INDEX_VERSION_GROUP_INDEX = "ix_index_versions_document_group_id"
INDEX_VERSION_GROUP_FK = "fk_index_versions_document_group_id_document_groups"
ACTIVE_INDEX_VERSION_CONSTRAINT = "uq_index_versions_document_group_id"

DOCUMENT_VERSION_HASH_INDEX = "ix_document_versions_normalized_content_hash"

# GitBook 문서 키는 페이지 URL 경로 하나로 고정한다.
GITBOOK_DOCUMENT_KEY_SQL = (
    r"regexp_replace("
    r"regexp_replace(canonical_uri, '^https?://docs\.riido\.io/', ''), '\.md$', '')"
)


def upgrade() -> None:
    connection = op.get_bind()

    op.create_table(
        "document_groups",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("group_key", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("consumer_key", sa.String(length=50), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_document_groups"),
        sa.UniqueConstraint("group_key", name="uq_document_groups_group_key"),
    )
    connection.execute(
        sa.text(
            "INSERT INTO document_groups (group_key, name, consumer_key)"
            " VALUES (:group_key, :name, :consumer_key)"
        ),
        {
            "group_key": DEFAULT_DOCUMENT_GROUP_KEY,
            "name": DEFAULT_DOCUMENT_GROUP_NAME,
            "consumer_key": DEFAULT_DOCUMENT_GROUP_CONSUMER_KEY,
        },
    )

    op.add_column(
        "document_sources",
        sa.Column("document_group_id", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "document_sources",
        sa.Column("document_key", sa.String(length=300), nullable=True),
    )

    _check_backfillable_source_types(connection)
    _check_gitbook_canonical_uri_format(connection)

    # 그룹 소속과 GitBook 문서 키를 채운다.
    connection.execute(
        sa.text(
            "UPDATE document_sources SET"
            " document_group_id ="
            " (SELECT id FROM document_groups WHERE group_key = :group_key),"
            " document_key = CASE"
            f" WHEN source_type = :gitbook THEN {GITBOOK_DOCUMENT_KEY_SQL}"
            " ELSE document_key END"
        ),
        {
            "group_key": DEFAULT_DOCUMENT_GROUP_KEY,
            "gitbook": LEGACY_GITBOOK_SOURCE_TYPE,
        },
    )
    # 콘솔 업로드 문서 키는 서비스와 같은 정규화 함수로 만든다.
    _backfill_upload_document_keys(connection)

    connection.execute(
        sa.text(
            "UPDATE document_sources SET source_type = CASE source_type"
            " WHEN :legacy_gitbook THEN :gitbook"
            " WHEN :legacy_upload THEN :upload"
            " ELSE source_type END"
        ),
        {
            "legacy_gitbook": LEGACY_GITBOOK_SOURCE_TYPE,
            "gitbook": SOURCE_TYPE_GITBOOK,
            "legacy_upload": LEGACY_UPLOAD_SOURCE_TYPE,
            "upload": SOURCE_TYPE_UPLOAD,
        },
    )
    _rewrite_upload_canonical_uris(connection)

    _check_no_duplicate_document_key(connection)
    _check_no_duplicate_group_canonical_uri(connection)

    op.alter_column("document_sources", "document_group_id", nullable=False)
    op.alter_column("document_sources", "document_key", nullable=False)
    op.create_foreign_key(
        DOCUMENT_SOURCE_GROUP_FK,
        "document_sources",
        "document_groups",
        ["document_group_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        DOCUMENT_SOURCE_GROUP_INDEX,
        "document_sources",
        ["document_group_id"],
    )
    op.create_unique_constraint(
        DOCUMENT_SOURCE_KEY_CONSTRAINT,
        "document_sources",
        ["document_group_id", "document_key"],
    )
    op.drop_constraint(
        DOCUMENT_SOURCE_URI_CONSTRAINT,
        "document_sources",
        type_="unique",
    )
    op.create_unique_constraint(
        DOCUMENT_SOURCE_GROUP_URI_CONSTRAINT,
        "document_sources",
        ["document_group_id", "canonical_uri"],
    )

    op.add_column(
        "index_versions",
        sa.Column("document_group_id", sa.BigInteger(), nullable=True),
    )
    connection.execute(
        sa.text(
            "UPDATE index_versions SET document_group_id ="
            " (SELECT id FROM document_groups WHERE group_key = :group_key)"
        ),
        {"group_key": DEFAULT_DOCUMENT_GROUP_KEY},
    )
    op.alter_column("index_versions", "document_group_id", nullable=False)
    op.create_foreign_key(
        INDEX_VERSION_GROUP_FK,
        "index_versions",
        "document_groups",
        ["document_group_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        INDEX_VERSION_GROUP_INDEX,
        "index_versions",
        ["document_group_id"],
    )

    _check_single_active_index_version(connection)
    op.create_index(
        ACTIVE_INDEX_VERSION_CONSTRAINT,
        "index_versions",
        ["document_group_id"],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVE'"),
    )

    op.create_index(
        DOCUMENT_VERSION_HASH_INDEX,
        "document_versions",
        ["normalized_content_hash"],
    )


def downgrade() -> None:
    connection = op.get_bind()

    op.drop_index(DOCUMENT_VERSION_HASH_INDEX, table_name="document_versions")

    op.drop_index(ACTIVE_INDEX_VERSION_CONSTRAINT, table_name="index_versions")
    op.drop_index(INDEX_VERSION_GROUP_INDEX, table_name="index_versions")
    op.drop_constraint(
        INDEX_VERSION_GROUP_FK,
        "index_versions",
        type_="foreignkey",
    )
    op.drop_column("index_versions", "document_group_id")

    op.drop_constraint(
        DOCUMENT_SOURCE_GROUP_URI_CONSTRAINT,
        "document_sources",
        type_="unique",
    )
    op.drop_constraint(
        DOCUMENT_SOURCE_KEY_CONSTRAINT,
        "document_sources",
        type_="unique",
    )
    op.drop_index(DOCUMENT_SOURCE_GROUP_INDEX, table_name="document_sources")
    op.drop_constraint(
        DOCUMENT_SOURCE_GROUP_FK,
        "document_sources",
        type_="foreignkey",
    )
    # 콘솔 문서의 riido-doc canonical_uri는 그 자체로 유일하므로 전역 unique를 복원할 수 있다.
    op.create_unique_constraint(
        DOCUMENT_SOURCE_URI_CONSTRAINT,
        "document_sources",
        ["canonical_uri"],
    )
    connection.execute(
        sa.text(
            "UPDATE document_sources SET source_type = CASE source_type"
            " WHEN :gitbook THEN :legacy_gitbook"
            " WHEN :upload THEN :legacy_upload"
            " ELSE source_type END"
        ),
        {
            "gitbook": SOURCE_TYPE_GITBOOK,
            "legacy_gitbook": LEGACY_GITBOOK_SOURCE_TYPE,
            "upload": SOURCE_TYPE_UPLOAD,
            "legacy_upload": LEGACY_UPLOAD_SOURCE_TYPE,
        },
    )
    op.drop_column("document_sources", "document_key")
    op.drop_column("document_sources", "document_group_id")

    op.drop_table("document_groups")


def _check_backfillable_source_types(connection: sa.engine.Connection) -> None:
    """문서 키 규칙이 정해지지 않은 source_type이 남아 있으면 중단한다."""

    unknown = connection.execute(
        sa.text(
            "SELECT DISTINCT source_type FROM document_sources"
            " WHERE source_type NOT IN (:gitbook, :upload)"
        ),
        {
            "gitbook": LEGACY_GITBOOK_SOURCE_TYPE,
            "upload": LEGACY_UPLOAD_SOURCE_TYPE,
        },
    ).scalars().all()
    if unknown:
        raise RuntimeError(
            "document_key 규칙이 없는 source_type이 있어 migration을 중단합니다: "
            f"{sorted(unknown)}"
        )


def _check_gitbook_canonical_uri_format(connection: sa.engine.Connection) -> None:
    """GitBook 문서 키를 URL 경로로 만들 수 없는 행이 있으면 중단한다."""

    invalid = connection.execute(
        sa.text(
            "SELECT count(*) FROM document_sources"
            " WHERE source_type = :gitbook"
            r" AND canonical_uri !~ '^https?://docs\.riido\.io/.+'"
        ),
        {"gitbook": LEGACY_GITBOOK_SOURCE_TYPE},
    ).scalar_one()
    if invalid:
        raise RuntimeError(
            "docs.riido.io 형식이 아닌 GitBook document_sources가 "
            f"{invalid}건 있어 migration을 중단합니다."
        )


def _backfill_upload_document_keys(connection: sa.engine.Connection) -> None:
    """콘솔 업로드 문서의 document_key를 title 정규화로 채운다."""

    rows = connection.execute(
        sa.text(
            "SELECT id, title FROM document_sources"
            " WHERE source_type = :upload ORDER BY id"
        ),
        {"upload": LEGACY_UPLOAD_SOURCE_TYPE},
    ).all()
    for source_id, title in rows:
        if not title:
            raise RuntimeError(
                "title이 없어 document_key를 만들 수 없는 업로드 문서가 있습니다: "
                f"document_sources.id={source_id}"
            )
        connection.execute(
            sa.text(
                "UPDATE document_sources SET document_key = :document_key"
                " WHERE id = :id"
            ),
            {"document_key": build_upload_document_key(title), "id": source_id},
        )


def _rewrite_upload_canonical_uris(connection: sa.engine.Connection) -> None:
    """콘솔 업로드 문서의 canonical_uri를 서버 생성 내부 스킴으로 바꾼다."""

    rows = connection.execute(
        sa.text(
            "SELECT id, document_key FROM document_sources"
            " WHERE source_type = :upload ORDER BY id"
        ),
        {"upload": SOURCE_TYPE_UPLOAD},
    ).all()
    for source_id, document_key in rows:
        connection.execute(
            sa.text(
                "UPDATE document_sources SET canonical_uri = :canonical_uri"
                " WHERE id = :id"
            ),
            {
                "canonical_uri": build_console_canonical_uri(
                    DEFAULT_DOCUMENT_GROUP_KEY,
                    document_key,
                ),
                "id": source_id,
            },
        )


def _check_no_duplicate_document_key(connection: sa.engine.Connection) -> None:
    duplicates = connection.execute(
        sa.text(
            "SELECT count(*) FROM ("
            " SELECT document_group_id, document_key FROM document_sources"
            " GROUP BY document_group_id, document_key HAVING count(*) > 1"
            ") AS duplicated"
        )
    ).scalar_one()
    if duplicates:
        raise RuntimeError(
            "(document_group_id, document_key)가 중복된 문서가 "
            f"{duplicates}건 있어 migration을 중단합니다."
        )


def _check_no_duplicate_group_canonical_uri(
    connection: sa.engine.Connection,
) -> None:
    duplicates = connection.execute(
        sa.text(
            "SELECT count(*) FROM ("
            " SELECT document_group_id, canonical_uri FROM document_sources"
            " GROUP BY document_group_id, canonical_uri HAVING count(*) > 1"
            ") AS duplicated"
        )
    ).scalar_one()
    if duplicates:
        raise RuntimeError(
            "(document_group_id, canonical_uri)가 중복된 문서가 "
            f"{duplicates}건 있어 migration을 중단합니다."
        )


def _check_single_active_index_version(connection: sa.engine.Connection) -> None:
    duplicates = connection.execute(
        sa.text(
            "SELECT count(*) FROM ("
            " SELECT document_group_id FROM index_versions"
            " WHERE status = 'ACTIVE' GROUP BY document_group_id"
            " HAVING count(*) > 1"
            ") AS duplicated"
        )
    ).scalar_one()
    if duplicates:
        raise RuntimeError(
            "ACTIVE 색인 버전이 2건 이상인 문서 그룹이 있어 migration을 중단합니다."
        )
