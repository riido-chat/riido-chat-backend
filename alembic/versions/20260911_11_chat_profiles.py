"""Add versioned chat profiles and pin conversations to a profile revision.

Existing conversations are pinned to the published HELP_CHATBOT revision while
the new profile tables are introduced.  The migration deliberately does not
create a HELP_CHATBOT_TEST profile; test fixtures can create a TESTING revision
when a disposable test profile is needed.
"""

from typing import Optional, Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260911_11"
down_revision: Optional[str] = "20260908_10"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


PROFILE_PUBLISHED_INDEX = "uq_chat_profile_revisions_profile_published"
PROFILE_TESTING_INDEX = "uq_chat_profile_revisions_profile_testing"
PROFILE_VERSION_CONSTRAINT = "uq_chat_profile_revisions_profile_id_version"
PROFILE_STATUS_CONSTRAINT = "chat_profile_revision_status"
CONVERSATION_PROFILE_FK = (
    "fk_conversations_profile_revision_id_chat_profile_revisions"
)
CONVERSATION_CHANNEL_CONSTRAINT = "conversation_channel"


def _profile_status() -> sa.Enum:
    return sa.Enum(
        "DRAFT",
        "TESTING",
        "PUBLISHED",
        "RETIRED",
        name=PROFILE_STATUS_CONSTRAINT,
        native_enum=False,
        create_constraint=True,
        length=20,
    )


def _conversation_channel() -> sa.Enum:
    return sa.Enum(
        "PUBLIC",
        "INTERNAL_TEST",
        name=CONVERSATION_CHANNEL_CONSTRAINT,
        native_enum=False,
        create_constraint=True,
        length=20,
    )


def _published_revision_id(connection, profile_id: int) -> int:
    revision_id = connection.execute(
        sa.text(
            "SELECT id FROM chat_profile_revisions "
            "WHERE profile_id = :profile_id AND status = 'PUBLISHED'"
        ),
        {"profile_id": profile_id},
    ).scalar_one_or_none()
    if revision_id is not None:
        return int(revision_id)

    group_id = connection.execute(
        sa.text(
            "SELECT id FROM document_groups WHERE group_key = 'HELP_CHATBOT'"
        )
    ).scalar_one_or_none()
    if group_id is None:
        raise RuntimeError(
            "HELP_CHATBOT document group이 없어 기본 chat profile을 만들 수 없습니다."
        )

    revision_id = connection.execute(
        sa.text(
            "INSERT INTO chat_profile_revisions "
            "(profile_id, version, document_group_id, status, "
            "generation_model_name, generation_prompt_version, "
            "query_rewrite_model_name, query_rewrite_prompt_version, "
            "semantic_cache_enabled) "
            "VALUES (:profile_id, 1, :group_id, 'PUBLISHED', "
            "'gpt-5.6-terra', 'v24', 'gpt-5.4-mini', 'v7', false) "
            "RETURNING id"
        ),
        {"profile_id": profile_id, "group_id": group_id},
    ).scalar_one()
    return int(revision_id)


def upgrade() -> None:
    op.create_table(
        "chat_profiles",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("profile_key", sa.String(length=80), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chat_profiles"),
        sa.UniqueConstraint("profile_key", name="uq_chat_profiles_profile_key"),
    )
    op.create_table(
        "chat_profile_revisions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("profile_id", sa.BigInteger(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("document_group_id", sa.BigInteger(), nullable=False),
        sa.Column("status", _profile_status(), nullable=False),
        sa.Column("generation_model_name", sa.String(length=150), nullable=False),
        sa.Column("generation_prompt_version", sa.String(length=50), nullable=False),
        sa.Column("query_rewrite_model_name", sa.String(length=150), nullable=False),
        sa.Column("query_rewrite_prompt_version", sa.String(length=50), nullable=False),
        sa.Column(
            "semantic_cache_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("verifier_model_name", sa.String(length=150), nullable=True),
        sa.Column("verifier_prompt_version", sa.String(length=50), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chat_profile_revisions"),
        sa.ForeignKeyConstraint(
            ["profile_id"],
            ["chat_profiles.id"],
            name="fk_chat_profile_revisions_profile_id_chat_profiles",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["document_group_id"],
            ["document_groups.id"],
            name="fk_chat_profile_revisions_document_group_id_document_groups",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "profile_id",
            "version",
            name=PROFILE_VERSION_CONSTRAINT,
        ),
    )
    op.create_index(
        PROFILE_PUBLISHED_INDEX,
        "chat_profile_revisions",
        ["profile_id"],
        unique=True,
        postgresql_where=sa.text("status = 'PUBLISHED'"),
    )
    op.create_index(
        PROFILE_TESTING_INDEX,
        "chat_profile_revisions",
        ["profile_id"],
        unique=True,
        postgresql_where=sa.text("status = 'TESTING'"),
    )

    connection = op.get_bind()
    connection.execute(
        sa.text(
            "INSERT INTO chat_profiles (profile_key, name) "
            "VALUES ('HELP_CHATBOT', '도움말 챗봇') "
            "ON CONFLICT (profile_key) DO NOTHING"
        )
    )
    profile_id = connection.execute(
        sa.text("SELECT id FROM chat_profiles WHERE profile_key = 'HELP_CHATBOT'")
    ).scalar_one()
    default_revision_id = _published_revision_id(connection, int(profile_id))

    # Add nullable first so the operation remains safe even if a deployment is
    # interrupted between DDL and the backfill.  It is made NOT NULL in the same
    # migration after every existing row has a stable pin.
    op.add_column(
        "conversations",
        sa.Column("chat_profile_revision_id", sa.BigInteger(), nullable=True),
    )
    op.create_foreign_key(
        CONVERSATION_PROFILE_FK,
        "conversations",
        "chat_profile_revisions",
        ["chat_profile_revision_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    connection.execute(
        sa.text(
            "UPDATE conversations SET chat_profile_revision_id = :revision_id "
            "WHERE chat_profile_revision_id IS NULL"
        ),
        {"revision_id": default_revision_id},
    )
    op.alter_column("conversations", "chat_profile_revision_id", nullable=False)
    op.add_column(
        "conversations",
        sa.Column("channel", _conversation_channel(), nullable=True),
    )
    connection.execute(
        sa.text("UPDATE conversations SET channel = 'PUBLIC' WHERE channel IS NULL")
    )
    op.alter_column(
        "conversations",
        "channel",
        nullable=False,
        server_default=sa.text("'PUBLIC'"),
    )


def downgrade() -> None:
    op.drop_constraint(CONVERSATION_PROFILE_FK, "conversations", type_="foreignkey")
    op.drop_column("conversations", "channel")
    op.drop_column("conversations", "chat_profile_revision_id")
    op.drop_index(PROFILE_TESTING_INDEX, table_name="chat_profile_revisions")
    op.drop_index(PROFILE_PUBLISHED_INDEX, table_name="chat_profile_revisions")
    op.drop_table("chat_profile_revisions")
    op.drop_table("chat_profiles")
