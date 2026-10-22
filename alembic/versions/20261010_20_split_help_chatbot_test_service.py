"""Split HELP_CHATBOT_TEST into its own chat profile with a 1:1 document group.

서비스 하나는 chat_profiles 행 하나이고 전용 문서 그룹 하나를 쓴다(#211 1단계, 확장만).

- chat_profiles.document_group_id: 필수, UNIQUE. 기존 프로필은 PUBLISHED 프로필 버전의 문서 그룹으로,
  PUBLISHED 가 없으면 모든 프로필 버전이 쓰는 단 하나의 문서 그룹으로 채운다.
- HELP_CHATBOT_TEST 문서 그룹을 쓰는 프로필 버전이 있으면 같은 키의 프로필을 만들고 그 프로필 버전을
  옮긴다. 옮긴 TESTING 프로필 버전은 그 서비스가 서빙하던 버전이라 PUBLISHED 로 바꾼다. 다른 상태는
  그대로 두고, id 와 version 도 그대로 둔다(새 프로필에는 겹칠 version 이 없다). 대화와 턴 행은
  프로필 버전 id 를 그대로 가리킨다.
- 프로필 버전의 (profile_id, document_group_id) 는 프로필의 (id, document_group_id) 를 참조한다.
  다른 서비스의 문서 그룹을 가리키는 프로필 버전은 DB 가 거부한다.
- conversations.profile_id: 필수. 대화의 프로필 버전이 속한 프로필로 채운다.
- conversations.channel 칸은 4단계에서 지운다. document_groups.consumer_key 는 실행 코드가 읽지 않아
  바꾸지 않는다.

HELP_CHATBOT_TEST 문서 그룹이나 그 그룹을 쓰는 프로필 버전이 없으면(빈 DB) 프로필을 만들지 않는다.

downgrade 는 HELP_CHATBOT_TEST 프로필의 프로필 버전을 HELP_CHATBOT 으로 되돌리고(PUBLISHED 는
TESTING 으로), 그 프로필과 새 칸, 제약을 지운다. 되돌릴 version 이 HELP_CHATBOT 에 이미 있으면
HELP_CHATBOT 의 마지막 version 뒤로 번호를 새로 매긴다.

Revision ID: 20261010_20
Revises: 20261001_19
Create Date: 2026-10-10
"""

from typing import Optional, Sequence, Union

from alembic import context, op
import sqlalchemy as sa


revision: str = "20261010_20"
down_revision: Optional[str] = "20261001_19"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


PUBLIC_PROFILE_KEY = "HELP_CHATBOT"
TEST_PROFILE_KEY = "HELP_CHATBOT_TEST"
TEST_PROFILE_NAME = "테스트 서비스"
TEST_GROUP_KEY = "HELP_CHATBOT_TEST"

PROFILE_GROUP_FK = "fk_chat_profiles_document_group_id_document_groups"
PROFILE_GROUP_UNIQUE = "uq_chat_profiles_document_group_id"
PROFILE_ID_GROUP_UNIQUE = "uq_chat_profiles_id_document_group_id"
# 명명 규칙대로 붙이면 63자를 넘어 이름을 직접 지정한다(app/database/models.py 와 같다).
REVISION_PROFILE_GROUP_FK = "fk_chat_profile_revisions_profile_id_document_group_id"
CONVERSATION_PROFILE_FK = "fk_conversations_profile_id_chat_profiles"

_TEST_GROUP_ID = f"(SELECT id FROM document_groups WHERE group_key = '{TEST_GROUP_KEY}')"


def _check_test_group_revisions(connection) -> None:
    """옮길 프로필 버전이 한 서비스로 모일 수 있는지 확인한다."""

    rows = connection.execute(
        sa.text(
            "SELECT r.id, r.status, p.profile_key "
            "FROM chat_profile_revisions r "
            "JOIN chat_profiles p ON p.id = r.profile_id "
            f"WHERE r.document_group_id = {_TEST_GROUP_ID} "
            "AND p.profile_key <> :test_key"
        ),
        {"test_key": TEST_PROFILE_KEY},
    ).all()
    if any(row.status == "PUBLISHED" for row in rows):
        raise RuntimeError(
            f"{TEST_GROUP_KEY} 문서 그룹을 쓰는 다른 프로필의 PUBLISHED 프로필 버전이 있어 옮기지 않습니다: "
            + ", ".join(f"{row.profile_key}#{row.id}" for row in rows)
        )
    if sum(row.status == "TESTING" for row in rows) > 1:
        raise RuntimeError(
            f"{TEST_GROUP_KEY} 문서 그룹을 쓰는 TESTING 프로필 버전이 둘 이상입니다."
        )
    collisions = connection.execute(
        sa.text(
            "SELECT r.id FROM chat_profile_revisions r "
            "JOIN chat_profiles p ON p.id = r.profile_id "
            "JOIN chat_profile_revisions existing ON existing.version = r.version "
            "JOIN chat_profiles target ON target.id = existing.profile_id "
            f"WHERE r.document_group_id = {_TEST_GROUP_ID} "
            "AND p.profile_key <> :test_key AND target.profile_key = :test_key"
        ),
        {"test_key": TEST_PROFILE_KEY},
    ).scalars().all()
    if collisions:
        raise RuntimeError(
            f"{TEST_PROFILE_KEY} 에 같은 version 의 프로필 버전이 이미 있습니다: {list(collisions)}"
        )


def _check_profile_groups(connection) -> None:
    """NOT NULL, UNIQUE, 복합 외래키를 걸기 전에 어긋난 행을 이름으로 알린다."""

    missing = connection.execute(
        sa.text("SELECT profile_key FROM chat_profiles WHERE document_group_id IS NULL")
    ).scalars().all()
    if missing:
        raise RuntimeError(f"문서 그룹을 정할 수 없는 프로필입니다: {list(missing)}")
    shared = connection.execute(
        sa.text(
            "SELECT document_group_id FROM chat_profiles "
            "GROUP BY document_group_id HAVING count(*) > 1"
        )
    ).scalars().all()
    if shared:
        raise RuntimeError(f"여러 프로필이 같은 문서 그룹을 씁니다: {list(shared)}")
    mismatched = connection.execute(
        sa.text(
            "SELECT r.id FROM chat_profile_revisions r "
            "JOIN chat_profiles p ON p.id = r.profile_id "
            "WHERE r.document_group_id <> p.document_group_id ORDER BY r.id"
        )
    ).scalars().all()
    if mismatched:
        raise RuntimeError(
            f"프로필과 다른 문서 그룹을 쓰는 프로필 버전입니다: {list(mismatched)}"
        )


def upgrade() -> None:
    online = not context.is_offline_mode()

    op.add_column(
        "chat_profiles",
        sa.Column("document_group_id", sa.BigInteger(), nullable=True),
    )
    op.create_foreign_key(
        PROFILE_GROUP_FK,
        "chat_profiles",
        "document_groups",
        ["document_group_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    if online:
        _check_test_group_revisions(op.get_bind())
    # 그 문서 그룹을 쓰는 프로필 버전이 있을 때만 서비스를 만든다.
    op.execute(
        sa.text(
            "INSERT INTO chat_profiles (profile_key, name, document_group_id) "
            "SELECT :test_key, :test_name, g.id FROM document_groups g "
            "WHERE g.group_key = :group_key AND EXISTS ("
            "SELECT 1 FROM chat_profile_revisions r WHERE r.document_group_id = g.id) "
            "ON CONFLICT (profile_key) DO NOTHING"
        ).bindparams(
            test_key=TEST_PROFILE_KEY,
            test_name=TEST_PROFILE_NAME,
            group_key=TEST_GROUP_KEY,
        )
    )
    op.execute(
        sa.text(
            "UPDATE chat_profile_revisions r SET profile_id = p.id, "
            "status = CASE WHEN r.status = 'TESTING' THEN 'PUBLISHED' ELSE r.status END "
            "FROM chat_profiles p "
            f"WHERE p.profile_key = :test_key AND r.document_group_id = {_TEST_GROUP_ID} "
            "AND r.profile_id <> p.id"
        ).bindparams(test_key=TEST_PROFILE_KEY)
    )

    op.execute(
        "UPDATE chat_profiles p SET document_group_id = r.document_group_id "
        "FROM chat_profile_revisions r "
        "WHERE p.document_group_id IS NULL AND r.profile_id = p.id "
        "AND r.status = 'PUBLISHED'"
    )
    op.execute(
        "UPDATE chat_profiles p SET document_group_id = ("
        "SELECT min(r.document_group_id) FROM chat_profile_revisions r "
        "WHERE r.profile_id = p.id) "
        "WHERE p.document_group_id IS NULL AND ("
        "SELECT count(DISTINCT r.document_group_id) FROM chat_profile_revisions r "
        "WHERE r.profile_id = p.id) = 1"
    )
    if online:
        _check_profile_groups(op.get_bind())

    op.alter_column("chat_profiles", "document_group_id", nullable=False)
    op.create_unique_constraint(
        PROFILE_GROUP_UNIQUE, "chat_profiles", ["document_group_id"]
    )
    op.create_unique_constraint(
        PROFILE_ID_GROUP_UNIQUE, "chat_profiles", ["id", "document_group_id"]
    )
    op.create_foreign_key(
        REVISION_PROFILE_GROUP_FK,
        "chat_profile_revisions",
        "chat_profiles",
        ["profile_id", "document_group_id"],
        ["id", "document_group_id"],
        ondelete="RESTRICT",
    )

    op.add_column(
        "conversations",
        sa.Column("profile_id", sa.BigInteger(), nullable=True),
    )
    op.create_foreign_key(
        CONVERSATION_PROFILE_FK,
        "conversations",
        "chat_profiles",
        ["profile_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.execute(
        "UPDATE conversations c SET profile_id = r.profile_id "
        "FROM chat_profile_revisions r WHERE r.id = c.chat_profile_revision_id"
    )
    op.alter_column("conversations", "profile_id", nullable=False)


def _restore_test_revisions(connection) -> None:
    """HELP_CHATBOT_TEST 의 프로필 버전을 HELP_CHATBOT 으로 되돌린다."""

    test_profile_id = connection.execute(
        sa.text("SELECT id FROM chat_profiles WHERE profile_key = :key"),
        {"key": TEST_PROFILE_KEY},
    ).scalar_one_or_none()
    if test_profile_id is None:
        return
    public_profile_id = connection.execute(
        sa.text("SELECT id FROM chat_profiles WHERE profile_key = :key"),
        {"key": PUBLIC_PROFILE_KEY},
    ).scalar_one_or_none()
    if public_profile_id is None:
        raise RuntimeError(f"{PUBLIC_PROFILE_KEY} 프로필이 없어 되돌릴 수 없습니다.")

    revisions = connection.execute(
        sa.text(
            "SELECT id, version, status FROM chat_profile_revisions "
            "WHERE profile_id = :profile_id ORDER BY version, id"
        ),
        {"profile_id": test_profile_id},
    ).all()
    if any(row.status == "TESTING" for row in revisions):
        raise RuntimeError(
            f"{TEST_PROFILE_KEY} 에 TESTING 프로필 버전이 있어 되돌릴 수 없습니다."
        )
    if any(row.status == "PUBLISHED" for row in revisions):
        public_testing = connection.execute(
            sa.text(
                "SELECT id FROM chat_profile_revisions "
                "WHERE profile_id = :profile_id AND status = 'TESTING'"
            ),
            {"profile_id": public_profile_id},
        ).scalar_one_or_none()
        if public_testing is not None:
            raise RuntimeError(
                f"{PUBLIC_PROFILE_KEY} 에 TESTING 프로필 버전({public_testing})이 있어 "
                f"{TEST_PROFILE_KEY} 의 PUBLISHED 프로필 버전을 TESTING 으로 되돌릴 수 없습니다."
            )

    taken = set(
        connection.execute(
            sa.text(
                "SELECT version FROM chat_profile_revisions WHERE profile_id = :profile_id"
            ),
            {"profile_id": public_profile_id},
        ).scalars()
    )
    next_version = max(taken | {row.version for row in revisions}, default=0) + 1
    for row in revisions:
        version = row.version
        if version in taken:
            version = next_version
            next_version += 1
        taken.add(version)
        connection.execute(
            sa.text(
                "UPDATE chat_profile_revisions SET profile_id = :profile_id, "
                "version = :version, "
                "status = CASE WHEN status = 'PUBLISHED' THEN 'TESTING' ELSE status END "
                "WHERE id = :id"
            ),
            {"profile_id": public_profile_id, "version": version, "id": row.id},
        )
    connection.execute(
        sa.text("DELETE FROM chat_profiles WHERE id = :id"),
        {"id": test_profile_id},
    )


def downgrade() -> None:
    op.drop_constraint(CONVERSATION_PROFILE_FK, "conversations", type_="foreignkey")
    op.drop_column("conversations", "profile_id")
    op.drop_constraint(
        REVISION_PROFILE_GROUP_FK, "chat_profile_revisions", type_="foreignkey"
    )
    op.drop_constraint(PROFILE_ID_GROUP_UNIQUE, "chat_profiles", type_="unique")
    op.drop_constraint(PROFILE_GROUP_UNIQUE, "chat_profiles", type_="unique")

    _restore_test_revisions(op.get_bind())

    op.drop_constraint(PROFILE_GROUP_FK, "chat_profiles", type_="foreignkey")
    op.drop_column("chat_profiles", "document_group_id")
