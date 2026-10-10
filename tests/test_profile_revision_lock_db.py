"""프로필 버전 잠금 트리거(20261010_21) 통합 테스트.

구성 칸 불변, 상태 전이, 구성 행 불변, DRAFT 삭제 규칙과 downgrade 를 확인한다.
로컬 PostgreSQL 에 연결할 수 없으면 skip 한다. 일회용 DB 를 만들고 지운다.
"""

import asyncio
import os
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.database.models import (
    ChatProfileRevision,
    ChatProfileRevisionStatus,
    DocumentGroup,
    ProfileRevisionComponent,
    ProfileRevisionComponentRecordedBy,
)
from app.ops.bootstrap_testing_corpus import _test_service_revision


REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_REVISION = "20261010_20"
LOCK_REVISION = "20261010_21"

SUB_STAGES = ("source_planning", "source_planning_repair", "answer", "answer_repair")
STATUSES = ("DRAFT", "TESTING", "PUBLISHED", "RETIRED")
ALLOWED_TRANSITIONS = {
    ("DRAFT", "TESTING"),
    ("TESTING", "PUBLISHED"),
    ("PUBLISHED", "RETIRED"),
    ("TESTING", "RETIRED"),
    ("RETIRED", "PUBLISHED"),
}
# status 외 칸은 모두 구성 칸이다. 칸을 더하면 이 목록과 잠금 범위를 함께 확인한다.
CONFIGURATION_COLUMNS = {
    "id",
    "profile_id",
    "version",
    "document_group_id",
    "generation_model_name",
    "generation_prompt_version",
    "query_rewrite_model_name",
    "query_rewrite_prompt_version",
    "semantic_cache_enabled",
    "exact_cache_enabled",
    "created_at",
}
COMPONENT_COLUMNS = {
    "id",
    "revision_id",
    "stage",
    "model_name",
    "prompt_key",
    "prompt_version",
    "prompt_sha256",
    "params",
    "recorded_by",
    "created_at",
}
TRIGGERS = {
    "trg_chat_profile_revisions_lock_update",
    "trg_chat_profile_revisions_lock_delete",
    "trg_chat_profile_revisions_note_insert",
    "trg_profile_revision_components_lock",
}
FUNCTIONS = {
    "chat_profile_revision_lock_update",
    "chat_profile_revision_lock_delete",
    "chat_profile_revision_note_insert",
    "profile_revision_component_lock",
}
# 칸 종류마다 확실히 다른 값을 만드는 식.
CHANGED_VALUE = {
    "bigint": "{column} + 1000",
    "integer": "{column} + 1000",
    "character varying": "{column} || 'x'",
    "boolean": "NOT {column}",
    "timestamp with time zone": "{column} + interval '1 day'",
    "jsonb": "coalesce({column}, '{{}}'::jsonb) || '{{\"changed\": true}}'::jsonb",
}
CONFIGURATION_LOCKED = "구성 칸은 바꿀 수 없습니다"
TRANSITION_REJECTED = "허용하지 않는 프로필 버전 상태 전이"
COMPONENT_LOCKED = "구성 행은 바꿀 수 없습니다"
COMPONENT_INSERT_REJECTED = "같은 트랜잭션에서 만든 프로필 버전에만 넣을 수 있습니다"
COMPONENT_DELETE_REJECTED = "DRAFT 프로필 버전의 구성 행만 지울 수 있습니다"
DELETE_DRAFT_ONLY = "DRAFT 프로필 버전만 지울 수 있습니다"
DELETE_SERVED = "서빙된 프로필 버전은 지울 수 없습니다"


async def _available(url: str) -> bool:
    engine = create_async_engine(url)
    try:
        async with engine.connect():
            return True
    except Exception:
        return False
    finally:
        await engine.dispose()


def _alembic(url: str, *args: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO_ROOT,
        env={**os.environ, "DATABASE_URL": url},
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise AssertionError(f"alembic {' '.join(args)} 실패:\n{result.stderr}")


class ProfileRevisionLockDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_available(cls.database_url)):
            raise unittest.SkipTest("로컬 PostgreSQL에 연결할 수 없습니다.")

    async def asyncSetUp(self) -> None:
        self.database_name = f"riido_revision_lock_{uuid.uuid4().hex[:12]}"
        self.scratch_url = make_url(self.database_url).set(
            database=self.database_name
        ).render_as_string(hide_password=False)
        await self._maintenance(f'CREATE DATABASE "{self.database_name}"')
        self.engine = create_async_engine(self.scratch_url)

    async def asyncTearDown(self) -> None:
        await self.engine.dispose()
        await self._maintenance(
            f'DROP DATABASE IF EXISTS "{self.database_name}" WITH (FORCE)'
        )

    async def _upgrade_and_seed(self) -> dict:
        _alembic(self.scratch_url, "upgrade", LOCK_REVISION)
        seed = {
            "profile_id": await self._scalar(
                "SELECT id FROM chat_profiles WHERE profile_key = 'HELP_CHATBOT'"
            ),
            "group_id": await self._scalar(
                "SELECT id FROM document_groups WHERE group_key = 'HELP_CHATBOT'"
            ),
            "published_id": await self._scalar(
                "SELECT id FROM chat_profile_revisions WHERE status = 'PUBLISHED'"
            ),
        }
        # 빈 DB 의 PUBLISHED 프로필 버전은 백필로 rewrite, generation, judge 구성 행을 가진다.
        self.assertEqual(
            ["generation", "judge", "rewrite"], await self._stages(seed["published_id"])
        )
        return seed

    # chat_profile_revisions UPDATE

    async def test_every_configuration_column_rejects_update(self) -> None:
        seed = await self._upgrade_and_seed()
        draft_id = await self._insert_revision(seed, 2, "DRAFT")
        columns = dict(
            await self._rows(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = 'chat_profile_revisions'"
            )
        )
        self.assertEqual(CONFIGURATION_COLUMNS | {"status"}, set(columns))

        # DRAFT 도 PUBLISHED 와 같이 잠긴다.
        for revision_id in (draft_id, seed["published_id"]):
            for column in sorted(CONFIGURATION_COLUMNS):
                with self.subTest(revision_id=revision_id, column=column):
                    expression = CHANGED_VALUE[columns[column]].format(column=column)
                    with self.assertRaisesRegex(
                        IntegrityError, f"{CONFIGURATION_LOCKED}.*칸={column}"
                    ):
                        await self._execute(
                            f"UPDATE chat_profile_revisions SET {column} = {expression} "
                            "WHERE id = :id",
                            {"id": revision_id},
                        )
        # 상태 전이와 함께 구성 칸을 바꿔도 거부한다.
        with self.assertRaisesRegex(IntegrityError, "칸=generation_prompt_version"):
            await self._execute(
                "UPDATE chat_profile_revisions "
                "SET status = 'TESTING', generation_prompt_version = 'v99' WHERE id = :id",
                {"id": draft_id},
            )
        # 여러 칸이면 모두 이름을 댄다.
        with self.assertRaisesRegex(
            IntegrityError, "칸=generation_model_name, semantic_cache_enabled"
        ):
            await self._execute(
                "UPDATE chat_profile_revisions SET generation_model_name = 'other', "
                "semantic_cache_enabled = NOT semantic_cache_enabled WHERE id = :id",
                {"id": seed["published_id"]},
            )
        # 같은 값으로 쓰는 것은 변경이 아니다.
        await self._execute(
            "UPDATE chat_profile_revisions SET generation_model_name = generation_model_name, "
            "status = status WHERE id = :id",
            {"id": seed["published_id"]},
        )
        self.assertEqual(
            "DRAFT",
            await self._scalar(
                "SELECT status FROM chat_profile_revisions WHERE id = :id", {"id": draft_id}
            ),
        )

    async def test_allowed_status_transitions_pass(self) -> None:
        seed = await self._upgrade_and_seed()
        v1 = seed["published_id"]
        v2 = await self._insert_revision(seed, 2, "DRAFT")
        v3 = await self._insert_revision(seed, 3, "DRAFT")
        steps = (
            (v2, "TESTING"),  # DRAFT → TESTING
            (v1, "RETIRED"),  # PUBLISHED → RETIRED
            (v2, "PUBLISHED"),  # TESTING → PUBLISHED
            (v2, "RETIRED"),  # PUBLISHED → RETIRED
            (v1, "PUBLISHED"),  # RETIRED → PUBLISHED (되돌리기)
            (v3, "TESTING"),
            (v3, "RETIRED"),  # TESTING → RETIRED
        )
        seen = set()
        for revision_id, status in steps:
            before = await self._status(revision_id)
            seen.add((before, status))
            await self._set_status(revision_id, status)
            self.assertEqual(status, await self._status(revision_id))
        self.assertEqual(ALLOWED_TRANSITIONS, seen)

    async def test_disallowed_status_transitions_are_rejected(self) -> None:
        seed = await self._upgrade_and_seed()
        revision_ids = {
            "PUBLISHED": seed["published_id"],
            "DRAFT": await self._insert_revision(seed, 2, "DRAFT"),
            "TESTING": await self._insert_revision(seed, 3, "TESTING"),
            "RETIRED": await self._insert_revision(seed, 4, "RETIRED"),
        }
        rejected = {
            (before, after)
            for before in STATUSES
            for after in STATUSES
            if before != after and (before, after) not in ALLOWED_TRANSITIONS
        }
        # DRAFT → RETIRED 도 막는다. 열면 DRAFT → RETIRED → PUBLISHED 로 TESTING 을 건너뛴다.
        self.assertIn(("DRAFT", "RETIRED"), rejected)
        self.assertEqual(7, len(rejected))
        for before, after in sorted(rejected):
            with self.subTest(transition=f"{before}->{after}"):
                with self.assertRaisesRegex(
                    IntegrityError, f"{TRANSITION_REJECTED}.*{before} → {after}"
                ):
                    await self._set_status(revision_ids[before], after)
                self.assertEqual(before, await self._status(revision_ids[before]))

    # profile_revision_components

    async def test_component_rows_reject_update(self) -> None:
        seed = await self._upgrade_and_seed()
        draft_id = await self._insert_revision(seed, 2, "DRAFT")
        await self._copy_components(seed["published_id"], draft_id, ("rewrite",))
        columns = dict(
            await self._rows(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = 'profile_revision_components'"
            )
        )
        self.assertEqual(COMPONENT_COLUMNS, set(columns))
        for revision_id in (draft_id, seed["published_id"]):
            for column in sorted(COMPONENT_COLUMNS):
                with self.subTest(revision_id=revision_id, column=column):
                    expression = CHANGED_VALUE[columns[column]].format(column=column)
                    with self.assertRaisesRegex(IntegrityError, COMPONENT_LOCKED):
                        await self._execute(
                            f"UPDATE profile_revision_components SET {column} = {expression} "
                            "WHERE revision_id = :id AND stage = 'rewrite'",
                            {"id": revision_id},
                        )
        # 같은 값으로 쓰는 것도 막는다(구성 행에는 바꿀 칸이 없다).
        with self.assertRaisesRegex(IntegrityError, COMPONENT_LOCKED):
            await self._execute(
                "UPDATE profile_revision_components SET model_name = model_name "
                "WHERE revision_id = :id",
                {"id": draft_id},
            )

    async def test_component_insert_and_delete_follow_parent_status(self) -> None:
        seed = await self._upgrade_and_seed()
        draft_id = await self._insert_revision(seed, 2, "DRAFT")
        # DRAFT 는 다음 트랜잭션에서도 구성 행을 넣고 지울 수 있다.
        await self._copy_components(seed["published_id"], draft_id, ("rewrite", "generation"))
        await self._execute(
            "DELETE FROM profile_revision_components "
            "WHERE revision_id = :id AND stage = 'generation'",
            {"id": draft_id},
        )
        self.assertEqual(
            ["rewrite"],
            await self._stages(draft_id),
        )

        await self._set_status(draft_id, "TESTING")
        for status, revision_id in (
            ("TESTING", draft_id),
            ("PUBLISHED", seed["published_id"]),
        ):
            with self.subTest(status=status, operation="insert"):
                with self.assertRaisesRegex(IntegrityError, COMPONENT_INSERT_REJECTED):
                    await self._insert_component(revision_id, "judge")
            with self.subTest(status=status, operation="delete"):
                with self.assertRaisesRegex(IntegrityError, COMPONENT_DELETE_REJECTED):
                    await self._execute(
                        "DELETE FROM profile_revision_components WHERE revision_id = :id",
                        {"id": revision_id},
                    )
        retired_id = await self._insert_revision(seed, 3, "RETIRED")
        with self.assertRaisesRegex(IntegrityError, COMPONENT_INSERT_REJECTED):
            await self._insert_component(retired_id, "judge")
        self.assertEqual(["rewrite"], await self._stages(draft_id))

    async def test_new_version_with_components_in_one_transaction(self) -> None:
        seed = await self._upgrade_and_seed()
        sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        # 게시 경로: 기존 PUBLISHED 를 내리고 새 PUBLISHED 와 구성 행을 한 트랜잭션에 만든다.
        async with sessions() as session, session.begin():
            published = await session.get(ChatProfileRevision, seed["published_id"])
            published.status = ChatProfileRevisionStatus.RETIRED
            await session.flush()
            revision = ChatProfileRevision(
                profile_id=published.profile_id,
                version=2,
                document_group_id=published.document_group_id,
                status=ChatProfileRevisionStatus.PUBLISHED,
                generation_model_name=published.generation_model_name,
                generation_prompt_version=published.generation_prompt_version,
                query_rewrite_model_name=published.query_rewrite_model_name,
                query_rewrite_prompt_version=published.query_rewrite_prompt_version,
            )
            session.add(revision)
            await session.flush()
            components = (
                await session.scalars(
                    select(ProfileRevisionComponent).where(
                        ProfileRevisionComponent.revision_id == published.id
                    )
                )
            ).all()
            for component in components:
                session.add(
                    ProfileRevisionComponent(
                        revision_id=revision.id,
                        stage=component.stage,
                        model_name=component.model_name,
                        prompt_key=component.prompt_key,
                        prompt_version=component.prompt_version,
                        prompt_sha256=component.prompt_sha256,
                        params=component.params,
                        recorded_by=ProfileRevisionComponentRecordedBy.PUBLISH,
                    )
                )
            await session.flush()
            new_id = revision.id
        self.assertEqual(
            await self._stages(seed["published_id"]), await self._stages(new_id)
        )
        self.assertEqual("PUBLISHED", await self._status(new_id))

        # TESTING 도 만든 트랜잭션 안에서는 구성 행을 받고, 커밋 뒤에는 거부한다.
        async with self.engine.begin() as connection:
            testing_id = (
                await connection.execute(
                    text(
                        "INSERT INTO chat_profile_revisions "
                        "(profile_id, version, document_group_id, status, "
                        "generation_model_name, generation_prompt_version, "
                        "query_rewrite_model_name, query_rewrite_prompt_version) "
                        "SELECT profile_id, 3, document_group_id, 'TESTING', "
                        "generation_model_name, generation_prompt_version, "
                        "query_rewrite_model_name, query_rewrite_prompt_version "
                        "FROM chat_profile_revisions WHERE id = :id RETURNING id"
                    ),
                    {"id": new_id},
                )
            ).scalar_one()
            for stage in ("rewrite", "generation"):
                await connection.execute(
                    text(
                        "INSERT INTO profile_revision_components "
                        "(revision_id, stage, model_name, prompt_key, prompt_version, "
                        "recorded_by) SELECT :target, stage, model_name, prompt_key, "
                        "prompt_version, 'PUBLISH' FROM profile_revision_components "
                        "WHERE revision_id = :source AND stage = :stage"
                    ),
                    {"target": testing_id, "source": new_id, "stage": stage},
                )
        self.assertEqual(["generation", "rewrite"], await self._stages(testing_id))
        with self.assertRaisesRegex(IntegrityError, COMPONENT_INSERT_REJECTED):
            await self._insert_component(testing_id, "judge")

        # 저장점을 되돌리면 그 안에서 만든 프로필 버전 표시도 사라진다.
        async with sessions() as session, session.begin():
            savepoint = await session.begin_nested()
            draft = ChatProfileRevision(
                profile_id=seed["profile_id"],
                version=4,
                document_group_id=seed["group_id"],
                status=ChatProfileRevisionStatus.DRAFT,
                generation_model_name="m",
                generation_prompt_version="v1",
                query_rewrite_model_name="m",
                query_rewrite_prompt_version="v1",
            )
            session.add(draft)
            await session.flush()
            await savepoint.rollback()
            setting = await session.scalar(
                text(
                    "SELECT coalesce(current_setting('riido.created_profile_revision_ids', true), '')"
                )
            )
            self.assertEqual("", setting)

    async def test_testing_corpus_bootstrap_still_replaces_published_version(self) -> None:
        seed = await self._upgrade_and_seed()
        sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with sessions() as session, session.begin():
            group = DocumentGroup(
                group_key="HELP_CHATBOT_TEST", name="테스트", consumer_key="HELP_CHATBOT"
            )
            session.add(group)
            await session.flush()
            group_id = group.id
            profile, first = await _test_service_revision(session, group)
            first_id = first.id
        async with self.engine.begin() as connection:
            # 지금 코드와 다른 PUBLISHED 프로필 버전을 둔다(구성 칸은 고칠 수 없어 새로 만든다).
            await connection.execute(
                text("UPDATE chat_profile_revisions SET status = 'RETIRED' WHERE id = :id"),
                {"id": first_id},
            )
            await connection.execute(
                text(
                    "INSERT INTO chat_profile_revisions "
                    "(profile_id, version, document_group_id, status, "
                    "generation_model_name, generation_prompt_version, "
                    "query_rewrite_model_name, query_rewrite_prompt_version) "
                    "VALUES (:profile_id, 2, :group_id, 'PUBLISHED', 'old', 'v1', 'old', 'v1')"
                ),
                {"profile_id": profile.id, "group_id": group_id},
            )
        async with sessions() as session, session.begin():
            group = await session.get(DocumentGroup, group_id)
            _, replacement = await _test_service_revision(session, group)
            replacement_id = replacement.id
        self.assertEqual(
            [(1, "RETIRED"), (2, "RETIRED"), (3, "PUBLISHED")],
            [
                tuple(row)
                for row in await self._rows(
                    "SELECT version, status FROM chat_profile_revisions "
                    "WHERE profile_id = :id ORDER BY version",
                    {"id": profile.id},
                )
            ],
        )
        self.assertEqual("PUBLISHED", await self._status(replacement_id))
        self.assertEqual("PUBLISHED", await self._status(seed["published_id"]))

    # chat_profile_revisions DELETE

    async def test_only_never_served_draft_can_be_deleted(self) -> None:
        seed = await self._upgrade_and_seed()
        draft_id = await self._insert_revision(seed, 2, "DRAFT")
        await self._copy_components(seed["published_id"], draft_id, ("rewrite",))
        # 구성 행부터 지우면 프로필 버전도 지울 수 있다.
        await self._execute(
            "DELETE FROM profile_revision_components WHERE revision_id = :id", {"id": draft_id}
        )
        await self._execute(
            "DELETE FROM chat_profile_revisions WHERE id = :id", {"id": draft_id}
        )
        self.assertEqual(
            0,
            await self._scalar(
                "SELECT count(*) FROM chat_profile_revisions WHERE id = :id", {"id": draft_id}
            ),
        )

        for version, status in ((3, "TESTING"), (4, "RETIRED")):
            revision_id = await self._insert_revision(seed, version, status)
            with self.subTest(status=status):
                with self.assertRaisesRegex(IntegrityError, f"{DELETE_DRAFT_ONLY}.*{status}"):
                    await self._delete_revision(revision_id)
        with self.subTest(status="PUBLISHED"):
            with self.assertRaisesRegex(IntegrityError, DELETE_DRAFT_ONLY):
                await self._delete_revision(seed["published_id"])

        # 대화가 가리키는 DRAFT.
        conversation_draft = await self._insert_revision(seed, 5, "DRAFT")
        conversation_id = uuid.uuid4()
        await self._execute(
            "INSERT INTO conversations (id, chat_profile_revision_id, profile_id, status) "
            "VALUES (:id, :revision_id, :profile_id, 'ACTIVE')",
            {
                "id": conversation_id,
                "revision_id": conversation_draft,
                "profile_id": seed["profile_id"],
            },
        )
        with self.assertRaisesRegex(
            IntegrityError, f"{DELETE_SERVED}.*conversations.chat_profile_revision_id"
        ):
            await self._delete_revision(conversation_draft)

        # 턴 기록만 가리키는 DRAFT(대화는 다른 프로필 버전).
        run_draft = await self._insert_revision(seed, 6, "DRAFT")
        other_conversation = uuid.uuid4()
        await self._execute(
            "INSERT INTO conversations (id, chat_profile_revision_id, profile_id, status) "
            "VALUES (:id, :revision_id, :profile_id, 'ACTIVE')",
            {
                "id": other_conversation,
                "revision_id": seed["published_id"],
                "profile_id": seed["profile_id"],
            },
        )
        index_id = await self._insert_index_version(seed["group_id"])
        await self._execute(
            "INSERT INTO rag_runs "
            "(id, trace_id, conversation_id, turn_no, index_version_id, user_query, "
            "context_strategy, status, profile_revision_id) "
            "VALUES (:id, :trace_id, :conversation_id, 1, :index_id, '질문', "
            "'NEW_TOPIC', 'COMPLETED', :revision_id)",
            {
                "id": uuid.uuid4(),
                "trace_id": uuid.uuid4(),
                "conversation_id": other_conversation,
                "index_id": index_id,
                "revision_id": run_draft,
            },
        )
        with self.assertRaisesRegex(
            IntegrityError, f"{DELETE_SERVED}.*rag_runs.profile_revision_id"
        ):
            await self._delete_revision(run_draft)
        for revision_id in (conversation_draft, run_draft):
            self.assertEqual("DRAFT", await self._status(revision_id))

    # 기존 데이터와 downgrade

    async def test_upgrade_keeps_dev_shaped_rows_and_locks_them(self) -> None:
        # DEV: 서비스 둘, PUBLISHED 프로필 버전 둘, 각각 BACKFILL 구성 행 7개.
        _alembic(self.scratch_url, "upgrade", PARENT_REVISION)
        public_id = await self._scalar(
            "SELECT id FROM chat_profile_revisions WHERE status = 'PUBLISHED'"
        )
        group_id = await self._scalar_write(
            "INSERT INTO document_groups (group_key, name, consumer_key) "
            "VALUES ('HELP_CHATBOT_TEST', '테스트', 'HELP_CHATBOT') RETURNING id"
        )
        profile_id = await self._scalar_write(
            "INSERT INTO chat_profiles (profile_key, name, document_group_id) "
            "VALUES ('HELP_CHATBOT_TEST', '테스트 서비스', :group_id) RETURNING id",
            {"group_id": group_id},
        )
        # 빈 DB 는 백필 구성 행이 3개라 DEV 처럼 생성 하위 단계 4개를 더한다.
        prompt_version = await self._scalar(
            "SELECT generation_prompt_version FROM chat_profile_revisions WHERE id = :id",
            {"id": public_id},
        )
        for stage in SUB_STAGES:
            await self._execute(
                "INSERT INTO profile_revision_components "
                "(revision_id, stage, model_name, prompt_key, prompt_version, recorded_by) "
                "SELECT :id, :stage, generation_model_name, :prompt_key, :prompt_version, "
                "'BACKFILL' FROM chat_profile_revisions WHERE id = :id",
                {
                    "id": public_id,
                    "stage": stage,
                    "prompt_key": f"generation.{stage}@{prompt_version}",
                    "prompt_version": prompt_version,
                },
            )
        test_id = await self._scalar_write(
            "INSERT INTO chat_profile_revisions "
            "(profile_id, version, document_group_id, status, "
            "generation_model_name, generation_prompt_version, "
            "query_rewrite_model_name, query_rewrite_prompt_version) "
            "SELECT :profile_id, 2, :group_id, 'PUBLISHED', generation_model_name, "
            "generation_prompt_version, query_rewrite_model_name, "
            "query_rewrite_prompt_version FROM chat_profile_revisions WHERE id = :id "
            "RETURNING id",
            {"profile_id": profile_id, "group_id": group_id, "id": public_id},
        )
        await self._execute(
            "INSERT INTO profile_revision_components "
            "(revision_id, stage, model_name, prompt_key, prompt_version, prompt_sha256, "
            "params, recorded_by) SELECT :target, stage, model_name, prompt_key, "
            "prompt_version, prompt_sha256, params, 'BACKFILL' "
            "FROM profile_revision_components WHERE revision_id = :source",
            {"target": test_id, "source": public_id},
        )
        snapshot = (
            "SELECT r.id, r.status, r.generation_prompt_version, c.stage, c.prompt_key, "
            "c.recorded_by FROM chat_profile_revisions r "
            "JOIN profile_revision_components c ON c.revision_id = r.id ORDER BY r.id, c.stage"
        )
        before = await self._rows(snapshot)
        self.assertEqual(14, len(before))
        self.assertEqual({"BACKFILL"}, {row.recorded_by for row in before})

        _alembic(self.scratch_url, "upgrade", LOCK_REVISION)
        self.assertEqual(before, await self._rows(snapshot))

        for revision_id in (public_id, test_id):
            with self.subTest(revision_id=revision_id):
                with self.assertRaisesRegex(IntegrityError, CONFIGURATION_LOCKED):
                    await self._execute(
                        "UPDATE chat_profile_revisions SET generation_prompt_version = 'v99' "
                        "WHERE id = :id",
                        {"id": revision_id},
                    )
                with self.assertRaisesRegex(IntegrityError, COMPONENT_LOCKED):
                    await self._execute(
                        "UPDATE profile_revision_components SET prompt_version = prompt_version "
                        "WHERE revision_id = :id",
                        {"id": revision_id},
                    )
                with self.assertRaisesRegex(IntegrityError, COMPONENT_DELETE_REJECTED):
                    await self._execute(
                        "DELETE FROM profile_revision_components WHERE revision_id = :id",
                        {"id": revision_id},
                    )
                # 내리고 되돌리기는 된다.
                await self._set_status(revision_id, "RETIRED")
                await self._set_status(revision_id, "PUBLISHED")
        self.assertEqual(before, await self._rows(snapshot))

    async def test_downgrade_removes_triggers_and_functions(self) -> None:
        seed = await self._upgrade_and_seed()
        self.assertEqual(TRIGGERS, await self._trigger_names())
        self.assertEqual(FUNCTIONS, await self._function_names())

        _alembic(self.scratch_url, "downgrade", PARENT_REVISION)
        self.assertEqual(set(), await self._trigger_names())
        self.assertEqual(set(), await self._function_names())
        # 잠금 전처럼 구성 칸과 구성 행을 바꿀 수 있다.
        await self._execute(
            "UPDATE chat_profile_revisions SET semantic_cache_enabled = NOT semantic_cache_enabled "
            "WHERE id = :id",
            {"id": seed["published_id"]},
        )
        await self._execute(
            "UPDATE profile_revision_components SET model_name = model_name "
            "WHERE revision_id = :id",
            {"id": seed["published_id"]},
        )

        _alembic(self.scratch_url, "upgrade", LOCK_REVISION)
        self.assertEqual(TRIGGERS, await self._trigger_names())

    # helpers

    async def _insert_revision(self, seed: dict, version: int, status: str) -> int:
        return await self._scalar_write(
            "INSERT INTO chat_profile_revisions "
            "(profile_id, version, document_group_id, status, "
            "generation_model_name, generation_prompt_version, "
            "query_rewrite_model_name, query_rewrite_prompt_version) "
            "SELECT profile_id, :version, document_group_id, :status, "
            "generation_model_name, generation_prompt_version, "
            "query_rewrite_model_name, query_rewrite_prompt_version "
            "FROM chat_profile_revisions WHERE id = :id RETURNING id",
            {"version": version, "status": status, "id": seed["published_id"]},
        )

    async def _copy_components(self, source_id: int, target_id: int, stages) -> None:
        for stage in stages:
            await self._execute(
                "INSERT INTO profile_revision_components "
                "(revision_id, stage, model_name, prompt_key, prompt_version, recorded_by) "
                "SELECT :target, stage, model_name, prompt_key, prompt_version, 'PUBLISH' "
                "FROM profile_revision_components WHERE revision_id = :source AND stage = :stage",
                {"target": target_id, "source": source_id, "stage": stage},
            )

    async def _insert_component(self, revision_id: int, stage: str) -> None:
        await self._execute(
            "INSERT INTO profile_revision_components "
            "(revision_id, stage, model_name, prompt_key, prompt_version, recorded_by) "
            "VALUES (:id, :stage, 'm', :prompt_key, 'v1', 'PUBLISH')",
            {"id": revision_id, "stage": stage, "prompt_key": f"{stage}@v1"},
        )

    async def _insert_index_version(self, group_id: int) -> int:
        chunking_id = await self._scalar_write(
            "INSERT INTO chunking_configs (version, strategy, max_tokens) "
            "VALUES ('lock-test', 'section', 500) RETURNING id"
        )
        embedding_id = await self._scalar_write(
            "INSERT INTO embedding_configs "
            "(version, provider, model_name, dimensions, input_template_version) "
            "VALUES ('lock-test', 'openai', 'text-embedding-3-large', 1536, 'v1') "
            "RETURNING id"
        )
        return await self._scalar_write(
            "INSERT INTO index_versions "
            "(document_group_id, version, status, chunking_config_id, embedding_config_id) "
            "VALUES (:group_id, 'lock-test', 'ACTIVE', :chunking_id, :embedding_id) "
            "RETURNING id",
            {"group_id": group_id, "chunking_id": chunking_id, "embedding_id": embedding_id},
        )

    async def _set_status(self, revision_id: int, status: str) -> None:
        await self._execute(
            "UPDATE chat_profile_revisions SET status = :status WHERE id = :id",
            {"status": status, "id": revision_id},
        )

    async def _delete_revision(self, revision_id: int) -> None:
        await self._execute(
            "DELETE FROM chat_profile_revisions WHERE id = :id", {"id": revision_id}
        )

    async def _status(self, revision_id: int) -> str:
        return await self._scalar(
            "SELECT status FROM chat_profile_revisions WHERE id = :id", {"id": revision_id}
        )

    async def _stages(self, revision_id: int) -> list:
        return [
            row.stage
            for row in await self._rows(
                "SELECT stage FROM profile_revision_components WHERE revision_id = :id "
                "ORDER BY stage",
                {"id": revision_id},
            )
        ]

    async def _trigger_names(self) -> set:
        return {
            row.tgname
            for row in await self._rows(
                "SELECT tgname FROM pg_trigger WHERE NOT tgisinternal AND tgrelid IN "
                "('chat_profile_revisions'::regclass, 'profile_revision_components'::regclass)"
            )
        }

    async def _function_names(self) -> set:
        return {
            row.proname
            for row in await self._rows(
                "SELECT proname FROM pg_proc WHERE proname = ANY(:names)",
                {"names": list(FUNCTIONS)},
            )
        }

    async def _maintenance(self, statement: str) -> None:
        url = make_url(self.database_url).set(database="postgres")
        engine = create_async_engine(
            url.render_as_string(hide_password=False),
            isolation_level="AUTOCOMMIT",
        )
        try:
            async with engine.connect() as connection:
                await connection.execute(text(statement))
        finally:
            await engine.dispose()

    async def _execute(self, statement: str, parameters=None) -> None:
        async with self.engine.begin() as connection:
            await connection.execute(text(statement), parameters or {})

    async def _scalar_write(self, statement: str, parameters=None):
        async with self.engine.begin() as connection:
            return (await connection.execute(text(statement), parameters or {})).scalar_one()

    async def _scalar(self, statement: str, parameters=None):
        async with self.engine.connect() as connection:
            return (await connection.execute(text(statement), parameters or {})).scalar_one()

    async def _rows(self, statement: str, parameters=None):
        async with self.engine.connect() as connection:
            return (await connection.execute(text(statement), parameters or {})).fetchall()


if __name__ == "__main__":
    unittest.main()
