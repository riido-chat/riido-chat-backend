"""Disposable PostgreSQL checks for the chat profile migration."""

import asyncio
import os
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings


REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_REVISION = "20260908_10"
CHAT_PROFILE_REVISION = "20260911_11"
CONVERSATION_PROFILE_FK = (
    "fk_conversations_profile_revision_id_chat_profile_revisions"
)


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


class ChatProfileMigrationDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_available(cls.database_url)):
            raise unittest.SkipTest("로컬 PostgreSQL에 연결할 수 없습니다.")

    async def asyncSetUp(self) -> None:
        self.database_name = f"riido_profile_{uuid.uuid4().hex[:12]}"
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

    async def test_empty_upgrade_and_downgrade(self) -> None:
        _alembic(self.scratch_url, "upgrade", CHAT_PROFILE_REVISION)
        self.assertEqual(CHAT_PROFILE_REVISION, await self._scalar("SELECT version_num FROM alembic_version"))
        self.assertTrue(await self._table("chat_profiles"))
        self.assertTrue(await self._column("conversations", "channel"))
        self.assertEqual(
            CONVERSATION_PROFILE_FK,
            await self._scalar(
                "SELECT tc.constraint_name "
                "FROM information_schema.table_constraints tc "
                "JOIN information_schema.key_column_usage kcu "
                "ON kcu.constraint_name = tc.constraint_name "
                "AND kcu.table_schema = tc.table_schema "
                "WHERE tc.table_name = 'conversations' "
                "AND tc.constraint_type = 'FOREIGN KEY' "
                "AND kcu.column_name = 'chat_profile_revision_id'"
            ),
        )
        _alembic(self.scratch_url, "downgrade", PARENT_REVISION)
        self.assertFalse(await self._table("chat_profiles"))

    async def test_existing_conversations_are_backfilled_to_public_channel(self) -> None:
        _alembic(self.scratch_url, "upgrade", PARENT_REVISION)
        conversation_id = str(uuid.uuid4())
        async with self.engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO conversations "
                    "(id, status, created_at, last_active_at) "
                    "VALUES (:id, 'ACTIVE', now(), now())"
                ),
                {"id": conversation_id},
            )

        _alembic(self.scratch_url, "upgrade", CHAT_PROFILE_REVISION)
        row = await self._row(
            "SELECT c.chat_profile_revision_id, c.channel, p.profile_key "
            "FROM conversations c "
            "JOIN chat_profile_revisions r ON r.id = c.chat_profile_revision_id "
            "JOIN chat_profiles p ON p.id = r.profile_id "
            "WHERE c.id = :id",
            {"id": conversation_id},
        )
        self.assertEqual("PUBLIC", row[1])
        self.assertEqual("HELP_CHATBOT", row[2])
        new_conversation_id = str(uuid.uuid4())
        async with self.engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO conversations "
                    "(id, chat_profile_revision_id, status, created_at, last_active_at) "
                    "VALUES (:id, :revision_id, 'ACTIVE', now(), now())"
                ),
                {"id": new_conversation_id, "revision_id": row[0]},
            )
        self.assertEqual(
            "PUBLIC",
            await self._scalar(
                "SELECT channel FROM conversations WHERE id = :id",
                {"id": new_conversation_id},
            ),
        )
        self.assertEqual(1, await self._scalar(
            "SELECT count(*) FROM chat_profile_revisions WHERE status = 'PUBLISHED'"
        ))
        # The migration seeds the production profile only.  TESTING revisions
        # are fixture-owned so every environment can choose its test config.
        self.assertEqual(0, await self._scalar(
            "SELECT count(*) FROM chat_profile_revisions WHERE status = 'TESTING'"
        ))

        profile_id = await self._scalar(
            "SELECT id FROM chat_profiles WHERE profile_key = 'HELP_CHATBOT'"
        )
        group_id = await self._scalar(
            "SELECT id FROM document_groups WHERE group_key = 'HELP_CHATBOT'"
        )
        revision_values = {
            "profile_id": profile_id,
            "group_id": group_id,
        }
        async with self.engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO chat_profile_revisions "
                    "(profile_id, version, document_group_id, status, "
                    "generation_model_name, generation_prompt_version, "
                    "query_rewrite_model_name, query_rewrite_prompt_version) "
                    "VALUES (:profile_id, 2, :group_id, 'TESTING', "
                    "'gpt-5.6-terra', 'v24', 'gpt-5.4-mini', 'v7')"
                ),
                revision_values,
            )
        with self.assertRaises(IntegrityError):
            async with self.engine.begin() as connection:
                await connection.execute(
                    text(
                        "INSERT INTO chat_profile_revisions "
                        "(profile_id, version, document_group_id, status, "
                        "generation_model_name, generation_prompt_version, "
                        "query_rewrite_model_name, query_rewrite_prompt_version) "
                        "VALUES (:profile_id, 3, :group_id, 'TESTING', "
                        "'gpt-5.6-terra', 'v24', 'gpt-5.4-mini', 'v7')"
                    ),
                    revision_values,
                )

        with self.assertRaises(IntegrityError):
            async with self.engine.begin() as connection:
                await connection.execute(
                    text(
                        "INSERT INTO conversations "
                        "(id, chat_profile_revision_id, status, created_at, last_active_at) "
                        "VALUES (:id, 999999999, 'ACTIVE', now(), now())"
                    ),
                    {"id": str(uuid.uuid4())},
                )

        with self.assertRaises(IntegrityError):
            async with self.engine.begin() as connection:
                await connection.execute(
                    text(
                        "INSERT INTO chat_profile_revisions "
                        "(profile_id, version, document_group_id, status, "
                        "generation_model_name, generation_prompt_version, "
                        "query_rewrite_model_name, query_rewrite_prompt_version) "
                        "VALUES (:profile_id, 4, 999999999, 'DRAFT', "
                        "'gpt-5.6-terra', 'v24', 'gpt-5.4-mini', 'v7')"
                    ),
                    revision_values,
                )

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

    async def _scalar(self, statement: str, parameters=None):
        async with self.engine.connect() as connection:
            return (await connection.execute(text(statement), parameters or {})).scalar_one()

    async def _row(self, statement: str, parameters=None):
        async with self.engine.connect() as connection:
            return (await connection.execute(text(statement), parameters or {})).one()

    async def _table(self, name: str) -> bool:
        return await self._scalar("SELECT to_regclass(:name)", {"name": name}) is not None

    async def _column(self, table: str, column: str) -> bool:
        return bool(await self._scalar(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = :table AND column_name = :column",
            {"table": table, "column": column},
        ))


if __name__ == "__main__":
    unittest.main()
