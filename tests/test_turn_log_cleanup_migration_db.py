"""턴 기록 미사용 칸 정리(20261001_19) migration 의 upgrade, 값 보호, downgrade 통합 테스트.

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
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.config import get_settings
from app.database.models import AnswerStatus, ContextStrategy, RagRun


REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_REVISION = "20261001_18"
CLEANUP_REVISION = "20261001_19"
DROPPED_COLUMNS = (
    ("model_calls", "estimated_cost"),
    ("conversations", "summary_text"),
    ("conversations", "summary_version"),
    ("conversations", "summary_updated_turn_no"),
    ("chat_profile_revisions", "verifier_model_name"),
    ("chat_profile_revisions", "verifier_prompt_version"),
)
SNAPSHOT = '{"schemaVersion": "v2", "selectedTurns": [{"turnNo": 1}]}'


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


class TurnLogCleanupMigrationDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_available(cls.database_url)):
            raise unittest.SkipTest("로컬 PostgreSQL에 연결할 수 없습니다.")

    async def asyncSetUp(self) -> None:
        self.database_name = f"riido_turn_log_cleanup_{uuid.uuid4().hex[:12]}"
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

    async def test_upgrade_drops_columns_and_unifies_missing_snapshot(self) -> None:
        _alembic(self.scratch_url, "upgrade", PARENT_REVISION)
        seed = await self._seed()
        json_null_run = await self._insert_run(seed, 1, "'null'::jsonb")
        object_run = await self._insert_run(seed, 2, f"'{SNAPSHOT}'::jsonb")
        sql_null_run = await self._insert_run(seed, 3, "NULL")
        await self._execute(
            "INSERT INTO model_calls (rag_run_id, purpose, provider, model_name, status) "
            "VALUES (:id, 'GENERATION', 'openai', 'gpt-test', 'SUCCESS')",
            {"id": object_run},
        )

        _alembic(self.scratch_url, "upgrade", CLEANUP_REVISION)

        for table, column in DROPPED_COLUMNS:
            with self.subTest(table=table, column=column):
                self.assertFalse(await self._column(table, column))
        self.assertEqual(
            {json_null_run: "SQL NULL", object_run: "object", sql_null_run: "SQL NULL"},
            await self._snapshot_kinds(),
        )
        self.assertEqual(
            {"schemaVersion": "v2", "selectedTurns": [{"turnNo": 1}]},
            await self._scalar(
                "SELECT context_snapshot FROM rag_runs WHERE id = :id",
                {"id": object_run},
            ),
        )

        # 대화 요약 용도는 더 쓰지 않는다. 남은 용도는 그대로 받는다.
        with self.assertRaises(IntegrityError):
            await self._execute(
                "INSERT INTO model_calls (rag_run_id, purpose, provider, model_name, status) "
                "VALUES (:id, 'CONVERSATION_SUMMARY', 'openai', 'gpt-test', 'SUCCESS')",
                {"id": object_run},
            )
        await self._execute(
            "INSERT INTO model_calls (rag_run_id, purpose, provider, model_name, status) "
            "VALUES (:id, 'QUERY_REWRITE', 'openai', 'gpt-test', 'SUCCESS')",
            {"id": object_run},
        )

        # ORM 이 None 을 쓰면 SQL NULL 이다.
        orm_run = uuid.uuid4()
        async with AsyncSession(self.engine) as session:
            session.add(
                RagRun(
                    id=orm_run,
                    conversation_id=seed["conversation_id"],
                    turn_no=4,
                    index_version_id=seed["index_version_id"],
                    user_query="질문",
                    context_strategy=ContextStrategy.NEW_TOPIC,
                    context_turn_count=0,
                    context_snapshot=None,
                    status=AnswerStatus.COMPLETED,
                )
            )
            await session.commit()
        self.assertEqual("SQL NULL", (await self._snapshot_kinds())[orm_run])

        # 배포 중 이전 코드가 JSON null 을 써도 읽는 쪽은 SQL NULL 과 같이 None 으로 읽는다.
        late_json_null_run = await self._insert_run(seed, 5, "'null'::jsonb")
        async with AsyncSession(self.engine) as session:
            snapshots = dict(
                (
                    await session.execute(
                        select(RagRun.id, RagRun.context_snapshot).where(
                            RagRun.id.in_([late_json_null_run, sql_null_run])
                        )
                    )
                ).all()
            )
        self.assertEqual({late_json_null_run: None, sql_null_run: None}, snapshots)

        _alembic(self.scratch_url, "downgrade", PARENT_REVISION)
        for table, column in DROPPED_COLUMNS:
            with self.subTest(table=table, column=column):
                self.assertTrue(await self._column(table, column))
                self.assertEqual(
                    "YES",
                    await self._scalar(
                        "SELECT is_nullable FROM information_schema.columns "
                        "WHERE table_name = :table AND column_name = :column",
                        {"table": table, "column": column},
                    ),
                )
        await self._execute(
            "INSERT INTO model_calls (rag_run_id, purpose, provider, model_name, status) "
            "VALUES (:id, 'CONVERSATION_SUMMARY', 'openai', 'gpt-test', 'SUCCESS')",
            {"id": object_run},
        )
        # JSON null 과 SQL NULL 의 구분은 되살리지 않는다.
        self.assertEqual("SQL NULL", (await self._snapshot_kinds())[json_null_run])
        self.assertEqual(
            PARENT_REVISION, await self._scalar("SELECT version_num FROM alembic_version")
        )

    async def test_upgrade_stops_when_dropped_column_has_value(self) -> None:
        _alembic(self.scratch_url, "upgrade", PARENT_REVISION)
        seed = await self._seed()
        await self._execute(
            "UPDATE conversations SET summary_text = '요약' WHERE id = :id",
            {"id": seed["conversation_id"]},
        )

        with self.assertRaisesRegex(AssertionError, "conversations.summary_text=1"):
            _alembic(self.scratch_url, "upgrade", CLEANUP_REVISION)
        self.assertTrue(await self._column("conversations", "summary_text"))
        self.assertEqual(
            PARENT_REVISION, await self._scalar("SELECT version_num FROM alembic_version")
        )

    async def test_upgrade_stops_when_conversation_summary_call_exists(self) -> None:
        _alembic(self.scratch_url, "upgrade", PARENT_REVISION)
        seed = await self._seed()
        run_id = await self._insert_run(seed, 1, "NULL")
        await self._execute(
            "INSERT INTO model_calls (rag_run_id, purpose, provider, model_name, status) "
            "VALUES (:id, 'CONVERSATION_SUMMARY', 'openai', 'gpt-test', 'SUCCESS')",
            {"id": run_id},
        )

        with self.assertRaisesRegex(
            AssertionError, "model_calls.purpose=CONVERSATION_SUMMARY=1"
        ):
            _alembic(self.scratch_url, "upgrade", CLEANUP_REVISION)
        self.assertTrue(await self._column("model_calls", "estimated_cost"))

    # ------------------------------------------------------------------
    # 준비
    # ------------------------------------------------------------------

    async def _seed(self) -> dict:
        seed = {}
        seed["group_id"] = await self._scalar(
            "SELECT id FROM document_groups WHERE group_key = 'HELP_CHATBOT'"
        )
        seed["profile_revision_id"] = await self._scalar(
            "SELECT r.id FROM chat_profile_revisions r "
            "JOIN chat_profiles p ON p.id = r.profile_id "
            "WHERE p.profile_key = 'HELP_CHATBOT' AND r.status = 'PUBLISHED'"
        )
        seed["chunking_config_id"] = await self._scalar_write(
            "INSERT INTO chunking_configs (version, strategy, max_tokens) "
            "VALUES ('cleanup-test', 'section', 500) RETURNING id"
        )
        seed["embedding_config_id"] = await self._scalar_write(
            "INSERT INTO embedding_configs "
            "(version, provider, model_name, dimensions, input_template_version) "
            "VALUES ('cleanup-test', 'openai', 'text-embedding-3-large', 1536, 'v1') "
            "RETURNING id"
        )
        seed["index_version_id"] = await self._scalar_write(
            "INSERT INTO index_versions "
            "(document_group_id, version, status, chunking_config_id, embedding_config_id) "
            "VALUES (:group_id, 'cleanup-test', 'READY', :chunking_config_id, "
            ":embedding_config_id) RETURNING id",
            seed,
        )
        seed["conversation_id"] = uuid.uuid4()
        await self._execute(
            "INSERT INTO conversations (id, chat_profile_revision_id, status) "
            "VALUES (:conversation_id, :profile_revision_id, 'ACTIVE')",
            seed,
        )
        return seed

    async def _insert_run(self, seed: dict, turn_no: int, snapshot_sql: str) -> uuid.UUID:
        run_id = uuid.uuid4()
        await self._execute(
            "INSERT INTO rag_runs "
            "(id, trace_id, conversation_id, turn_no, index_version_id, user_query, "
            "context_strategy, context_snapshot, status) "
            "VALUES (:id, :trace_id, :conversation_id, :turn_no, :index_version_id, "
            f"'질문', 'NEW_TOPIC', {snapshot_sql}, 'COMPLETED')",
            {
                "id": run_id,
                "trace_id": uuid.uuid4(),
                "turn_no": turn_no,
                "conversation_id": seed["conversation_id"],
                "index_version_id": seed["index_version_id"],
            },
        )
        return run_id

    async def _snapshot_kinds(self) -> dict:
        rows = await self._rows(
            "SELECT id, CASE WHEN context_snapshot IS NULL THEN 'SQL NULL' "
            "WHEN jsonb_typeof(context_snapshot) = 'null' THEN 'JSON null' "
            "ELSE jsonb_typeof(context_snapshot) END FROM rag_runs"
        )
        return {row[0]: row[1] for row in rows}

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

    async def _rows(self, statement: str, parameters=None):
        async with self.engine.connect() as connection:
            result = await connection.execute(text(statement), parameters or {})
            return result.fetchall()

    async def _scalar(self, statement: str, parameters=None):
        async with self.engine.connect() as connection:
            return (await connection.execute(text(statement), parameters or {})).scalar_one()

    async def _column(self, table: str, column: str) -> bool:
        return bool(await self._scalar(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = :table AND column_name = :column",
            {"table": table, "column": column},
        ))


if __name__ == "__main__":
    unittest.main()
