"""프로필 판 구성(20261001_17)과 실행 출처 칸(20261001_18) migration 의 upgrade, 백필, downgrade 통합 테스트.

로컬 PostgreSQL 에 연결할 수 없으면 skip 한다. 일회용 DB 를 만들고 지운다.
"""

import asyncio
import json
import os
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from app.core.prompt_fingerprints import load_fingerprint_table


REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_REVISION = "20260918_16"
COMPONENTS_REVISION = "20261001_17"
PROVENANCE_REVISION = "20261001_18"
SUB_STAGES = ("source_planning", "source_planning_repair", "answer", "answer_repair")
NEW_COLUMNS = (
    ("rag_runs", "profile_revision_id"),
    ("rag_runs", "build_version"),
    ("question_classifications", "presented_canonical_answer_id"),
    ("canonical_answers", "generation_prompt_version"),
    ("canonical_answers", "generation_prompt_sha256"),
    ("canonical_answers", "generation_model_name"),
)
NEW_FOREIGN_KEYS = (
    ("rag_runs", "fk_rag_runs_profile_revision_id_chat_profile_revisions"),
    (
        "question_classifications",
        "fk_question_classifications_presented_canonical_answer_id",
    ),
    ("profile_revision_components", "fk_profile_revision_components_revision_id"),
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


class ProfileRevisionProvenanceMigrationDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_available(cls.database_url)):
            raise unittest.SkipTest("로컬 PostgreSQL에 연결할 수 없습니다.")

    async def asyncSetUp(self) -> None:
        self.database_name = f"riido_provenance_{uuid.uuid4().hex[:12]}"
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
        _alembic(self.scratch_url, "upgrade", PROVENANCE_REVISION)
        self.assertTrue(await self._table("profile_revision_components"))
        for table, column in NEW_COLUMNS:
            with self.subTest(table=table, column=column):
                self.assertTrue(await self._column(table, column))
        for table, name in NEW_FOREIGN_KEYS:
            with self.subTest(constraint=name):
                self.assertTrue(await self._constraint(table, name))

        _alembic(self.scratch_url, "downgrade", COMPONENTS_REVISION)
        for table, column in NEW_COLUMNS:
            with self.subTest(table=table, column=column):
                self.assertFalse(await self._column(table, column))
        self.assertTrue(await self._table("profile_revision_components"))

        _alembic(self.scratch_url, "downgrade", PARENT_REVISION)
        self.assertFalse(await self._table("profile_revision_components"))
        self.assertEqual(
            PARENT_REVISION, await self._scalar("SELECT version_num FROM alembic_version")
        )

    async def test_backfill_writes_only_known_components(self) -> None:
        _alembic(self.scratch_url, "upgrade", PARENT_REVISION)
        profile_id = await self._scalar(
            "SELECT id FROM chat_profiles WHERE profile_key = 'HELP_CHATBOT'"
        )
        group_id = await self._scalar(
            "SELECT id FROM document_groups WHERE group_key = 'HELP_CHATBOT'"
        )
        async with self.engine.begin() as connection:
            # DEV 처럼 PUBLISHED 판을 현재 코드 버전으로 제자리 갱신한 상태를 만든다.
            await connection.execute(
                text(
                    "UPDATE chat_profile_revisions "
                    "SET generation_prompt_version = 'v40', "
                    "    query_rewrite_prompt_version = 'v10' "
                    "WHERE profile_id = :profile_id AND status = 'PUBLISHED'"
                ),
                {"profile_id": profile_id},
            )
            for version, status, generation, rewrite in (
                (2, "TESTING", "v40", "v8"),
                (3, "DRAFT", "v40", "v10"),
                (4, "RETIRED", "v38", "v8"),
            ):
                await connection.execute(
                    text(
                        "INSERT INTO chat_profile_revisions "
                        "(profile_id, version, document_group_id, status, "
                        "generation_model_name, generation_prompt_version, "
                        "query_rewrite_model_name, query_rewrite_prompt_version) "
                        "VALUES (:profile_id, :version, :group_id, :status, "
                        "'gpt-5.6-terra', :generation, 'gpt-5.4-mini', :rewrite)"
                    ),
                    {
                        "profile_id": profile_id,
                        "version": version,
                        "group_id": group_id,
                        "status": status,
                        "generation": generation,
                        "rewrite": rewrite,
                    },
                )
        before = await self._rows(
            "SELECT id, status, generation_model_name, generation_prompt_version, "
            "query_rewrite_model_name, query_rewrite_prompt_version "
            "FROM chat_profile_revisions ORDER BY id"
        )

        _alembic(self.scratch_url, "upgrade", COMPONENTS_REVISION)

        # 기존 판 칸은 그대로다(확장만).
        self.assertEqual(
            before,
            await self._rows(
                "SELECT id, status, generation_model_name, generation_prompt_version, "
                "query_rewrite_model_name, query_rewrite_prompt_version "
                "FROM chat_profile_revisions ORDER BY id"
            ),
        )
        table = load_fingerprint_table()
        components = {
            (row[0], row[1]): row[2:]
            for row in await self._rows(
                "SELECT r.status, c.stage, c.model_name, c.prompt_key, c.prompt_version, "
                "c.prompt_sha256, c.params, c.recorded_by "
                "FROM profile_revision_components c "
                "JOIN chat_profile_revisions r ON r.id = c.revision_id"
            )
        }
        self.assertEqual({"BACKFILL"}, {value[-1] for value in components.values()})

        expected_stages = {
            "PUBLISHED": {"rewrite", "generation", "judge", *SUB_STAGES},
            "TESTING": {"rewrite", "generation", "judge", *SUB_STAGES},
            "DRAFT": {"rewrite", "generation", *SUB_STAGES},
            "RETIRED": {"rewrite", "generation"},
        }
        for status, stages in expected_stages.items():
            with self.subTest(status=status):
                self.assertEqual(
                    stages, {stage for (row_status, stage) in components if row_status == status}
                )

        published_rewrite = components[("PUBLISHED", "rewrite")]
        self.assertEqual(
            ("gpt-5.4-mini", "rewrite@v10", "v10", table["rewrite@v10"], None),
            tuple(published_rewrite[:5]),
        )
        # 코드에 없는 옛 버전은 지문을 비운다.
        self.assertEqual(("rewrite@v8", "v8", None), tuple(components[("TESTING", "rewrite")][1:4]))
        self.assertEqual(("generation@v38", "v38", None), tuple(components[("RETIRED", "generation")][1:4]))
        self.assertEqual(table["generation@v40"], components[("DRAFT", "generation")][3])
        for stage in SUB_STAGES:
            with self.subTest(stage=stage):
                model_name, prompt_key, _, sha256, params, _ = components[("PUBLISHED", stage)]
                self.assertEqual("gpt-5.6-terra", model_name)
                self.assertEqual(table[prompt_key], sha256)
                self.assertIsNone(params)
        judge = components[("TESTING", "judge")]
        self.assertEqual(
            ("gpt-5.6-luna", "judge@question-grouping-v7-2", "question-grouping-v7-2"),
            tuple(judge[:3]),
        )
        self.assertEqual(table["judge@question-grouping-v7-2"], judge[3])
        params = judge[4] if isinstance(judge[4], dict) else json.loads(judge[4])
        self.assertEqual({"reasoning": {"effort": "low"}, "max_output_tokens": 1024}, params)

        revision_id = await self._scalar(
            "SELECT id FROM chat_profile_revisions WHERE status = 'PUBLISHED'"
        )
        # RETIRED 판은 rewrite, generation 행만 있어 나머지 단계는 유일 제약에 걸리지 않는다.
        retired_id = await self._scalar(
            "SELECT id FROM chat_profile_revisions WHERE status = 'RETIRED'"
        )
        rejected = (
            # 같은 판의 같은 단계는 한 행뿐이다.
            (revision_id, "rewrite", "rewrite@v10", "v10", None),
            # prompt_key 는 단계의 component 와 버전으로 정해진다.
            (retired_id, "answer_repair", "generation.answer@v23", "v23", None),
            # 정한 단계만 쓴다.
            (retired_id, "verifier", "verifier@v1", "v1", None),
            # 지문은 64자리 소문자 hex 다.
            (retired_id, "judge", "judge@v1", "v1", "A" * 64),
        )
        for target_id, stage, prompt_key, version, sha256 in rejected:
            with self.subTest(stage=stage, prompt_key=prompt_key):
                with self.assertRaises(IntegrityError):
                    async with self.engine.begin() as connection:
                        await connection.execute(
                            text(
                                "INSERT INTO profile_revision_components "
                                "(revision_id, stage, model_name, prompt_key, prompt_version, "
                                "prompt_sha256, recorded_by) "
                                "VALUES (:revision_id, :stage, 'm', :prompt_key, :version, "
                                ":sha256, 'PUBLISH')"
                            ),
                            {
                                "revision_id": target_id,
                                "stage": stage,
                                "prompt_key": prompt_key,
                                "version": version,
                                "sha256": sha256,
                            },
                        )
        # 지문은 64자리 소문자 hex 다.
        with self.assertRaises(IntegrityError):
            async with self.engine.begin() as connection:
                await connection.execute(
                    text(
                        "UPDATE profile_revision_components SET prompt_sha256 = 'ABC' "
                        "WHERE revision_id = :revision_id AND stage = 'rewrite'"
                    ),
                    {"revision_id": revision_id},
                )
        # 구성 행이 있는 판은 지울 수 없다.
        with self.assertRaises(IntegrityError):
            async with self.engine.begin() as connection:
                await connection.execute(
                    text("DELETE FROM chat_profile_revisions WHERE id = :revision_id"),
                    {"revision_id": revision_id},
                )

        _alembic(self.scratch_url, "upgrade", PROVENANCE_REVISION)
        # 실행 출처 칸은 백필하지 않는다.
        for table_name, column in NEW_COLUMNS:
            with self.subTest(table=table_name, column=column):
                self.assertEqual(
                    0,
                    await self._scalar(
                        f"SELECT count(*) FROM {table_name} WHERE {column} IS NOT NULL"
                    ),
                )
        self.assertTrue(
            await self._constraint(
                "canonical_answers", "ck_canonical_answers_generation_prompt_sha256"
            )
        )

        _alembic(self.scratch_url, "downgrade", PARENT_REVISION)
        self.assertFalse(await self._table("profile_revision_components"))
        self.assertEqual(
            before,
            await self._rows(
                "SELECT id, status, generation_model_name, generation_prompt_version, "
                "query_rewrite_model_name, query_rewrite_prompt_version "
                "FROM chat_profile_revisions ORDER BY id"
            ),
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

    async def _rows(self, statement: str, parameters=None):
        async with self.engine.connect() as connection:
            result = await connection.execute(text(statement), parameters or {})
            return result.fetchall()

    async def _scalar(self, statement: str, parameters=None):
        async with self.engine.connect() as connection:
            return (await connection.execute(text(statement), parameters or {})).scalar_one()

    async def _table(self, name: str) -> bool:
        return await self._scalar("SELECT to_regclass(:name)", {"name": name}) is not None

    async def _column(self, table: str, column: str) -> bool:
        return bool(await self._scalar(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = :table AND column_name = :column",
            {"table": table, "column": column},
        ))

    async def _constraint(self, table: str, name: str) -> bool:
        return bool(await self._scalar(
            "SELECT count(*) FROM information_schema.table_constraints "
            "WHERE table_name = :table AND constraint_name = :name",
            {"table": table, "name": name},
        ))


if __name__ == "__main__":
    unittest.main()
