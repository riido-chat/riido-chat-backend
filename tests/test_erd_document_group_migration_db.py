"""문서 그룹 migration(20260904_06~08)의 upgrade, downgrade, backfill 통합 테스트."""

import asyncio
import json
import os
import subprocess
import sys
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from app.document.document_key import (
    DEFAULT_DOCUMENT_GROUP_KEY,
    SOURCE_TYPE_GITBOOK,
    SOURCE_TYPE_UPLOAD,
    build_console_canonical_uri,
    build_upload_document_key,
)
from app.core.config import get_settings


REPO_ROOT = Path(__file__).resolve().parents[1]
REVISION_BEFORE_GROUPS = "20260831_05"
HEAD_REVISION = "20260904_08"

LEGACY_GITBOOK_SOURCE_TYPE = "GITBOOK_MARKDOWN"
LEGACY_UPLOAD_SOURCE_TYPE = "ADMIN_MARKDOWN"
UPLOAD_DOCUMENT_TITLE = "자주 묻는 질문"


async def _check_database_available(url: str) -> bool:
    engine = create_async_engine(url)
    try:
        async with engine.connect():
            return True
    except Exception:
        return False
    finally:
        await engine.dispose()


def _run_alembic(database_url: str, *args: str) -> None:
    """migration CLI를 별도 process로 실행해 scratch DB에 적용한다."""

    completed = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=str(REPO_ROOT),
        env={**os.environ, "DATABASE_URL": database_url},
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"alembic {' '.join(args)} 실패:\n{completed.stdout}\n{completed.stderr}"
        )


class ErdDocumentGroupMigrationDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_check_database_available(cls.database_url)):
            raise unittest.SkipTest(
                "로컬 DB에 연결할 수 없어 migration 통합 테스트를 건너뜁니다."
            )

    async def asyncSetUp(self) -> None:
        self.database_name = f"riido_migration_{uuid.uuid4().hex[:12]}"
        # str(URL)은 password를 가리므로 접속 가능한 문자열로 만든다.
        self.scratch_url = make_url(self.database_url).set(
            database=self.database_name
        ).render_as_string(hide_password=False)
        await self._execute_on_maintenance_database(
            f'CREATE DATABASE "{self.database_name}"'
        )
        self.engine = create_async_engine(self.scratch_url)

    async def asyncTearDown(self) -> None:
        await self.engine.dispose()
        await self._execute_on_maintenance_database(
            f'DROP DATABASE IF EXISTS "{self.database_name}" WITH (FORCE)'
        )

    async def test_empty_database_upgrades_and_downgrades(self) -> None:
        _run_alembic(self.scratch_url, "upgrade", "head")

        self.assertEqual(HEAD_REVISION, await self._current_revision())
        groups = await self._fetch_all(
            "SELECT group_key, name, consumer_key FROM document_groups"
        )
        self.assertEqual(1, len(groups))
        self.assertEqual(DEFAULT_DOCUMENT_GROUP_KEY, groups[0][0])

        _run_alembic(self.scratch_url, "downgrade", REVISION_BEFORE_GROUPS)

        self.assertEqual(REVISION_BEFORE_GROUPS, await self._current_revision())
        self.assertFalse(await self._table_exists("document_groups"))
        self.assertFalse(
            await self._column_exists("document_sources", "document_group_id")
        )

        _run_alembic(self.scratch_url, "upgrade", "head")
        self.assertEqual(HEAD_REVISION, await self._current_revision())

    async def test_existing_rows_are_backfilled_and_restored(self) -> None:
        _run_alembic(self.scratch_url, "upgrade", REVISION_BEFORE_GROUPS)
        await self._insert_legacy_rows()

        _run_alembic(self.scratch_url, "upgrade", "head")

        group_id = await self._fetch_one(
            "SELECT id FROM document_groups WHERE group_key = :group_key",
            {"group_key": DEFAULT_DOCUMENT_GROUP_KEY},
        )
        self.assertIsNotNone(group_id)

        gitbook = await self._fetch_row(
            "SELECT document_group_id, document_key, source_type, canonical_uri"
            " FROM document_sources WHERE title = :title",
            {"title": "자동화"},
        )
        self.assertEqual(group_id, gitbook[0])
        # GitBook 문서 키는 페이지 URL 경로다.
        self.assertEqual("meetings/automations", gitbook[1])
        self.assertEqual(SOURCE_TYPE_GITBOOK, gitbook[2])
        self.assertEqual("https://docs.riido.io/meetings/automations.md", gitbook[3])

        upload = await self._fetch_row(
            "SELECT document_group_id, document_key, source_type, canonical_uri"
            " FROM document_sources WHERE title = :title",
            {"title": UPLOAD_DOCUMENT_TITLE},
        )
        document_key = build_upload_document_key(UPLOAD_DOCUMENT_TITLE)
        self.assertEqual(group_id, upload[0])
        self.assertEqual(document_key, upload[1])
        self.assertEqual(SOURCE_TYPE_UPLOAD, upload[2])
        self.assertEqual(
            build_console_canonical_uri(DEFAULT_DOCUMENT_GROUP_KEY, document_key),
            upload[3],
        )

        # 적용된 적이 있는 검색 버전에만 생성 순서대로 번호가 붙는다.
        index_versions = await self._fetch_all(
            "SELECT version, version_no, document_group_id FROM index_versions"
            " ORDER BY version"
        )
        numbers = {row[0]: row[1] for row in index_versions}
        self.assertEqual(1, numbers["idx-first"])
        self.assertEqual(2, numbers["idx-second"])
        self.assertIsNone(numbers["idx-building"])
        self.assertTrue(all(row[2] == group_id for row in index_versions))

        index_runs = await self._fetch_all(
            "SELECT trigger_type, stage, operation_type FROM index_runs"
            " ORDER BY trigger_type"
        )
        self.assertEqual(
            [
                ("ACTIVATED", "APPLYING", "BUILD_AND_APPLY"),
                ("FAILED_EMBEDDING", "BUILDING", "BUILD_AND_APPLY"),
                ("FAILED_VALIDATING", "VALIDATING", "BUILD_AND_APPLY"),
            ],
            index_runs,
        )

        ingestion_runs = await self._fetch_all(
            "SELECT trigger_type, result_code, stage, error_code FROM ingestion_runs"
            " ORDER BY trigger_type"
        )
        self.assertEqual(
            [
                ("FIRST_VERSION", "CREATED", "PERSISTING", None),
                ("SECOND_VERSION", "UPDATED", "PERSISTING", None),
                ("UPLOAD_FAILED", None, "RECEIVING", "INVALID_FILE"),
            ],
            ingestion_runs,
        )

        _run_alembic(self.scratch_url, "downgrade", REVISION_BEFORE_GROUPS)

        source_types = await self._fetch_all(
            "SELECT DISTINCT source_type FROM document_sources"
        )
        self.assertEqual(
            {LEGACY_GITBOOK_SOURCE_TYPE, LEGACY_UPLOAD_SOURCE_TYPE},
            {row[0] for row in source_types},
        )
        self.assertFalse(await self._table_exists("document_groups"))
        self.assertFalse(await self._column_exists("index_runs", "stage"))
        self.assertFalse(await self._column_exists("ingestion_runs", "result_code"))

    # ------------------------------------------------------------------

    async def _insert_legacy_rows(self) -> None:
        async with self.engine.begin() as connection:
            gitbook_id = (
                await connection.execute(
                    text(
                        "INSERT INTO document_sources"
                        " (source_type, canonical_uri, title, metadata, enabled)"
                        " VALUES (:source_type, :canonical_uri, :title,"
                        " CAST(:metadata AS jsonb), true) RETURNING id"
                    ),
                    {
                        "source_type": LEGACY_GITBOOK_SOURCE_TYPE,
                        "canonical_uri": (
                            "https://docs.riido.io/meetings/automations.md"
                        ),
                        "title": "자동화",
                        "metadata": json.dumps({"document_id": "gitbook-1"}),
                    },
                )
            ).scalar_one()
            await connection.execute(
                text(
                    "INSERT INTO document_sources"
                    " (source_type, canonical_uri, title, metadata, enabled)"
                    " VALUES (:source_type, :canonical_uri, :title,"
                    " CAST(:metadata AS jsonb), true)"
                ),
                {
                    "source_type": LEGACY_UPLOAD_SOURCE_TYPE,
                    "canonical_uri": "https://console.example.com/faq",
                    "title": UPLOAD_DOCUMENT_TITLE,
                    "metadata": json.dumps({"document_id": "admin-1"}),
                },
            )

            version_ids = []
            for version_no in (1, 2):
                version_ids.append(
                    (
                        await connection.execute(
                            text(
                                "INSERT INTO document_versions"
                                " (document_source_id, version_no, raw_content_uri,"
                                " mime_type, raw_content_hash,"
                                " normalized_content_hash, parser_name,"
                                " parser_version, status, collected_at)"
                                " VALUES (:source_id, :version_no, :uri,"
                                " 'text/markdown', :raw_hash, :norm_hash,"
                                " 'gitbook-markdown', '1', 'READY', now())"
                                " RETURNING id"
                            ),
                            {
                                "source_id": gitbook_id,
                                "version_no": version_no,
                                "uri": f"raw/meetings/automations-v{version_no}.md",
                                "raw_hash": f"raw-{version_no}",
                                "norm_hash": f"norm-{version_no}",
                            },
                        )
                    ).scalar_one()
                )

            for trigger_type, version_id in (
                ("FIRST_VERSION", version_ids[0]),
                ("SECOND_VERSION", version_ids[1]),
            ):
                await connection.execute(
                    text(
                        "INSERT INTO ingestion_runs"
                        " (document_source_id, produced_version_id, trigger_type,"
                        " parser_name, parser_version, status, summary, started_at)"
                        " VALUES (:source_id, :version_id, :trigger_type,"
                        " 'gitbook-markdown', '1', 'SUCCESS',"
                        " CAST(:summary AS jsonb), now())"
                    ),
                    {
                        "source_id": gitbook_id,
                        "version_id": version_id,
                        "trigger_type": trigger_type,
                        "summary": json.dumps({"stage": "COMPLETED"}),
                    },
                )
            await connection.execute(
                text(
                    "INSERT INTO ingestion_runs"
                    " (document_source_id, trigger_type, parser_name,"
                    " parser_version, status, summary, started_at)"
                    " VALUES (:source_id, 'UPLOAD_FAILED', 'gitbook-markdown', '1',"
                    " 'FAILED', CAST(:summary AS jsonb), now())"
                ),
                {
                    "source_id": gitbook_id,
                    "summary": json.dumps(
                        {"failed_stage": "LOADING", "error_code": "INVALID_FILE"}
                    ),
                },
            )

            chunking_id = (
                await connection.execute(
                    text(
                        "INSERT INTO chunking_configs (version, strategy, max_tokens)"
                        " VALUES ('migration-chunking', 'SECTION', 512) RETURNING id"
                    )
                )
            ).scalar_one()
            embedding_id = (
                await connection.execute(
                    text(
                        "INSERT INTO embedding_configs (version, provider,"
                        " model_name, dimensions, input_template_version)"
                        " VALUES ('migration-embedding', 'openai', 'test', 1536, 'v1')"
                        " RETURNING id"
                    )
                )
            ).scalar_one()

            index_version_ids = {}
            for version, status, created_at in (
                ("idx-first", "INACTIVE", datetime(2026, 9, 1, tzinfo=timezone.utc)),
                ("idx-second", "ACTIVE", datetime(2026, 9, 2, tzinfo=timezone.utc)),
                ("idx-building", "BUILDING", datetime(2026, 9, 3, tzinfo=timezone.utc)),
            ):
                index_version_ids[version] = (
                    await connection.execute(
                        text(
                            "INSERT INTO index_versions (version, status,"
                            " chunking_config_id, embedding_config_id, created_at)"
                            " VALUES (:version, :status, :chunking_id, :embedding_id,"
                            " :created_at) RETURNING id"
                        ),
                        {
                            "version": version,
                            "status": status,
                            "chunking_id": chunking_id,
                            "embedding_id": embedding_id,
                            "created_at": created_at,
                        },
                    )
                ).scalar_one()

            for trigger_type, version, summary in (
                ("ACTIVATED", "idx-second", {"stage": "ACTIVE"}),
                (
                    "FAILED_EMBEDDING",
                    "idx-building",
                    {"stage": "FAILED", "failed_stage": "EMBEDDING"},
                ),
                (
                    "FAILED_VALIDATING",
                    "idx-building",
                    {"stage": "FAILED", "failed_stage": "VALIDATING"},
                ),
            ):
                await connection.execute(
                    text(
                        "INSERT INTO index_runs (index_version_id, trigger_type,"
                        " status, summary, started_at)"
                        " VALUES (:index_version_id, :trigger_type, 'SUCCESS',"
                        " CAST(:summary AS jsonb), now())"
                    ),
                    {
                        "index_version_id": index_version_ids[version],
                        "trigger_type": trigger_type,
                        "summary": json.dumps(summary),
                    },
                )

    async def _execute_on_maintenance_database(self, statement: str) -> None:
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

    async def _current_revision(self):
        return await self._fetch_one("SELECT version_num FROM alembic_version")

    async def _table_exists(self, table_name: str) -> bool:
        found = await self._fetch_one(
            "SELECT to_regclass(:table_name)", {"table_name": table_name}
        )
        return found is not None

    async def _column_exists(self, table_name: str, column_name: str) -> bool:
        found = await self._fetch_one(
            "SELECT count(*) FROM information_schema.columns"
            " WHERE table_name = :table_name AND column_name = :column_name",
            {"table_name": table_name, "column_name": column_name},
        )
        return bool(found)

    async def _fetch_one(self, statement: str, parameters=None):
        async with self.engine.connect() as connection:
            result = await connection.execute(text(statement), parameters or {})
            return result.scalar_one()

    async def _fetch_row(self, statement: str, parameters=None):
        async with self.engine.connect() as connection:
            result = await connection.execute(text(statement), parameters or {})
            return result.one()

    async def _fetch_all(self, statement: str, parameters=None):
        async with self.engine.connect() as connection:
            result = await connection.execute(text(statement), parameters or {})
            return [tuple(row) for row in result.all()]


if __name__ == "__main__":
    unittest.main()
