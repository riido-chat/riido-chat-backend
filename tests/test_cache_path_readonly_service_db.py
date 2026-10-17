"""Prove PostgreSQL rejects writes inside cache-path service snapshots.

Runs only against the explicitly configured loopback CI PostgreSQL service. It
creates and drops a uniquely named probe table and never reads application .env
or general DATABASE_URL settings.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from evaluation.cache_path_eval.readonly_service import (
    EmbeddingConfigIdentity,
    ReadOnlyRiidoServiceData,
    SelectedServiceSettings,
)


_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class _ObservedSession(AsyncSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.was_closed = False

    async def close(self) -> None:
        self.was_closed = True
        await super().close()


class ReadOnlyRiidoServicePostgresTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("APP_ENV") != "test":
            raise unittest.SkipTest("APP_ENV=test is required for the PostgreSQL guard test.")
        url_value = os.environ.get("PROFILE_LIFECYCLE_TEST_DATABASE_URL")
        if not url_value:
            raise unittest.SkipTest(
                "PROFILE_LIFECYCLE_TEST_DATABASE_URL must explicitly select the CI test database."
            )
        cls.url = make_url(url_value)
        if cls.url.host not in _LOOPBACK_HOSTS:
            raise unittest.SkipTest("The configured PostgreSQL test URL must use a loopback host.")
        if not cls.url.drivername.startswith("postgresql+"):
            raise unittest.SkipTest("The configured test URL must use PostgreSQL.")

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(self.url)
        self.sessions = async_sessionmaker(
            self.engine,
            class_=_ObservedSession,
            expire_on_commit=False,
            autoflush=True,
        )
        self.table_name = "cache_path_readonly_probe_" + uuid.uuid4().hex
        async with self.engine.begin() as connection:
            await connection.execute(
                text(f'CREATE TABLE "{self.table_name}" (value integer NOT NULL)')
            )

    async def asyncTearDown(self) -> None:
        try:
            async with self.engine.begin() as connection:
                await connection.execute(text(f'DROP TABLE IF EXISTS "{self.table_name}"'))
        finally:
            await self.engine.dispose()

    async def test_database_rejects_insert_and_snapshot_rolls_back_and_closes(self) -> None:
        sessions: list[_ObservedSession] = []

        def session_factory() -> _ObservedSession:
            session = self.sessions()
            sessions.append(session)
            return session

        data = ReadOnlyRiidoServiceData(session_factory)
        selection = SelectedServiceSettings(
            service_key="SYNTHETIC_TEST",
            profile_revision_id=1,
            document_group_id=1,
            index_version_id=1,
            embedding_config_id=1,
            embedding_config=EmbeddingConfigIdentity(
                id=1,
                version="probe",
                provider="probe",
                model_name="probe",
                dimensions=1,
                input_template_version="probe",
            ),
            exact_cache_enabled=False,
            semantic_cache_enabled=False,
            stage_settings=SimpleNamespace(),
        )
        with patch.object(
            data,
            "_select_settings",
            new=AsyncMock(return_value=(selection, object())),
        ):
            async with data.open_snapshot(SimpleNamespace()) as snapshot:
                self.assertIs(selection, snapshot.selection)
                with self.assertRaises(DBAPIError) as raised:
                    await sessions[0].execute(
                        text(f'INSERT INTO "{self.table_name}" (value) VALUES (7)')
                    )
                original = raised.exception.orig
                sqlstate = getattr(original, "sqlstate", None) or getattr(
                    original, "pgcode", None
                )
                self.assertEqual("25006", sqlstate)

        self.assertEqual(1, len(sessions))
        self.assertFalse(sessions[0].in_transaction())
        self.assertTrue(sessions[0].was_closed)
        async with self.engine.connect() as connection:
            count = await connection.scalar(
                text(f'SELECT count(*) FROM "{self.table_name}"')
            )
        self.assertEqual(0, count)


if __name__ == "__main__":
    unittest.main()
