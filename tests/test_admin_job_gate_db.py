"""문서 그룹 단위 작업 잠금이 실제로 동시 실행을 막는지 검증한다."""

import asyncio
import unittest
import uuid

import asyncpg
from sqlalchemy import delete, make_url, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.database.models import DocumentSource, DocumentVersion, IngestionRun
from app.database.session import dispose_engine
from app.document.document_group import get_document_group
from app.document.ingestion_service import (
    AdminIngestionService,
    AdminJobInProgressError,
)
from app.document.job_gate import ADMIN_JOB_LOCK_NAMESPACE, acquire_group_job_gate


async def _check_database_available(url: str) -> bool:
    engine = create_async_engine(url)
    try:
        async with engine.connect():
            return True
    except Exception:
        return False
    finally:
        await engine.dispose()


def _asyncpg_dsn(database_url: str) -> str:
    """asyncpg 가 직접 쓸 수 있는 DSN 으로 바꾼다."""

    url = make_url(database_url).set(drivername="postgresql")
    return url.render_as_string(hide_password=False)


class AdminJobGateDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_check_database_available(cls.database_url)):
            raise unittest.SkipTest(
                "로컬 DB에 연결할 수 없어 작업 잠금 통합 테스트를 건너뜁니다."
            )

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(self.database_url)
        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )
        self.titles = []
        async with self.session_factory() as session:
            self.group_id = (await get_document_group(session)).id

    async def asyncTearDown(self) -> None:
        async with self.session_factory() as session:
            for title in self.titles:
                source = await session.scalar(
                    select(DocumentSource).where(DocumentSource.title == title)
                )
                if source is None:
                    continue
                await session.execute(
                    delete(IngestionRun).where(
                        IngestionRun.document_source_id == source.id
                    )
                )
                await session.execute(
                    delete(DocumentVersion).where(
                        DocumentVersion.document_source_id == source.id
                    )
                )
                await session.delete(source)
            await session.commit()
        await self.engine.dispose()
        await dispose_engine()

    def _new_title(self) -> str:
        title = f"gate-test-{uuid.uuid4().hex[:10]}"
        self.titles.append(title)
        return title

    async def test_gate_blocks_the_same_group_and_frees_on_commit(self) -> None:
        observer = await asyncpg.connect(_asyncpg_dsn(self.database_url))
        other_group_id = self.group_id + 1000
        try:
            async with self.session_factory() as session:
                await acquire_group_job_gate(session, self.group_id)

                async with observer.transaction():
                    same_group = await observer.fetchval(
                        "select pg_try_advisory_xact_lock($1, $2)",
                        ADMIN_JOB_LOCK_NAMESPACE,
                        self.group_id,
                    )
                    other_group = await observer.fetchval(
                        "select pg_try_advisory_xact_lock($1, $2)",
                        ADMIN_JOB_LOCK_NAMESPACE,
                        other_group_id,
                    )

                # 같은 그룹은 막히고 다른 그룹은 막히지 않는다
                self.assertFalse(same_group)
                self.assertTrue(other_group)
                await session.commit()

            async with observer.transaction():
                after_commit = await observer.fetchval(
                    "select pg_try_advisory_xact_lock($1, $2)",
                    ADMIN_JOB_LOCK_NAMESPACE,
                    self.group_id,
                )
            # 트랜잭션이 끝나면 잠금도 함께 풀린다
            self.assertTrue(after_commit)
        finally:
            await observer.close()

    async def test_concurrent_accepts_leave_exactly_one_running_job(self) -> None:
        first_title = self._new_title()
        second_title = self._new_title()

        async def accept(title: str):
            async with self.session_factory() as session:
                return await AdminIngestionService(session).start_new_document(
                    group_id=self.group_id,
                    title=title,
                    category="test",
                    filename="guide.md",
                )

        results = await asyncio.gather(
            accept(first_title),
            accept(second_title),
            return_exceptions=True,
        )

        accepted = [r for r in results if not isinstance(r, Exception)]
        rejected = [r for r in results if isinstance(r, AdminJobInProgressError)]
        unexpected = [
            r
            for r in results
            if isinstance(r, Exception)
            and not isinstance(r, AdminJobInProgressError)
        ]

        self.assertEqual([], unexpected, f"예상하지 못한 예외: {unexpected}")
        # 잠금이 확인과 삽입 사이를 막아 한 건만 접수된다
        self.assertEqual(1, len(accepted))
        self.assertEqual(1, len(rejected))

        async with self.session_factory() as session:
            processing = await session.scalar(
                select(IngestionRun.id).where(
                    IngestionRun.id == accepted[0].ingestion_run_id
                )
            )
        self.assertIsNotNone(processing)


if __name__ == "__main__":
    unittest.main()
