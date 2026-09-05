"""기동 시 중단된 실행 정리를 실제 DB로 검증한다."""

import asyncio
import unittest
import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.database.models import (
    DocumentSource,
    ExecutionStatus,
    IndexOperationType,
    IndexRun,
    IndexRunStage,
    IndexVersion,
    IndexVersionStatus,
    IngestionRun,
    IngestionStage,
)
from app.document.chunking_config import get_or_create_chunking_config
from app.document.document_group import get_document_group
from app.document.document_key import SOURCE_TYPE_GITBOOK
from app.document.job_gate import find_processing_job
from app.document.stale_runs import close_interrupted_runs
from app.retrieval.embedding_config import get_or_create_embedding_config


async def _check_database_available(url: str) -> bool:
    engine = create_async_engine(url)
    try:
        async with engine.connect():
            return True
    except Exception:
        return False
    finally:
        await engine.dispose()


class StaleRunsDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_check_database_available(cls.database_url)):
            raise unittest.SkipTest(
                "로컬 DB에 연결할 수 없어 중단 실행 정리 테스트를 건너뜁니다."
            )

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(self.database_url)
        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )
        self.suffix = uuid.uuid4().hex[:8]
        self.source_id = None
        self.index_version_ids = []
        async with self.session_factory() as session:
            self.group_id = (await get_document_group(session)).id

    async def asyncTearDown(self) -> None:
        async with self.session_factory() as session:
            if self.source_id is not None:
                await session.execute(
                    delete(IngestionRun).where(
                        IngestionRun.document_source_id == self.source_id
                    )
                )
                source = await session.get(DocumentSource, self.source_id)
                if source is not None:
                    await session.delete(source)
            for index_version_id in self.index_version_ids:
                await session.execute(
                    delete(IndexVersion).where(
                        IndexVersion.id == index_version_id
                    )
                )
            await session.commit()
        await self.engine.dispose()

    async def _make_processing_ingestion(self) -> int:
        now = datetime.now(timezone.utc)
        async with self.session_factory() as session:
            source = DocumentSource(
                document_group_id=self.group_id,
                document_key=f"stale-test-{self.suffix}",
                source_type=SOURCE_TYPE_GITBOOK,
                canonical_uri=f"https://docs.riido.io/stale-{self.suffix}.md",
                title=f"중단 테스트 {self.suffix}",
                metadata_={"document_id": f"stale-{self.suffix}"},
                enabled=True,
                created_at=now,
                updated_at=now,
            )
            session.add(source)
            await session.flush()
            self.source_id = source.id

            run = IngestionRun(
                document_source_id=source.id,
                trigger_type="RECOLLECT",
                parser_name="gitbook-markdown",
                parser_version="1",
                status=ExecutionStatus.PROCESSING,
                stage=IngestionStage.EMBEDDING,
                started_at=now,
            )
            session.add(run)
            await session.flush()
            await session.commit()
            return run.id

    async def _make_processing_index(self, status: IndexVersionStatus) -> int:
        now = datetime.now(timezone.utc)
        async with self.session_factory() as session:
            chunking = await get_or_create_chunking_config(session, now)
            embedding = await get_or_create_embedding_config(session, now)
            index_version = IndexVersion(
                document_group_id=self.group_id,
                version=f"stale-{self.suffix}-{uuid.uuid4().hex[:6]}",
                status=status,
                chunking_config_id=chunking.id,
                embedding_config_id=embedding.id,
                created_at=now,
            )
            session.add(index_version)
            await session.flush()
            self.index_version_ids.append(index_version.id)

            run = IndexRun(
                index_version_id=index_version.id,
                trigger_type="MANUAL",
                operation_type=IndexOperationType.BUILD_AND_APPLY,
                stage=IndexRunStage.BUILDING,
                status=ExecutionStatus.PROCESSING,
                started_at=now,
            )
            session.add(run)
            await session.flush()
            await session.commit()
            return run.id

    async def test_interrupted_ingestion_no_longer_blocks_the_group(self) -> None:
        run_id = await self._make_processing_ingestion()

        async with self.session_factory() as session:
            blocked = await find_processing_job(session, self.group_id)
        self.assertEqual("INGESTION", blocked)

        async with self.session_factory() as session:
            counts = await close_interrupted_runs(session)

        self.assertEqual((1, 0), counts)
        async with self.session_factory() as session:
            run = await session.get(IngestionRun, run_id)
            # 정리 뒤에는 그룹이 다시 열린다
            self.assertIsNone(await find_processing_job(session, self.group_id))

        self.assertEqual(ExecutionStatus.FAILED, run.status)
        self.assertEqual("INTERNAL_ERROR", run.error_code)
        self.assertIsNotNone(run.finished_at)
        # 실패 지점은 끊긴 단계 그대로 남는다
        self.assertEqual(IngestionStage.EMBEDDING, run.stage)

    async def test_building_candidate_is_marked_failed(self) -> None:
        run_id = await self._make_processing_index(IndexVersionStatus.BUILDING)

        async with self.session_factory() as session:
            await close_interrupted_runs(session)

        async with self.session_factory() as session:
            run = await session.get(IndexRun, run_id)
            index_version = await session.get(IndexVersion, run.index_version_id)

        self.assertEqual(ExecutionStatus.FAILED, run.status)
        self.assertEqual(IndexVersionStatus.FAILED, index_version.status)

    async def test_ready_candidate_survives_for_retry(self) -> None:
        run_id = await self._make_processing_index(IndexVersionStatus.READY)

        async with self.session_factory() as session:
            await close_interrupted_runs(session)

        async with self.session_factory() as session:
            run = await session.get(IndexRun, run_id)
            index_version = await session.get(IndexVersion, run.index_version_id)

        self.assertEqual(ExecutionStatus.FAILED, run.status)
        # 적용 단계에서 끊긴 후보는 살려 두어야 재시도할 수 있다
        self.assertEqual(IndexVersionStatus.READY, index_version.status)

    async def test_nothing_to_close_is_a_no_op(self) -> None:
        async with self.session_factory() as session:
            self.assertEqual((0, 0), await close_interrupted_runs(session))


if __name__ == "__main__":
    unittest.main()
