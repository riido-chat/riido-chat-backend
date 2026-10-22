"""문서 그룹 도입으로 추가된 제약과 상태 값의 로컬 DB 통합 테스트."""

import asyncio
import unittest
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.config import get_settings
from app.database.models import (
    ChunkingConfig,
    DocumentGroup,
    DocumentSource,
    EmbeddingConfig,
    ExecutionStatus,
    IndexOperationType,
    IndexRun,
    IndexRunStage,
    IndexVersion,
    IndexVersionStatus,
    IngestionResultCode,
    IngestionRun,
    IngestionStage,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _check_database_available(url: str) -> bool:
    engine = create_async_engine(url)
    try:
        async with engine.connect():
            return True
    except Exception:
        return False
    finally:
        await engine.dispose()


class ErdDocumentGroupSchemaDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_check_database_available(cls.database_url)):
            raise unittest.SkipTest(
                "로컬 DB에 연결할 수 없어 문서 그룹 스키마 통합 테스트를 건너뜁니다."
            )

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(self.database_url)
        self.connection = await self.engine.connect()
        self.transaction = await self.connection.begin()
        self.session = AsyncSession(bind=self.connection, expire_on_commit=False)

        self.suffix = uuid.uuid4().hex
        self.group = await self._create_group("A")
        self.chunking = ChunkingConfig(
            version=f"schema-chunking-{self.suffix}",
            strategy="SECTION",
            max_tokens=512,
            created_at=_now(),
        )
        self.embedding = EmbeddingConfig(
            version=f"schema-embedding-{self.suffix}",
            provider="openai",
            model_name="text-embedding-test",
            dimensions=1536,
            input_template_version="v1",
            created_at=_now(),
        )
        self.session.add_all([self.chunking, self.embedding])
        await self.session.flush()

    async def asyncTearDown(self) -> None:
        await self.session.close()
        await self.transaction.rollback()
        await self.connection.close()
        await self.engine.dispose()

    # ------------------------------------------------------------------
    # document_sources

    async def test_document_key_is_unique_within_a_group(self) -> None:
        await self._create_source(self.group, "upload/중복-키", "riido-doc://a/1")

        with self.assertRaises(IntegrityError):
            async with self.session.begin_nested():
                await self._create_source(
                    self.group,
                    "upload/중복-키",
                    "riido-doc://a/2",
                )

    async def test_same_document_key_is_allowed_in_another_group(self) -> None:
        other = await self._create_group("B")

        await self._create_source(self.group, "upload/같은-키", "riido-doc://a/1")
        source = await self._create_source(
            other,
            "upload/같은-키",
            "riido-doc://b/1",
        )

        self.assertIsNotNone(source.id)

    async def test_canonical_uri_is_unique_within_a_group(self) -> None:
        await self._create_source(self.group, "upload/키-1", "riido-doc://a/1")

        with self.assertRaises(IntegrityError):
            async with self.session.begin_nested():
                await self._create_source(
                    self.group,
                    "upload/키-2",
                    "riido-doc://a/1",
                )

    # ------------------------------------------------------------------
    # index_versions

    async def test_group_allows_only_one_active_index_version(self) -> None:
        await self._create_index_version(
            self.group,
            "active-1",
            IndexVersionStatus.ACTIVE,
        )

        with self.assertRaises(IntegrityError):
            async with self.session.begin_nested():
                await self._create_index_version(
                    self.group,
                    "active-2",
                    IndexVersionStatus.ACTIVE,
                )

    async def test_another_group_can_have_its_own_active_index_version(self) -> None:
        other = await self._create_group("B")

        await self._create_index_version(
            self.group,
            "active-1",
            IndexVersionStatus.ACTIVE,
        )
        index_version = await self._create_index_version(
            other,
            "active-2",
            IndexVersionStatus.ACTIVE,
        )

        self.assertIsNotNone(index_version.id)

    async def test_index_version_accepts_ready_status(self) -> None:
        index_version = await self._create_index_version(
            self.group,
            "ready-1",
            IndexVersionStatus.READY,
        )

        self.assertEqual(IndexVersionStatus.READY, index_version.status)

    async def test_index_version_rejects_unknown_status(self) -> None:
        with self.assertRaises(IntegrityError):
            async with self.session.begin_nested():
                await self._insert_index_version_status("PUBLISHED")

    async def test_version_no_is_unique_within_a_group_when_present(self) -> None:
        await self._create_index_version(
            self.group,
            "numbered-1",
            IndexVersionStatus.INACTIVE,
            version_no=1,
        )

        with self.assertRaises(IntegrityError):
            async with self.session.begin_nested():
                await self._create_index_version(
                    self.group,
                    "numbered-2",
                    IndexVersionStatus.INACTIVE,
                    version_no=1,
                )

    async def test_multiple_index_versions_can_have_no_version_no(self) -> None:
        await self._create_index_version(
            self.group,
            "unnumbered-1",
            IndexVersionStatus.BUILDING,
        )
        index_version = await self._create_index_version(
            self.group,
            "unnumbered-2",
            IndexVersionStatus.BUILDING,
        )

        self.assertIsNone(index_version.version_no)

    # ------------------------------------------------------------------
    # index_runs

    async def test_index_run_accepts_every_operation_type(self) -> None:
        index_version = await self._create_index_version(
            self.group,
            "operations",
            IndexVersionStatus.BUILDING,
        )

        for operation_type in IndexOperationType:
            run = IndexRun(
                index_version_id=index_version.id,
                trigger_type="MANUAL",
                operation_type=operation_type,
                stage=IndexRunStage.BUILDING,
                status=ExecutionStatus.PROCESSING,
                started_at=_now(),
            )
            self.session.add(run)
            await self.session.flush()
            self.assertIsNotNone(run.id)

    async def test_index_run_rejects_unknown_operation_type(self) -> None:
        index_version = await self._create_index_version(
            self.group,
            "bad-operation",
            IndexVersionStatus.BUILDING,
        )

        with self.assertRaises(DBAPIError):
            async with self.session.begin_nested():
                await self.session.execute(
                    IndexRun.__table__.insert().values(
                        index_version_id=index_version.id,
                        trigger_type="MANUAL",
                        operation_type="REBUILD",
                        stage="BUILDING",
                        status="PROCESSING",
                        started_at=_now(),
                    )
                )

    async def test_index_run_rejects_unknown_stage(self) -> None:
        index_version = await self._create_index_version(
            self.group,
            "bad-stage",
            IndexVersionStatus.BUILDING,
        )

        with self.assertRaises(DBAPIError):
            async with self.session.begin_nested():
                await self.session.execute(
                    IndexRun.__table__.insert().values(
                        index_version_id=index_version.id,
                        trigger_type="MANUAL",
                        operation_type="BUILD_AND_APPLY",
                        stage="ACTIVATING",
                        status="PROCESSING",
                        started_at=_now(),
                    )
                )

    # ------------------------------------------------------------------
    # ingestion_runs

    async def test_ingestion_run_accepts_result_code_and_stage(self) -> None:
        source = await self._create_source(
            self.group,
            "upload/결과-코드",
            "riido-doc://a/result",
        )

        run = IngestionRun(
            document_source_id=source.id,
            trigger_type="ADMIN_UPLOAD",
            parser_name="gitbook-markdown",
            parser_version="1",
            status=ExecutionStatus.SUCCESS,
            result_code=IngestionResultCode.DUPLICATE_CONTENT,
            stage=IngestionStage.PERSISTING,
            error_code=None,
            batch_id=uuid.uuid4(),
            duplicate_of_document_source_id=source.id,
            started_at=_now(),
        )
        self.session.add(run)
        await self.session.flush()

        self.assertIsNotNone(run.id)

    async def test_ingestion_run_rejects_unknown_result_code(self) -> None:
        source = await self._create_source(
            self.group,
            "upload/잘못된-결과",
            "riido-doc://a/bad-result",
        )

        with self.assertRaises(DBAPIError):
            async with self.session.begin_nested():
                await self.session.execute(
                    IngestionRun.__table__.insert().values(
                        document_source_id=source.id,
                        trigger_type="ADMIN_UPLOAD",
                        parser_name="gitbook-markdown",
                        parser_version="1",
                        status="SUCCESS",
                        result_code="REPLACED",
                        started_at=_now(),
                    )
                )

    async def test_ingestion_run_rejects_unknown_stage(self) -> None:
        source = await self._create_source(
            self.group,
            "upload/잘못된-단계",
            "riido-doc://a/bad-stage",
        )

        with self.assertRaises(DBAPIError):
            async with self.session.begin_nested():
                await self.session.execute(
                    IngestionRun.__table__.insert().values(
                        document_source_id=source.id,
                        trigger_type="ADMIN_UPLOAD",
                        parser_name="gitbook-markdown",
                        parser_version="1",
                        status="FAILED",
                        stage="LOADING",
                        started_at=_now(),
                    )
                )

    # ------------------------------------------------------------------

    async def _create_group(self, label: str) -> DocumentGroup:
        group = DocumentGroup(
            group_key=f"SCHEMA-{label}-{uuid.uuid4().hex}"[:50],
            name=f"스키마 테스트 그룹 {label}",
            consumer_key="TEST",
        )
        self.session.add(group)
        await self.session.flush()
        return group

    async def _create_source(
        self,
        group: DocumentGroup,
        document_key: str,
        canonical_uri: str,
    ) -> DocumentSource:
        source = DocumentSource(
            document_group_id=group.id,
            document_key=document_key,
            source_type="UPLOAD",
            canonical_uri=canonical_uri,
            title="스키마 테스트 문서",
            created_at=_now(),
            updated_at=_now(),
        )
        self.session.add(source)
        await self.session.flush()
        return source

    async def _create_index_version(
        self,
        group: DocumentGroup,
        label: str,
        status: IndexVersionStatus,
        version_no: Optional[int] = None,
    ) -> IndexVersion:
        index_version = IndexVersion(
            document_group_id=group.id,
            version=f"schema-{label}-{uuid.uuid4().hex}"[:50],
            version_no=version_no,
            status=status,
            chunking_config_id=self.chunking.id,
            embedding_config_id=self.embedding.id,
            created_at=_now(),
        )
        self.session.add(index_version)
        await self.session.flush()
        return index_version

    async def _insert_index_version_status(self, status: str) -> None:
        await self.session.execute(
            IndexVersion.__table__.insert().values(
                document_group_id=self.group.id,
                version=f"schema-bad-{uuid.uuid4().hex}"[:50],
                status=status,
                chunking_config_id=self.chunking.id,
                embedding_config_id=self.embedding.id,
                created_at=_now(),
            )
        )


if __name__ == "__main__":
    unittest.main()
