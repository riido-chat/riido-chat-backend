"""Admin Markdown 신규 업로드 수명주기의 로컬 DB 통합 테스트."""

import asyncio
import unittest
import uuid

from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.admin.ingestion_service import (
    AdminIngestionService,
    AdminJobInProgressError,
    DocumentAlreadyExistsError,
    run_admin_ingestion,
)
from app.admin.document_key import (
    DEFAULT_DOCUMENT_GROUP_KEY,
    SOURCE_TYPE_UPLOAD,
    build_console_canonical_uri,
    build_upload_document_key,
)
from app.core.config import get_settings
from app.database.models import (
    ContentNode,
    DocumentGroup,
    DocumentChunk,
    DocumentSource,
    DocumentVersion,
    DocumentVersionStatus,
    ExecutionStatus,
    IndexVersion,
    IndexVersionStatus,
    IngestionRun,
)
from app.database.session import dispose_engine
from app.main import create_app


async def _check_database_available(url: str) -> bool:
    engine = create_async_engine(url)
    try:
        async with engine.connect():
            return True
    except Exception:
        return False
    finally:
        await engine.dispose()


class AdminIngestionDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_check_database_available(cls.database_url)):
            raise unittest.SkipTest(
                "로컬 DB에 연결할 수 없어 Admin 수집 통합 테스트를 건너뜁니다."
            )

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(self.database_url)
        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )
        self.titles: list[str] = []

    async def asyncTearDown(self) -> None:
        async with self.session_factory() as session:
            group_id = await session.scalar(
                select(DocumentGroup.id).where(
                    DocumentGroup.group_key == DEFAULT_DOCUMENT_GROUP_KEY
                )
            )
            for title in self.titles:
                source = await session.scalar(
                    select(DocumentSource).where(
                        DocumentSource.document_group_id == group_id,
                        DocumentSource.document_key
                        == build_upload_document_key(title),
                    )
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
        # run_admin_ingestion이 사용하는 전역 engine도 테스트별 event loop와 함께 닫는다.
        await dispose_engine()

    async def test_upload_persists_ready_version_and_keeps_active_index(self) -> None:
        title = self._new_title()
        raw_content = "# 새 문서\n\n## 이용 방법\n\n관리자 업로드 본문\n"
        active_before = await self._active_index_ids()

        accepted = await self._start(title)
        await run_admin_ingestion(accepted.ingestion_run_id, raw_content)

        async with self.session_factory() as session:
            run = await session.get(IngestionRun, accepted.ingestion_run_id)
            version = await session.get(DocumentVersion, run.produced_version_id)
            node_count = await session.scalar(
                select(func.count())
                .select_from(ContentNode)
                .where(ContentNode.document_version_id == version.id)
            )
            chunk_count = await session.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .join(ContentNode, ContentNode.id == DocumentChunk.id)
                .where(ContentNode.document_version_id == version.id)
            )

        self.assertEqual(ExecutionStatus.SUCCESS, run.status)
        self.assertEqual(DocumentVersionStatus.READY, version.status)
        self.assertEqual(1, version.version_no)
        self.assertIsNone(version.raw_content_uri)
        self.assertEqual(raw_content, version.raw_content)
        self.assertEqual(1, node_count)
        self.assertEqual(1, chunk_count)
        self.assertEqual(active_before, await self._active_index_ids())

    async def test_http_upload_and_polling_complete_end_to_end(self) -> None:
        title = self._new_title()
        raw_content = "# API 업로드\n\n## 이용 방법\n\n실제 multipart 본문\n"
        active_before = await self._active_index_ids()
        app = create_app()

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            accepted_response = await client.post(
                "/api/admin/documents",
                data={"title": title, "category": "test"},
                files={
                    "file": (
                        "api-smoke.md",
                        raw_content.encode("utf-8"),
                        "text/markdown",
                    )
                },
            )
            self.assertEqual(202, accepted_response.status_code)
            accepted = accepted_response.json()

            result = None
            for _ in range(100):
                status_response = await client.get(
                    f"/api/admin/ingestion-runs/{accepted['ingestionRunId']}"
                )
                self.assertEqual(200, status_response.status_code)
                result = status_response.json()
                if result["status"] != "PROCESSING":
                    break
                await asyncio.sleep(0.05)

        self.assertIsNotNone(result)
        self.assertEqual("SUCCESS", result["status"])
        self.assertEqual(accepted["documentId"], result["documentId"])

        async with self.session_factory() as session:
            version = await session.get(
                DocumentVersion,
                result["documentVersionId"],
            )
        self.assertEqual(DocumentVersionStatus.READY, version.status)
        self.assertEqual(raw_content, version.raw_content)
        self.assertEqual(active_before, await self._active_index_ids())

    async def test_upload_source_uses_document_key_and_console_uri(self) -> None:
        title = self._new_title()
        accepted = await self._start(title)
        await run_admin_ingestion(
            accepted.ingestion_run_id,
            "# 문서\n\n## 본문\n\n키 확인\n",
        )

        document_key = build_upload_document_key(title)
        async with self.session_factory() as session:
            source = await session.get(DocumentSource, accepted.document_source_id)

        self.assertEqual(document_key, source.document_key)
        self.assertEqual(SOURCE_TYPE_UPLOAD, source.source_type)
        self.assertEqual(
            build_console_canonical_uri(DEFAULT_DOCUMENT_GROUP_KEY, document_key),
            source.canonical_uri,
        )

    async def test_failed_empty_pipeline_can_retry_same_source(self) -> None:
        title = self._new_title()
        first = await self._start(title)

        await run_admin_ingestion(first.ingestion_run_id, "# 제목만 있는 문서\n")

        async with self.session_factory() as session:
            failed = await session.get(IngestionRun, first.ingestion_run_id)
            version_count = await session.scalar(
                select(func.count())
                .select_from(DocumentVersion)
                .where(DocumentVersion.document_source_id == first.document_source_id)
            )
        self.assertEqual(ExecutionStatus.FAILED, failed.status)
        self.assertEqual("INVALID_FILE", failed.summary["error_code"])
        self.assertEqual(0, version_count)

        second = await self._start(title)
        self.assertEqual(first.document_source_id, second.document_source_id)
        self.assertNotEqual(first.ingestion_run_id, second.ingestion_run_id)

        await run_admin_ingestion(
            second.ingestion_run_id,
            "# 문서\n\n## 본문\n\n재시도 성공\n",
        )
        async with self.session_factory() as session:
            succeeded = await session.get(IngestionRun, second.ingestion_run_id)
        self.assertEqual(ExecutionStatus.SUCCESS, succeeded.status)

    async def test_successful_source_is_rejected_as_duplicate(self) -> None:
        title = self._new_title()
        accepted = await self._start(title)
        await run_admin_ingestion(
            accepted.ingestion_run_id,
            "# 문서\n\n## 본문\n\n성공\n",
        )

        with self.assertRaises(DocumentAlreadyExistsError):
            await self._start(title)

    async def test_processing_run_blocks_another_admin_job(self) -> None:
        first_title = self._new_title()
        second_title = self._new_title()
        first = await self._start(first_title)

        with self.assertRaises(AdminJobInProgressError):
            await self._start(second_title)

        await run_admin_ingestion(
            first.ingestion_run_id,
            "# 문서\n\n## 본문\n\n처리 완료\n",
        )

    async def _start(self, title: str):
        async with self.session_factory() as session:
            return await AdminIngestionService(session).start_new_document(
                title=title,
                category="test",
                filename="guide.md",
            )

    async def _active_index_ids(self) -> set[int]:
        async with self.session_factory() as session:
            return set(
                (
                    await session.execute(
                        select(IndexVersion.id).where(
                            IndexVersion.status == IndexVersionStatus.ACTIVE
                        )
                    )
                ).scalars()
            )

    def _new_title(self) -> str:
        title = f"admin-test-{uuid.uuid4().hex}"
        self.titles.append(title)
        return title


if __name__ == "__main__":
    unittest.main()
