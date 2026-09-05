"""문서 그룹 목록과 상세 계산을 실제 DB로 검증한다."""

import asyncio
import unittest
import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.admin.group_service import (
    SEARCH_STATUS_IN_PROGRESS,
    SEARCH_STATUS_NO_DOCUMENTS,
    SEARCH_STATUS_REINDEX_REQUIRED,
    DocumentGroupService,
)
from app.core.config import get_settings
from app.database.models import (
    DocumentGroup,
    DocumentSource,
    DocumentVersion,
    ExecutionStatus,
    IndexOperationType,
    IndexRun,
    IndexRunStage,
    IndexVersion,
    IndexVersionStatus,
    IngestionRun,
    IngestionStage,
)
from app.database.session import dispose_engine
from app.document.chunking_config import get_or_create_chunking_config
from app.document.document_group import get_default_document_group
from app.document.document_key import SOURCE_TYPE_UPLOAD
from app.document.ingestion_service import (
    AdminIngestionService,
    DocumentGroupNotFoundError,
    run_admin_ingestion,
)
from app.retrieval.embedding_config import get_or_create_embedding_config
from app.retrieval.embedding import (
    OPENAI_EMBEDDING_DIMENSIONS,
    EmbeddingResponse,
)


async def _check_database_available(url: str) -> bool:
    engine = create_async_engine(url)
    try:
        async with engine.connect():
            return True
    except Exception:
        return False
    finally:
        await engine.dispose()


class _StubEmbedder:
    def embed_many_with_usage(self, texts):
        return EmbeddingResponse(
            embeddings=[[0.1] * OPENAI_EMBEDDING_DIMENSIONS for _ in texts],
            input_tokens=len(texts) * 3,
            retry_count=0,
        )


class DocumentGroupDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_check_database_available(cls.database_url)):
            raise unittest.SkipTest(
                "로컬 DB에 연결할 수 없어 문서 그룹 조회 테스트를 건너뜁니다."
            )

    async def asyncSetUp(self) -> None:
        await dispose_engine()
        self.engine = create_async_engine(self.database_url)
        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )
        self.suffix = uuid.uuid4().hex[:8]
        self.titles = []
        self.index_version_ids = []
        self.other_group_id = None
        async with self.session_factory() as session:
            self.group_id = (await get_default_document_group(session)).id

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
            for index_version_id in self.index_version_ids:
                await session.execute(
                    delete(IndexVersion).where(
                        IndexVersion.id == index_version_id
                    )
                )
            if self.other_group_id is not None:
                await session.execute(
                    delete(DocumentGroup).where(
                        DocumentGroup.id == self.other_group_id
                    )
                )
            await session.commit()
        await self.engine.dispose()
        await dispose_engine()

    async def _service(self, session) -> DocumentGroupService:
        return DocumentGroupService(session)

    async def _upload(self, title: str) -> int:
        self.titles.append(title)
        async with self.session_factory() as session:
            accepted = await AdminIngestionService(session).start_new_document(
                group_id=self.group_id,
                title=title,
                category="test",
                filename="guide.md",
            )
        await run_admin_ingestion(
            accepted.ingestion_run_id,
            f"# 그룹 조회\n\n## 본문\n\n{title}\n",
            _StubEmbedder,
        )
        return accepted.document_source_id

    async def _detail(self):
        async with self.session_factory() as session:
            return await (await self._service(session)).get_group_detail(
                self.group_id
            )

    async def test_list_matches_detail_document_count(self) -> None:
        async with self.session_factory() as session:
            groups = await (await self._service(session)).list_groups()
        detail = await self._detail()

        group = next(g for g in groups if g.group_id == self.group_id)
        # 목록의 문서 수와 상세 표의 행 수는 같은 기준이어야 한다
        self.assertEqual(group.document_count, len(detail.documents))
        self.assertEqual(group.search_status, detail.search_status)

    async def test_new_document_appears_as_pending_new(self) -> None:
        title = f"group-test-{self.suffix}"
        document_id = await self._upload(title)

        detail = await self._detail()

        pending = {item.document_id: item for item in detail.pending_documents}
        self.assertIn(document_id, pending)
        self.assertEqual("NEW", pending[document_id].change_type)
        self.assertEqual(title, pending[document_id].title)
        self.assertEqual(
            SEARCH_STATUS_REINDEX_REQUIRED,
            detail.search_status,
        )

        row = next(d for d in detail.documents if d.document_id == document_id)
        self.assertEqual(SOURCE_TYPE_UPLOAD, row.source_type)
        self.assertEqual(1, row.document_version_no)
        # 아직 색인에 들어가지 않았으므로 반영 버전이 없다
        self.assertIsNone(row.applied_version_no)
        self.assertEqual("READY", row.processing_status)

    async def test_running_ingestion_is_reported(self) -> None:
        title = f"group-running-{self.suffix}"
        self.titles.append(title)
        async with self.session_factory() as session:
            accepted = await AdminIngestionService(session).start_new_document(
                group_id=self.group_id,
                title=title,
                category="test",
                filename="guide.md",
            )

        detail = await self._detail()

        self.assertIsNotNone(detail.running_job)
        self.assertEqual("INGESTION", detail.running_job.job_type)
        self.assertEqual(
            accepted.ingestion_run_id,
            detail.running_job.ingestion_run_id,
        )
        self.assertEqual(
            IngestionStage.RECEIVING.value,
            detail.running_job.stage,
        )

    async def test_running_index_makes_status_in_progress(self) -> None:
        now = datetime.now(timezone.utc)
        async with self.session_factory() as session:
            # ACTIVE 색인이 없는 빈 DB 에서도 돌아야 하므로 설정을 직접 만든다
            chunking = await get_or_create_chunking_config(session, now)
            embedding = await get_or_create_embedding_config(session, now)
            index_version = IndexVersion(
                document_group_id=self.group_id,
                version=f"group-test-{self.suffix}",
                status=IndexVersionStatus.BUILDING,
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
            await session.commit()
            run_id = run.id

        detail = await self._detail()

        self.assertEqual(SEARCH_STATUS_IN_PROGRESS, detail.search_status)
        self.assertEqual("INDEX", detail.running_job.job_type)
        self.assertEqual(run_id, detail.running_job.index_run_id)
        self.assertEqual("BUILDING", detail.running_job.stage)
        # 재진입 복원 근거도 같은 실행을 가리킨다
        self.assertEqual(run_id, detail.latest_index_run.index_run_id)
        self.assertEqual(
            ExecutionStatus.PROCESSING,
            detail.latest_index_run.status,
        )

    async def test_gitbook_documents_show_applied_version(self) -> None:
        detail = await self._detail()

        gitbook = [d for d in detail.documents if d.source_type == "GITBOOK"]
        if not gitbook:
            self.skipTest("GitBook 문서가 없어 건너뜁니다.")

        applied = [d for d in gitbook if d.applied_version_no is not None]
        # ACTIVE 색인에 든 문서는 반영 버전이 채워진다
        self.assertTrue(applied)
        for document in applied:
            self.assertLessEqual(
                document.applied_version_no,
                document.document_version_no,
            )

    async def test_second_group_is_served_by_its_own_id(self) -> None:
        """요청의 groupId 로 동작해야 한다.

        상수로 그룹을 찾으면 새 그룹이 있어도 코드가 보지 못하고,
        존재하는 그룹에 404 를 낸다.
        """

        now = datetime.now(timezone.utc)
        async with self.session_factory() as session:
            other = DocumentGroup(
                group_key=f"TEST_GROUP_{self.suffix}",
                name=f"테스트 그룹 {self.suffix}",
                consumer_key=f"TEST_{self.suffix}",
                created_at=now,
                updated_at=now,
            )
            session.add(other)
            await session.commit()
            self.other_group_id = other.id

        async with self.session_factory() as session:
            service = await self._service(session)
            groups = await service.list_groups()
            detail = await service.get_group_detail(self.other_group_id)

        self.assertIn(self.other_group_id, {g.group_id for g in groups})
        self.assertEqual(self.other_group_id, detail.group_id)
        # 새 그룹에는 문서가 없다. 기본 그룹의 문서가 섞이면 안 된다
        self.assertEqual([], detail.documents)
        self.assertEqual([], detail.sources)
        self.assertEqual(SEARCH_STATUS_NO_DOCUMENTS, detail.search_status)

    async def test_unknown_group_is_not_found(self) -> None:
        async with self.session_factory() as session:
            service = await self._service(session)
            with self.assertRaises(DocumentGroupNotFoundError):
                await service.get_group_detail(987654321)


if __name__ == "__main__":
    unittest.main()
