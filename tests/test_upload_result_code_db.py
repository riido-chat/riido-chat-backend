"""업로드 결과 판정과 단계 기록을 실제 DB로 검증한다."""

import asyncio
import unittest
import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.database.models import (
    DocumentSource,
    DocumentVersion,
    DocumentVersionStatus,
    ExecutionStatus,
    IngestionResultCode,
    IngestionRun,
    IngestionStage,
)
from app.database.session import dispose_engine
from app.document.document_group import get_document_group
from app.document.ingestion_service import (
    AdminIngestionService,
    DocumentNotFoundError,
    DocumentNotRevisableError,
    run_admin_ingestion,
)
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


class UploadResultCodeDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_check_database_available(cls.database_url)):
            raise unittest.SkipTest(
                "로컬 DB에 연결할 수 없어 업로드 판정 통합 테스트를 건너뜁니다."
            )

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(self.database_url)
        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )
        self.suffix = uuid.uuid4().hex[:8]
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
        # run_admin_ingestion 이 쓰는 전역 engine 도 테스트별 loop 와 함께 닫는다
        await dispose_engine()

    def _new_title(self) -> str:
        title = f"result-test-{self.suffix}-{uuid.uuid4().hex[:6]}"
        self.titles.append(title)
        return title

    async def _upload(self, title: str, body: str):
        async with self.session_factory() as session:
            accepted = await AdminIngestionService(session).start_new_document(
                group_id=self.group_id,
                title=title,
                category="test",
                filename="guide.md",
            )
        await run_admin_ingestion(
            accepted.ingestion_run_id,
            body,
            _StubEmbedder,
        )
        return accepted

    async def _revise(self, document_id: int, body: str):
        async with self.session_factory() as session:
            accepted = await AdminIngestionService(
                session
            ).start_document_revision(
                document_id=document_id,
                filename="guide.md",
            )
        await run_admin_ingestion(
            accepted.ingestion_run_id,
            body,
            _StubEmbedder,
        )
        return accepted

    async def _run(self, ingestion_run_id: int) -> IngestionRun:
        async with self.session_factory() as session:
            return await session.get(IngestionRun, ingestion_run_id)

    def _body(self, marker: str) -> str:
        return f"# 결과 판정\n\n## 본문\n\n{marker} {self.suffix}\n"

    async def test_first_upload_is_created_with_chunk_stats(self) -> None:
        title = self._new_title()
        accepted = await self._upload(title, self._body("첫 판"))

        run = await self._run(accepted.ingestion_run_id)

        self.assertEqual(ExecutionStatus.SUCCESS, run.status)
        self.assertEqual(IngestionResultCode.CREATED, run.result_code)
        self.assertEqual(IngestionStage.PERSISTING, run.stage)
        self.assertIsNotNone(run.produced_version_id)
        # 첫 판은 전부 추가다
        self.assertEqual(0, run.summary["reused"])
        self.assertEqual(0, run.summary["changed"])
        self.assertEqual(0, run.summary["deleted"])
        self.assertGreater(run.summary["added"], 0)

    async def test_same_content_upload_is_no_change(self) -> None:
        title = self._new_title()
        body = self._body("같은 내용")
        await self._upload(title, body)

        second = await self._upload(title, body)
        run = await self._run(second.ingestion_run_id)

        self.assertEqual(ExecutionStatus.SUCCESS, run.status)
        self.assertEqual(IngestionResultCode.NO_CHANGE, run.result_code)
        # 새 판을 만들지 않는다
        self.assertIsNone(run.produced_version_id)

    async def test_changed_content_upload_is_updated(self) -> None:
        title = self._new_title()
        await self._upload(title, self._body("처음"))

        second = await self._upload(title, self._body("고친 뒤"))
        run = await self._run(second.ingestion_run_id)

        self.assertEqual(IngestionResultCode.UPDATED, run.result_code)
        self.assertIsNotNone(run.produced_version_id)

        async with self.session_factory() as session:
            version = await session.get(
                DocumentVersion,
                run.produced_version_id,
            )
        self.assertEqual(2, version.version_no)
        self.assertEqual(DocumentVersionStatus.READY, version.status)
        # 같은 섹션의 내용만 바뀌었다
        self.assertEqual(1, run.summary["changed"])
        self.assertEqual(0, run.summary["added"])
        self.assertEqual(0, run.summary["deleted"])

    async def test_same_content_in_another_document_is_duplicate(self) -> None:
        body = self._body("공유 본문")
        first_title = self._new_title()
        first = await self._upload(first_title, body)

        second = await self._upload(self._new_title(), body)
        run = await self._run(second.ingestion_run_id)

        self.assertEqual(ExecutionStatus.SUCCESS, run.status)
        self.assertEqual(
            IngestionResultCode.DUPLICATE_CONTENT,
            run.result_code,
        )
        self.assertIsNone(run.produced_version_id)
        self.assertEqual(
            first.document_source_id,
            run.duplicate_of_document_source_id,
        )

    async def test_revision_creates_next_version(self) -> None:
        title = self._new_title()
        first = await self._upload(title, self._body("원본"))

        revised = await self._revise(
            first.document_source_id,
            self._body("수정본"),
        )
        run = await self._run(revised.ingestion_run_id)

        self.assertEqual(IngestionResultCode.UPDATED, run.result_code)
        self.assertEqual(first.document_source_id, run.document_source_id)

    async def test_revision_rejects_gitbook_document(self) -> None:
        async with self.session_factory() as session:
            gitbook_source = await session.scalar(
                select(DocumentSource)
                .where(DocumentSource.source_type == "GITBOOK")
                .limit(1)
            )
        if gitbook_source is None:
            self.skipTest("GitBook 문서가 없어 건너뜁니다.")

        async with self.session_factory() as session:
            with self.assertRaises(DocumentNotRevisableError):
                await AdminIngestionService(session).start_document_revision(
                    document_id=gitbook_source.id,
                    filename="guide.md",
                )

    async def test_body_less_markdown_fails_at_chunking_stage(self) -> None:
        title = self._new_title()
        # 제목만 있으면 정제는 통과하고 검색 단위를 만들 때 실패한다
        accepted = await self._upload(title, "# 제목만 있는 문서\n")

        run = await self._run(accepted.ingestion_run_id)

        self.assertEqual(ExecutionStatus.FAILED, run.status)
        # FE 는 이 stage 로 3-4 원인 문구를 고른다
        self.assertEqual(IngestionStage.CHUNKING, run.stage)
        self.assertEqual("INVALID_FILE", run.error_code)
        self.assertIsNone(run.produced_version_id)

    async def test_revision_rejects_unknown_document(self) -> None:
        async with self.session_factory() as session:
            with self.assertRaises(DocumentNotFoundError):
                await AdminIngestionService(session).start_document_revision(
                    document_id=987654321,
                    filename="guide.md",
                )


if __name__ == "__main__":
    unittest.main()
