"""GitBook 수집 배치를 실제 DB로 검증한다.

외부 네트워크는 모두 stub 으로 대체한다. 이 기능이 바꾼 것은 페이지별
실행을 만들고 결과를 집계하는 방식이며 HTTP 조회 자체가 아니다.
"""

import asyncio
import unittest
import uuid
from unittest.mock import patch

from sqlalchemy import delete, or_, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.admin.group_service import DocumentGroupService
from app.core.config import get_settings
from app.database.models import (
    DocumentGroupSource,
    DocumentSource,
    DocumentVersion,
    ExecutionStatus,
    IngestionResultCode,
    IngestionRun,
    IngestionStage,
)
from app.database.session import dispose_engine
from app.document.document_group import get_document_group
from app.document.document_key import SOURCE_TYPE_GITBOOK
from app.document.gitbook.client import GitBookListError, GitBookPage
from app.document.ingestion_service import AdminIngestionService
from app.document.recollect import run_recollect_batch
from app.document.recollect_service import (
    RecollectBatchNotFoundError,
    RecollectService,
    SourceListFailedError,
)
from app.retrieval.embedding import (
    OPENAI_EMBEDDING_DIMENSIONS,
    EmbeddingResponse,
)


ROOT_URL = "https://docs.riido.io"


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


class GitBookSyncDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_check_database_available(cls.database_url)):
            raise unittest.SkipTest(
                "로컬 DB에 연결할 수 없어 GitBook 수집 통합 테스트를 건너뜁니다."
            )

    async def asyncSetUp(self) -> None:
        # 앞선 테스트가 다른 event loop 에서 만든 전역 engine 을 물려받지 않는다
        await dispose_engine()
        self.engine = create_async_engine(self.database_url)
        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )
        self.suffix = uuid.uuid4().hex[:8]
        self.keys = []
        self.console_titles = []
        self.other_root = None
        async with self.session_factory() as session:
            self.group_id = (await get_document_group(session)).id
            # 수집은 목록에 없는 GitBook 문서를 사라진 것으로 보고 비활성화한다.
            # stub 목록에는 이 테스트의 페이지만 있으므로 기존 문서가 전부
            # 꺼진다. 원래 켜져 있던 문서를 기억해 두었다가 teardown 에서
            # 되돌린다.
            self.enabled_before = set(
                (
                    await session.execute(
                        select(DocumentSource.id).where(
                            DocumentSource.source_type == SOURCE_TYPE_GITBOOK,
                            DocumentSource.enabled.is_(True),
                        )
                    )
                ).scalars()
            )

    async def asyncTearDown(self) -> None:
        async with self.session_factory() as session:
            # 같은 키가 원천마다 하나씩 있을 수 있으므로 전부 지운다.
            # 하나만 집으면 다른 원천의 문서가 DB 에 남는다.
            statement = select(DocumentSource).where(
                or_(
                    DocumentSource.document_key.in_(self.keys or [""]),
                    DocumentSource.title.in_(self.console_titles or [""]),
                )
            )
            for source in (await session.execute(statement)).scalars().all():
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
            await session.flush()
            if self.other_root is not None:
                other_source = await session.scalar(
                    select(DocumentGroupSource).where(
                        DocumentGroupSource.root_url == self.other_root
                    )
                )
                if other_source is not None:
                    for source in (
                        await session.execute(
                            select(DocumentSource).where(
                                DocumentSource.group_source_id == other_source.id
                            )
                        )
                    ).scalars():
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
                    await session.flush()
                    await session.delete(other_source)
            if self.enabled_before:
                await session.execute(
                    update(DocumentSource)
                    .where(DocumentSource.id.in_(self.enabled_before))
                    .values(enabled=True)
                )
            await session.commit()
        await self.engine.dispose()
        await dispose_engine()

    def _page(self, name: str) -> GitBookPage:
        key = f"synctest-{self.suffix}/{name}"
        self.keys.append(key)
        return GitBookPage(
            title=f"동기화 {name} {self.suffix}",
            url=f"{ROOT_URL}/{key}.md",
            category="synctest",
        )

    def _body(self, marker: str) -> str:
        return f"# 수집 문서\n\n## 본문\n\n{marker} {self.suffix}\n"

    async def _sync(self, pages, bodies, root_url: str = ROOT_URL):
        """목록과 본문을 stub 으로 대체해 배치를 끝까지 돌린다."""

        with patch(
            "app.document.recollect_service.list_pages",
            return_value=pages,
        ):
            async with self.session_factory() as session:
                accepted = await RecollectService(session).start_sync(
                    self.group_id,
                    root_url,
                )

        def fake_fetch(url: str) -> str:
            body = bodies.get(url)
            if body is None:
                raise RuntimeError(f"페이지를 읽지 못했습니다: {url}")
            return body

        with patch("app.document.recollect.fetch_page", side_effect=fake_fetch):
            await run_recollect_batch(accepted.batch_id, _StubEmbedder)
        return accepted

    async def _batch(self, batch_id):
        async with self.session_factory() as session:
            return await RecollectService(session).get_batch(batch_id)

    async def _source_of(self, page: GitBookPage) -> DocumentSource:
        async with self.session_factory() as session:
            return await session.scalar(
                select(DocumentSource).where(
                    DocumentSource.canonical_uri == page.url
                )
            )

    async def test_first_sync_creates_every_page(self) -> None:
        pages = [self._page("intro"), self._page("guide")]
        bodies = {
            pages[0].url: self._body("소개"),
            pages[1].url: self._body("안내"),
        }

        accepted = await self._sync(pages, bodies)
        batch = await self._batch(accepted.batch_id)

        self.assertEqual(ExecutionStatus.SUCCESS, batch.status)
        # removed 는 그룹에 이미 있던 문서에 좌우되므로 단정하지 않는다
        self.assertEqual(2, batch.counts["total"])
        self.assertEqual(2, batch.counts["created"])
        self.assertEqual(0, batch.counts["updated"])
        self.assertEqual(0, batch.counts["no_change"])
        self.assertEqual(0, batch.counts["failed"])
        self.assertEqual((), batch.failures)

    async def test_second_sync_reports_no_change_and_updated(self) -> None:
        pages = [self._page("intro"), self._page("guide")]
        first_bodies = {
            pages[0].url: self._body("소개"),
            pages[1].url: self._body("안내"),
        }
        await self._sync(pages, first_bodies)

        second_bodies = dict(first_bodies)
        second_bodies[pages[1].url] = self._body("안내 고침")
        accepted = await self._sync(pages, second_bodies)
        batch = await self._batch(accepted.batch_id)

        self.assertEqual(1, batch.counts["no_change"])
        self.assertEqual(1, batch.counts["updated"])
        self.assertEqual(0, batch.counts["created"])

    async def test_disappeared_page_is_disabled_and_restored(self) -> None:
        pages = [self._page("intro"), self._page("guide")]
        bodies = {
            pages[0].url: self._body("소개"),
            pages[1].url: self._body("안내"),
        }
        await self._sync(pages, bodies)

        # 두 번째 수집에서 guide 가 목록에서 사라진다
        accepted = await self._sync(pages[:1], bodies)
        batch = await self._batch(accepted.batch_id)
        disabled = await self._source_of(pages[1])

        self.assertEqual(1, batch.counts["removed"])
        self.assertEqual(1, batch.counts["total"])
        self.assertFalse(disabled.enabled)

        # 다시 나타나면 되살아난다
        restored_batch = await self._sync(pages, bodies)
        restored = await self._source_of(pages[1])
        counts = (await self._batch(restored_batch.batch_id)).counts

        self.assertTrue(restored.enabled)
        self.assertEqual(0, counts["removed"])

    async def test_one_page_failure_does_not_stop_the_batch(self) -> None:
        pages = [self._page("intro"), self._page("broken"), self._page("guide")]
        bodies = {
            pages[0].url: self._body("소개"),
            pages[2].url: self._body("안내"),
        }

        accepted = await self._sync(pages, bodies)
        batch = await self._batch(accepted.batch_id)

        self.assertEqual(ExecutionStatus.SUCCESS, batch.status)
        self.assertEqual(2, batch.counts["created"])
        self.assertEqual(1, batch.counts["failed"])
        self.assertEqual(1, len(batch.failures))

        failure = batch.failures[0]
        self.assertIn("broken", failure.document_key)
        self.assertEqual(IngestionStage.RECEIVING.value, failure.stage)
        self.assertEqual("UPSTREAM_ERROR", failure.error_code)
        # 실패 상세는 기존 업로드 실행 조회로 볼 수 있다
        async with self.session_factory() as session:
            detail = await AdminIngestionService(session).get_ingestion_run(
                failure.ingestion_run_id
            )
        self.assertEqual(ExecutionStatus.FAILED, detail.status)

    async def test_console_documents_are_untouched(self) -> None:
        title = f"sync-console-{self.suffix}"
        self.console_titles.append(title)
        async with self.session_factory() as session:
            await AdminIngestionService(session).start_new_document(
                group_id=self.group_id,
                title=title,
                category="test",
                filename="guide.md",
            )
        async with self.session_factory() as session:
            console_source = await session.scalar(
                select(DocumentSource).where(DocumentSource.title == title)
            )
            await session.execute(
                delete(IngestionRun).where(
                    IngestionRun.document_source_id == console_source.id
                )
            )
            await session.commit()

        page = self._page("intro")
        await self._sync([page], {page.url: self._body("소개")})

        async with self.session_factory() as session:
            after = await session.get(DocumentSource, console_source.id)
        # 콘솔 업로드 문서는 수집 대상이 아니므로 그대로 남는다
        self.assertTrue(after.enabled)

    async def test_source_list_failure_creates_no_run(self) -> None:
        async with self.session_factory() as session:
            before = await session.scalar(
                select(IngestionRun.id).order_by(IngestionRun.id.desc()).limit(1)
            )

        with patch(
            "app.document.recollect_service.list_pages",
            side_effect=GitBookListError("목록 조회 실패"),
        ):
            async with self.session_factory() as session:
                with self.assertRaises(SourceListFailedError):
                    await RecollectService(session).start_sync(
                        self.group_id,
                        ROOT_URL,
                    )

        async with self.session_factory() as session:
            after = await session.scalar(
                select(IngestionRun.id).order_by(IngestionRun.id.desc()).limit(1)
            )
        self.assertEqual(before, after)

    async def test_second_gitbook_keeps_the_first_untouched(self) -> None:
        page = self._page("intro")
        await self._sync([page], {page.url: self._body("소개")})

        other_root = f"https://docs.example-{self.suffix}.com"
        # 두 GitBook 이 같은 경로 키를 가져도 원천이 달라 서로 다른 문서다
        other = GitBookPage(
            title=f"다른 GitBook {self.suffix}",
            url=f"{other_root}/{self.keys[0]}.md",
            category="synctest",
        )
        self.other_root = other_root
        await self._sync(
            [other],
            {other.url: self._body("다른 GitBook 본문")},
            root_url=other_root,
        )

        first = await self._source_of(page)
        second = await self._source_of(other)

        self.assertNotEqual(first.id, second.id)
        self.assertEqual(first.document_key, second.document_key)
        self.assertNotEqual(first.group_source_id, second.group_source_id)
        # 다른 원천을 수집해도 앞선 GitBook 문서는 그대로다
        self.assertTrue(first.enabled)

        # 응답에 원천이 없으면 표에서 두 문서를 구분할 수 없다
        async with self.session_factory() as session:
            detail = await DocumentGroupService(session).get_group_detail(
                self.group_id
            )
        roots = {source.root_url for source in detail.sources}
        self.assertIn(other_root, roots)

        by_id = {row.document_id: row for row in detail.documents}
        self.assertEqual(
            first.group_source_id,
            by_id[first.id].group_source_id,
        )
        self.assertEqual(
            second.group_source_id,
            by_id[second.id].group_source_id,
        )

    async def test_batch_reports_which_gitbook_was_collected(self) -> None:
        page = self._page("intro")
        accepted = await self._sync([page], {page.url: self._body("소개")})

        batch = await self._batch(accepted.batch_id)

        # 새로고침 뒤 재진입해도 어느 GitBook 수집인지 알 수 있어야 한다
        self.assertEqual(ROOT_URL, batch.root_url)
        self.assertEqual(accepted.group_source_id, batch.group_source_id)

    async def test_unknown_batch_is_not_found(self) -> None:
        async with self.session_factory() as session:
            with self.assertRaises(RecollectBatchNotFoundError):
                await RecollectService(session).get_batch(uuid.uuid4())


if __name__ == "__main__":
    unittest.main()
