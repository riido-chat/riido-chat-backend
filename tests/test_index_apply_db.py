"""검색 반영 상태 전이와 적용 재시도를 실제 DB로 검증한다."""

import asyncio
import hashlib
import unittest
import uuid

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.database.models import (
    DocumentSource,
    DocumentVersion,
    ExecutionStatus,
    IndexOperationType,
    IndexRun,
    IndexRunStage,
    IndexVersion,
    IndexVersionStatus,
    IngestionRun,
)
from app.document.document_store import DocumentStore
from app.document.models import NormalizedDocument
from app.indexing.index_builder import (
    CORPUS_RELOAD_FAILED,
    run_index_job,
    start_reindex_run,
    start_retry_apply_run,
)
from app.indexing.index_writer import IndexWriter
from app.retrieval.corpus import build_document_retrieval_chunks
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
    """색인 단계에서 누락 임베딩을 채울 때 쓰는 stub."""

    def __init__(self) -> None:
        self.calls = []

    def embed_many_with_usage(self, texts):
        self.calls.append(list(texts))
        return EmbeddingResponse(
            embeddings=[
                [0.05] * OPENAI_EMBEDDING_DIMENSIONS for _ in texts
            ],
            input_tokens=len(texts) * 5,
            retry_count=0,
        )


class _StubCorpusState:
    """BM25 교체를 대신하는 stub. 실패를 주입할 수 있다."""

    def __init__(self, error: Exception = None, fail_times: int = 1) -> None:
        self.error = error
        self.remaining_failures = fail_times if error is not None else 0
        self.replaced_index_version_ids = []

    def replace(self, chunks):
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise self.error
        self.replaced_index_version_ids.append(chunks[0].index_version_id)
        return None


class IndexApplyDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_check_database_available(cls.database_url)):
            raise unittest.SkipTest(
                "로컬 DB에 연결할 수 없어 검색 반영 통합 테스트를 건너뜁니다."
            )

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(self.database_url)
        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )
        self.suffix = uuid.uuid4().hex[:8]
        self.source_url = f"https://docs.riido.io/apply-test-{self.suffix}.md"
        self.index_version_ids = []
        self.active_before = await self._active_index_ids()
        await self._ingest_document()

    async def asyncTearDown(self) -> None:
        async with self.session_factory() as session:
            for index_version_id in self.index_version_ids:
                await session.execute(
                    delete(IndexVersion).where(
                        IndexVersion.id == index_version_id
                    )
                )
            source = await session.scalar(
                select(DocumentSource).where(
                    DocumentSource.canonical_uri == self.source_url
                )
            )
            if source is not None:
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
            if self.active_before:
                await session.execute(
                    update(IndexVersion)
                    .where(IndexVersion.id.in_(self.active_before))
                    .values(status=IndexVersionStatus.ACTIVE)
                )
            await session.commit()
        await self.engine.dispose()

    async def _active_index_ids(self) -> set:
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

    async def _ingest_document(self) -> None:
        content = f"# 반영 테스트 {self.suffix}\n\n## 본문\n\n적용 검증 본문"
        document = NormalizedDocument(
            document_id=f"apply-test-{self.suffix}",
            title=f"반영 테스트 {self.suffix}",
            source_url=self.source_url,
            category="test",
            content=content,
            raw_content_uri=f"raw/apply-test-{self.suffix}.md",
            raw_content_hash=hashlib.sha256(
                f"raw {self.suffix}".encode("utf-8")
            ).hexdigest(),
            normalized_content_hash=hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest(),
        )
        chunks = build_document_retrieval_chunks(document)
        async with self.session_factory() as session:
            store = DocumentStore(session)
            run = await store.start_ingestion(document)
            await store.complete_ingestion(
                run.id,
                document,
                chunks,
                [[0.05] * OPENAI_EMBEDDING_DIMENSIONS for _ in chunks],
            )
            await session.commit()

    async def _start_and_run(self, corpus_state) -> int:
        async with self.session_factory() as session:
            run = await start_reindex_run(session)
            index_run_id = run.id
            self.index_version_ids.append(run.index_version_id)
            await session.commit()
        async with self.session_factory() as session:
            await run_index_job(
                session,
                corpus_state,
                index_run_id,
                _StubEmbedder,
            )
        return index_run_id

    async def test_ready_gets_version_no_then_becomes_active(self) -> None:
        corpus_state = _StubCorpusState()

        index_run_id = await self._start_and_run(corpus_state)

        async with self.session_factory() as session:
            run = await session.get(IndexRun, index_run_id)
            index_version = await session.get(IndexVersion, run.index_version_id)

        self.assertEqual(ExecutionStatus.SUCCESS, run.status)
        self.assertEqual(IndexRunStage.APPLYING, run.stage)
        self.assertEqual(
            IndexOperationType.BUILD_AND_APPLY,
            run.operation_type,
        )
        self.assertEqual(IndexVersionStatus.ACTIVE, index_version.status)
        self.assertIsNotNone(index_version.version_no)
        self.assertIsNotNone(index_version.activated_at)
        # corpus 도 같은 세대로 교체된다
        self.assertEqual(
            [index_version.id],
            corpus_state.replaced_index_version_ids,
        )

    async def test_previous_active_is_recorded_on_the_run(self) -> None:
        first_run_id = await self._start_and_run(_StubCorpusState())
        async with self.session_factory() as session:
            first_run = await session.get(IndexRun, first_run_id)
        first_version_id = first_run.index_version_id

        second_run_id = await self._start_and_run(_StubCorpusState())

        async with self.session_factory() as session:
            second_run = await session.get(IndexRun, second_run_id)

        # 현재 상태 역산이 아니라 적용 시점 기록이므로 나중에도 변하지 않는다
        self.assertEqual(
            first_version_id,
            second_run.summary["previous_index_version_id"],
        )

    async def test_apply_failure_keeps_candidate_ready_and_allows_retry(
        self,
    ) -> None:
        failing = _StubCorpusState(RuntimeError("corpus 교체 실패"))
        index_run_id = await self._start_and_run(failing)

        async with self.session_factory() as session:
            failed_run = await session.get(IndexRun, index_run_id)
            candidate = await session.get(
                IndexVersion,
                failed_run.index_version_id,
            )

        self.assertEqual(ExecutionStatus.FAILED, failed_run.status)
        self.assertEqual(IndexRunStage.APPLYING, failed_run.stage)
        self.assertEqual(CORPUS_RELOAD_FAILED, failed_run.error_code)
        # 후보는 READY 로 남아 다시 시도할 수 있다
        self.assertEqual(IndexVersionStatus.READY, candidate.status)
        self.assertIsNotNone(candidate.version_no)
        self.assertEqual(self.active_before, await self._active_index_ids())

        corpus_state = _StubCorpusState()
        async with self.session_factory() as session:
            retry = await start_retry_apply_run(
                session,
                await session.get(IndexRun, index_run_id),
            )
            retry_run_id = retry.id
            await session.commit()
        async with self.session_factory() as session:
            await run_index_job(
                session,
                corpus_state,
                retry_run_id,
                _StubEmbedder,
            )

        async with self.session_factory() as session:
            retry_run = await session.get(IndexRun, retry_run_id)
            applied = await session.get(IndexVersion, retry_run.index_version_id)

        self.assertEqual(candidate.id, retry_run.index_version_id)
        self.assertEqual(IndexOperationType.APPLY, retry_run.operation_type)
        self.assertEqual("RETRY", retry_run.trigger_type)
        self.assertEqual(ExecutionStatus.SUCCESS, retry_run.status)
        self.assertEqual(IndexVersionStatus.ACTIVE, applied.status)

    async def test_leftover_ready_candidate_becomes_inactive_on_apply(
        self,
    ) -> None:
        leftover_run_id = await self._start_and_run(
            _StubCorpusState(RuntimeError("corpus 교체 실패"))
        )
        async with self.session_factory() as session:
            leftover_run = await session.get(IndexRun, leftover_run_id)
        leftover_version_id = leftover_run.index_version_id

        await self._start_and_run(_StubCorpusState())

        async with self.session_factory() as session:
            leftover = await session.get(IndexVersion, leftover_version_id)

        # 적용되지 못한 후보는 새 후보가 ACTIVE 가 될 때 정리된다
        self.assertEqual(IndexVersionStatus.INACTIVE, leftover.status)


if __name__ == "__main__":
    unittest.main()
