"""수집·색인 실패 이력이 checkpoint commit 뒤 보존되는지 검증한다."""

import asyncio
import hashlib
import unittest
import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.database.models import (
    DocumentSource,
    DocumentVersion,
    ExecutionStatus,
    IngestionRun,
    IndexRun,
    IndexVersion,
    IndexVersionStatus,
)
from app.document.models import NormalizedDocument
from app.indexing.index_vector_corpus import run_reindex


async def _check_database_available(url: str) -> bool:
    engine = create_async_engine(url)
    try:
        async with engine.connect():
            return True
    except Exception:
        return False
    finally:
        await engine.dispose()


class _FailingEmbedder:
    def embed_many(self, texts):
        raise RuntimeError("embedding unavailable")

    def embed_many_with_usage(self, texts):
        raise RuntimeError("embedding unavailable")


class IndexRunLoggingDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_check_database_available(cls.database_url)):
            raise unittest.SkipTest(
                "로컬 DB에 연결할 수 없어 색인 로그 통합 테스트를 건너뜁니다."
            )

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(self.database_url)
        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )
        self.source_url = None
        self.index_version_id = None

    async def asyncTearDown(self) -> None:
        async with self.session_factory() as session:
            if self.index_version_id is not None:
                await session.execute(
                    delete(IndexVersion).where(
                        IndexVersion.id == self.index_version_id
                    )
                )

            if self.source_url is not None:
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
            await session.commit()
        await self.engine.dispose()

    async def test_embedding_failure_fails_ingestion_before_any_index_run(
        self,
    ) -> None:
        suffix = uuid.uuid4().hex[:8]
        self.source_url = f"https://docs.riido.io/log-test-{suffix}.md"
        document = NormalizedDocument(
            document_id=f"log-test-{suffix}",
            title="로그 통합 테스트",
            source_url=self.source_url,
            category="test",
            content="# 로그 통합 테스트\n\n## 실행\n\n실패 로그 본문",
            raw_content_uri=f"raw/log-test-{suffix}.md",
            raw_content_hash=hashlib.sha256(
                f"raw log test {suffix}".encode("utf-8")
            ).hexdigest(),
            normalized_content_hash=hashlib.sha256(
                "# 로그 통합 테스트\n\n## 실행\n\n실패 로그 본문".encode(
                    "utf-8"
                )
            ).hexdigest(),
        )

        async with self.session_factory() as session:
            active_before = set(
                (
                    await session.execute(
                        select(IndexVersion.id).where(
                            IndexVersion.status == IndexVersionStatus.ACTIVE
                        )
                    )
                ).scalars()
            )
            index_run_ids_before = set(
                (await session.execute(select(IndexRun.id))).scalars()
            )

        async with self.session_factory() as session:
            with self.assertRaisesRegex(RuntimeError, "embedding unavailable"):
                await run_reindex([document], _FailingEmbedder(), session)

        async with self.session_factory() as session:
            source = await session.scalar(
                select(DocumentSource).where(
                    DocumentSource.canonical_uri == self.source_url
                )
            )
            self.assertIsNotNone(source)
            ingestion_run = await session.scalar(
                select(IngestionRun).where(
                    IngestionRun.document_source_id == source.id
                )
            )
            # embedding은 접수 시점에 만들어지므로 실패도 수집 실행에서 마감된다
            self.assertEqual(ExecutionStatus.FAILED, ingestion_run.status)
            self.assertIsNone(ingestion_run.produced_version_id)
            self.assertIsNotNone(ingestion_run.finished_at)
            self.assertIn("embedding unavailable", ingestion_run.error_message)

            index_runs = list((await session.execute(select(IndexRun))).scalars())
            created_runs = [
                run for run in index_runs if run.id not in index_run_ids_before
            ]
            self.assertEqual([], created_runs)
            active_after = set(
                (
                    await session.execute(
                        select(IndexVersion.id).where(
                            IndexVersion.status == IndexVersionStatus.ACTIVE
                        )
                    )
                ).scalars()
            )
            self.assertEqual(active_before, active_after)


if __name__ == "__main__":
    unittest.main()
