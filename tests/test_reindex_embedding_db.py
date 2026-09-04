"""접수 시점 embedding 적재와 재사용이 실제 DB에서 동작하는지 검증한다.

외부 embedding API를 호출하지 않는다. 이 PR이 바꾼 것은 embedding을
만드는 시점과 재사용 여부이고, 모델 호출 자체의 계약은 바뀌지 않았다.
"""

import asyncio
import hashlib
import unittest
import uuid

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.database.models import (
    ChunkEmbedding,
    ContentNode,
    DocumentChunk,
    DocumentSource,
    DocumentVersion,
    DocumentVersionStatus,
    ExecutionStatus,
    IndexRun,
    IndexVersion,
    IndexVersionStatus,
    IngestionRun,
    ModelCall,
    ModelCallPurpose,
)
from app.document.models import NormalizedDocument
from app.indexing.index_vector_corpus import run_reindex
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


class _CountingStubEmbedder:
    """호출 횟수를 세는 stub. 같은 입력에는 같은 vector를 돌려준다."""

    def __init__(self) -> None:
        self.calls = []

    def _vector(self, text: str) -> list:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        value = (seed % 1000) / 1000.0
        return [value] * OPENAI_EMBEDDING_DIMENSIONS

    def embed_many(self, texts):
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_many_with_usage(self, texts):
        self.calls.append(list(texts))
        return EmbeddingResponse(
            embeddings=[self._vector(text) for text in texts],
            input_tokens=len(texts) * 7,
            retry_count=0,
        )


class ReindexEmbeddingDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_check_database_available(cls.database_url)):
            raise unittest.SkipTest(
                "로컬 DB에 연결할 수 없어 재색인 embedding 통합 테스트를 건너뜁니다."
            )

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(self.database_url)
        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )
        self.suffix = uuid.uuid4().hex[:8]
        self.source_url = f"https://docs.riido.io/embed-test-{self.suffix}.md"
        self.index_version_ids = []
        self.active_before = await self._active_index_ids()

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
            # 이 테스트가 INACTIVE 로 바꾼 기존 ACTIVE 버전을 되돌린다
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

    def _document(self, body: str) -> NormalizedDocument:
        content = f"# embedding 통합 테스트\n\n## 본문\n\n{body}"
        return NormalizedDocument(
            document_id=f"embed-test-{self.suffix}",
            title="embedding 통합 테스트",
            source_url=self.source_url,
            category="test",
            content=content,
            raw_content_uri=f"raw/embed-test-{self.suffix}.md",
            raw_content_hash=hashlib.sha256(
                f"raw {self.suffix} {body}".encode("utf-8")
            ).hexdigest(),
            normalized_content_hash=hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest(),
        )

    async def test_ingestion_stores_embeddings_and_second_run_reuses_them(
        self,
    ) -> None:
        document = self._document("첫 번째 본문")
        embedder = _CountingStubEmbedder()

        async with self.session_factory() as session:
            result = await run_reindex([document], embedder, session)
        self.index_version_ids.append(result.index_version.id)

        # 접수 단계에서 한 번만 호출하고 색인 단계에서는 채울 것이 없다
        self.assertEqual(1, len(embedder.calls))

        async with self.session_factory() as session:
            source = await session.scalar(
                select(DocumentSource).where(
                    DocumentSource.canonical_uri == self.source_url
                )
            )
            version = await session.scalar(
                select(DocumentVersion).where(
                    DocumentVersion.document_source_id == source.id
                )
            )
            embedding_count = await session.scalar(
                select(func.count())
                .select_from(ChunkEmbedding)
                .join(DocumentChunk, DocumentChunk.id == ChunkEmbedding.chunk_id)
                .join(ContentNode, ContentNode.id == DocumentChunk.id)
                .where(ContentNode.document_version_id == version.id)
            )
            ingestion_run = await session.scalar(
                select(IngestionRun).where(
                    IngestionRun.document_source_id == source.id
                )
            )
            model_calls = list(
                (
                    await session.execute(
                        select(ModelCall).where(
                            ModelCall.ingestion_run_id == ingestion_run.id
                        )
                    )
                ).scalars()
            )
            index_run = await session.scalar(
                select(IndexRun).where(
                    IndexRun.index_version_id == result.index_version.id
                )
            )
            index_version = await session.get(
                IndexVersion,
                result.index_version.id,
            )

        self.assertEqual(ExecutionStatus.SUCCESS, ingestion_run.status)
        self.assertEqual(DocumentVersionStatus.READY, version.status)
        self.assertEqual(result.chunk_count, embedding_count)
        self.assertEqual(1, len(model_calls))
        self.assertEqual(
            ModelCallPurpose.CHUNK_EMBEDDING,
            model_calls[0].purpose,
        )
        self.assertEqual(ExecutionStatus.SUCCESS, model_calls[0].status)
        self.assertEqual(ExecutionStatus.SUCCESS, index_run.status)
        self.assertEqual(IndexVersionStatus.ACTIVE, index_version.status)

        # 내용이 같은 문서를 다시 수집하면 embedding 을 새로 만들지 않는다
        second_embedder = _CountingStubEmbedder()
        async with self.session_factory() as session:
            second = await run_reindex([document], second_embedder, session)
        self.index_version_ids.append(second.index_version.id)

        self.assertEqual([], second_embedder.calls)

        async with self.session_factory() as session:
            reused_calls = await session.scalar(
                select(func.count())
                .select_from(ModelCall)
                .join(
                    IngestionRun,
                    IngestionRun.id == ModelCall.ingestion_run_id,
                )
                .where(IngestionRun.document_source_id == source.id)
            )
        # 재사용만으로 채워졌으므로 모델 호출 기록이 늘지 않는다
        self.assertEqual(1, reused_calls)


if __name__ == "__main__":
    unittest.main()
