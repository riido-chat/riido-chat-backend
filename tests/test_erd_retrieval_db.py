"""신규 ERD 전체 재색인과 ACTIVE Retrieval의 로컬 DB 통합 테스트."""

import asyncio
import unittest
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.config import get_settings
from app.database.models import (
    ChunkEmbedding,
    ContentNode,
    DocumentChunk,
    ExecutionStatus,
    IngestionRun,
    IndexDocument,
    IndexRun,
    IndexVersion,
    IndexVersionStatus,
)
from retrieval.bm25_retriever import BM25Retriever
from retrieval.embedding import OPENAI_EMBEDDING_DIMENSIONS, EmbeddingResponse
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import RetrievalChunk
from retrieval.pgvector_store import PgVectorStore
from retrieval.vector_retriever import VectorRetriever


class _StubEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1] * OPENAI_EMBEDDING_DIMENSIONS

    def embed_many_with_usage(
        self,
        texts,
        *,
        sdk_max_retries=None,
    ) -> EmbeddingResponse:
        if sdk_max_retries != 0:
            raise AssertionError("Query Embedding SDK retry가 비활성화되지 않았습니다.")
        return EmbeddingResponse(
            embeddings=[[0.1] * OPENAI_EMBEDDING_DIMENSIONS for _ in texts],
            input_tokens=len(texts),
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


class ErdRetrievalDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_check_database_available(cls.database_url)):
            raise unittest.SkipTest(
                "로컬 DB에 연결할 수 없어 신규 ERD Retrieval 통합 테스트를 건너뜁니다."
            )

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(self.database_url)
        self.connection = await self.engine.connect()
        self.transaction = await self.connection.begin()
        self.session = AsyncSession(bind=self.connection, expire_on_commit=False)
        self.store = PgVectorStore(self.session)

    async def asyncTearDown(self) -> None:
        await self.session.close()
        await self.transaction.rollback()
        await self.connection.close()
        await self.engine.dispose()

    async def test_reindexes_full_erd_chain_and_switches_single_active_index(
        self,
    ) -> None:
        suffix = uuid.uuid4().hex[:8]
        chunks = [
            self._source_chunk(suffix, document=1, section=1),
            self._source_chunk(suffix, document=1, section=2),
            self._source_chunk(suffix, document=2, section=1),
        ]
        items = [
            (chunk, [0.1 + index * 0.1] * OPENAI_EMBEDDING_DIMENSIONS)
            for index, chunk in enumerate(chunks)
        ]

        first_index = await self.store.replace_all(items)

        self.assertEqual(IndexVersionStatus.ACTIVE, first_index.status)
        self.assertEqual(first_index.id, await self.store.get_active_index_version_id())
        await self._assert_active_chain(first_index, chunks)
        await self._assert_successful_run_logs(first_index, document_count=2)

        active_chunks = await self.store.load_active_chunks()
        self.assertEqual(3, len(active_chunks))
        self.assertEqual(
            [chunk.section_id for chunk in chunks],
            [chunk.section_id for chunk in active_chunks],
        )
        self.assertTrue(all(isinstance(chunk.chunk_id, int) for chunk in active_chunks))
        self.assertTrue(
            all(isinstance(chunk.document_version_id, int) for chunk in active_chunks)
        )
        self.assertEqual(
            {first_index.id},
            {chunk.index_version_id for chunk in active_chunks},
        )

        vector_results = await self.store.similarity_search(
            [0.1] * OPENAI_EMBEDDING_DIMENSIONS,
            top_k=3,
        )
        self.assertEqual(3, len(vector_results))
        self.assertTrue(
            all(result[0].index_version_id == first_index.id for result in vector_results)
        )

        hybrid_results = await HybridRetriever(
            BM25Retriever(active_chunks),
            VectorRetriever(_StubEmbedder(), self.store),
        ).search("본문", top_k=3)
        self.assertEqual(3, len(hybrid_results))
        self.assertTrue(
            all(isinstance(result.chunk.chunk_id, int) for result in hybrid_results)
        )
        self.assertEqual(
            {first_index.id},
            {result.chunk.index_version_id for result in hybrid_results},
        )

        second_index = await self.store.replace_all(items)

        self.assertNotEqual(first_index.id, second_index.id)
        self.assertEqual(IndexVersionStatus.INACTIVE, first_index.status)
        self.assertEqual(IndexVersionStatus.ACTIVE, second_index.status)
        self.assertEqual(second_index.id, await self.store.get_active_index_version_id())
        active_count = await self.session.scalar(
            select(func.count())
            .select_from(IndexVersion)
            .where(IndexVersion.status == IndexVersionStatus.ACTIVE)
        )
        self.assertEqual(1, active_count)
        await self._assert_active_chain(second_index, chunks)
        await self._assert_successful_run_logs(second_index, document_count=2)

    async def _assert_active_chain(
        self,
        index_version: IndexVersion,
        source_chunks: list[RetrievalChunk],
    ) -> None:
        document_count = len({chunk.document_id for chunk in source_chunks})
        linked_document_count = await self.session.scalar(
            select(func.count())
            .select_from(IndexDocument)
            .where(IndexDocument.index_version_id == index_version.id)
        )
        self.assertEqual(document_count, linked_document_count)

        shared_primary_keys = (
            await self.session.execute(
                select(ContentNode.id, DocumentChunk.id)
                .join(DocumentChunk, DocumentChunk.id == ContentNode.id)
                .join(
                    IndexDocument,
                    IndexDocument.document_version_id
                    == ContentNode.document_version_id,
                )
                .where(IndexDocument.index_version_id == index_version.id)
            )
        ).all()
        self.assertEqual(len(source_chunks), len(shared_primary_keys))
        self.assertTrue(
            all(node_id == chunk_id for node_id, chunk_id in shared_primary_keys)
        )

        embedding_count = await self.session.scalar(
            select(func.count())
            .select_from(ChunkEmbedding)
            .join(DocumentChunk, DocumentChunk.id == ChunkEmbedding.chunk_id)
            .join(ContentNode, ContentNode.id == DocumentChunk.id)
            .join(
                IndexDocument,
                IndexDocument.document_version_id
                == ContentNode.document_version_id,
            )
            .where(
                IndexDocument.index_version_id == index_version.id,
                ChunkEmbedding.embedding_config_id
                == index_version.embedding_config_id,
            )
        )
        self.assertEqual(len(source_chunks), embedding_count)

    async def _assert_successful_run_logs(
        self,
        index_version: IndexVersion,
        *,
        document_count: int,
    ) -> None:
        index_run = await self.session.scalar(
            select(IndexRun).where(
                IndexRun.index_version_id == index_version.id
            )
        )
        self.assertEqual(ExecutionStatus.SUCCESS, index_run.status)
        self.assertEqual("ACTIVE", index_run.summary["stage"])
        self.assertIsNotNone(index_run.finished_at)

        ingestion_runs = list(
            (
                await self.session.execute(
                    select(IngestionRun)
                    .join(
                        IndexDocument,
                        IndexDocument.document_version_id
                        == IngestionRun.produced_version_id,
                    )
                    .where(IndexDocument.index_version_id == index_version.id)
                )
            ).scalars()
        )
        self.assertEqual(document_count, len(ingestion_runs))
        self.assertTrue(
            all(run.status == ExecutionStatus.SUCCESS for run in ingestion_runs)
        )
        self.assertTrue(
            all(run.produced_version_id is not None for run in ingestion_runs)
        )

    @staticmethod
    def _source_chunk(
        suffix: str,
        *,
        document: int,
        section: int,
    ) -> RetrievalChunk:
        document_id = f"document-{suffix}-{document}"
        document_title = f"문서 {document}"
        return RetrievalChunk(
            document_id=document_id,
            section_id=f"{document_id}:section-{section}",
            document_title=document_title,
            section_path=(document_title, f"섹션 {section}"),
            source_url=f"https://docs.riido.io/test/{suffix}/{document}.md",
            category="guide",
            content=f"문서 {document}의 본문 {section}",
        )


if __name__ == "__main__":
    unittest.main()
