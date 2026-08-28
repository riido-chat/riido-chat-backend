"""신규 ERD 전체 재색인과 ACTIVE Retrieval의 로컬 DB 통합 테스트."""

import asyncio
import hashlib
import unittest
import uuid
from dataclasses import replace

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.config import get_settings
from app.database.models import (
    ChunkEmbedding,
    ContentNode,
    DocumentChunk,
    DocumentVersion,
    ExecutionStatus,
    IngestionRun,
    IndexDocument,
    IndexRun,
    IndexVersion,
    IndexVersionStatus,
)
from pipeline.document.models import NormalizedDocument
from pipeline.document.section_parser import create_section_identity_hash
from retrieval.bm25_retriever import BM25Retriever
from retrieval.embedding import OPENAI_EMBEDDING_DIMENSIONS, EmbeddingResponse
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import RetrievalChunk
from retrieval.pgvector_store import PgVectorStore
from retrieval.vector_retriever import (
    QUERY_EMBEDDING_TIMEOUT_SECONDS,
    VectorRetriever,
)


class _StubEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1] * OPENAI_EMBEDDING_DIMENSIONS

    def embed_many_with_usage(
        self,
        texts,
        *,
        sdk_max_retries=None,
        timeout=None,
    ) -> EmbeddingResponse:
        if sdk_max_retries != 0:
            raise AssertionError("Query Embedding SDK retry가 비활성화되지 않았습니다.")
        if timeout != QUERY_EMBEDDING_TIMEOUT_SECONDS:
            raise AssertionError("Query Embedding timeout이 30초가 아닙니다.")
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
        documents = [
            self._source_document(suffix, document=1),
            self._source_document(suffix, document=2),
        ]

        first_index = await self.store.replace_all(items, documents)

        self.assertEqual(IndexVersionStatus.ACTIVE, first_index.status)
        self.assertEqual(first_index.id, await self.store.get_active_index_version_id())
        await self._assert_active_chain(first_index, chunks)
        await self._assert_successful_run_logs(first_index, document_count=2)
        first_node_hashes = await self._active_node_hashes(first_index)

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

        changed_chunks = [
            replace(chunks[0], content="수정된 본문"),
            *chunks[1:],
        ]
        changed_items = [
            (chunk, [0.1 + index * 0.1] * OPENAI_EMBEDDING_DIMENSIONS)
            for index, chunk in enumerate(changed_chunks)
        ]
        changed_documents = [
            self._source_document(suffix, document=1, revision="changed"),
            documents[1],
        ]

        second_index = await self.store.replace_all(
            changed_items,
            changed_documents,
        )

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
        await self._assert_active_chain(second_index, changed_chunks)
        await self._assert_successful_run_logs(second_index, document_count=2)
        second_node_hashes = await self._active_node_hashes(second_index)
        changed_section_id = chunks[0].section_id
        self.assertEqual(
            first_node_hashes[changed_section_id][0],
            second_node_hashes[changed_section_id][0],
        )
        self.assertNotEqual(
            first_node_hashes[changed_section_id][1],
            second_node_hashes[changed_section_id][1],
        )

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

        document_versions = list(
            (
                await self.session.execute(
                    select(DocumentVersion)
                    .join(
                        IndexDocument,
                        IndexDocument.document_version_id == DocumentVersion.id,
                    )
                    .where(IndexDocument.index_version_id == index_version.id)
                )
            ).scalars()
        )
        self.assertTrue(
            all(
                version.raw_content_uri.startswith("raw/")
                for version in document_versions
            )
        )
        self.assertTrue(
            all(
                version.raw_content_hash != version.normalized_content_hash
                for version in document_versions
            )
        )

        content_nodes = list(
            (
                await self.session.execute(
                    select(ContentNode)
                    .join(
                        IndexDocument,
                        IndexDocument.document_version_id
                        == ContentNode.document_version_id,
                    )
                    .where(IndexDocument.index_version_id == index_version.id)
                )
            ).scalars()
        )
        self.assertTrue(
            all(node.node_identity_kind == "path" for node in content_nodes)
        )
        for node in content_nodes:
            self.assertEqual(
                create_section_identity_hash(
                    node.metadata_["document_id"],
                    tuple(node.metadata_["section_path"][1:]),
                ),
                node.node_identity_hash,
            )

    async def _active_node_hashes(
        self,
        index_version: IndexVersion,
    ) -> dict[str, tuple[str, str]]:
        nodes = list(
            (
                await self.session.execute(
                    select(ContentNode)
                    .join(
                        IndexDocument,
                        IndexDocument.document_version_id
                        == ContentNode.document_version_id,
                    )
                    .where(IndexDocument.index_version_id == index_version.id)
                )
            ).scalars()
        )
        return {
            node.metadata_["section_id"]: (
                node.node_identity_hash,
                node.content_hash,
            )
            for node in nodes
        }

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

    @staticmethod
    def _source_document(
        suffix: str,
        *,
        document: int,
        revision: str = "initial",
    ) -> NormalizedDocument:
        document_id = f"document-{suffix}-{document}"
        normalized_content = f"# 문서 {document}\n\n{revision} 정제 본문"
        return NormalizedDocument(
            document_id=document_id,
            title=f"문서 {document}",
            source_url=f"https://docs.riido.io/test/{suffix}/{document}.md",
            category="guide",
            content=normalized_content,
            raw_content_uri=f"raw/test/{suffix}/{document}.md",
            raw_content_hash=hashlib.sha256(
                f"{revision} raw {document}".encode("utf-8")
            ).hexdigest(),
            normalized_content_hash=hashlib.sha256(
                normalized_content.encode("utf-8")
            ).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
