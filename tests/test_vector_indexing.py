import hashlib
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock, patch

from app.document.models import NormalizedDocument
from app.retrieval.embedding import OPENAI_EMBEDDING_DIMENSIONS
from app.indexing.index_vector_corpus import (
    ReindexResult,
    build_index_items,
    main,
    reindex_vector_corpus,
    run_reindex,
)
from app.retrieval.models import RetrievalChunk


class VectorIndexItemTest(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [self._chunk(0), self._chunk(1), self._chunk(2)]
        self.embeddings = [
            [float(index)] * OPENAI_EMBEDDING_DIMENSIONS
            for index in range(len(self.chunks))
        ]
        self.embedder = Mock()

    def test_builds_embedding_items_for_every_chunk_in_order(self) -> None:
        self.embedder.embed_many.return_value = self.embeddings

        items = build_index_items(self.chunks, self.embedder)

        self.embedder.embed_many.assert_called_once_with(
            [
                "문서 0\n섹션 0\n본문 0",
                "문서 1\n섹션 1\n본문 1",
                "문서 2\n섹션 2\n본문 2",
            ]
        )
        self.embedder.embed.assert_not_called()
        self.assertEqual(self.chunks, [chunk for chunk, _ in items])
        self.assertEqual(self.embeddings, [embedding for _, embedding in items])

    def test_skips_embedding_when_nothing_is_missing(self) -> None:
        # embedding은 접수 시점에 만들어지므로 색인 단계에서 채울 것이 없는 상태가 정상이다
        self.assertEqual([], build_index_items([], self.embedder))
        self.embedder.embed_many.assert_not_called()

    def test_propagates_multi_input_embedding_failure(self) -> None:
        failure = RuntimeError("embedding unavailable")
        self.embedder.embed_many.side_effect = failure

        with self.assertRaises(RuntimeError) as context:
            build_index_items(self.chunks, self.embedder)

        self.assertIs(failure, context.exception)
        self.embedder.embed_many.assert_called_once()

    def test_rejects_mismatched_embedding_count(self) -> None:
        self.embedder.embed_many.return_value = self.embeddings[:-1]

        with self.assertRaisesRegex(RuntimeError, "개수가 Chunk 개수"):
            build_index_items(self.chunks, self.embedder)

    @staticmethod
    def _chunk(index: int) -> RetrievalChunk:
        return RetrievalChunk(
            document_id=f"document-{index}",
            section_id=f"section-{index}",
            document_title=f"문서 {index}",
            section_path=(f"문서 {index}", f"섹션 {index}"),
            source_url=f"https://docs.riido.io/document-{index}.md",
            category="guide",
            content=f"본문 {index}",
        )


class VectorReindexRunnerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        normalized_content = "# 문서 1\n\n## 섹션 1\n\n본문 1"
        self.document = NormalizedDocument(
            document_id="document-1",
            title="문서 1",
            source_url="https://docs.riido.io/document-1.md",
            category="guide",
            content=normalized_content,
            raw_content_uri="raw/document-1.md",
            raw_content_hash=hashlib.sha256(b"raw document 1").hexdigest(),
            normalized_content_hash=hashlib.sha256(
                normalized_content.encode("utf-8")
            ).hexdigest(),
        )
        self.chunk = VectorIndexItemTest._chunk(1)
        self.persisted_chunk = replace(
            self.chunk,
            chunk_id=101,
            document_version_id=201,
        )
        self.embedding = [0.1] * OPENAI_EMBEDDING_DIMENSIONS
        self.embedder = Mock()
        self.embedder.embed_many.return_value = [self.embedding]
        self.session = AsyncMock()

    @patch("app.indexing.index_vector_corpus.prepare_chunk_embeddings")
    @patch("app.indexing.index_vector_corpus.build_document_retrieval_chunks")
    @patch("app.indexing.index_vector_corpus.IndexWriter")
    @patch("app.indexing.index_vector_corpus.DocumentStore")
    async def test_commits_each_checkpoint_and_finishes_success(
        self,
        store_class: Mock,
        writer_class: Mock,
        build_chunks: Mock,
        prepare_embeddings: Mock,
    ) -> None:
        prepare_embeddings.return_value = [self.embedding]
        store = store_class.return_value
        writer = writer_class.return_value
        store.start_ingestion = AsyncMock(return_value=SimpleNamespace(id=11))
        store.complete_ingestion = AsyncMock(
            return_value=[self.persisted_chunk]
        )
        writer.start_index = AsyncMock(return_value=SimpleNamespace(id=21))
        writer.list_chunks_missing_embedding = AsyncMock(return_value=[])
        writer.store_index_items = AsyncMock()
        index_version = SimpleNamespace(id=77)
        writer.mark_index_ready = AsyncMock()
        writer.apply_index = AsyncMock(return_value=index_version)
        writer.finish_apply_run = AsyncMock()
        build_chunks.return_value = [self.chunk]

        result = await run_reindex(
            [self.document],
            self.embedder,
            self.session,
        )

        # 수집 2회, 색인 시작·저장·READY·적용으로 6회다
        self.assertEqual(6, self.session.commit.await_count)
        self.session.rollback.assert_not_awaited()
        store.start_ingestion.assert_awaited_once_with(
            self.document,
            group_id=ANY,
        )
        store.complete_ingestion.assert_awaited_once_with(
            11,
            self.document,
            [self.chunk],
            [self.embedding],
        )
        writer.start_index.assert_awaited_once_with(
            [self.persisted_chunk],
            group_id=ANY,
        )
        writer.store_index_items.assert_awaited_once()
        writer.mark_index_ready.assert_awaited_once_with(21)
        writer.apply_index.assert_awaited_once_with(21)
        self.assertIs(index_version, result.index_version)
        self.assertEqual(1, result.document_count)
        self.assertEqual(1, result.chunk_count)

    @patch("app.indexing.index_vector_corpus.prepare_chunk_embeddings")
    @patch("app.indexing.index_vector_corpus.build_document_retrieval_chunks")
    @patch("app.indexing.index_vector_corpus.IndexWriter")
    @patch("app.indexing.index_vector_corpus.DocumentStore")
    async def test_marks_ingestion_failed_when_document_pipeline_fails(
        self,
        store_class: Mock,
        writer_class: Mock,
        build_chunks: Mock,
        prepare_embeddings: Mock,
    ) -> None:
        prepare_embeddings.return_value = [self.embedding]
        failure = RuntimeError("parser unavailable")
        store = store_class.return_value
        writer = writer_class.return_value
        store.start_ingestion = AsyncMock(return_value=SimpleNamespace(id=11))
        store.fail_ingestion = AsyncMock()
        build_chunks.side_effect = failure

        with self.assertRaises(RuntimeError) as context:
            await run_reindex([self.document], self.embedder, self.session)

        self.assertIs(failure, context.exception)
        store.fail_ingestion.assert_awaited_once_with(11, failure)
        writer.start_index.assert_not_called()
        self.assertEqual(2, self.session.commit.await_count)
        self.session.rollback.assert_awaited_once_with()

    @patch("app.indexing.index_vector_corpus.prepare_chunk_embeddings")
    @patch("app.indexing.index_vector_corpus.build_document_retrieval_chunks")
    @patch("app.indexing.index_vector_corpus.IndexWriter")
    @patch("app.indexing.index_vector_corpus.DocumentStore")
    async def test_marks_index_failed_when_embedding_fails(
        self,
        store_class: Mock,
        writer_class: Mock,
        build_chunks: Mock,
        prepare_embeddings: Mock,
    ) -> None:
        prepare_embeddings.return_value = [self.embedding]
        failure = RuntimeError("embedding unavailable")
        self.embedder.embed_many.side_effect = failure
        store = store_class.return_value
        writer = writer_class.return_value
        store.start_ingestion = AsyncMock(return_value=SimpleNamespace(id=11))
        store.complete_ingestion = AsyncMock(
            return_value=[self.persisted_chunk]
        )
        writer.start_index = AsyncMock(return_value=SimpleNamespace(id=21))
        writer.list_chunks_missing_embedding = AsyncMock(
            return_value=[self.persisted_chunk]
        )
        writer.fail_index = AsyncMock()
        build_chunks.return_value = [self.chunk]

        with self.assertRaises(RuntimeError) as context:
            await run_reindex([self.document], self.embedder, self.session)

        self.assertIs(failure, context.exception)
        writer.fail_index.assert_awaited_once_with(
            21,
            failure,
            failed_stage="EMBEDDING",
        )
        writer.store_index_items.assert_not_called()
        self.assertEqual(4, self.session.commit.await_count)
        self.session.rollback.assert_awaited_once_with()

    @patch("app.indexing.index_vector_corpus.prepare_chunk_embeddings")
    @patch("app.indexing.index_vector_corpus.build_document_retrieval_chunks")
    @patch("app.indexing.index_vector_corpus.IndexWriter")
    @patch("app.indexing.index_vector_corpus.DocumentStore")
    async def test_marks_index_failed_when_validation_fails(
        self,
        store_class: Mock,
        writer_class: Mock,
        build_chunks: Mock,
        prepare_embeddings: Mock,
    ) -> None:
        prepare_embeddings.return_value = [self.embedding]
        failure = RuntimeError("stored count mismatch")
        store = store_class.return_value
        writer = writer_class.return_value
        store.start_ingestion = AsyncMock(return_value=SimpleNamespace(id=11))
        store.complete_ingestion = AsyncMock(
            return_value=[self.persisted_chunk]
        )
        writer.start_index = AsyncMock(return_value=SimpleNamespace(id=21))
        writer.list_chunks_missing_embedding = AsyncMock(return_value=[])
        writer.store_index_items = AsyncMock()
        writer.mark_index_ready = AsyncMock(side_effect=failure)
        writer.fail_index = AsyncMock()
        build_chunks.return_value = [self.chunk]

        with self.assertRaises(RuntimeError) as context:
            await run_reindex([self.document], self.embedder, self.session)

        self.assertIs(failure, context.exception)
        writer.fail_index.assert_awaited_once_with(
            21,
            failure,
            failed_stage="VALIDATING",
        )
        self.assertEqual(5, self.session.commit.await_count)
        self.session.rollback.assert_awaited_once_with()


class VectorIndexMainTest(unittest.TestCase):
    @patch("app.indexing.index_vector_corpus.reindex_vector_corpus")
    @patch("app.indexing.index_vector_corpus.OpenAIEmbedder")
    @patch("app.indexing.index_vector_corpus.load_normalized_documents")
    def test_runs_reindex_and_prints_active_index(
        self,
        load_documents: Mock,
        embedder_class: Mock,
        reindex_corpus: AsyncMock,
    ) -> None:
        documents = [Mock(spec=NormalizedDocument)]
        load_documents.return_value = documents
        reindex_corpus.return_value = ReindexResult(
            index_version=SimpleNamespace(id=77),
            document_count=1,
            chunk_count=2,
        )

        with patch("builtins.print") as print_result:
            main()

        reindex_corpus.assert_awaited_once_with(
            documents,
            embedder_class.return_value,
        )
        print_result.assert_called_once_with(
            "Vector corpus indexing 완료: 2개 Chunk, ACTIVE index=77"
        )


class VectorCorpusReplacementTest(unittest.IsolatedAsyncioTestCase):
    @patch("app.indexing.index_vector_corpus.dispose_engine")
    @patch("app.indexing.index_vector_corpus.run_reindex")
    @patch("app.indexing.index_vector_corpus.get_session_factory")
    async def test_reuses_session_and_disposes_engine(
        self,
        get_session_factory: Mock,
        run: AsyncMock,
        dispose_engine: AsyncMock,
    ) -> None:
        session = AsyncMock()
        session_context = AsyncMock()
        session_context.__aenter__.return_value = session
        session_factory = Mock(return_value=session_context)
        get_session_factory.return_value = session_factory
        documents = [Mock(spec=NormalizedDocument)]
        embedder = Mock()
        expected = ReindexResult(
            index_version=SimpleNamespace(id=7),
            document_count=1,
            chunk_count=1,
        )
        run.return_value = expected

        result = await reindex_vector_corpus(documents, embedder)

        get_session_factory.assert_called_once_with()
        session_factory.assert_called_once_with()
        run.assert_awaited_once_with(documents, embedder, session)
        self.assertIs(expected, result)
        dispose_engine.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
