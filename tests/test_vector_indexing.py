import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from retrieval.embedding import OPENAI_EMBEDDING_DIMENSIONS
from retrieval.index_vector_corpus import (
    build_index_items,
    main,
    replace_vector_corpus,
)
from retrieval.models import RetrievalChunk


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

    def test_rejects_empty_corpus_without_embedding(self) -> None:
        with self.assertRaisesRegex(ValueError, "하나 이상"):
            build_index_items([], self.embedder)

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


class VectorIndexMainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [VectorIndexItemTest._chunk(0), VectorIndexItemTest._chunk(1)]
        self.embeddings = [
            [float(index)] * OPENAI_EMBEDDING_DIMENSIONS
            for index in range(len(self.chunks))
        ]

    @patch("retrieval.index_vector_corpus.replace_vector_corpus")
    @patch("retrieval.index_vector_corpus.OpenAIEmbedder")
    @patch("retrieval.index_vector_corpus.build_retrieval_chunks")
    def test_stores_only_after_all_embeddings_succeed(
        self,
        build_chunks: Mock,
        embedder_class: Mock,
        replace_corpus: AsyncMock,
    ) -> None:
        events = []

        def embed_many(_: object) -> list[list[float]]:
            events.append("embed_many")
            return self.embeddings

        async def replace(_: object) -> None:
            events.append("replace")
            return SimpleNamespace(id=77)

        build_chunks.return_value = self.chunks
        embedder = embedder_class.return_value
        embedder.embed_many.side_effect = embed_many
        replace_corpus.side_effect = replace

        with patch("builtins.print") as print_result:
            main()

        replace_corpus.assert_awaited_once()
        items = replace_corpus.await_args.args[0]
        embedder.embed_many.assert_called_once_with(
            [
                "문서 0\n섹션 0\n본문 0",
                "문서 1\n섹션 1\n본문 1",
            ]
        )
        embedder.embed.assert_not_called()
        self.assertEqual(
            list(zip(self.chunks, self.embeddings)),
            items,
        )
        self.assertEqual(["embed_many", "replace"], events)
        print_result.assert_called_once_with(
            "Vector corpus indexing 완료: 2개 Chunk, ACTIVE index=77"
        )

    @patch("retrieval.index_vector_corpus.replace_vector_corpus")
    @patch("retrieval.index_vector_corpus.OpenAIEmbedder")
    @patch("retrieval.index_vector_corpus.build_retrieval_chunks")
    def test_does_not_enter_db_stage_when_embedding_fails(
        self,
        build_chunks: Mock,
        embedder_class: Mock,
        replace_corpus: AsyncMock,
    ) -> None:
        failure = RuntimeError("embedding unavailable")
        build_chunks.return_value = self.chunks
        embedder_class.return_value.embed_many.side_effect = failure

        with self.assertRaises(RuntimeError) as context:
            main()

        self.assertIs(failure, context.exception)
        replace_corpus.assert_not_awaited()

    @patch("retrieval.index_vector_corpus.replace_vector_corpus")
    @patch("retrieval.index_vector_corpus.OpenAIEmbedder")
    @patch("retrieval.index_vector_corpus.build_retrieval_chunks")
    def test_does_not_enter_db_stage_when_embedding_count_mismatches(
        self,
        build_chunks: Mock,
        embedder_class: Mock,
        replace_corpus: AsyncMock,
    ) -> None:
        build_chunks.return_value = self.chunks
        embedder_class.return_value.embed_many.return_value = [
            self.embeddings[0]
        ]

        with self.assertRaisesRegex(RuntimeError, "개수가 Chunk 개수"):
            main()

        replace_corpus.assert_not_awaited()

    @patch("retrieval.index_vector_corpus.replace_vector_corpus")
    @patch("retrieval.index_vector_corpus.OpenAIEmbedder")
    @patch("retrieval.index_vector_corpus.build_retrieval_chunks")
    def test_rejects_empty_canonical_corpus_before_dependencies(
        self,
        build_chunks: Mock,
        embedder_class: Mock,
        replace_corpus: AsyncMock,
    ) -> None:
        build_chunks.return_value = []

        with self.assertRaisesRegex(ValueError, "하나 이상"):
            main()

        embedder_class.assert_not_called()
        replace_corpus.assert_not_awaited()


class VectorCorpusReplacementTest(unittest.IsolatedAsyncioTestCase):
    @patch("retrieval.index_vector_corpus.dispose_engine")
    @patch("retrieval.index_vector_corpus.PgVectorStore")
    @patch("retrieval.index_vector_corpus.get_session_factory")
    async def test_reuses_session_store_and_disposes_engine(
        self,
        get_session_factory: Mock,
        store_class: Mock,
        dispose_engine: AsyncMock,
    ) -> None:
        session = AsyncMock()
        session_context = AsyncMock()
        session_context.__aenter__.return_value = session
        session_factory = Mock(return_value=session_context)
        get_session_factory.return_value = session_factory
        items = [(VectorIndexItemTest._chunk(0), [0.1] * 1536)]
        store = store_class.return_value
        expected = SimpleNamespace(id=7)
        store.replace_all = AsyncMock(return_value=expected)

        result = await replace_vector_corpus(items)

        get_session_factory.assert_called_once_with()
        session_factory.assert_called_once_with()
        store_class.assert_called_once_with(session)
        store.replace_all.assert_awaited_once_with(items)
        self.assertIs(expected, result)
        dispose_engine.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
