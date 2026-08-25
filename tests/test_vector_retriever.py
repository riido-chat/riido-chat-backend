import unittest
from unittest.mock import AsyncMock, Mock

from retrieval.embedding import (
    OPENAI_EMBEDDING_DIMENSIONS,
    OPENAI_EMBEDDING_MODEL,
    EmbeddingResponse,
    OpenAIEmbedder,
)
from retrieval.models import RetrievalChunk, RetrievalResult
from retrieval.pgvector_store import PgVectorStore
from retrieval.vector_retriever import VectorRetriever


class VectorRetrieverTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.embedder = Mock(spec=OpenAIEmbedder)
        self.store = AsyncMock(spec=PgVectorStore)
        self.retriever = VectorRetriever(self.embedder, self.store)
        self.embedding = [0.1] * OPENAI_EMBEDDING_DIMENSIONS
        self.embedder.embed_many_with_usage.return_value = EmbeddingResponse(
            embeddings=[self.embedding],
            input_tokens=7,
        )
        self.chunks = [self._chunk(0), self._chunk(1)]

    async def test_injects_embedder_and_store(self) -> None:
        self.assertIs(self.embedder, self.retriever._embedder)
        self.assertIs(self.store, self.retriever._store)

    async def test_searches_original_query_and_builds_ranked_results(self) -> None:
        self.store.similarity_search.return_value = [
            (self.chunks[1], 0.91),
            (self.chunks[0], 0.73),
        ]

        results = await self.retriever.search("사용자 원문 질문", top_k=2)

        self.embedder.embed_many_with_usage.assert_called_once_with(
            ["사용자 원문 질문"]
        )
        self.store.similarity_search.assert_awaited_once_with(
            self.embedding,
            2,
        )
        self.assertTrue(all(isinstance(result, RetrievalResult) for result in results))
        self.assertEqual(
            [self.chunks[1], self.chunks[0]],
            [result.chunk for result in results],
        )
        self.assertEqual([0.91, 0.73], [result.score for result in results])
        self.assertEqual([1, 2], [result.rank for result in results])

    async def test_returns_empty_list_when_store_has_no_matches(self) -> None:
        self.store.similarity_search.return_value = []

        results = await self.retriever.search("검색 결과 없는 질문")

        self.assertEqual([], results)
        self.embedder.embed_many_with_usage.assert_called_once_with(
            ["검색 결과 없는 질문"]
        )
        self.store.similarity_search.assert_awaited_once_with(
            self.embedding,
            10,
        )

    async def test_returns_empty_list_without_dependencies_for_blank_query(self) -> None:
        for query in ("", "   "):
            with self.subTest(query=query):
                self.assertEqual([], await self.retriever.search(query))

        self.embedder.embed_many_with_usage.assert_not_called()
        self.store.similarity_search.assert_not_awaited()

    async def test_rejects_invalid_top_k_before_embedding(self) -> None:
        for top_k in (0, -1):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, "top_k"):
                    await self.retriever.search("질문", top_k=top_k)

        self.embedder.embed_many_with_usage.assert_not_called()
        self.store.similarity_search.assert_not_awaited()

    async def test_propagates_embedding_failure_without_searching_store(self) -> None:
        failure = RuntimeError("embedding unavailable")
        self.embedder.embed_many_with_usage.side_effect = failure

        with self.assertRaisesRegex(RuntimeError, "embedding unavailable"):
            await self.retriever.search("질문")

        self.store.similarity_search.assert_not_awaited()

    async def test_propagates_store_failure(self) -> None:
        failure = RuntimeError("database unavailable")
        self.store.similarity_search.side_effect = failure

        with self.assertRaisesRegex(RuntimeError, "database unavailable"):
            await self.retriever.search("질문")

        self.embedder.embed_many_with_usage.assert_called_once_with(["질문"])

    async def test_trace_records_successful_embedding_call(self) -> None:
        self.store.similarity_search.return_value = [(self.chunks[0], 0.9)]

        call = await self.retriever.search_with_trace("질문")

        self.assertIsNone(call.error)
        self.assertEqual(1, len(call.results))
        self.assertIsNotNone(call.embedding_call)
        self.assertTrue(call.embedding_call.succeeded)
        self.assertEqual(7, call.embedding_call.input_tokens)
        self.assertEqual(OPENAI_EMBEDDING_MODEL, call.embedding_call.model_name)
        self.assertIsNone(call.embedding_call.error_message)

    async def test_trace_keeps_failed_embedding_call_for_logging(self) -> None:
        self.embedder.embed_many_with_usage.side_effect = RuntimeError(
            "embedding unavailable"
        )

        call = await self.retriever.search_with_trace("질문")

        self.assertIsInstance(call.error, RuntimeError)
        self.assertEqual((), call.results)
        self.assertFalse(call.embedding_call.succeeded)
        self.assertIn("embedding unavailable", call.embedding_call.error_message)
        self.store.similarity_search.assert_not_awaited()

    async def test_trace_keeps_embedding_call_when_store_fails(self) -> None:
        self.store.similarity_search.side_effect = RuntimeError("database unavailable")

        call = await self.retriever.search_with_trace("질문")

        self.assertIsInstance(call.error, RuntimeError)
        self.assertTrue(call.embedding_call.succeeded)

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
            chunk_id=index + 1,
            document_version_id=index + 101,
            index_version_id=1,
        )


if __name__ == "__main__":
    unittest.main()
