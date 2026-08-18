import unittest
from unittest.mock import AsyncMock, Mock

from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.dml import Delete

from app.database.models import ChunkEmbedding, DocumentChunk
from retrieval.embedding import OPENAI_EMBEDDING_DIMENSIONS
from retrieval.models import RetrievalChunk
from retrieval.pgvector_store import PgVectorStore


class PgVectorStoreTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.session = AsyncMock(spec=AsyncSession)
        self.session.in_transaction.return_value = False
        self.transaction = AsyncMock()
        self.transaction.__aexit__.return_value = False
        self.session.begin.return_value = self.transaction
        self.store = PgVectorStore(self.session)
        self.chunk = RetrievalChunk(
            chunk_id="document-id:0",
            document_id="document-id",
            section_id="section-id",
            document_title="문서 제목",
            section_path=("문서 제목", "상위 섹션", "하위 섹션"),
            source_url="https://docs.riido.io/document.md",
            category="guide",
            content="Chunk 본문",
        )
        self.embedding = [0.1] * OPENAI_EMBEDDING_DIMENSIONS

    async def test_injects_async_session(self) -> None:
        self.assertIs(self.session, self.store._session)

    async def test_replaces_existing_corpus_with_orm_models(self) -> None:
        events = []

        async def record_delete(statement: Delete) -> Mock:
            events.append("delete")
            return Mock()

        def record_add_all(models: list[object]) -> None:
            events.append(type(models[0]).__name__)

        async def record_flush() -> None:
            events.append("flush")

        self.session.execute.side_effect = record_delete
        self.session.add_all.side_effect = record_add_all
        self.session.flush.side_effect = record_flush

        await self.store.replace_all([(self.chunk, self.embedding)])

        delete_statement = self.session.execute.await_args.args[0]
        self.assertIsInstance(delete_statement, Delete)
        self.assertEqual(DocumentChunk.__table__, delete_statement.table)

        self.assertEqual(2, self.session.add_all.call_count)
        stored_chunk = self.session.add_all.call_args_list[0].args[0][0]
        stored_embedding = self.session.add_all.call_args_list[1].args[0][0]

        self.assertIsInstance(stored_chunk, DocumentChunk)
        self.assertEqual(self.chunk.chunk_id, stored_chunk.chunk_id)
        self.assertEqual(self.chunk.document_id, stored_chunk.document_id)
        self.assertEqual(self.chunk.section_id, stored_chunk.section_id)
        self.assertEqual(self.chunk.document_title, stored_chunk.document_title)
        self.assertEqual(list(self.chunk.section_path), stored_chunk.section_path)
        self.assertEqual(self.chunk.source_url, stored_chunk.source_url)
        self.assertEqual(self.chunk.category, stored_chunk.category)
        self.assertEqual(self.chunk.content, stored_chunk.content)

        self.assertIsInstance(stored_embedding, ChunkEmbedding)
        self.assertEqual(self.chunk.chunk_id, stored_embedding.chunk_id)
        self.assertEqual(self.embedding, stored_embedding.embedding)
        self.assertEqual(2, self.session.flush.await_count)
        self.assertEqual(
            [
                "delete",
                "DocumentChunk",
                "flush",
                "ChunkEmbedding",
                "flush",
            ],
            events,
        )

    async def test_owns_transaction_when_session_has_no_active_transaction(self) -> None:
        await self.store.replace_all([(self.chunk, self.embedding)])

        self.session.begin.assert_called_once_with()
        self.transaction.__aenter__.assert_awaited_once_with()
        self.transaction.__aexit__.assert_awaited_once_with(None, None, None)

    async def test_uses_callers_active_transaction_without_nesting(self) -> None:
        self.session.in_transaction.return_value = True

        await self.store.replace_all([(self.chunk, self.embedding)])

        self.session.begin.assert_not_called()
        self.session.execute.assert_awaited_once()
        self.assertEqual(2, self.session.flush.await_count)

    async def test_propagates_failure_inside_owned_transaction(self) -> None:
        failure = RuntimeError("embedding insert failed")
        self.session.flush.side_effect = [None, failure]

        with self.assertRaisesRegex(RuntimeError, "embedding insert failed"):
            await self.store.replace_all([(self.chunk, self.embedding)])

        exit_args = self.transaction.__aexit__.await_args.args
        self.assertIs(RuntimeError, exit_args[0])
        self.assertIs(failure, exit_args[1])

    async def test_rejects_empty_or_invalid_reindex_input_before_deleting(self) -> None:
        with self.assertRaisesRegex(ValueError, "하나 이상"):
            await self.store.replace_all([])
        with self.assertRaisesRegex(ValueError, "1536차원"):
            await self.store.replace_all([(self.chunk, [0.1])])

        self.session.execute.assert_not_awaited()
        self.session.begin.assert_not_called()

    async def test_builds_exact_cosine_similarity_query(self) -> None:
        result = Mock()
        result.all.return_value = []
        self.session.execute.return_value = result

        await self.store.similarity_search(self.embedding, top_k=3)

        statement = self.session.execute.await_args.args[0]
        compiled = statement.compile(dialect=postgresql.dialect())
        sql = " ".join(str(compiled).split())

        self.assertIn("JOIN chunk_embeddings", sql)
        self.assertIn("chunk_embeddings.embedding <=>", sql)
        self.assertIn("ORDER BY cosine_distance ASC", sql)
        self.assertIn("LIMIT", sql)
        self.assertIn(3, compiled.params.values())

    async def test_restores_retrieval_chunk_and_converts_distance_to_similarity(
        self,
    ) -> None:
        stored_chunk = DocumentChunk(
            chunk_id=self.chunk.chunk_id,
            document_id=self.chunk.document_id,
            section_id=self.chunk.section_id,
            document_title=self.chunk.document_title,
            section_path=list(self.chunk.section_path),
            source_url=self.chunk.source_url,
            category=self.chunk.category,
            content=self.chunk.content,
        )
        result = Mock()
        result.all.return_value = [(stored_chunk, 0.25)]
        self.session.execute.return_value = result

        results = await self.store.similarity_search(self.embedding)

        restored_chunk, score = results[0]
        self.assertEqual(self.chunk, restored_chunk)
        self.assertIsInstance(restored_chunk.section_path, tuple)
        self.assertAlmostEqual(0.75, score)

    async def test_validates_similarity_search_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k"):
            await self.store.similarity_search(self.embedding, top_k=0)
        with self.assertRaisesRegex(ValueError, "1536차원"):
            await self.store.similarity_search([0.1])

        self.session.execute.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
