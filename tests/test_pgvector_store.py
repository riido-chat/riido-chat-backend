import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    ContentNode,
    DocumentChunk,
    DocumentSource,
    DocumentVersion,
    IndexVersion,
)
from retrieval.embedding import OPENAI_EMBEDDING_DIMENSIONS
from retrieval.models import RetrievalChunk
from retrieval.pgvector_store import ActiveIndexNotFoundError, PgVectorStore


class PgVectorStoreTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.session = AsyncMock(spec=AsyncSession)
        self.session.in_transaction.return_value = False
        self.transaction = AsyncMock()
        self.transaction.__aexit__.return_value = False
        self.session.begin.return_value = self.transaction
        self.store = PgVectorStore(self.session)
        self.chunk = RetrievalChunk(
            document_id="document-id",
            section_id="section-id",
            document_title="문서 제목",
            section_path=("문서 제목", "상위 섹션", "하위 섹션"),
            source_url="https://docs.riido.io/document.md",
            category="guide",
            content="Chunk 본문",
        )
        self.embedding = [0.1] * OPENAI_EMBEDDING_DIMENSIONS
        self.active_index = SimpleNamespace(
            id=7,
            chunking_config_id=11,
            embedding_config_id=12,
        )

    async def test_injects_async_session(self) -> None:
        self.assertIs(self.session, self.store._session)

    async def test_owns_transaction_when_session_has_no_active_transaction(self) -> None:
        expected = Mock(spec=IndexVersion)

        with patch.object(
            self.store,
            "_replace_rows",
            new=AsyncMock(return_value=expected),
        ) as replace_rows:
            result = await self.store.replace_all([(self.chunk, self.embedding)])

        self.assertIs(expected, result)
        replace_rows.assert_awaited_once()
        self.session.begin.assert_called_once_with()
        self.transaction.__aenter__.assert_awaited_once_with()
        self.transaction.__aexit__.assert_awaited_once_with(None, None, None)

    async def test_uses_callers_active_transaction_without_nesting(self) -> None:
        self.session.in_transaction.return_value = True
        expected = Mock(spec=IndexVersion)

        with patch.object(
            self.store,
            "_replace_rows",
            new=AsyncMock(return_value=expected),
        ) as replace_rows:
            result = await self.store.replace_all([(self.chunk, self.embedding)])

        self.assertIs(expected, result)
        replace_rows.assert_awaited_once()
        self.session.begin.assert_not_called()

    async def test_propagates_failure_inside_owned_transaction(self) -> None:
        failure = RuntimeError("document chunk insert failed")

        with patch.object(
            self.store,
            "_replace_rows",
            new=AsyncMock(side_effect=failure),
        ):
            with self.assertRaises(RuntimeError) as context:
                await self.store.replace_all([(self.chunk, self.embedding)])

        self.assertIs(failure, context.exception)
        exit_args = self.transaction.__aexit__.await_args.args
        self.assertIs(RuntimeError, exit_args[0])
        self.assertIs(failure, exit_args[1])

    async def test_validates_reindex_input_before_transaction(self) -> None:
        with self.assertRaisesRegex(ValueError, "하나 이상"):
            await self.store.replace_all([])
        with self.assertRaisesRegex(ValueError, "1536차원"):
            await self.store.replace_all([(self.chunk, [0.1])])

        persisted = self._runtime_chunk()
        with self.assertRaisesRegex(ValueError, "DB 식별자"):
            await self.store.replace_all([(persisted, self.embedding)])

        with self.assertRaisesRegex(ValueError, "중복 section_id"):
            await self.store.replace_all(
                [(self.chunk, self.embedding), (self.chunk, self.embedding)]
            )

        self.session.begin.assert_not_called()

    async def test_gets_exactly_one_active_index_version(self) -> None:
        result = Mock()
        result.scalars.return_value.all.return_value = [self.active_index]
        self.session.execute.return_value = result

        index_version_id = await self.store.get_active_index_version_id()

        self.assertEqual(7, index_version_id)

    async def test_rejects_missing_or_multiple_active_index_versions(self) -> None:
        result = Mock()
        self.session.execute.return_value = result

        result.scalars.return_value.all.return_value = []
        with self.assertRaises(ActiveIndexNotFoundError):
            await self.store.get_active_index_version_id()

        result.scalars.return_value.all.return_value = [
            self.active_index,
            SimpleNamespace(id=8),
        ]
        with self.assertRaisesRegex(RuntimeError, "둘 이상"):
            await self.store.get_active_index_version_id()

    async def test_builds_new_erd_cosine_similarity_query(self) -> None:
        result = Mock()
        result.all.return_value = []
        self.session.execute.return_value = result

        with patch.object(
            self.store,
            "_get_active_index_version",
            new=AsyncMock(return_value=self.active_index),
        ):
            await self.store.similarity_search(self.embedding, top_k=3)

        statement = self.session.execute.await_args.args[0]
        compiled = statement.compile(dialect=postgresql.dialect())
        sql = " ".join(str(compiled).split())

        self.assertIn("JOIN content_nodes", sql)
        self.assertIn("JOIN document_versions", sql)
        self.assertIn("JOIN document_sources", sql)
        self.assertIn("JOIN index_documents", sql)
        self.assertIn("JOIN chunk_embeddings", sql)
        self.assertIn("chunk_embeddings.embedding <=>", sql)
        self.assertIn("ORDER BY cosine_distance ASC", sql)
        self.assertNotIn("legacy_", sql)
        self.assertIn(3, compiled.params.values())

    async def test_loads_active_chunks_with_runtime_identifiers(self) -> None:
        row = self._stored_row()
        result = Mock()
        result.all.return_value = [row]
        self.session.execute.return_value = result

        with patch.object(
            self.store,
            "_get_active_index_version",
            new=AsyncMock(return_value=self.active_index),
        ):
            chunks = await self.store.load_active_chunks()

        self.assertEqual([self._runtime_chunk()], chunks)
        statement = self.session.execute.await_args.args[0]
        sql = " ".join(
            str(statement.compile(dialect=postgresql.dialect())).split()
        )
        self.assertIn("JOIN index_documents", sql)
        self.assertIn("JOIN chunk_embeddings", sql)
        self.assertNotIn("legacy_", sql)

    async def test_restores_chunk_and_converts_distance_to_similarity(self) -> None:
        result = Mock()
        result.all.return_value = [(*self._stored_row(), 0.25)]
        self.session.execute.return_value = result

        with patch.object(
            self.store,
            "_get_active_index_version",
            new=AsyncMock(return_value=self.active_index),
        ):
            results = await self.store.similarity_search(self.embedding)

        restored_chunk, score = results[0]
        self.assertEqual(self._runtime_chunk(), restored_chunk)
        self.assertEqual(101, restored_chunk.chunk_id)
        self.assertEqual(201, restored_chunk.document_version_id)
        self.assertEqual(7, restored_chunk.index_version_id)
        self.assertAlmostEqual(0.75, score)

    async def test_validates_similarity_search_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k"):
            await self.store.similarity_search(self.embedding, top_k=0)
        with self.assertRaisesRegex(ValueError, "1536차원"):
            await self.store.similarity_search([0.1])

        self.session.execute.assert_not_awaited()

    def _stored_row(
        self,
    ) -> tuple[DocumentChunk, ContentNode, DocumentVersion, DocumentSource]:
        document_chunk = DocumentChunk(
            id=101,
            chunking_config_id=11,
            chunk_index=0,
        )
        content_node = ContentNode(
            id=101,
            document_version_id=201,
            node_type="SECTION",
            node_order=0,
            normalized_content="Chunk 본문",
            content_hash="hash",
            metadata_={
                "document_id": "document-id",
                "section_id": "section-id",
                "section_path": ["문서 제목", "상위 섹션", "하위 섹션"],
            },
        )
        document_version = DocumentVersion(id=201, document_source_id=301)
        document_source = DocumentSource(
            id=301,
            source_type="GITBOOK_MARKDOWN",
            canonical_uri="https://docs.riido.io/document.md",
            title="문서 제목",
            metadata_={"document_id": "document-id", "category": "guide"},
        )
        return document_chunk, content_node, document_version, document_source

    @staticmethod
    def _runtime_chunk() -> RetrievalChunk:
        return RetrievalChunk(
            document_id="document-id",
            section_id="section-id",
            document_title="문서 제목",
            section_path=("문서 제목", "상위 섹션", "하위 섹션"),
            source_url="https://docs.riido.io/document.md",
            category="guide",
            content="Chunk 본문",
            chunk_id=101,
            document_version_id=201,
            index_version_id=7,
        )


if __name__ == "__main__":
    unittest.main()
