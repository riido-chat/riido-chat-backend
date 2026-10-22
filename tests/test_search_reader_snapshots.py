import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.retrieval.models import RetrievalChunk
from app.retrieval.search_reader import SearchReader


class _ScalarResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _RowsResult:
    def __init__(self, rows=()):
        self._rows = rows

    def all(self):
        return self._rows


class SearchReaderSnapshotTest(unittest.IsolatedAsyncioTestCase):
    async def test_exact_snapshot_is_read_even_after_it_is_inactive(self) -> None:
        session = SimpleNamespace(
            execute=AsyncMock(
                return_value=_ScalarResult(
                    [
                        SimpleNamespace(
                            id=41,
                            document_group_id=7,
                            status="INACTIVE",
                            activated_at=None,
                        )
                    ]
                )
            )
        )

        reader = SearchReader(
            session,
            document_group_id=7,
            index_version_id=41,
        )
        result = await reader._get_active_index_version()

        self.assertEqual(41, result.id)
        session.execute.assert_awaited_once()

    async def test_exact_snapshot_rejects_a_different_group(self) -> None:
        session = SimpleNamespace(
            execute=AsyncMock(return_value=_ScalarResult([]))
        )
        reader = SearchReader(
            session,
            document_group_id=8,
            index_version_id=41,
        )

        with self.assertRaises(RuntimeError):
            await reader._get_active_index_version()

    async def test_public_load_and_vector_paths_keep_group_snapshot_scope(self) -> None:
        chunk = RetrievalChunk(
            document_id="doc",
            section_id="section",
            document_title="문서",
            section_path=("문서",),
            source_url="https://docs.example/doc",
            category="guide",
            content="본문",
            chunk_id=1,
            document_version_id=2,
            index_version_id=71,
        )

        for group_id, index_id in ((7, 71), (8, 81)):
            with self.subTest(group_id=group_id):
                active = SimpleNamespace(
                    id=index_id,
                    document_group_id=group_id,
                    embedding_config_id=31,
                    chunking_config_id=41,
                    activated_at=None,
                )
                session = SimpleNamespace(
                    execute=AsyncMock(
                        side_effect=[
                            _ScalarResult([active]),
                            _RowsResult(),
                            _ScalarResult([active]),
                            _RowsResult(),
                        ]
                    )
                )
                reader = SearchReader(session, document_group_id=group_id)

                with patch.object(
                    SearchReader,
                    "_to_retrieval_chunk",
                    return_value=chunk,
                ):
                    self.assertEqual([], await reader.load_active_chunks())
                    self.assertEqual([], await reader.similarity_search([0.0] * 1536))

                self.assertEqual(group_id, reader._document_group_id)
                self.assertEqual(index_id, active.id)
                self.assertEqual(4, session.execute.await_count)


if __name__ == "__main__":
    unittest.main()
