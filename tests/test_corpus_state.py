import unittest
from pathlib import Path

from app.rag.corpus_state import CorpusNotLoadedError, CorpusState
from retrieval.models import RetrievalChunk


class CorpusStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = CorpusState(Path("data"))
        self.chunks = [
            self._chunk(1, "document-1"),
            self._chunk(2, "document-1"),
            self._chunk(3, "document-2"),
        ]

    def test_starts_unloaded(self) -> None:
        snapshot = self.state.snapshot()

        self.assertFalse(self.state.is_loaded)
        self.assertFalse(snapshot.loaded)
        self.assertEqual(0, snapshot.chunk_count)
        self.assertEqual(0, snapshot.document_count)
        self.assertIsNone(snapshot.loaded_at)

    def test_get_retriever_raises_when_corpus_is_not_loaded(self) -> None:
        with self.assertRaises(CorpusNotLoadedError):
            self.state.get_retriever()

    def test_replace_rejects_empty_or_unpersisted_corpus(self) -> None:
        with self.assertRaisesRegex(ValueError, "적재된 Chunk"):
            self.state.replace([])

        unpersisted = RetrievalChunk(
            document_id="document",
            section_id="section",
            document_title="문서",
            section_path=("문서", "섹션"),
            source_url="https://docs.riido.io/test.md",
            category="guide",
            content="본문",
        )
        with self.assertRaisesRegex(ValueError, "index_version_id"):
            self.state.replace([unpersisted])

    def test_replace_builds_index_and_reports_counts(self) -> None:
        snapshot = self.state.replace(self.chunks)

        self.assertTrue(self.state.is_loaded)
        self.assertTrue(snapshot.loaded)
        self.assertEqual(3, snapshot.chunk_count)
        self.assertEqual(2, snapshot.document_count)
        self.assertIsNotNone(snapshot.loaded_at)
        self.assertEqual("index_version:7", snapshot.source)

    def test_replace_rejects_mixed_index_versions(self) -> None:
        mixed = [self.chunks[0], self._chunk(4, "document-2", index_version_id=8)]

        with self.assertRaisesRegex(ValueError, "유일"):
            self.state.replace(mixed)

    def test_replace_swaps_completed_bm25_index(self) -> None:
        self.state.replace(self.chunks)
        first_retriever = self.state.get_retriever()

        self.state.replace(self.chunks)

        self.assertIsNot(first_retriever, self.state.get_retriever())

    @staticmethod
    def _chunk(
        chunk_id: int,
        document_id: str,
        *,
        index_version_id: int = 7,
    ) -> RetrievalChunk:
        return RetrievalChunk(
            document_id=document_id,
            section_id=f"section-{chunk_id}",
            document_title=f"문서 {document_id}",
            section_path=(f"문서 {document_id}", f"섹션 {chunk_id}"),
            source_url=f"https://docs.riido.io/{document_id}.md",
            category="guide",
            content=f"본문 {chunk_id}",
            chunk_id=chunk_id,
            document_version_id=100 + chunk_id,
            index_version_id=index_version_id,
        )


if __name__ == "__main__":
    unittest.main()
