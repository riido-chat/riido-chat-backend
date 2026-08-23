import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.rag.corpus_state import CorpusNotLoadedError, CorpusState


MARKDOWN = """# 워크스페이스 안내

## 멤버 초대

워크스페이스에 팀원을 초대하는 방법을 안내합니다.

## 구독 결제

구독 요금제와 결제 수단을 변경할 수 있습니다.

## 스프린트 관리

반복 주기로 스프린트를 만들고 진행 상황을 확인합니다.
"""


class CorpusStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = TemporaryDirectory()
        self.corpus_dir = Path(self._directory.name)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_starts_unloaded_when_corpus_is_absent(self) -> None:
        state = CorpusState(self.corpus_dir)
        snapshot = state.snapshot()

        self.assertFalse(state.is_loaded)
        self.assertFalse(snapshot.loaded)
        self.assertEqual(0, snapshot.chunk_count)
        self.assertEqual(0, snapshot.document_count)
        self.assertIsNone(snapshot.loaded_at)

    def test_get_retriever_raises_when_corpus_is_not_loaded(self) -> None:
        state = CorpusState(self.corpus_dir)

        with self.assertRaises(CorpusNotLoadedError):
            state.get_retriever()

    def test_load_raises_when_manifest_is_missing(self) -> None:
        state = CorpusState(self.corpus_dir)

        with self.assertRaises(FileNotFoundError):
            state.load()

        self.assertFalse(state.is_loaded)

    def test_load_builds_index_and_reports_counts(self) -> None:
        self._write_corpus()
        state = CorpusState(self.corpus_dir)

        snapshot = state.load()

        self.assertTrue(state.is_loaded)
        self.assertTrue(snapshot.loaded)
        self.assertEqual(3, snapshot.chunk_count)
        self.assertEqual(1, snapshot.document_count)
        self.assertIsNotNone(snapshot.loaded_at)
        self.assertEqual(str(state.manifest_path), snapshot.source)

    def test_reload_replaces_previous_index(self) -> None:
        self._write_corpus()
        state = CorpusState(self.corpus_dir)
        state.load()
        first_retriever = state.get_retriever()

        state.load()

        self.assertIsNot(first_retriever, state.get_retriever())

    def _write_corpus(self) -> None:
        markdown_path = self.corpus_dir / "clean" / "guide.md"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(MARKDOWN, encoding="utf-8")

        manifest = [
            {
                "doc_id": "document-id",
                "title": "워크스페이스 안내",
                "url": "https://docs.riido.io/guide.md",
                "category": "guide",
                "path": "clean/guide.md",
            }
        ]
        (self.corpus_dir / "clean_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
