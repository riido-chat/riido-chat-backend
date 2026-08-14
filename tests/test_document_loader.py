import json
import tempfile
import unittest
from pathlib import Path

from pipeline.document.loader import load_normalized_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NormalizedDocumentLoaderTest(unittest.TestCase):
    def test_loads_current_canonical_documents(self) -> None:
        manifest_path = PROJECT_ROOT / "data/clean_manifest.json"
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))

        documents = load_normalized_documents(manifest_path)

        self.assertEqual(39, len(documents))
        self.assertEqual(entries[0]["doc_id"], documents[0].document_id)
        self.assertEqual(entries[0]["title"], documents[0].title)
        self.assertEqual(entries[0]["url"], documents[0].source_url)
        self.assertEqual(entries[0]["category"], documents[0].category)
        self.assertEqual(
            (manifest_path.parent / entries[0]["path"]).read_text(encoding="utf-8"),
            documents[0].content,
        )

    def test_maps_manifest_metadata_and_markdown_content(self) -> None:
        content = "# 테스트 문서\n\n본문입니다.\n"
        entry = {
            "doc_id": "test-document",
            "url": "https://docs.riido.io/test.md",
            "title": "테스트 문서",
            "category": "test",
            "path": "clean/test.md",
        }

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._write_fixture(Path(directory), entry, content)

            document = load_normalized_documents(manifest_path)[0]

        self.assertEqual("test-document", document.document_id)
        self.assertEqual("테스트 문서", document.title)
        self.assertEqual("https://docs.riido.io/test.md", document.source_url)
        self.assertEqual("test", document.category)
        self.assertEqual(content, document.content)

    def test_raises_when_markdown_file_does_not_exist(self) -> None:
        entry = {
            "doc_id": "missing-document",
            "url": "https://docs.riido.io/missing.md",
            "title": "없는 문서",
            "category": None,
            "path": "clean/missing.md",
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "clean_manifest.json"
            manifest_path.write_text(
                json.dumps([entry], ensure_ascii=False), encoding="utf-8"
            )

            with self.assertRaises(FileNotFoundError):
                load_normalized_documents(manifest_path)

    @staticmethod
    def _write_fixture(root: Path, entry: dict, content: str) -> Path:
        markdown_path = root / entry["path"]
        markdown_path.parent.mkdir(parents=True)
        markdown_path.write_text(content, encoding="utf-8")

        manifest_path = root / "clean_manifest.json"
        manifest_path.write_text(
            json.dumps([entry], ensure_ascii=False), encoding="utf-8"
        )
        return manifest_path


if __name__ == "__main__":
    unittest.main()
