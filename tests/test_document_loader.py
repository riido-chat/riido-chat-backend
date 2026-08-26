import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.document.loader import load_normalized_documents


class NormalizedDocumentLoaderTest(unittest.TestCase):
    def test_maps_manifest_metadata_and_markdown_content(self) -> None:
        content = "# 테스트 문서\n\n본문입니다.\n"
        entry = {
            "doc_id": "test-document",
            "url": "https://docs.riido.io/test.md",
            "title": "테스트 문서",
            "category": "test",
            "path": "clean/test.md",
            "raw_content_uri": "raw/test.md",
            "raw_content_hash": self._sha256("수집 원문"),
            "normalized_content_hash": self._sha256(content),
        }

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._write_fixture(Path(directory), entry, content)

            document = load_normalized_documents(manifest_path)[0]

        self.assertEqual("test-document", document.document_id)
        self.assertEqual("테스트 문서", document.title)
        self.assertEqual("https://docs.riido.io/test.md", document.source_url)
        self.assertEqual("test", document.category)
        self.assertEqual(content, document.content)
        self.assertEqual("raw/test.md", document.raw_content_uri)
        self.assertEqual(entry["raw_content_hash"], document.raw_content_hash)
        self.assertEqual(
            entry["normalized_content_hash"],
            document.normalized_content_hash,
        )

    def test_rejects_mismatched_normalized_content_hash(self) -> None:
        content = "# 테스트 문서\n\n본문입니다.\n"
        entry = {
            "doc_id": "test-document",
            "url": "https://docs.riido.io/test.md",
            "title": "테스트 문서",
            "category": "test",
            "path": "clean/test.md",
            "raw_content_uri": "raw/test.md",
            "raw_content_hash": self._sha256("수집 원문"),
            "normalized_content_hash": self._sha256("다른 정제 문서"),
        }

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._write_fixture(Path(directory), entry, content)

            with self.assertRaisesRegex(ValueError, "정제 문서 hash"):
                load_normalized_documents(manifest_path)

    def test_raises_when_markdown_file_does_not_exist(self) -> None:
        entry = {
            "doc_id": "missing-document",
            "url": "https://docs.riido.io/missing.md",
            "title": "없는 문서",
            "category": None,
            "path": "clean/missing.md",
            "raw_content_uri": "raw/missing.md",
            "raw_content_hash": self._sha256("수집 원문"),
            "normalized_content_hash": self._sha256("정제 문서"),
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

    @staticmethod
    def _sha256(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
