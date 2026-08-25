import json
import tempfile
import unittest
from pathlib import Path

from retrieval.corpus import build_retrieval_chunks


class RetrievalCorpusTest(unittest.TestCase):
    def test_builds_retrieval_chunks_through_document_pipeline(self) -> None:
        entry = {
            "doc_id": "document-id",
            "url": "https://docs.riido.io/test.md",
            "title": "테스트 문서",
            "category": "test",
            "path": "clean/test.md",
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            markdown_path = root / entry["path"]
            markdown_path.parent.mkdir(parents=True)
            markdown_path.write_text(
                "# 테스트 문서\n\n## 개요\n\n검색할 본문\n",
                encoding="utf-8",
            )
            manifest_path = root / "clean_manifest.json"
            manifest_path.write_text(
                json.dumps([entry], ensure_ascii=False),
                encoding="utf-8",
            )

            chunks = build_retrieval_chunks(manifest_path)

        self.assertEqual(1, len(chunks))
        chunk = chunks[0]
        self.assertIsNone(chunk.chunk_id)
        self.assertIsNone(chunk.document_version_id)
        self.assertIsNone(chunk.index_version_id)
        self.assertEqual("document-id", chunk.document_id)
        self.assertEqual("document-id:a8d0dd1a5c1a", chunk.section_id)
        self.assertEqual("테스트 문서", chunk.document_title)
        self.assertEqual(("테스트 문서", "개요"), chunk.section_path)
        self.assertEqual("검색할 본문", chunk.content)
        self.assertEqual("https://docs.riido.io/test.md", chunk.source_url)


if __name__ == "__main__":
    unittest.main()
