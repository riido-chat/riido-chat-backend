import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.preprocess import clean


class CleanManifestTest(unittest.TestCase):
    def test_carries_raw_metadata_and_normalized_hash(self) -> None:
        raw_content = "# 테스트 문서\n\n원문 본문\n"

        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "data"
            raw_path = data_dir / "raw/test.md"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_text(raw_content, encoding="utf-8")
            manifest_path = data_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "doc_id": "test-document",
                            "url": "https://docs.riido.io/test.md",
                            "path": "raw/test.md",
                            "title": "테스트 문서",
                            "category": "test",
                            "order": 1,
                            "sha256": self._sha256(raw_content),
                            "bytes": len(raw_content.encode("utf-8")),
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            clean_manifest_path = data_dir / "clean_manifest.json"

            with patch.multiple(
                clean,
                MANIFEST_PATH=manifest_path,
                DATA_DIR=data_dir,
                CLEAN_DIR=data_dir / "clean",
                CLEAN_MANIFEST_PATH=clean_manifest_path,
            ), patch("builtins.print"):
                clean.main()

            entry = json.loads(clean_manifest_path.read_text(encoding="utf-8"))[0]
            normalized_content = (data_dir / entry["path"]).read_text(
                encoding="utf-8"
            )

        self.assertEqual("raw/test.md", entry["raw_content_uri"])
        self.assertEqual(self._sha256(raw_content), entry["raw_content_hash"])
        self.assertEqual(
            self._sha256(normalized_content),
            entry["normalized_content_hash"],
        )

    @staticmethod
    def _sha256(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
