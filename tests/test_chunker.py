import unittest
from pathlib import Path

from pipeline.document.chunker import create_chunks
from pipeline.document.loader import load_normalized_documents
from pipeline.document.models import Section, Subsection
from pipeline.document.section_parser import parse_sections


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ChunkerTest(unittest.TestCase):
    def test_maps_one_section_to_one_chunk(self) -> None:
        section = self._section(sequence=1, body="Section 전체 본문")

        chunks = create_chunks([section])

        self.assertEqual(1, len(chunks))
        chunk = chunks[0]
        self.assertEqual(section.section_id, chunk.chunk_id)
        self.assertEqual(section.document_id, chunk.document_id)
        self.assertEqual(section.section_id, chunk.section_id)
        self.assertEqual(section.section_path, chunk.section_path)
        self.assertEqual(section.body, chunk.content)
        self.assertEqual(section.sequence, chunk.sequence)

    def test_preserves_count_and_order_for_multiple_sections(self) -> None:
        sections = [
            self._section(sequence=0, body="첫 번째 본문"),
            self._section(sequence=1, body="두 번째 본문"),
            self._section(sequence=2, body="세 번째 본문"),
        ]

        chunks = create_chunks(sections)

        self.assertEqual(len(sections), len(chunks))
        self.assertEqual(
            [section.section_id for section in sections],
            [chunk.chunk_id for chunk in chunks],
        )
        self.assertEqual(
            [section.body for section in sections],
            [chunk.content for chunk in chunks],
        )

    def test_does_not_split_section_with_subsections(self) -> None:
        section = Section(
            section_id="document-id:0",
            document_id="document-id",
            title="설치 가이드",
            section_path=("문서 제목", "설치 가이드"),
            body="### 첫 번째\n\n첫 번째 설명\n\n### 두 번째\n\n두 번째 설명",
            subsections=(
                Subsection(title="첫 번째", content="첫 번째 설명", sequence=0),
                Subsection(title="두 번째", content="두 번째 설명", sequence=1),
            ),
            sequence=0,
        )

        chunks = create_chunks([section])

        self.assertEqual(1, len(chunks))
        self.assertEqual(section.body, chunks[0].content)

    def test_creates_one_chunk_per_section_for_canonical_corpus(self) -> None:
        documents = load_normalized_documents(PROJECT_ROOT / "data/clean_manifest.json")
        sections = [section for document in documents for section in parse_sections(document)]

        chunks = create_chunks(sections)

        self.assertEqual(39, len(documents))
        self.assertEqual(141, len(sections))
        self.assertEqual(141, len(chunks))
        self.assertEqual(
            [section.section_id for section in sections],
            [chunk.chunk_id for chunk in chunks],
        )
        self.assertEqual(
            [section.body for section in sections],
            [chunk.content for chunk in chunks],
        )

    @staticmethod
    def _section(sequence: int, body: str) -> Section:
        return Section(
            section_id=f"document-id:{sequence}",
            document_id="document-id",
            title=f"Section {sequence}",
            section_path=("문서 제목", f"Section {sequence}"),
            body=body,
            subsections=(),
            sequence=sequence,
        )


if __name__ == "__main__":
    unittest.main()
