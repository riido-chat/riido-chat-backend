"""Canonical corpus가 로딩 → 섹션 파싱 → 청킹 → BM25 색인을 통과하는지,
문서 39개 / 섹션·청크 142개라는 구조 수치가 유지되는지 검증한다.
"""

import json
import unittest
from pathlib import Path

from pipeline.document.chunker import create_chunks
from pipeline.document.loader import load_normalized_documents
from pipeline.document.section_parser import parse_sections
from retrieval.bm25_retriever import BM25Retriever
from retrieval.models import RetrievalChunk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data/clean_manifest.json"


class CanonicalDocumentLoadingCheck(unittest.TestCase):
    def test_loads_current_canonical_documents(self) -> None:
        entries = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        documents = load_normalized_documents(MANIFEST_PATH)

        self.assertEqual(39, len(documents))
        self.assertEqual(entries[0]["doc_id"], documents[0].document_id)
        self.assertEqual(entries[0]["title"], documents[0].title)
        self.assertEqual(entries[0]["url"], documents[0].source_url)
        self.assertEqual(entries[0]["category"], documents[0].category)
        self.assertEqual(
            (MANIFEST_PATH.parent / entries[0]["path"]).read_text(encoding="utf-8"),
            documents[0].content,
        )


class CanonicalSectionParsingCheck(unittest.TestCase):
    def test_parses_all_canonical_documents(self) -> None:
        documents = load_normalized_documents(MANIFEST_PATH)

        sections = [section for document in documents for section in parse_sections(document)]
        subsections = [
            subsection for section in sections for subsection in section.subsections
        ]

        self.assertEqual(39, len(documents))
        self.assertEqual(142, len(sections))
        self.assertEqual(43, len(subsections))

        google_calendar = next(
            document
            for document in documents
            if document.source_url.endswith("/integrations/google-calendar.md")
        )
        google_sections = parse_sections(google_calendar)
        self.assertEqual(4, len(google_sections))
        self.assertEqual("개요", google_sections[0].title)
        self.assertIn("양방향 동기화", google_sections[0].body)
        self.assertEqual("연동 설정 가이드", google_sections[1].title)


class CanonicalChunkingCheck(unittest.TestCase):
    def test_creates_one_chunk_per_section_for_canonical_corpus(self) -> None:
        documents = load_normalized_documents(MANIFEST_PATH)
        sections = [section for document in documents for section in parse_sections(document)]

        chunks = create_chunks(sections)

        self.assertEqual(39, len(documents))
        self.assertEqual(142, len(sections))
        self.assertEqual(142, len(chunks))
        self.assertEqual(
            [section.section_id for section in sections],
            [chunk.chunk_id for chunk in chunks],
        )
        self.assertEqual(
            [section.body for section in sections],
            [chunk.content for chunk in chunks],
        )


class CanonicalBM25IndexCheck(unittest.TestCase):
    def test_creates_index_for_canonical_corpus(self) -> None:
        documents = load_normalized_documents(MANIFEST_PATH)
        retrieval_chunks = []

        for document in documents:
            sections = parse_sections(document)
            chunks = create_chunks(sections)
            retrieval_chunks.extend(
                RetrievalChunk.from_document_chunk(document, chunk)
                for chunk in chunks
            )

        retriever = BM25Retriever(retrieval_chunks)

        self.assertEqual(142, retriever._index.corpus_size)


if __name__ == "__main__":
    unittest.main()
