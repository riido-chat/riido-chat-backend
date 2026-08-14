import unittest
from pathlib import Path

from pipeline.document.loader import load_normalized_documents
from pipeline.document.models import NormalizedDocument
from pipeline.document.section_parser import parse_sections


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SectionParserTest(unittest.TestCase):
    def test_creates_a_section_for_each_h2(self) -> None:
        document = self._document(
            """# 문서 제목

## 첫 번째

첫 번째 본문

## 두 번째

두 번째 본문
"""
        )

        sections = parse_sections(document)

        self.assertEqual(2, len(sections))
        self.assertEqual([0, 1], [section.sequence for section in sections])
        self.assertEqual(
            ["document-id:0", "document-id:1"],
            [section.section_id for section in sections],
        )
        self.assertEqual(
            [("문서 제목", "첫 번째"), ("문서 제목", "두 번째")],
            [section.section_path for section in sections],
        )

    def test_preserves_h3_and_h4_structure(self) -> None:
        document = self._document(
            """# 문서 제목

## 설치 가이드

공통 설명

### 첫 번째 환경

첫 번째 설명

#### 세부 설정

세부 설명

### 두 번째 환경

두 번째 설명
"""
        )

        sections = parse_sections(document)

        self.assertEqual(1, len(sections))
        section = sections[0]
        self.assertIn("### 첫 번째 환경", section.body)
        self.assertIn("#### 세부 설정", section.body)
        self.assertEqual(2, len(section.subsections))
        self.assertEqual(
            ["첫 번째 환경", "두 번째 환경"],
            [subsection.title for subsection in section.subsections],
        )
        self.assertEqual([0, 1], [subsection.sequence for subsection in section.subsections])
        self.assertIn("#### 세부 설정", section.subsections[0].content)
        self.assertIn("세부 설명", section.subsections[0].content)

    def test_excludes_document_preamble(self) -> None:
        document = self._document(
            """# 문서 제목

문서 소개 문장

### H2 밖의 보조 제목

보조 설명

## 실제 Section

Section 본문
"""
        )

        sections = parse_sections(document)

        self.assertEqual(1, len(sections))
        self.assertEqual("실제 Section", sections[0].title)
        self.assertEqual("Section 본문", sections[0].body)
        self.assertEqual((), sections[0].subsections)

    def test_ignores_heading_like_text_inside_fenced_code_blocks(self) -> None:
        document = self._document(
            """# 문서 제목

```markdown
## 가짜 Section
```

## 실제 Section

~~~markdown
### 가짜 Subsection
~~~

### 실제 Subsection

실제 본문
"""
        )

        sections = parse_sections(document)

        self.assertEqual(1, len(sections))
        self.assertEqual("실제 Section", sections[0].title)
        self.assertEqual(1, len(sections[0].subsections))
        self.assertEqual("실제 Subsection", sections[0].subsections[0].title)
        self.assertIn("### 가짜 Subsection", sections[0].body)

    def test_parses_all_canonical_documents(self) -> None:
        documents = load_normalized_documents(PROJECT_ROOT / "data/clean_manifest.json")

        sections = [section for document in documents for section in parse_sections(document)]
        subsections = [
            subsection for section in sections for subsection in section.subsections
        ]

        self.assertEqual(39, len(documents))
        self.assertEqual(141, len(sections))
        self.assertEqual(43, len(subsections))

        google_calendar = next(
            document
            for document in documents
            if document.source_url.endswith("/integrations/google-calendar.md")
        )
        google_sections = parse_sections(google_calendar)
        self.assertEqual(3, len(google_sections))
        self.assertEqual("연동 설정 가이드", google_sections[0].title)
        self.assertNotIn("개요", [section.title for section in google_sections])

    @staticmethod
    def _document(content: str) -> NormalizedDocument:
        return NormalizedDocument(
            document_id="document-id",
            title="문서 제목",
            source_url="https://docs.riido.io/test.md",
            category="test",
            content=content,
        )


if __name__ == "__main__":
    unittest.main()
