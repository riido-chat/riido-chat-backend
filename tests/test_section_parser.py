import unittest

from pipeline.document.models import NormalizedDocument
from pipeline.document.section_parser import parse_sections


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
            ["document-id:bc1229b52429", "document-id:88cfc51b3532"],
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

    def test_preserves_meaningful_h3_preamble_as_section(self) -> None:
        document = self._document(
            """# 문서 제목

### 개요

문서 소개 문장

## 실제 Section

Section 본문
"""
        )

        sections = parse_sections(document)

        self.assertEqual(2, len(sections))
        self.assertEqual("개요", sections[0].title)
        self.assertEqual(("문서 제목", "개요"), sections[0].section_path)
        self.assertEqual("문서 소개 문장", sections[0].body)
        self.assertEqual((), sections[0].subsections)
        self.assertEqual("실제 Section", sections[1].title)
        self.assertEqual("Section 본문", sections[1].body)
        self.assertEqual([0, 1], [section.sequence for section in sections])
        self.assertEqual(
            ["document-id:a8d0dd1a5c1a", "document-id:ec58eafcaf5f"],
            [section.section_id for section in sections],
        )

    def test_preserves_headingless_preamble_as_document_section(self) -> None:
        document = self._document(
            """# 문서 제목

문서 전체에 적용되는 소개 문장

## 실제 Section

Section 본문
"""
        )

        sections = parse_sections(document)

        self.assertEqual(2, len(sections))
        self.assertEqual("문서 제목", sections[0].title)
        self.assertEqual(("문서 제목",), sections[0].section_path)
        self.assertEqual("문서 전체에 적용되는 소개 문장", sections[0].body)

    def test_does_not_create_section_for_h1_and_whitespace_only(self) -> None:
        document = self._document(
            """# 문서 제목


## 실제 Section

Section 본문
"""
        )

        sections = parse_sections(document)

        self.assertEqual(1, len(sections))
        self.assertEqual("실제 Section", sections[0].title)
        self.assertEqual("document-id:ec58eafcaf5f", sections[0].section_id)

    def test_preserves_meaningful_document_without_h2(self) -> None:
        document = self._document(
            """# 문서 제목

문서 전체 본문
"""
        )

        sections = parse_sections(document)

        self.assertEqual(1, len(sections))
        self.assertEqual("문서 제목", sections[0].title)
        self.assertEqual(("문서 제목",), sections[0].section_path)
        self.assertEqual("문서 전체 본문", sections[0].body)

    def test_keeps_section_id_when_preceding_section_is_added(self) -> None:
        original = self._document(
            """# 문서 제목

## 설정

설정 본문
"""
        )
        with_preceding_section = self._document(
            """# 문서 제목

## 개요

개요 본문

## 설정

설정 본문
"""
        )

        original_section = parse_sections(original)[0]
        moved_section = parse_sections(with_preceding_section)[1]

        self.assertEqual(original_section.section_id, moved_section.section_id)
        self.assertEqual(0, original_section.sequence)
        self.assertEqual(1, moved_section.sequence)

    def test_keeps_section_id_when_sections_are_reordered(self) -> None:
        original = self._document(
            """# 문서 제목

## 첫 번째

첫 번째 본문

## 두 번째

두 번째 본문
"""
        )
        reordered = self._document(
            """# 문서 제목

## 두 번째

두 번째 본문

## 첫 번째

첫 번째 본문
"""
        )

        original_by_title = {
            section.title: section for section in parse_sections(original)
        }
        reordered_by_title = {
            section.title: section for section in parse_sections(reordered)
        }

        self.assertEqual(
            original_by_title["첫 번째"].section_id,
            reordered_by_title["첫 번째"].section_id,
        )
        self.assertEqual(
            original_by_title["두 번째"].section_id,
            reordered_by_title["두 번째"].section_id,
        )
        self.assertEqual(0, reordered_by_title["두 번째"].sequence)
        self.assertEqual(1, reordered_by_title["첫 번째"].sequence)

    def test_keeps_section_id_when_only_body_changes(self) -> None:
        original = self._document("## 설정\n\n기존 본문")
        changed = self._document("## 설정\n\n수정된 본문")

        original_section = parse_sections(original)[0]
        changed_section = parse_sections(changed)[0]

        self.assertEqual(original_section.section_id, changed_section.section_id)
        self.assertNotEqual(original_section.body, changed_section.body)

    def test_changes_section_id_when_heading_changes(self) -> None:
        original = self._document("## 설정\n\n본문")
        changed = self._document("## 환경 설정\n\n본문")

        self.assertNotEqual(
            parse_sections(original)[0].section_id,
            parse_sections(changed)[0].section_id,
        )

    def test_distinguishes_same_local_path_in_different_documents(self) -> None:
        first = self._document("## 개요\n\n본문")
        second = NormalizedDocument(
            document_id="other-document-id",
            title="다른 문서",
            source_url="https://docs.riido.io/other.md",
            category="test",
            content="## 개요\n\n본문",
        )

        self.assertNotEqual(
            parse_sections(first)[0].section_id,
            parse_sections(second)[0].section_id,
        )

    def test_excludes_document_title_from_section_identity(self) -> None:
        original = self._document("## 개요\n\n본문")
        renamed_document = NormalizedDocument(
            document_id="document-id",
            title="변경된 문서 제목",
            source_url="https://docs.riido.io/test.md",
            category="test",
            content="## 개요\n\n본문",
        )

        self.assertEqual(
            parse_sections(original)[0].section_id,
            parse_sections(renamed_document)[0].section_id,
        )

    def test_keeps_root_section_id_when_only_body_changes(self) -> None:
        original = self._document("# 문서 제목\n\n기존 본문")
        changed = self._document("# 문서 제목\n\n수정된 본문")

        original_section = parse_sections(original)[0]
        changed_section = parse_sections(changed)[0]

        self.assertEqual("document-id:fbbcd684667d", original_section.section_id)
        self.assertEqual(original_section.section_id, changed_section.section_id)

    def test_rejects_duplicate_semantic_path_in_same_document(self) -> None:
        document = self._document(
            """# 문서 제목

## 설정

첫 번째 본문

## 설정

두 번째 본문
"""
        )

        with self.assertRaisesRegex(ValueError, "중복된 semantic Section path"):
            parse_sections(document)

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

        self.assertEqual(2, len(sections))
        self.assertEqual(("문서 제목",), sections[0].section_path)
        self.assertIn("## 가짜 Section", sections[0].body)
        self.assertEqual("실제 Section", sections[1].title)
        self.assertEqual(1, len(sections[1].subsections))
        self.assertEqual("실제 Subsection", sections[1].subsections[0].title)
        self.assertIn("### 가짜 Subsection", sections[1].body)


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
