"""NormalizedDocument의 Markdown을 H2 Section으로 변환한다."""

import re
from typing import List, Optional, Tuple

from pipeline.document.models import NormalizedDocument, Section, Subsection


_FENCE_PATTERN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_HEADING_PATTERN = re.compile(r"^ {0,3}(#{2,3})(?!#)(?:[ \t]+(.*)|[ \t]*)$")
_CLOSING_HASH_PATTERN = re.compile(r"[ \t]+#+[ \t]*$")


def parse_sections(document: NormalizedDocument) -> List[Section]:
    """문서에서 fenced code block 밖의 H2를 Section으로 파싱한다."""

    lines = document.content.splitlines(keepends=True)
    headings = _find_headings(lines)
    h2_headings = [heading for heading in headings if heading[1] == 2]
    sections = []

    for sequence, (start, _, title) in enumerate(h2_headings):
        end = h2_headings[sequence + 1][0] if sequence + 1 < len(h2_headings) else len(lines)
        subsection_headings = [
            heading
            for heading in headings
            if start < heading[0] < end and heading[1] == 3
        ]

        sections.append(
            Section(
                section_id=f"{document.document_id}:{sequence}",
                document_id=document.document_id,
                title=title,
                section_path=(document.title, title),
                body="".join(lines[start + 1 : end]).strip(),
                subsections=_parse_subsections(lines, subsection_headings, end),
                sequence=sequence,
            )
        )

    return sections


def _find_headings(lines: List[str]) -> List[Tuple[int, int, str]]:
    headings = []
    fence: Optional[Tuple[str, int]] = None

    for index, line in enumerate(lines):
        content = line.rstrip("\r\n")
        fence_match = _FENCE_PATTERN.match(content)

        if fence_match:
            marker = fence_match.group(1)
            remainder = fence_match.group(2)

            if fence is None:
                if marker[0] == "`" and "`" in remainder:
                    continue
                fence = (marker[0], len(marker))
            elif (
                marker[0] == fence[0]
                and len(marker) >= fence[1]
                and not remainder.strip()
            ):
                fence = None
            continue

        if fence is not None:
            continue

        heading_match = _HEADING_PATTERN.match(content)
        if heading_match is None:
            continue

        level = len(heading_match.group(1))
        title = heading_match.group(2) or ""
        title = _CLOSING_HASH_PATTERN.sub("", title).strip()
        headings.append((index, level, title))

    return headings


def _parse_subsections(
    lines: List[str],
    headings: List[Tuple[int, int, str]],
    section_end: int,
) -> Tuple[Subsection, ...]:
    subsections = []

    for sequence, (start, _, title) in enumerate(headings):
        end = headings[sequence + 1][0] if sequence + 1 < len(headings) else section_end
        subsections.append(
            Subsection(
                title=title,
                content="".join(lines[start + 1 : end]).strip(),
                sequence=sequence,
            )
        )

    return tuple(subsections)
