"""NormalizedDocument의 Markdown을 H2 Section으로 변환한다."""

import hashlib
import json
import re
from typing import List, Optional, Tuple

from app.document.models import NormalizedDocument, Section, Subsection


_FENCE_PATTERN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_HEADING_PATTERN = re.compile(r"^ {0,3}(#{1,3})(?!#)(?:[ \t]+(.*)|[ \t]*)$")
_CLOSING_HASH_PATTERN = re.compile(r"[ \t]+#+[ \t]*$")
_SECTION_ID_HASH_LENGTH = 12


def parse_sections(document: NormalizedDocument) -> List[Section]:
    """H2와 그 이전의 의미 콘텐츠를 Section으로 파싱한다."""

    lines = document.content.splitlines(keepends=True)
    headings = _find_headings(lines)
    h2_headings = [heading for heading in headings if heading[1] == 2]
    sections = []

    first_h2_start = h2_headings[0][0] if h2_headings else len(lines)
    preamble_section = _parse_preamble_section(
        document,
        lines,
        [heading for heading in headings if heading[0] < first_h2_start],
        first_h2_start,
    )
    if preamble_section is not None:
        sections.append(preamble_section)

    sequence_offset = len(sections)
    for index, (start, _, title) in enumerate(h2_headings):
        sequence = index + sequence_offset
        end = (
            h2_headings[index + 1][0]
            if index + 1 < len(h2_headings)
            else len(lines)
        )
        subsection_headings = [
            heading
            for heading in headings
            if start < heading[0] < end and heading[1] == 3
        ]

        sections.append(
            Section(
                section_id=_create_section_id(document.document_id, (title,)),
                document_id=document.document_id,
                title=title,
                section_path=(document.title, title),
                body="".join(lines[start + 1 : end]).strip(),
                subsections=_parse_subsections(lines, subsection_headings, end),
                sequence=sequence,
            )
        )

    _validate_unique_semantic_paths(sections)
    return sections


def _parse_preamble_section(
    document: NormalizedDocument,
    lines: List[str],
    headings: List[Tuple[int, int, str]],
    preamble_end: int,
) -> Optional[Section]:
    h1_indices = {heading[0] for heading in headings if heading[1] == 1}
    h3_headings = [heading for heading in headings if heading[1] == 3]
    promoted_heading = h3_headings[0] if h3_headings else None
    excluded_indices = set(h1_indices)
    if promoted_heading is not None:
        excluded_indices.add(promoted_heading[0])

    preamble_content = "".join(
        line
        for index, line in enumerate(lines[:preamble_end])
        if index not in h1_indices
    ).strip()
    if not preamble_content:
        return None

    if promoted_heading is not None:
        title = promoted_heading[2]
        section_path = (document.title, title)
        subsection_headings = h3_headings[1:]
    else:
        title = document.title
        section_path = (document.title,)
        subsection_headings = []

    body = "".join(
        line
        for index, line in enumerate(lines[:preamble_end])
        if index not in excluded_indices
    ).strip()

    return Section(
        section_id=_create_section_id(
            document.document_id,
            section_path[1:],
        ),
        document_id=document.document_id,
        title=title,
        section_path=section_path,
        body=body,
        subsections=_parse_subsections(lines, subsection_headings, preamble_end),
        sequence=0,
    )


def _create_section_id(
    document_id: str,
    local_section_path: Tuple[str, ...],
) -> str:
    digest = create_section_identity_hash(document_id, local_section_path)
    return f"{document_id}:{digest[:_SECTION_ID_HASH_LENGTH]}"


def create_section_identity_hash(
    document_id: str,
    local_section_path: Tuple[str, ...],
) -> str:
    """문서 ID와 문서 내부 경로로 재색인 간 유지되는 Section 신원을 만든다."""

    if local_section_path:
        identity_source = {
            "document_id": document_id,
            "kind": "path",
            "path": local_section_path,
        }
    else:
        identity_source = {
            "document_id": document_id,
            "kind": "root",
        }

    serialized = json.dumps(
        identity_source,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_unique_semantic_paths(sections: List[Section]) -> None:
    seen_paths = set()

    for section in sections:
        local_section_path = section.section_path[1:]
        if local_section_path in seen_paths:
            path = " > ".join(local_section_path) or "<root>"
            raise ValueError(
                f"동일 문서에 중복된 semantic Section path가 있습니다: {path}"
            )
        seen_paths.add(local_section_path)


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
