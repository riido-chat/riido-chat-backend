"""정제 문서 파이프라인에서 사용하는 데이터 모델."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class NormalizedDocument:
    """정제된 공식 문서 한 개를 나타낸다."""

    document_id: str
    title: str
    source_url: str
    category: Optional[str]
    content: str


@dataclass(frozen=True)
class Subsection:
    """Section 안의 H3 구조를 보존한다."""

    title: str
    content: str
    sequence: int


@dataclass(frozen=True)
class Section:
    """문서의 H2 의미 구조 단위를 나타낸다."""

    section_id: str
    document_id: str
    title: str
    section_path: Tuple[str, ...]
    body: str
    subsections: Tuple[Subsection, ...]
    sequence: int


@dataclass(frozen=True)
class Chunk:
    """검색에 사용하는 최소 단위를 나타낸다."""

    chunk_id: str
    document_id: str
    section_id: str
    section_path: Tuple[str, ...]
    content: str
    sequence: int
