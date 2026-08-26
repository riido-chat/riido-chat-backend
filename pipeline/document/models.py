"""정제 문서 파이프라인에서 사용하는 데이터 모델."""

from dataclasses import dataclass
from typing import Optional, Tuple

# frozen=True: 객체 생성 후 필드값 변경을 막는다.
@dataclass(frozen=True)
class NormalizedDocument:
    """정제 본문과 이를 만든 수집 원본의 추적 metadata를 나타낸다."""

    document_id: str
    title: str
    source_url: str
    category: Optional[str]
    content: str
    raw_content_uri: str
    raw_content_hash: str
    normalized_content_hash: str


@dataclass(frozen=True)
class Subsection:
    """Section 안의 H3 구조를 보존한다."""

    title: str
    content: str
    sequence: int


@dataclass(frozen=True)
class Section:
    """문서의 의미 구조 단위를 나타낸다."""

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
