"""Section을 검색 단위인 Chunk로 변환한다."""

from typing import List

from pipeline.document.models import Chunk, Section


def create_chunks(sections: List[Section]) -> List[Chunk]:
    """각 Section을 내용 분할 없이 Chunk 하나로 변환한다."""

    return [
        Chunk(
            chunk_id=section.section_id,
            document_id=section.document_id,
            section_id=section.section_id,
            section_path=section.section_path,
            content=section.body,
            sequence=section.sequence,
        )
        for section in sections
    ]
