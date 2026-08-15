"""문서 파이프라인 결과를 공통 검색 corpus로 조립한다."""

from pathlib import Path
from typing import List, Union

from pipeline.document.chunker import create_chunks
from pipeline.document.loader import (
    DEFAULT_MANIFEST_PATH,
    load_normalized_documents,
)
from pipeline.document.section_parser import parse_sections
from retrieval.models import RetrievalChunk


def build_retrieval_chunks(
    manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
) -> List[RetrievalChunk]:
    """정제 문서를 동일한 Document → Section → Chunk 경로로 변환한다."""

    retrieval_chunks = []

    for document in load_normalized_documents(manifest_path):
        sections = parse_sections(document)
        chunks = create_chunks(sections)
        retrieval_chunks.extend(
            RetrievalChunk.from_document_chunk(document, chunk)
            for chunk in chunks
        )

    return retrieval_chunks
