"""문서 파이프라인 결과를 공통 검색 corpus로 조립한다."""

from pathlib import Path
from typing import List, Union

from pipeline.document.chunker import create_chunks
from pipeline.document.loader import (
    DEFAULT_MANIFEST_PATH,
    load_normalized_documents,
)
from pipeline.document.models import NormalizedDocument
from pipeline.document.section_parser import parse_sections
from retrieval.models import RetrievalChunk


def build_document_retrieval_chunks(
    document: NormalizedDocument,
) -> List[RetrievalChunk]:
    """정제 문서 한 건을 Section → Chunk 검색 객체로 변환한다."""

    sections = parse_sections(document)
    chunks = create_chunks(sections)
    return [RetrievalChunk.from_document_chunk(document, chunk) for chunk in chunks]


def build_retrieval_chunks(
    manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
) -> List[RetrievalChunk]:
    """정제 문서를 동일한 Document → Section → Chunk 경로로 변환한다."""

    retrieval_chunks = []

    for document in load_normalized_documents(manifest_path):
        retrieval_chunks.extend(build_document_retrieval_chunks(document))

    return retrieval_chunks
