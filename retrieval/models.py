"""Retrieval 계층에서 사용하는 데이터 모델"""

from dataclasses import dataclass
from typing import Optional, Tuple

from pipeline.document.models import Chunk, NormalizedDocument


@dataclass(frozen=True)
class RetrievalChunk:
    """Chunk와 출처 추적에 필요한 문서 메타데이터를 결합한다."""

    chunk_id: str
    document_id: str
    section_id: str
    document_title: str
    section_path: Tuple[str, ...]
    source_url: str
    category: Optional[str]
    content: str

    @classmethod
    def from_document_chunk(
        cls,
        document: NormalizedDocument,
        chunk: Chunk,
    ) -> "RetrievalChunk":
        """동일한 문서에 속한 Document와 Chunk를 검색 객체로 변환한다."""

        if document.document_id != chunk.document_id:
            raise ValueError("Document와 Chunk의 document_id가 일치하지 않습니다.")

        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            section_id=chunk.section_id,
            document_title=document.title,
            section_path=chunk.section_path,
            source_url=document.source_url,
            category=document.category,
            content=chunk.content,
        )


@dataclass(frozen=True)
class RetrievalResult:
    """BM25 검색 결과와 점수, 순위를 함께 반환한다."""

    chunk: RetrievalChunk
    score: float
    rank: int


@dataclass(frozen=True)
class HybridRetrievalResult:
    """RRF로 결합한 검색 결과와 Retriever별 순위를 반환한다."""

    chunk: RetrievalChunk
    rrf_score: float
    final_rank: int
    bm25_rank: Optional[int]
    vector_rank: Optional[int]
