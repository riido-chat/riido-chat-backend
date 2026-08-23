"""BM25 검색 corpus의 적재 상태를 보관하고 갱신한다."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from retrieval.bm25_retriever import BM25Retriever
from retrieval.corpus import build_retrieval_chunks


CLEAN_MANIFEST_FILENAME = "clean_manifest.json"


class CorpusNotLoadedError(RuntimeError):
    """corpus가 적재되지 않은 상태에서 검색을 요청했을 때 발생한다."""


@dataclass(frozen=True)
class CorpusSnapshot:
    """corpus 적재 상태 조회 결과."""

    loaded: bool
    chunk_count: int
    document_count: int
    loaded_at: Optional[datetime]
    source: str


class CorpusState:
    """정제 문서에서 만든 BM25 인덱스를 보관한다.

    인덱스는 프로세스 메모리에만 존재하며 종료 시 함께 사라진다.
    """

    def __init__(self, corpus_dir: Union[str, Path]) -> None:
        self._manifest_path = Path(corpus_dir) / CLEAN_MANIFEST_FILENAME
        self._retriever: Optional[BM25Retriever] = None
        self._chunk_count = 0
        self._document_count = 0
        self._loaded_at: Optional[datetime] = None

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def is_loaded(self) -> bool:
        return self._retriever is not None

    def get_retriever(self) -> BM25Retriever:
        """적재된 BM25Retriever를 반환하고, 미적재면 예외를 발생시킨다."""

        if self._retriever is None:
            raise CorpusNotLoadedError(
                f"corpus가 적재되지 않았습니다: {self._manifest_path}"
            )
        return self._retriever

    def load(self) -> CorpusSnapshot:
        """정제 문서를 읽어 BM25 인덱스를 새로 만들고 기존 인덱스를 교체한다."""

        chunks = build_retrieval_chunks(self._manifest_path)
        if not chunks:
            raise ValueError(f"적재할 Chunk가 없습니다: {self._manifest_path}")

        self._retriever = BM25Retriever(chunks)
        self._chunk_count = len(chunks)
        self._document_count = len({chunk.document_id for chunk in chunks})
        self._loaded_at = datetime.now(timezone.utc)
        return self.snapshot()

    def snapshot(self) -> CorpusSnapshot:
        return CorpusSnapshot(
            loaded=self.is_loaded,
            chunk_count=self._chunk_count,
            document_count=self._document_count,
            loaded_at=self._loaded_at,
            source=str(self._manifest_path),
        )
