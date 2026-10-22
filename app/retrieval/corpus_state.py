"""BM25 검색 corpus의 적재 상태를 보관하고 갱신한다."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.models import RetrievalChunk


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


@dataclass(frozen=True)
class PreparedCorpusSnapshot:
    """BM25를 미리 완성해 commit 뒤 publish할 수 있는 immutable 값."""

    retriever: BM25Retriever
    index_version_id: int
    chunk_count: int
    document_count: int


class CorpusState:
    """정제 문서에서 만든 BM25 인덱스를 보관한다.

    인덱스는 프로세스 메모리에만 존재하며 종료 시 함께 사라진다.
    """

    def __init__(self, corpus_dir: Union[str, Path] = "data") -> None:
        self._manifest_path = Path(corpus_dir) / CLEAN_MANIFEST_FILENAME
        self._retriever: Optional[BM25Retriever] = None
        self._index_version_id: Optional[int] = None
        self._search_snapshot: Optional[tuple[BM25Retriever, int]] = None
        self._chunk_count = 0
        self._document_count = 0
        self._loaded_at: Optional[datetime] = None
        self._source = str(self._manifest_path)

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def is_loaded(self) -> bool:
        return self._retriever is not None

    @property
    def index_version_id(self) -> int:
        """적재된 corpus의 ACTIVE index version을 반환한다.

        rag_run은 어느 색인으로 답했는지 남겨야 하므로, 검색에 실제로 사용한
        BM25 corpus와 같은 index version을 그대로 쓴다.
        """

        if self._index_version_id is None:
            raise CorpusNotLoadedError(
                f"corpus가 적재되지 않았습니다: {self._manifest_path}"
            )
        return self._index_version_id

    def get_retriever(self) -> BM25Retriever:
        """적재된 BM25Retriever를 반환하고, 미적재면 예외를 발생시킨다."""

        if self._retriever is None:
            raise CorpusNotLoadedError(
                f"corpus가 적재되지 않았습니다: {self._manifest_path}"
            )
        return self._retriever

    def replace(self, chunks: Sequence[RetrievalChunk]) -> CorpusSnapshot:
        """ACTIVE index Chunk로 BM25를 완성한 뒤 기존 인덱스를 교체한다."""

        prepared = self.prepare(chunks)
        self.publish(prepared)
        return self.snapshot()

    def prepare(
        self, chunks: Sequence[RetrievalChunk]
    ) -> PreparedCorpusSnapshot:
        """Validate and build BM25 without making it visible to readers."""

        if not chunks:
            raise ValueError("ACTIVE index에 적재된 Chunk가 없습니다.")

        index_version_ids = {chunk.index_version_id for chunk in chunks}
        if None in index_version_ids or len(index_version_ids) != 1:
            raise ValueError("BM25 corpus의 index_version_id가 유일하지 않습니다.")
        if any(
            chunk.chunk_id is None or chunk.document_version_id is None
            for chunk in chunks
        ):
            raise ValueError("BM25 corpus에 신규 ERD 식별자가 없는 Chunk가 있습니다.")

        return PreparedCorpusSnapshot(
            retriever=BM25Retriever(chunks),
            index_version_id=next(iter(index_version_ids)),
            chunk_count=len(chunks),
            document_count=len({chunk.document_id for chunk in chunks}),
        )

    def publish(self, prepared: PreparedCorpusSnapshot) -> None:
        """Publish one prepared retriever/id pair as one immutable state."""

        # Publish the two values through one immutable reference.  Readers
        # cannot observe a retriever from one generation with an id from
        # another generation while reindex replaces this state.
        self._search_snapshot = (prepared.retriever, prepared.index_version_id)
        self._retriever = prepared.retriever
        self._index_version_id = prepared.index_version_id
        self._chunk_count = prepared.chunk_count
        self._document_count = prepared.document_count
        self._loaded_at = datetime.now(timezone.utc)
        self._source = f"index_version:{prepared.index_version_id}"

    def get_search_snapshot(self) -> tuple[BM25Retriever, int]:
        """Return the BM25 retriever and its exact index generation together."""

        snapshot = self._search_snapshot
        if snapshot is None:
            raise CorpusNotLoadedError(
                f"corpus가 적재되지 않았습니다: {self._manifest_path}"
            )
        return snapshot

    def snapshot(self) -> CorpusSnapshot:
        return CorpusSnapshot(
            loaded=self.is_loaded,
            chunk_count=self._chunk_count,
            document_count=self._document_count,
            loaded_at=self._loaded_at,
            source=self._source,
        )


class CorpusRegistry:
    """Group-scoped BM25 snapshots.

    Each entry is replaced atomically after its matching ACTIVE index has been
    read.  Keeping the index version on the same state lets ChatService pin the
    BM25 and vector half of a request to one snapshot.
    """

    def __init__(self, corpus_dir: Union[str, Path] = "data") -> None:
        self._corpus_dir = Path(corpus_dir)
        self._states: Dict[int, CorpusState] = {}
        self._legacy_compat = False

    @property
    def is_legacy_compat(self) -> bool:
        return self._legacy_compat

    def state_for(self, document_group_id: int) -> CorpusState:
        state = self._states.get(document_group_id)
        if state is None:
            state = CorpusState(self._corpus_dir / f"group-{document_group_id}")
            self._states[document_group_id] = state
        return state

    def get_search_snapshot(
        self, document_group_id: int
    ) -> tuple[BM25Retriever, int]:
        return self.state_for(document_group_id).get_search_snapshot()

    def replace(
        self,
        document_group_id: int,
        chunks: Sequence[RetrievalChunk],
    ) -> CorpusSnapshot:
        state = self.state_for(document_group_id)
        return state.replace(chunks)

    def prepare(
        self,
        document_group_id: int,
        chunks: Sequence[RetrievalChunk],
    ) -> PreparedCorpusSnapshot:
        return self.state_for(document_group_id).prepare(chunks)

    def publish(
        self,
        document_group_id: int,
        prepared: PreparedCorpusSnapshot,
    ) -> CorpusSnapshot:
        state = self.state_for(document_group_id)
        state.publish(prepared)
        return state.snapshot()

    def snapshot(self, document_group_id: int) -> CorpusSnapshot:
        return self.state_for(document_group_id).snapshot()

    def group_ids(self) -> Iterable[int]:
        return tuple(self._states)

    def replace_legacy(
        self,
        chunks: Sequence[RetrievalChunk],
        *,
        prepared: Optional[PreparedCorpusSnapshot] = None,
    ) -> CorpusSnapshot:
        """Populate the compatibility state used only by legacy tests."""

        self._legacy_compat = True
        if prepared is None:
            return self.replace(0, chunks)
        state = self.state_for(0)
        state.publish(prepared)
        return state.snapshot()
