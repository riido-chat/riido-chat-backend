"""문서 후보 outline 을 DB content_nodes 에서 읽는다.

문서 판과 청킹 설정이 같으면 절 목록이 바뀌지 않으므로 (document_version_id,
chunking_config_id) 로 outline 을 LRU 캐시한다. 청크를 다시 만들면 새 청킹 설정 행이 생기고,
문서가 바뀌면 새 문서 판이 생겨 키가 달라진다. 제목은 document_sources.title 을 읽을 때의
값이다. 같은 판을 가리키는 동안 제목만 바뀌면 캐시가 옛 제목을 보여줄 수 있다(판별 표시용).

이 계층은 쓰거나 commit 하지 않는다.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import ContentNode, DocumentChunk, DocumentSource, DocumentVersion
from app.question_grouping.constants import DOCUMENT_OUTLINE_CACHE_SIZE, HEADING_SEPARATOR
from app.question_grouping.document_candidates import (
    build_document_outline,
    headings_from_sections,
)
from app.question_grouping.models import DocumentOutline

OutlineCacheKey = Tuple[int, int]


class DocumentOutlineCache:
    """(document_version_id, chunking_config_id) → DocumentOutline LRU 캐시.

    한 이벤트 루프 안에서 await 없이 읽고 쓰므로 잠금을 두지 않는다.
    """

    def __init__(self, max_entries: int = DOCUMENT_OUTLINE_CACHE_SIZE) -> None:
        if max_entries <= 0:
            raise ValueError("outline 캐시 크기는 1 이상이어야 합니다.")
        self._max_entries = max_entries
        self._entries: "OrderedDict[OutlineCacheKey, DocumentOutline]" = OrderedDict()

    def get(self, key: OutlineCacheKey) -> Optional[DocumentOutline]:
        outline = self._entries.get(key)
        if outline is not None:
            self._entries.move_to_end(key)
        return outline

    def put(self, key: OutlineCacheKey, outline: DocumentOutline) -> None:
        self._entries[key] = outline
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries


_SHARED_OUTLINE_CACHE = DocumentOutlineCache()


def shared_outline_cache() -> DocumentOutlineCache:
    """프로세스 공용 outline 캐시. 요청마다 세션이 달라도 캐시는 이어진다."""

    return _SHARED_OUTLINE_CACHE


def _section_path(node_path: Optional[str], metadata: Any) -> Tuple[str, ...]:
    """적재 계층이 metadata.section_path 에 남긴 (문서 제목, H2) 경로. 없으면 node_path 를 나눈다."""

    if isinstance(metadata, dict):
        path = metadata.get("section_path")
        if isinstance(path, list) and all(isinstance(part, str) for part in path):
            return tuple(path)
    if node_path:
        return tuple(node_path.split(HEADING_SEPARATOR))
    return ()


class DocumentOutlineReader:
    """문서 판과 턴 색인의 청킹 설정으로 outline 을 만든다."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        cache: Optional[DocumentOutlineCache] = None,
    ) -> None:
        self._session = session
        self._cache = shared_outline_cache() if cache is None else cache

    async def load_outlines(
        self,
        document_version_ids: Sequence[int],
        *,
        chunking_config_id: int,
    ) -> Dict[int, DocumentOutline]:
        """문서 판별 outline. 문서 판이 없거나 그 청킹 설정의 절이 없으면 결과에서 빠진다.

        빠진 판은 document_candidates.attach_outlines 가 데이터 오류로 알린다. 절이 없는 판은
        색인이 아직 만들어지는 중일 수 있어 캐시하지 않는다.
        """

        outlines: Dict[int, DocumentOutline] = {}
        missing: List[int] = []
        for version_id in dict.fromkeys(document_version_ids):
            cached = self._cache.get((version_id, chunking_config_id))
            if cached is None:
                missing.append(version_id)
            else:
                outlines[version_id] = cached
        if not missing:
            return outlines

        documents = (
            await self._session.execute(
                select(
                    DocumentVersion.id,
                    DocumentSource.id.label("document_source_id"),
                    DocumentSource.document_key,
                    DocumentSource.title,
                )
                .join(DocumentSource, DocumentSource.id == DocumentVersion.document_source_id)
                .where(DocumentVersion.id.in_(missing))
            )
        ).all()
        if not documents:
            return outlines

        node_rows = (
            await self._session.execute(
                select(
                    ContentNode.document_version_id,
                    ContentNode.node_path,
                    ContentNode.metadata_,
                    ContentNode.normalized_content,
                )
                .join(DocumentChunk, DocumentChunk.id == ContentNode.id)
                .where(
                    ContentNode.document_version_id.in_([row.id for row in documents]),
                    DocumentChunk.chunking_config_id == chunking_config_id,
                )
                .order_by(
                    ContentNode.document_version_id,
                    ContentNode.node_order,
                    ContentNode.id,
                )
            )
        ).all()
        sections: Dict[int, List[Tuple[Tuple[str, ...], str]]] = {}
        for row in node_rows:
            sections.setdefault(row.document_version_id, []).append(
                (_section_path(row.node_path, row.metadata_), row.normalized_content)
            )

        for document in documents:
            version_sections = sections.get(document.id)
            if not version_sections:
                continue
            title = document.title or next(
                (path[0] for path, _ in version_sections if path), ""
            )
            outline = build_document_outline(
                document_source_id=document.document_source_id,
                document_version_id=document.id,
                document_key=document.document_key,
                title=title,
                headings=headings_from_sections(version_sections),
            )
            self._cache.put((document.id, chunking_config_id), outline)
            outlines[document.id] = outline
        return outlines
