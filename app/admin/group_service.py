"""운영자 콘솔의 문서 그룹 목록과 상세를 조립한다.

문서와 색인, 실행 데이터가 모두 필요한 읽기 전용 모델이라 두 도메인을
합치는 admin 계층에 둔다. 상태 값은 저장하지 않고 조회 시점에 계산한다.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    DocumentGroup,
    DocumentSource,
    DocumentVersion,
    ExecutionStatus,
    IndexRun,
    IndexVersion,
    IndexVersionStatus,
)
from app.document.ingestion_service import DocumentGroupNotFoundError
from app.document.group_source import GroupSourceView, list_group_sources
from app.document.job_gate import load_running_job
from app.indexing.index_builder import (
    load_active_document_versions,
    load_latest_ready_versions,
    load_pending_documents,
)


SEARCH_STATUS_UP_TO_DATE = "UP_TO_DATE"
SEARCH_STATUS_REINDEX_REQUIRED = "REINDEX_REQUIRED"
SEARCH_STATUS_IN_PROGRESS = "IN_PROGRESS"
SEARCH_STATUS_NO_DOCUMENTS = "NO_DOCUMENTS"
SEARCH_STATUS_FAILED = "FAILED"

APPLIED_STATUS_APPLIED = "APPLIED"
APPLIED_STATUS_UNAPPLIED = "UNAPPLIED"


@dataclass(frozen=True)
class GroupSummary:
    """목록 한 줄."""

    group_id: int
    group_key: str
    name: str
    consumer_key: str
    document_count: int
    active_index_version_no: Optional[int]
    search_status: str


@dataclass(frozen=True)
class ActiveIndexVersion:
    index_version_id: int
    version_no: Optional[int]


@dataclass(frozen=True)
class GroupDocument:
    """상세 표의 한 행."""

    document_id: int
    document_key: str
    title: str
    source_type: str
    group_source_id: Optional[int]
    document_version_no: int
    applied_version_no: Optional[int]
    applied_status: str


@dataclass(frozen=True)
class GroupDetail:
    """상세 화면 한 번의 응답."""

    group_id: int
    group_key: str
    name: str
    consumer_key: str
    sources: List[GroupSourceView]
    active_index_version: Optional[ActiveIndexVersion]
    pending_count: int
    search_status: str
    documents: List[GroupDocument]
    job_in_progress: bool


class DocumentGroupService:
    """문서 그룹 조회를 담당한다."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_groups(self) -> List[GroupSummary]:
        """그룹마다 문서 수와 검색 버전, 반영 상태를 계산한다."""

        groups = list(
            (
                await self._session.execute(
                    select(DocumentGroup).order_by(DocumentGroup.id)
                )
            ).scalars()
        )
        summaries = []
        for group in groups:
            latest = await load_latest_ready_versions(self._session, group.id)
            pending = await load_pending_documents(self._session, group.id)
            active = await self._active_index_version(group.id)
            summaries.append(
                GroupSummary(
                    group_id=group.id,
                    group_key=group.group_key,
                    name=group.name,
                    consumer_key=group.consumer_key,
                    document_count=len(latest),
                    active_index_version_no=(
                        None if active is None else active.version_no
                    ),
                    search_status=await self._search_status(
                        group.id,
                        len(latest),
                        len(pending),
                    ),
                )
            )
        return summaries

    async def get_group_detail(self, group_id: int) -> GroupDetail:
        """요약 카드와 문서 표를 한 번에 만든다."""

        group = await self._session.get(DocumentGroup, group_id)
        if group is None:
            raise DocumentGroupNotFoundError()

        latest = await load_latest_ready_versions(self._session, group.id)
        pending = await load_pending_documents(self._session, group.id)
        applied = await self._applied_version_no_by_source(group.id)
        documents = await self._documents(latest, applied)

        return GroupDetail(
            group_id=group.id,
            group_key=group.group_key,
            name=group.name,
            consumer_key=group.consumer_key,
            sources=await list_group_sources(self._session, group.id),
            active_index_version=await self._active_index_version(group.id),
            pending_count=sum(
                1
                for document in documents
                if document.applied_status == APPLIED_STATUS_UNAPPLIED
            ),
            search_status=await self._search_status(
                group.id,
                len(latest),
                len(pending),
            ),
            documents=documents,
            job_in_progress=(
                await load_running_job(self._session, group.id) is not None
            ),
        )

    async def _search_status(
        self,
        group_id: int,
        document_count: int,
        pending_count: int,
    ) -> str:
        """상태는 저장하지 않고 실행 이력과 문서 수로 계산한다."""

        latest_run = await self._latest_index_run(group_id)
        if latest_run is not None and latest_run.status == ExecutionStatus.PROCESSING:
            return SEARCH_STATUS_IN_PROGRESS
        if document_count == 0:
            return SEARCH_STATUS_NO_DOCUMENTS
        if latest_run is not None and latest_run.status == ExecutionStatus.FAILED:
            # 직전 반영이 실패했다는 뜻이고 반영이 필요한 상태를 포함한다.
            # 성공한 반영이 있기 전까지 유지된다.
            return SEARCH_STATUS_FAILED
        if pending_count == 0:
            return SEARCH_STATUS_UP_TO_DATE
        return SEARCH_STATUS_REINDEX_REQUIRED

    async def _active_index_version(
        self,
        group_id: int,
    ) -> Optional[ActiveIndexVersion]:
        index_version = await self._session.scalar(
            select(IndexVersion)
            .where(
                IndexVersion.document_group_id == group_id,
                IndexVersion.status == IndexVersionStatus.ACTIVE,
            )
            .limit(1)
        )
        if index_version is None:
            return None
        return ActiveIndexVersion(
            index_version_id=index_version.id,
            version_no=index_version.version_no,
        )

    async def _latest_index_run(self, group_id: int) -> Optional[IndexRun]:
        """검색 반영 상태 계산에만 쓴다. 응답으로 내보내지 않는다."""

        return await self._session.scalar(
            select(IndexRun)
            .join(IndexVersion, IndexVersion.id == IndexRun.index_version_id)
            .where(IndexVersion.document_group_id == group_id)
            .order_by(IndexRun.started_at.desc(), IndexRun.id.desc())
            .limit(1)
        )

    async def _applied_version_no_by_source(self, group_id: int) -> Dict[int, int]:
        """ACTIVE 색인에 든 문서의 판 번호를 원본별로 모은다."""

        active_version_ids = await load_active_document_versions(
            self._session,
            group_id,
        )
        if not active_version_ids:
            return {}

        rows = (
            await self._session.execute(
                select(
                    DocumentVersion.document_source_id,
                    DocumentVersion.version_no,
                ).where(DocumentVersion.id.in_(active_version_ids))
            )
        ).all()
        return {source_id: version_no for source_id, version_no in rows}

    async def _documents(
        self,
        latest: Dict[int, tuple],
        applied: Dict[int, int],
    ) -> List[GroupDocument]:
        """표에는 사용 중이고 READY 판이 있는 문서만 넣는다."""

        if not latest:
            return []

        sources = list(
            (
                await self._session.execute(
                    select(DocumentSource)
                    .where(DocumentSource.id.in_(list(latest)))
                    .order_by(DocumentSource.document_key)
                )
            ).scalars()
        )
        return [
            GroupDocument(
                document_id=source.id,
                document_key=source.document_key,
                title=source.title or "",
                source_type=source.source_type,
                group_source_id=source.group_source_id,
                document_version_no=latest[source.id][1],
                applied_version_no=applied.get(source.id),
                applied_status=(
                    APPLIED_STATUS_APPLIED
                    if applied.get(source.id) == latest[source.id][1]
                    else APPLIED_STATUS_UNAPPLIED
                ),
            )
            for source in sources
        ]
