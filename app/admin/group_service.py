"""운영자 콘솔의 문서 그룹 목록과 상세를 조립한다.

문서와 색인, 실행 데이터가 모두 필요한 읽기 전용 모델이라 두 도메인을
합치는 admin 계층에 둔다. 상태 값은 저장하지 않고 조회 시점에 계산한다.
"""

from dataclasses import dataclass
from datetime import datetime
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
from app.document.job_gate import RunningJob, load_running_job
from app.indexing.index_builder import (
    load_active_document_versions,
    load_latest_ready_versions,
    load_pending_documents,
)


SEARCH_STATUS_UP_TO_DATE = "UP_TO_DATE"
SEARCH_STATUS_REINDEX_REQUIRED = "REINDEX_REQUIRED"
SEARCH_STATUS_IN_PROGRESS = "IN_PROGRESS"
SEARCH_STATUS_NO_DOCUMENTS = "NO_DOCUMENTS"


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
    activated_at: Optional[datetime]


@dataclass(frozen=True)
class PendingDocumentView:
    document_id: int
    title: str
    change_type: str


@dataclass(frozen=True)
class GroupDocument:
    """상세 표의 한 행."""

    document_id: int
    document_key: str
    title: str
    source_type: str
    document_version_no: int
    applied_version_no: Optional[int]
    processing_status: str


@dataclass(frozen=True)
class LatestIndexRun:
    index_run_id: int
    index_version_id: int
    operation_type: str
    status: ExecutionStatus
    stage: str
    error_code: Optional[str]
    started_at: datetime
    finished_at: Optional[datetime]


@dataclass(frozen=True)
class GroupDetail:
    """상세 화면 한 번의 응답."""

    group_id: int
    group_key: str
    name: str
    consumer_key: str
    active_index_version: Optional[ActiveIndexVersion]
    pending_documents: List[PendingDocumentView]
    search_status: str
    documents: List[GroupDocument]
    running_job: Optional[RunningJob]
    latest_index_run: Optional[LatestIndexRun]


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
            latest = await load_latest_ready_versions(self._session)
            pending = await load_pending_documents(self._session)
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
        """요약 카드와 문서 표, 재진입 복원 정보를 한 번에 만든다."""

        group = await self._session.get(DocumentGroup, group_id)
        if group is None:
            raise DocumentGroupNotFoundError()

        latest = await load_latest_ready_versions(self._session)
        pending = await load_pending_documents(self._session)
        applied = await self._applied_version_no_by_source()
        documents = await self._documents(latest, applied)

        return GroupDetail(
            group_id=group.id,
            group_key=group.group_key,
            name=group.name,
            consumer_key=group.consumer_key,
            active_index_version=await self._active_index_version(group.id),
            pending_documents=[
                PendingDocumentView(
                    document_id=item.document_source_id,
                    title=item.title,
                    change_type=item.change_type,
                )
                for item in pending
            ],
            search_status=await self._search_status(
                group.id,
                len(latest),
                len(pending),
            ),
            documents=documents,
            running_job=await load_running_job(self._session, group.id),
            latest_index_run=await self._latest_index_run(group.id),
        )

    async def _search_status(
        self,
        group_id: int,
        document_count: int,
        pending_count: int,
    ) -> str:
        """부록 A 의 계산 규칙을 그대로 따른다."""

        latest_run = await self._latest_index_run(group_id)
        if latest_run is not None and latest_run.status == ExecutionStatus.PROCESSING:
            return SEARCH_STATUS_IN_PROGRESS
        if document_count == 0:
            return SEARCH_STATUS_NO_DOCUMENTS
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
            activated_at=index_version.activated_at,
        )

    async def _latest_index_run(
        self,
        group_id: int,
    ) -> Optional[LatestIndexRun]:
        run = await self._session.scalar(
            select(IndexRun)
            .join(IndexVersion, IndexVersion.id == IndexRun.index_version_id)
            .where(IndexVersion.document_group_id == group_id)
            .order_by(IndexRun.started_at.desc(), IndexRun.id.desc())
            .limit(1)
        )
        if run is None:
            return None
        return LatestIndexRun(
            index_run_id=run.id,
            index_version_id=run.index_version_id,
            operation_type=run.operation_type.value,
            status=run.status,
            stage=run.stage.value,
            error_code=run.error_code,
            started_at=run.started_at,
            finished_at=run.finished_at,
        )

    async def _applied_version_no_by_source(self) -> Dict[int, int]:
        """ACTIVE 색인에 든 문서의 판 번호를 원본별로 모은다."""

        active_version_ids = await load_active_document_versions(self._session)
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
                document_version_no=latest[source.id][1],
                applied_version_no=applied.get(source.id),
                processing_status="READY",
            )
            for source in sources
        ]
