"""운영자 콘솔의 검색 반영 API 를 뒷받침한다.

접수 단계 검증과 실행 행 생성, 실행 조회 응답 조립을 맡는다.
실제 단계 진행은 index_builder 가 background task 로 수행한다.
"""

from dataclasses import dataclass, replace
from datetime import datetime
from http import HTTPStatus
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    ExecutionStatus,
    IndexDocument,
    IndexRun,
    IndexRunStage,
    IndexVersion,
    IndexVersionStatus,
    IngestionRun,
)
from app.document.document_group import get_document_group
from app.document.ingestion_service import (
    AdminApiError,
    AdminJobInProgressError,
)
from app.document.job_gate import acquire_group_job_gate, find_processing_job
from app.indexing.index_builder import (
    compute_pending_documents,
    load_latest_ready_versions,
    start_retry_apply_run,
    start_reindex_run,
)


NOT_FOUND = "NOT_FOUND"
REINDEX_NOT_REQUIRED = "REINDEX_NOT_REQUIRED"
NO_READY_DOCUMENTS = "NO_READY_DOCUMENTS"
RETRY_NOT_ALLOWED = "RETRY_NOT_ALLOWED"


class DocumentGroupNotFoundError(AdminApiError):
    def __init__(self) -> None:
        super().__init__(
            NOT_FOUND,
            "존재하지 않는 문서 그룹입니다.",
            HTTPStatus.NOT_FOUND,
        )


class IndexRunNotFoundError(AdminApiError):
    def __init__(self) -> None:
        super().__init__(
            NOT_FOUND,
            "존재하지 않는 검색 반영 실행입니다.",
            HTTPStatus.NOT_FOUND,
        )


class ReindexNotRequiredError(AdminApiError):
    def __init__(self) -> None:
        super().__init__(
            REINDEX_NOT_REQUIRED,
            "검색에 반영할 변경이 없습니다.",
            HTTPStatus.CONFLICT,
        )


class NoReadyDocumentsError(AdminApiError):
    def __init__(self) -> None:
        super().__init__(
            NO_READY_DOCUMENTS,
            "검색에 반영할 준비된 문서가 없습니다.",
            HTTPStatus.CONFLICT,
        )


class RetryNotAllowedError(AdminApiError):
    def __init__(self) -> None:
        super().__init__(
            RETRY_NOT_ALLOWED,
            "적용 단계에서 실패한 실행만 다시 시도할 수 있습니다.",
            HTTPStatus.CONFLICT,
        )


@dataclass(frozen=True)
class AcceptedIndexRun:
    """반영 시작과 재시도 접수 결과."""

    index_run_id: int
    index_version_id: int
    group_id: int
    operation_type: str
    trigger_type: str
    stage: str
    retry_of_index_run_id: Optional[int] = None


@dataclass(frozen=True)
class IndexVersionSummary:
    index_version_id: int
    version_no: Optional[int]
    status: str
    activated_at: Optional[datetime]


@dataclass(frozen=True)
class IndexRunDetail:
    """검색 반영 실행 조회 결과."""

    index_run_id: int
    group_id: int
    index_version_id: int
    operation_type: str
    trigger_type: str
    status: ExecutionStatus
    stage: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    index_version: Optional[IndexVersionSummary] = None
    previous_index_version: Optional[IndexVersionSummary] = None
    document_count: Optional[int] = None
    chunk_count: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retryable: Optional[bool] = None


class IndexReindexService:
    """검색 반영 접수와 조회를 담당한다."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def start_reindex(
        self,
        group_id: int,
        *,
        actor_id: Optional[str] = None,
    ) -> AcceptedIndexRun:
        """반영 대기 변경을 확인하고 BUILD_AND_APPLY 실행을 시작한다."""

        group = await self._get_group(group_id)
        await self._acquire_group_gate(group.id)
        await self._ensure_no_processing_job(group.id)

        if not await load_latest_ready_versions(self._session, group.id):
            raise NoReadyDocumentsError()
        pending = await compute_pending_documents(self._session, group.id)
        if pending.total == 0:
            raise ReindexNotRequiredError()

        run = await start_reindex_run(
            self._session,
            group.id,
            actor_id=actor_id,
        )
        accepted = AcceptedIndexRun(
            index_run_id=run.id,
            index_version_id=run.index_version_id,
            group_id=group.id,
            operation_type=run.operation_type.value,
            trigger_type=run.trigger_type,
            stage=run.stage.value,
        )
        await self._session.commit()
        return accepted

    async def start_retry_apply(
        self,
        index_run_id: int,
        *,
        actor_id: Optional[str] = None,
    ) -> AcceptedIndexRun:
        """적용 단계에서 실패한 실행의 후보에 적용 전용 실행을 만든다."""

        failed_run = await self._session.get(IndexRun, index_run_id)
        if failed_run is None:
            raise IndexRunNotFoundError()

        index_version = await self._session.get(
            IndexVersion,
            failed_run.index_version_id,
        )
        if index_version is not None:
            await self._acquire_group_gate(index_version.document_group_id)
            await self._ensure_no_processing_job(index_version.document_group_id)

        if (
            failed_run.status != ExecutionStatus.FAILED
            or failed_run.stage != IndexRunStage.APPLYING
            or index_version is None
            or index_version.status != IndexVersionStatus.READY
        ):
            raise RetryNotAllowedError()

        run = await start_retry_apply_run(
            self._session,
            failed_run,
            actor_id=actor_id,
        )
        accepted = AcceptedIndexRun(
            index_run_id=run.id,
            index_version_id=run.index_version_id,
            group_id=index_version.document_group_id,
            operation_type=run.operation_type.value,
            trigger_type=run.trigger_type,
            stage=run.stage.value,
            retry_of_index_run_id=index_run_id,
        )
        await self._session.commit()
        return accepted

    async def get_index_run(self, index_run_id: int) -> IndexRunDetail:
        """실행 한 건의 진행과 결과를 조립한다."""

        run = await self._session.get(IndexRun, index_run_id)
        if run is None:
            raise IndexRunNotFoundError()

        index_version = await self._session.get(
            IndexVersion,
            run.index_version_id,
        )
        if index_version is None:
            raise IndexRunNotFoundError()

        detail = IndexRunDetail(
            index_run_id=run.id,
            group_id=index_version.document_group_id,
            index_version_id=index_version.id,
            operation_type=run.operation_type.value,
            trigger_type=run.trigger_type,
            status=run.status,
            stage=run.stage.value,
            started_at=run.started_at,
            finished_at=run.finished_at,
        )
        if run.status == ExecutionStatus.PROCESSING:
            return detail

        summary = IndexVersionSummary(
            index_version_id=index_version.id,
            version_no=index_version.version_no,
            status=index_version.status.value,
            activated_at=index_version.activated_at,
        )
        if run.status == ExecutionStatus.FAILED:
            return replace(
                detail,
                index_version=summary,
                error_code=run.error_code,
                error_message=run.error_message,
                retryable=(
                    run.stage == IndexRunStage.APPLYING
                    and index_version.status == IndexVersionStatus.READY
                ),
            )

        document_count = await self._session.scalar(
            select(func.count())
            .select_from(IndexDocument)
            .where(IndexDocument.index_version_id == index_version.id)
        )
        run_summary = run.summary or {}
        return replace(
            detail,
            index_version=summary,
            previous_index_version=await self._previous_active(run_summary),
            document_count=document_count,
            chunk_count=run_summary.get("chunk_count"),
        )

    async def _previous_active(
        self,
        run_summary: dict,
    ) -> Optional[IndexVersionSummary]:
        """이번 적용이 끌어내린 직전 ACTIVE 를 실행 기록에서 읽는다.

        현재 상태에서 역산하면 이후 다른 적용이 일어났을 때 과거 실행의
        조회 결과가 달라진다. 적용 시점에 남긴 값을 그대로 쓴다.
        """

        previous_id = run_summary.get("previous_index_version_id")
        if previous_id is None:
            return None

        previous = await self._session.get(IndexVersion, previous_id)
        if previous is None:
            return None
        return IndexVersionSummary(
            index_version_id=previous.id,
            version_no=previous.version_no,
            status=previous.status.value,
            activated_at=previous.activated_at,
        )

    async def _get_group(self, group_id: int):
        group = await get_document_group(self._session, group_id)
        if group is None:
            raise DocumentGroupNotFoundError()
        return group

    async def _acquire_group_gate(self, group_id: int) -> None:
        await acquire_group_job_gate(self._session, group_id)

    async def _ensure_no_processing_job(self, group_id: int) -> None:
        if await find_processing_job(self._session, group_id) is not None:
            raise AdminJobInProgressError()
