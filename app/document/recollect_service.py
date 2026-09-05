"""재탐색 접수와 배치 조회를 담당한다."""

import uuid
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    DocumentSource,
    ExecutionStatus,
    IngestionResultCode,
    IngestionRun,
)
from app.document.document_group import get_document_group
from app.document.document_key import normalize_gitbook_root_url
from app.document.gitbook.client import GitBookListError, list_pages
from app.document.ingestion_service import (
    AdminApiError,
    AdminJobInProgressError,
    DocumentGroupNotFoundError,
)
from app.document.job_gate import acquire_group_job_gate, find_processing_job
from app.document.recollect import (
    REMOVED_ACTION,
    AcceptedRecollect,
    accept_recollect_batch,
)


SOURCE_LIST_FAILED = "SOURCE_LIST_FAILED"
NOT_FOUND = "NOT_FOUND"


class SourceListFailedError(AdminApiError):
    def __init__(self, message: str) -> None:
        super().__init__(
            SOURCE_LIST_FAILED,
            f"docs.riido.io 페이지 목록을 읽지 못했습니다: {message}",
            HTTPStatus.BAD_GATEWAY,
        )


class RecollectBatchNotFoundError(AdminApiError):
    def __init__(self) -> None:
        super().__init__(
            NOT_FOUND,
            "존재하지 않는 재탐색 배치입니다.",
            HTTPStatus.NOT_FOUND,
        )


@dataclass(frozen=True)
class RecollectFailure:
    document_key: str
    title: str
    ingestion_run_id: int
    stage: str
    error_code: Optional[str]


@dataclass(frozen=True)
class RecollectBatchDetail:
    """재탐색 배치의 진행과 집계."""

    batch_id: uuid.UUID
    group_id: int
    status: ExecutionStatus
    total: int
    processed: int
    started_at: object
    finished_at: object = None
    counts: Optional[Dict[str, int]] = None
    failures: tuple = ()


class RecollectService:
    """GitBook 재탐색 접수와 배치 조회를 담당한다."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def start_sync(
        self,
        group_id: int,
        root_url: str,
    ) -> AcceptedRecollect:
        """GitBook 루트의 페이지 목록을 읽고 페이지별 실행을 만든다.

        목록 조회는 배치 전체의 전제라 접수 요청 안에서 끝낸다. 실패하면
        실행을 하나도 만들지 않고 요청 자체를 거절한다.
        """

        # 한 그룹이 여러 GitBook 을 가질 수 있다. 문서 키는 원천 안에서만
        # 유일하므로 서로 다른 GitBook 이 같은 경로를 가져도 겹치지 않는다.
        root = normalize_gitbook_root_url(root_url)
        group = await self._get_group(group_id)
        await acquire_group_job_gate(self._session, group.id)
        if await find_processing_job(self._session, group.id) is not None:
            raise AdminJobInProgressError()

        try:
            pages = list_pages(root)
        except GitBookListError as error:
            await self._session.rollback()
            raise SourceListFailedError(str(error)) from error

        return await accept_recollect_batch(
            self._session,
            group.id,
            root,
            pages,
        )

    async def get_batch(self, batch_id: uuid.UUID) -> RecollectBatchDetail:
        """배치에 묶인 수집 실행에서 진행과 집계를 조립한다."""

        rows = (
            await self._session.execute(
                select(IngestionRun, DocumentSource)
                .join(
                    DocumentSource,
                    DocumentSource.id == IngestionRun.document_source_id,
                )
                .where(IngestionRun.batch_id == batch_id)
                .order_by(IngestionRun.id)
            )
        ).all()
        if not rows:
            raise RecollectBatchNotFoundError()

        runs = [run for run, _ in rows]
        finished = [run for run in runs if run.status != ExecutionStatus.PROCESSING]
        detail = RecollectBatchDetail(
            batch_id=batch_id,
            group_id=rows[0][1].document_group_id,
            status=(
                ExecutionStatus.PROCESSING
                if len(finished) < len(runs)
                else ExecutionStatus.SUCCESS
            ),
            total=len(runs),
            processed=len(finished),
            started_at=min(run.started_at for run in runs),
        )
        if detail.status == ExecutionStatus.PROCESSING:
            return detail

        return RecollectBatchDetail(
            batch_id=detail.batch_id,
            group_id=detail.group_id,
            status=detail.status,
            total=detail.total,
            processed=detail.processed,
            started_at=detail.started_at,
            finished_at=max(
                run.finished_at for run in runs if run.finished_at is not None
            ),
            counts=_count_results(runs),
            failures=tuple(_failures_of(rows)),
        )

    async def _get_group(self, group_id: int):
        group = await get_document_group(self._session)
        if group.id != group_id:
            raise DocumentGroupNotFoundError()
        return group


def _is_removed(run: IngestionRun) -> bool:
    return (run.summary or {}).get("recollect_action") == REMOVED_ACTION


def _count_results(runs: List[IngestionRun]) -> Dict[str, int]:
    """배치 결과를 화면이 쓰는 다섯 갈래로 센다."""

    counts = {
        "total": 0,
        "created": 0,
        "updated": 0,
        "no_change": 0,
        "removed": 0,
        "failed": 0,
    }
    for run in runs:
        if _is_removed(run):
            counts["removed"] += 1
            continue

        # total 은 GitBook 에서 읽은 페이지 수다. 제거 표시는 세지 않는다
        counts["total"] += 1
        if run.status == ExecutionStatus.FAILED:
            counts["failed"] += 1
        elif run.result_code == IngestionResultCode.CREATED:
            counts["created"] += 1
        elif run.result_code == IngestionResultCode.UPDATED:
            counts["updated"] += 1
        elif run.result_code == IngestionResultCode.NO_CHANGE:
            counts["no_change"] += 1
    return counts


def _failures_of(rows) -> List[RecollectFailure]:
    return [
        RecollectFailure(
            document_key=source.document_key,
            title=source.title or source.document_key,
            ingestion_run_id=run.id,
            stage=run.stage.value,
            error_code=run.error_code,
        )
        for run, source in rows
        if run.status == ExecutionStatus.FAILED
    ]
