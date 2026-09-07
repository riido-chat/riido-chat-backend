"""재탐색 접수와 배치 조회를 담당한다."""

import logging
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    DocumentGroupSource,
    DocumentSource,
    ExecutionStatus,
    IngestionResultCode,
    IngestionRun,
)
from app.document.document_group import get_document_group
from app.document.document_key import normalize_gitbook_root_url
from app.document.gitbook.client import GitBookListError, list_pages
from app.document.ingestion_service import (
    INVALID_FILE,
    AdminApiError,
    AdminJobInProgressError,
    DocumentGroupNotFoundError,
)
from app.document.job_gate import acquire_group_job_gate, find_processing_job
from app.document.recollect import (
    REMOVED_ACTION,
    UPSTREAM_ERROR,
    AcceptedRecollect,
    accept_recollect_batch,
)


logger = logging.getLogger(__name__)

SOURCE_LIST_FAILED = "SOURCE_LIST_FAILED"
NOT_FOUND = "NOT_FOUND"


class SourceListFailedError(AdminApiError):
    """페이지 목록을 읽지 못했다. 수집을 시작하지 않는다.

    원인이 여럿이라 일시 장애만 가정하는 표현을 쓰지 않는다.
    루트 URL 은 요청마다 다르므로 문장에 넣지 않는다.
    """

    def __init__(self, reason: str) -> None:
        logger.warning("GitBook 페이지 목록 조회에 실패했습니다: %s", reason)
        super().__init__(
            SOURCE_LIST_FAILED,
            "GitBook 페이지 목록을 읽지 못했습니다."
            " 다시 시도하거나 GitBook 을 확인해 주세요.",
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
    """실패한 페이지 한 건. 목록 행에 그대로 그린다."""

    document_key: str
    title: str
    ingestion_run_id: int
    message: str


@dataclass(frozen=True)
class RecollectBatchDetail:
    """재탐색 배치의 진행과 집계."""

    batch_id: uuid.UUID
    group_id: int
    group_source_id: Optional[int]
    root_url: Optional[str]
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

    async def read_finished_batch(
        self,
        batch_id: uuid.UUID,
    ) -> RecollectBatchDetail:
        """수집이 다른 세션에서 마감한 배치를 읽는다.

        run_recollect_batch 가 자기 세션으로 커밋하므로 요청 세션의
        identity map 을 비운 뒤 다시 조회한다.
        """

        self._session.expire_all()
        return await self.get_batch(batch_id)

    async def get_batch(self, batch_id: uuid.UUID) -> RecollectBatchDetail:
        """배치에 묶인 수집 실행에서 진행과 집계를 조립한다."""

        rows = (
            await self._session.execute(
                select(IngestionRun, DocumentSource, DocumentGroupSource)
                .join(
                    DocumentSource,
                    DocumentSource.id == IngestionRun.document_source_id,
                )
                .outerjoin(
                    DocumentGroupSource,
                    DocumentGroupSource.id == DocumentSource.group_source_id,
                )
                .where(IngestionRun.batch_id == batch_id)
                .order_by(IngestionRun.id)
            )
        ).all()
        if not rows:
            raise RecollectBatchNotFoundError()

        runs = [run for run, _, _ in rows]
        group_source = next(
            (source for _, _, source in rows if source is not None),
            None,
        )
        finished = [run for run in runs if run.status != ExecutionStatus.PROCESSING]
        detail = RecollectBatchDetail(
            batch_id=batch_id,
            group_id=rows[0][1].document_group_id,
            group_source_id=None if group_source is None else group_source.id,
            root_url=None if group_source is None else group_source.root_url,
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
            group_source_id=detail.group_source_id,
            root_url=detail.root_url,
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
        group = await get_document_group(self._session, group_id)
        if group is None:
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
        elif run.result_code in (
            IngestionResultCode.NO_CHANGE,
            # 중복은 코퍼스에 아무 변화를 만들지 않으므로 변경 없음으로 센다.
            # 중복 여부는 개별 실행의 result_code 로 확인한다.
            IngestionResultCode.DUPLICATE_CONTENT,
        ):
            counts["no_change"] += 1
    return counts


# 실패 목록은 행 폭이 좁아 오류 모달의 전문이 들어가지 않는다.
# 원인 코드는 ingestion_runs 의 error_code 에만 남는다.
_FAILURE_MESSAGES = {
    UPSTREAM_ERROR: "페이지를 읽지 못했습니다.",
    INVALID_FILE: "문서 내용을 처리할 수 없습니다.",
}
_DEFAULT_FAILURE_MESSAGE = "문제가 발생했습니다."


def _failures_of(rows) -> List[RecollectFailure]:
    return [
        RecollectFailure(
            document_key=source.document_key,
            title=source.title or source.document_key,
            ingestion_run_id=run.id,
            message=_FAILURE_MESSAGES.get(
                run.error_code,
                _DEFAULT_FAILURE_MESSAGE,
            ),
        )
        for run, source, _ in rows
        if run.status == ExecutionStatus.FAILED
    ]
