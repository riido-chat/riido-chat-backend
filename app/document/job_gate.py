"""문서 그룹 단위 작업 잠금과 실행 중 작업 확인을 제공한다.

업로드, 수정본, 검색 반영, 재탐색은 한 그룹에서 한 번에 하나만 돌 수 있다.
잠금은 advisory lock 이라 컨테이너가 여럿이어도 성립한다.
"""

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    DocumentSource,
    ExecutionStatus,
    IndexRun,
    IndexVersion,
    IngestionRun,
)


INGESTION_JOB = "INGESTION"
RECOLLECT_JOB = "RECOLLECT"
INDEX_JOB = "INDEX"


@dataclass(frozen=True)
class RunningJob:
    """그룹에서 진행 중인 작업 한 건.

    콘솔이 재진입 시 어떤 모달을 복원할지 정하는 근거다.
    """

    job_type: str
    stage: str
    ingestion_run_id: Optional[int] = None
    document_source_id: Optional[int] = None
    index_run_id: Optional[int] = None
    batch_id: Optional[UUID] = None


# advisory lock 의 두 인자 형태를 쓴다. 첫 인자는 이 프로젝트의 잠금 종류,
# 둘째 인자는 문서 그룹 ID 다. 둘 다 int4 범위여야 한다.
ADMIN_JOB_LOCK_NAMESPACE = 0x5249


async def acquire_group_job_gate(session: AsyncSession, group_id: int) -> None:
    """문서 그룹의 작업 잠금을 트랜잭션 범위로 잡는다."""

    await session.execute(
        select(
            func.pg_advisory_xact_lock(ADMIN_JOB_LOCK_NAMESPACE, group_id)
        )
    )


async def load_running_job(
    session: AsyncSession,
    group_id: int,
) -> Optional[RunningJob]:
    """그룹에서 진행 중인 작업을 찾는다. 없으면 None 이다.

    재탐색 배치는 batch_id 가 붙은 수집 실행이므로 같은 조회에서 갈라낸다.
    """

    row = (
        await session.execute(
            select(
                IngestionRun.id,
                IngestionRun.document_source_id,
                IngestionRun.stage,
                IngestionRun.batch_id,
            )
            .join(
                DocumentSource,
                DocumentSource.id == IngestionRun.document_source_id,
            )
            .where(
                DocumentSource.document_group_id == group_id,
                IngestionRun.status == ExecutionStatus.PROCESSING,
            )
            .order_by(IngestionRun.id)
            .limit(1)
        )
    ).first()
    if row is not None:
        run_id, source_id, stage, batch_id = row
        if batch_id is not None:
            return RunningJob(
                job_type=RECOLLECT_JOB,
                stage="PROCESSING",
                batch_id=batch_id,
            )
        return RunningJob(
            job_type=INGESTION_JOB,
            stage=stage.value,
            ingestion_run_id=run_id,
            document_source_id=source_id,
        )

    index_row = (
        await session.execute(
            select(IndexRun.id, IndexRun.stage)
            .join(IndexVersion, IndexVersion.id == IndexRun.index_version_id)
            .where(
                IndexVersion.document_group_id == group_id,
                IndexRun.status == ExecutionStatus.PROCESSING,
            )
            .order_by(IndexRun.id)
            .limit(1)
        )
    ).first()
    if index_row is not None:
        run_id, stage = index_row
        return RunningJob(
            job_type=INDEX_JOB,
            stage=stage.value,
            index_run_id=run_id,
        )
    return None


async def find_processing_job(
    session: AsyncSession,
    group_id: int,
) -> Optional[str]:
    """그룹에 진행 중인 작업이 있는지만 확인한다."""

    job = await load_running_job(session, group_id)
    return None if job is None else job.job_type
