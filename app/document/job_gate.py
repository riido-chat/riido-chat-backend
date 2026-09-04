"""문서 그룹 단위 작업 잠금과 실행 중 작업 확인을 제공한다.

업로드, 수정본, 검색 반영, 재탐색은 한 그룹에서 한 번에 하나만 돌 수 있다.
잠금은 advisory lock 이라 컨테이너가 여럿이어도 성립한다.
"""

from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    DocumentSource,
    ExecutionStatus,
    IndexRun,
    IndexVersion,
    IngestionRun,
)


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


async def find_processing_job(
    session: AsyncSession,
    group_id: int,
) -> Optional[str]:
    """그룹에서 진행 중인 작업 종류를 돌려준다. 없으면 None 이다."""

    ingestion_run_id = await session.scalar(
        select(IngestionRun.id)
        .join(
            DocumentSource,
            DocumentSource.id == IngestionRun.document_source_id,
        )
        .where(
            DocumentSource.document_group_id == group_id,
            IngestionRun.status == ExecutionStatus.PROCESSING,
        )
        .limit(1)
    )
    if ingestion_run_id is not None:
        return "INGESTION"

    index_run_id = await session.scalar(
        select(IndexRun.id)
        .join(IndexVersion, IndexVersion.id == IndexRun.index_version_id)
        .where(
            IndexVersion.document_group_id == group_id,
            IndexRun.status == ExecutionStatus.PROCESSING,
        )
        .limit(1)
    )
    if index_run_id is not None:
        return "INDEX"
    return None
