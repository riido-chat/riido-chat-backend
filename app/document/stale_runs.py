"""기동 시 남아 있는 PROCESSING 실행을 정리한다.

실행 중에 프로세스가 죽으면 그 실행을 이어서 진행할 주체가 없는데도
PROCESSING 으로 남는다. 작업 잠금이 이 행을 보고 막으므로 그룹이 영구히
잠긴다. 재탐색 배치가 길어지면서 실제로 겪을 수 있는 상태가 됐다.

기동 시점에 살아 있는 실행은 없으므로 남은 것은 모두 중단된 실행이다.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    ExecutionStatus,
    IndexRun,
    IndexVersion,
    IndexVersionStatus,
    IngestionRun,
)


logger = logging.getLogger(__name__)

INTERRUPTED_ERROR_CODE = "INTERNAL_ERROR"
INTERRUPTED_MESSAGE = "실행 중 서버가 종료되어 마감하지 못했습니다."


async def close_interrupted_runs(session: AsyncSession) -> tuple:
    """중단된 수집 실행과 색인 실행을 FAILED 로 마감한다.

    검색 버전은 READY 를 건드리지 않는다. 적용 단계에서 끊긴 후보는 그대로
    두어야 적용 재시도가 가능하다.
    """

    ingestion_ids = list(
        (
            await session.execute(
                select(IngestionRun.id).where(
                    IngestionRun.status == ExecutionStatus.PROCESSING
                )
            )
        ).scalars()
    )
    index_ids = list(
        (
            await session.execute(
                select(IndexRun.id).where(
                    IndexRun.status == ExecutionStatus.PROCESSING
                )
            )
        ).scalars()
    )
    if not ingestion_ids and not index_ids:
        return (0, 0)

    now = datetime.now(timezone.utc)

    if ingestion_ids:
        await session.execute(
            update(IngestionRun)
            .where(IngestionRun.id.in_(ingestion_ids))
            .values(
                status=ExecutionStatus.FAILED,
                error_code=INTERRUPTED_ERROR_CODE,
                error_message=INTERRUPTED_MESSAGE,
                finished_at=now,
            )
        )

    if index_ids:
        # 후보를 만들다 끊긴 경우만 FAILED 로 내린다. READY 후보는 살려 둔다.
        await session.execute(
            update(IndexVersion)
            .where(
                IndexVersion.id.in_(
                    select(IndexRun.index_version_id).where(
                        IndexRun.id.in_(index_ids)
                    )
                ),
                IndexVersion.status.in_(
                    (
                        IndexVersionStatus.BUILDING,
                        IndexVersionStatus.VALIDATING,
                    )
                ),
            )
            .values(status=IndexVersionStatus.FAILED)
        )
        await session.execute(
            update(IndexRun)
            .where(IndexRun.id.in_(index_ids))
            .values(
                status=ExecutionStatus.FAILED,
                error_code=INTERRUPTED_ERROR_CODE,
                error_message=INTERRUPTED_MESSAGE,
                finished_at=now,
            )
        )

    await session.commit()
    logger.warning(
        "중단된 실행을 마감했습니다: 수집 %d건, 색인 %d건",
        len(ingestion_ids),
        len(index_ids),
    )
    return (len(ingestion_ids), len(index_ids))
