"""문서 청킹 설정 버전과 그 설정 행을 관리한다."""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import ChunkingConfig


CHUNKING_CONFIG_VERSION = "section-v1"
CHUNKING_STRATEGY = "SECTION"


async def get_or_create_chunking_config(
    session: AsyncSession,
    now: datetime,
) -> ChunkingConfig:
    """현재 SECTION 청킹 설정 행을 조회하고 없으면 만든다."""

    config = await session.scalar(
        select(ChunkingConfig).where(
            ChunkingConfig.version == CHUNKING_CONFIG_VERSION
        )
    )
    if config is not None:
        if (
            config.strategy != CHUNKING_STRATEGY
            or config.max_tokens != 0
            or config.overlap_tokens != 0
        ):
            raise ValueError("기존 ChunkingConfig가 현재 SECTION 설정과 다릅니다.")
        return config

    config = ChunkingConfig(
        version=CHUNKING_CONFIG_VERSION,
        strategy=CHUNKING_STRATEGY,
        max_tokens=0,
        overlap_tokens=0,
        parameters={"split": False, "section_boundary": "H2"},
        created_at=now,
    )
    session.add(config)
    await session.flush()
    return config
