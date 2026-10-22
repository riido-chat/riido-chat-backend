"""임베딩 설정 버전과 그 설정 행, 벡터 차원 검증을 관리한다."""

from datetime import datetime
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import EmbeddingConfig
from app.retrieval.embedding import (
    OPENAI_EMBEDDING_DIMENSIONS,
    OPENAI_EMBEDDING_MODEL,
)


EMBEDDING_CONFIG_VERSION = "openai-text-embedding-3-large-1536-v1"
EMBEDDING_INPUT_TEMPLATE_VERSION = "document-section-content-v1"


async def get_or_create_embedding_config(
    session: AsyncSession,
    now: datetime,
) -> EmbeddingConfig:
    """현재 OpenAI 임베딩 설정 행을 조회하고 없으면 만든다."""

    config = await session.scalar(
        select(EmbeddingConfig).where(
            EmbeddingConfig.version == EMBEDDING_CONFIG_VERSION
        )
    )
    if config is not None:
        if (
            config.provider != "openai"
            or config.model_name != OPENAI_EMBEDDING_MODEL
            or config.dimensions != OPENAI_EMBEDDING_DIMENSIONS
            or config.input_template_version
            != EMBEDDING_INPUT_TEMPLATE_VERSION
        ):
            raise ValueError("기존 EmbeddingConfig가 현재 OpenAI 설정과 다릅니다.")
        return config

    config = EmbeddingConfig(
        version=EMBEDDING_CONFIG_VERSION,
        provider="openai",
        model_name=OPENAI_EMBEDDING_MODEL,
        dimensions=OPENAI_EMBEDDING_DIMENSIONS,
        input_template_version=EMBEDDING_INPUT_TEMPLATE_VERSION,
        parameters={"encoding_format": "float"},
        created_at=now,
    )
    session.add(config)
    await session.flush()
    return config


def validate_embedding_dimension(embedding: Sequence[float]) -> None:
    """저장하거나 조회하는 벡터가 확정 차원인지 확인한다."""

    if len(embedding) != OPENAI_EMBEDDING_DIMENSIONS:
        raise ValueError(
            "embedding은 "
            f"{OPENAI_EMBEDDING_DIMENSIONS}차원이어야 합니다."
        )
