"""SQLAlchemy 비동기 엔진과 요청 단위 세션을 관리한다."""

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    """환경 변수의 DATABASE_URL을 사용하는 비동기 엔진을 반환한다."""

    return create_async_engine(
        get_settings().database_url,
        pool_pre_ping=True,
    )


@lru_cache
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """요청 단위 AsyncSession을 생성하는 팩토리를 반환한다."""

    return async_sessionmaker(
        bind=get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI 의존성으로 사용할 요청 단위 DB 세션을 제공한다."""

    async with get_session_factory()() as session:
        yield session


async def dispose_engine() -> None:
    """애플리케이션 종료 시 연결 풀을 정리한다."""

    if get_engine.cache_info().currsize:
        await get_engine().dispose()

    get_session_factory.cache_clear()
    get_engine.cache_clear()
