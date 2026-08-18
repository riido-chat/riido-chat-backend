from typing import Union

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db_session


router = APIRouter(tags=["health"])


@router.get("/health", summary="애플리케이션 상태 확인")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/health/db",
    response_model=None,
    summary="데이터베이스 연결 상태 확인",
)
async def database_health_check(
    session: AsyncSession = Depends(get_db_session),
) -> Union[dict[str, str], JSONResponse]:
    """간단한 쿼리로 데이터베이스 연결 상태를 확인한다."""

    try:
        await session.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "database": "unavailable"},
        )

    return {"status": "ok", "database": "connected"}
