"""Admin 서비스의 FastAPI 의존성을 구성한다."""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.ingestion_service import AdminIngestionService
from app.database.session import get_db_session


def get_admin_ingestion_service(
    session: AsyncSession = Depends(get_db_session),
) -> AdminIngestionService:
    return AdminIngestionService(session)
