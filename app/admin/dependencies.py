"""Admin 서비스의 FastAPI 의존성을 구성한다."""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.document.ingestion_service import AdminIngestionService
from app.database.session import get_db_session
from app.retrieval.embedding import OpenAIEmbedder


def get_admin_ingestion_service(
    session: AsyncSession = Depends(get_db_session),
) -> AdminIngestionService:
    return AdminIngestionService(session)


def get_chunk_embedder() -> OpenAIEmbedder:
    """접수 시점 청크 embedding에 쓸 embedder를 만든다.

    의존성으로 두어 테스트에서 외부 호출 없이 대체할 수 있게 한다.
    """

    return OpenAIEmbedder()
