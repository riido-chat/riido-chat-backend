"""Admin 서비스의 FastAPI 의존성을 구성한다."""

from typing import Callable

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.document.ingestion_service import AdminIngestionService
from app.document.recollect_service import RecollectService
from app.indexing.index_service import IndexReindexService
from app.database.session import get_db_session
from app.retrieval.embedding import OpenAIEmbedder


def get_admin_ingestion_service(
    session: AsyncSession = Depends(get_db_session),
) -> AdminIngestionService:
    return AdminIngestionService(session)


def get_index_reindex_service(
    session: AsyncSession = Depends(get_db_session),
) -> IndexReindexService:
    return IndexReindexService(session)


def get_recollect_service(
    session: AsyncSession = Depends(get_db_session),
) -> RecollectService:
    return RecollectService(session)


def get_chunk_embedder_factory() -> Callable[[], OpenAIEmbedder]:
    """접수 시점 청크 embedding에 쓸 embedder 생성자를 준다.

    embedder는 업로드를 접수할 때가 아니라 background task가 실제로
    embedding을 만들 때 필요하다. 요청 경로에서 미리 만들면 OpenAI 설정이
    없을 때 접수 자체가 실패하므로 생성자만 넘긴다.
    의존성으로 두어 테스트에서 외부 호출 없이 대체할 수 있게 한다.
    """

    return OpenAIEmbedder
