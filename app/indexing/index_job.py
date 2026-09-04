"""검색 반영 실행을 독립 세션의 background task 로 진행한다."""

import logging
from typing import Callable

from app.database.session import get_session_factory
from app.indexing.index_builder import run_index_job
from app.retrieval.corpus_state import CorpusState
from app.retrieval.embedding import OpenAIEmbedder


logger = logging.getLogger(__name__)


async def run_admin_index_job(
    index_run_id: int,
    corpus_state: CorpusState,
    embedder_factory: Callable[[], OpenAIEmbedder] = OpenAIEmbedder,
) -> None:
    """요청 세션과 분리된 세션에서 색인 생성과 적용을 끝까지 진행한다."""

    async with get_session_factory()() as session:
        try:
            await run_index_job(
                session,
                corpus_state,
                index_run_id,
                embedder_factory,
            )
        except Exception:
            logger.exception(
                "검색 반영 실행을 진행하지 못했습니다: index_run_id=%s",
                index_run_id,
            )
            await session.rollback()
