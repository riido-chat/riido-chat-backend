import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.chat import corpus_unavailable_response
from app.api.chat import router as chat_router
from app.api.feedback import (
    feedback_not_allowed_response,
    rag_run_not_found_response,
)
from app.api.feedback import router as feedback_router
from app.api.health import router as health_router
from app.api.internal import router as internal_router
from app.core.config import get_settings
from app.database.session import dispose_engine, get_session_factory
from app.rag.chat_service import (
    ConversationNotFoundError,
    conversation_not_found_response,
)
from app.rag.corpus_state import CorpusNotLoadedError, CorpusState
from app.rag.generation_service import GenerationService
from app.rag.log_store import FeedbackNotAllowedError, RagRunNotFoundError
from generation.generator import OpenAIGenerator
from retrieval.embedding import OpenAIEmbedder
from retrieval.pgvector_store import ActiveIndexNotFoundError, PgVectorStore


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        corpus_state = CorpusState(get_settings().corpus_dir)
        app.state.corpus_state = corpus_state
        await _load_corpus_if_available(corpus_state)
        app.state.embedder = OpenAIEmbedder()
        app.state.generation_service = GenerationService(OpenAIGenerator())
        yield
    finally:
        await dispose_engine()


async def _load_corpus_if_available(corpus_state: CorpusState) -> None:
    """ACTIVE index가 준비되어 있으면 BM25 corpus를 적재한다."""

    try:
        async with get_session_factory()() as session:
            chunks = await PgVectorStore(session).load_active_chunks()
        snapshot = corpus_state.replace(chunks)
    except (ActiveIndexNotFoundError, ValueError) as exc:
        logger.warning("corpus 미적재 상태로 기동합니다: %s", exc)
        return

    logger.info(
        "corpus 적재 완료: 문서 %d개, Chunk %d개",
        snapshot.document_count,
        snapshot.chunk_count,
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="Riido RAG Chatbot API",
        version="0.1.0",
        lifespan=lifespan,
    )
    # 브라우저에서 다른 오리진의 FE가 호출하므로 허용 목록을 명시한다.
    # 로그인이 없어 쿠키를 쓰지 않으므로 credentials는 허용하지 않는다.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(feedback_router)
    app.include_router(internal_router)

    @app.exception_handler(CorpusNotLoadedError)
    async def handle_corpus_not_loaded(
        _: Request,
        exc: CorpusNotLoadedError,
    ) -> JSONResponse:
        logger.warning("corpus 미적재 상태에서 요청을 받았습니다: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=corpus_unavailable_response().model_dump(
                mode="json",
                by_alias=True,
            ),
        )

    @app.exception_handler(ConversationNotFoundError)
    async def handle_conversation_not_found(
        _: Request,
        exc: ConversationNotFoundError,
    ) -> JSONResponse:
        logger.info("이어갈 수 없는 대화로 요청을 받았습니다: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=conversation_not_found_response().model_dump(
                mode="json",
                by_alias=True,
            ),
        )

    @app.exception_handler(RagRunNotFoundError)
    async def handle_rag_run_not_found(
        _: Request,
        exc: RagRunNotFoundError,
    ) -> JSONResponse:
        logger.info("존재하지 않는 답변에 피드백 요청을 받았습니다: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=rag_run_not_found_response().model_dump(
                mode="json",
                by_alias=True,
            ),
        )

    @app.exception_handler(FeedbackNotAllowedError)
    async def handle_feedback_not_allowed(
        _: Request,
        exc: FeedbackNotAllowedError,
    ) -> JSONResponse:
        logger.info("평가할 수 없는 답변에 피드백 요청을 받았습니다: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=feedback_not_allowed_response().model_dump(
                mode="json",
                by_alias=True,
            ),
        )

    return app


app = create_app()
