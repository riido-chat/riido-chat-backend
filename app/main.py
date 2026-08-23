import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.chat import corpus_unavailable_response
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.internal import router as internal_router
from app.core.config import get_settings
from app.database.session import dispose_engine
from app.rag.corpus_state import CorpusNotLoadedError, CorpusState
from app.rag.generation_service import GenerationService
from generation.generator import OpenAIGenerator
from retrieval.embedding import OpenAIEmbedder


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        corpus_state = CorpusState(get_settings().corpus_dir)
        app.state.corpus_state = corpus_state
        _load_corpus_if_available(corpus_state)
        app.state.embedder = OpenAIEmbedder()
        app.state.generation_service = GenerationService(OpenAIGenerator())
        yield
    finally:
        await dispose_engine()


def _load_corpus_if_available(corpus_state: CorpusState) -> None:
    """corpus가 준비되어 있으면 적재하고, 없으면 미적재 상태로 기동한다."""

    try:
        snapshot = corpus_state.load()
    except (FileNotFoundError, ValueError) as exc:
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
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    app.include_router(health_router)
    app.include_router(chat_router)
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

    return app


app = create_app()
