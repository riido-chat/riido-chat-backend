import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.document.ingestion_service import AdminApiError
from app.admin.router import router as admin_documents_router
from app.admin.schema import AdminErrorCode, AdminErrorResponse
from app.chat.router import corpus_unavailable_response
from app.chat.router import router as chat_router
from app.chat.feedback import (
    feedback_not_allowed_response,
    rag_run_not_found_response,
)
from app.chat.feedback import router as feedback_router
from app.api.health import router as health_router
from app.api.internal import router as internal_router
from app.chat.rag_run import rag_run_result_not_found_response
from app.chat.rag_run import router as rag_run_router
from app.core.config import get_settings
from app.database.session import dispose_engine, get_session_factory
from app.chat.service import (
    ConversationNotFoundError,
    conversation_busy_response,
    conversation_not_found_response,
)
from app.retrieval.corpus_state import CorpusNotLoadedError, CorpusState
from app.answering.service import GenerationService
from app.chat.log_store import (
    ConversationBusyError,
    FeedbackNotAllowedError,
    RagRunNotFoundError,
)
from app.chat.query_rewrite import QueryRewriteService
from app.chat.rag_run_view import RagRunResultNotFoundError
from app.answering.generator import OpenAIGenerator
from app.retrieval.embedding import OpenAIEmbedder
from app.retrieval.pgvector_store import ActiveIndexNotFoundError, PgVectorStore


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # 응답 스트림보다 오래 사는 파이프라인 task를 앱이 소유한다.
    app.state.pipeline_tasks = set()
    try:
        corpus_state = CorpusState(get_settings().corpus_dir)
        app.state.corpus_state = corpus_state
        await _load_corpus_if_available(corpus_state)
        app.state.embedder = OpenAIEmbedder()
        app.state.generation_service = GenerationService(OpenAIGenerator())
        app.state.query_rewrite_service = QueryRewriteService()
        yield
    finally:
        # 살아 있는 파이프라인이 끝난 뒤에 engine을 정리해야 세션이 깨지지 않는다.
        await _drain_pipeline_tasks(app)
        await dispose_engine()


async def _drain_pipeline_tasks(app: FastAPI) -> None:
    """실행 중인 파이프라인 task가 모두 끝날 때까지 기다린다."""

    pending = set(getattr(app.state, "pipeline_tasks", None) or ())
    if not pending:
        return

    logger.info("실행 중인 파이프라인 %d건을 기다립니다.", len(pending))
    # 하나가 실패해도 나머지 정리를 막지 않는다. 예외는 producer가 이미 처리했다.
    await asyncio.gather(*pending, return_exceptions=True)


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
    app.include_router(rag_run_router)
    app.include_router(internal_router)
    app.include_router(admin_documents_router)

    @app.exception_handler(AdminApiError)
    async def handle_admin_api_error(
        _: Request,
        exc: AdminApiError,
    ) -> JSONResponse:
        logger.info("Admin 요청을 처리할 수 없습니다: %s", exc)
        return JSONResponse(
            status_code=exc.status_code,
            content=AdminErrorResponse(
                code=AdminErrorCode(exc.code),
                message=exc.message,
            ).model_dump(mode="json", by_alias=True),
        )

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

    @app.exception_handler(ConversationBusyError)
    async def handle_conversation_busy(
        _: Request,
        exc: ConversationBusyError,
    ) -> JSONResponse:
        logger.info("처리 중인 대화로 중복 요청을 받았습니다: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=conversation_busy_response(exc.conversation_id).model_dump(
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

    @app.exception_handler(RagRunResultNotFoundError)
    async def handle_rag_run_result_not_found(
        _: Request,
        exc: RagRunResultNotFoundError,
    ) -> JSONResponse:
        logger.info("존재하지 않는 답변의 결과를 조회했습니다: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=rag_run_result_not_found_response().model_dump(
                mode="json",
                by_alias=True,
            ),
        )

    return app


app = create_app()
