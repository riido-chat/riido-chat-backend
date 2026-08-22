from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.database.session import dispose_engine
from app.rag.generation_service import GenerationService
from generation.generator import OpenAIGenerator
from retrieval.bm25_retriever import BM25Retriever
from retrieval.corpus import build_retrieval_chunks
from retrieval.embedding import OpenAIEmbedder


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        retrieval_corpus = build_retrieval_chunks()
        app.state.retrieval_corpus = retrieval_corpus
        app.state.bm25_retriever = BM25Retriever(retrieval_corpus)
        app.state.embedder = OpenAIEmbedder()
        app.state.generation_service = GenerationService(OpenAIGenerator())
        yield
    finally:
        await dispose_engine()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Riido RAG Chatbot API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(chat_router)
    return app


app = create_app()
