"""FastAPI 요청 범위에서 ChatService 의존성 그래프를 구성한다."""

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db_session
from app.rag.chat_service import ChatService
from app.rag.corpus_state import CorpusState
from app.rag.generation_service import GenerationService
from app.rag.log_store import RagLogStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.embedding import OpenAIEmbedder
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.pgvector_store import PgVectorStore
from retrieval.vector_retriever import VectorRetriever


def get_corpus_state(request: Request) -> CorpusState:
    """애플리케이션 시작 시 생성한 CorpusState를 반환한다."""

    return request.app.state.corpus_state


def get_bm25_retriever(
    corpus_state: CorpusState = Depends(get_corpus_state),
) -> BM25Retriever:
    """적재된 BM25Retriever를 반환하고, 미적재면 CorpusNotLoadedError를 발생시킨다."""

    return corpus_state.get_retriever()


def get_index_version_id(
    corpus_state: CorpusState = Depends(get_corpus_state),
) -> int:
    """검색에 사용 중인 ACTIVE index version을 반환한다.

    rag_run에 남길 값이므로 BM25 corpus가 실제로 쓰는 index version과 같아야 한다.
    """

    return corpus_state.index_version_id


def get_rag_log_store(
    session: AsyncSession = Depends(get_db_session),
) -> RagLogStore:
    """요청의 DB session으로 실행 로그 저장 계층을 만든다."""

    return RagLogStore(session)


def get_embedder(request: Request) -> OpenAIEmbedder:
    """애플리케이션 시작 시 생성한 OpenAIEmbedder를 반환한다."""

    return request.app.state.embedder


def get_generation_service(request: Request) -> GenerationService:
    """애플리케이션 시작 시 생성한 GenerationService를 반환한다."""

    return request.app.state.generation_service


def get_vector_retriever(
    session: AsyncSession = Depends(get_db_session),
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> VectorRetriever:
    """요청의 DB session과 공유 Embedder로 VectorRetriever를 생성한다."""

    return VectorRetriever(
        embedder=embedder,
        store=PgVectorStore(session),
    )


def get_hybrid_retriever(
    bm25_retriever: BM25Retriever = Depends(get_bm25_retriever),
    vector_retriever: VectorRetriever = Depends(get_vector_retriever),
) -> HybridRetriever:
    """공유 BM25와 요청별 VectorRetriever를 결합한다."""

    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
    )


def get_chat_service(
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    generation_service: GenerationService = Depends(get_generation_service),
    log_store: RagLogStore = Depends(get_rag_log_store),
    session: AsyncSession = Depends(get_db_session),
    index_version_id: int = Depends(get_index_version_id),
) -> ChatService:
    """요청별 HybridRetriever와 공유 GenerationService, 실행 로그 저장을 연결한다.

    검색과 로그가 같은 session을 공유하므로 트랜잭션 경계를 ChatService가 직접
    관리한다 (RagLogStore는 commit하지 않는다).
    """

    return ChatService(
        retriever=retriever,
        generation_service=generation_service,
        log_store=log_store,
        session=session,
        index_version_id=index_version_id,
    )
