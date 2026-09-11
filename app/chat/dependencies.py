"""FastAPI 요청 범위에서 ChatService 의존성 그래프를 구성한다."""

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db_session
from app.chat.service import ChatService
from app.retrieval.corpus_state import CorpusRegistry, CorpusState
from app.answering.service import GenerationService
from app.chat.log_store import RagLogStore
from app.chat.query_rewrite import QueryRewriteService
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.embedding import OpenAIEmbedder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.search_reader import SearchReader
from app.retrieval.vector_retriever import VectorRetriever
from app.database.models import ChatProfileRevisionStatus


def _build_group_retriever(
    corpus_registry: CorpusRegistry,
    session: AsyncSession,
    embedder: OpenAIEmbedder,
    document_group_id: int,
):
    bm25_retriever, snapshot_id = corpus_registry.get_search_snapshot(
        document_group_id
    )
    return (
        HybridRetriever(
            bm25_retriever=bm25_retriever,
            vector_retriever=VectorRetriever(
                embedder=embedder,
                store=SearchReader(
                    session,
                    document_group_id=document_group_id,
                    index_version_id=snapshot_id,
                ),
            ),
        ),
        snapshot_id,
    )


def get_corpus_state(request: Request) -> CorpusState:
    """애플리케이션 시작 시 생성한 CorpusState를 반환한다."""

    return request.app.state.corpus_state


def get_corpus_registry(request: Request) -> CorpusRegistry | None:
    """애플리케이션 시작 시 생성한 그룹별 BM25 registry를 반환한다."""

    registry = getattr(request.app.state, "corpus_registry", None)
    return registry


def get_bm25_retriever(
    corpus_state: CorpusState = Depends(get_corpus_state),
    corpus_registry: CorpusRegistry | None = Depends(get_corpus_registry),
) -> BM25Retriever | None:
    """적재된 BM25Retriever를 반환하고, 미적재면 CorpusNotLoadedError를 발생시킨다."""

    if corpus_registry is not None:
        if not corpus_registry.is_legacy_compat:
            return None
        return corpus_registry.state_for(0).get_retriever()
    return corpus_state.get_retriever()


def get_index_version_id(
    corpus_state: CorpusState = Depends(get_corpus_state),
    corpus_registry: CorpusRegistry | None = Depends(get_corpus_registry),
) -> int | None:
    """검색에 사용 중인 ACTIVE index version을 반환한다.

    rag_run에 남길 값이므로 BM25 corpus가 실제로 쓰는 index version과 같아야 한다.
    """

    if corpus_registry is not None:
        if not corpus_registry.is_legacy_compat:
            return None
        return corpus_registry.state_for(0).index_version_id
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


def get_query_rewrite_service(request: Request) -> QueryRewriteService:
    """애플리케이션 시작 시 생성한 QueryRewriteService를 반환한다."""

    return request.app.state.query_rewrite_service


def get_vector_retriever(
    session: AsyncSession = Depends(get_db_session),
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> VectorRetriever:
    """요청의 DB session과 공유 Embedder로 VectorRetriever를 생성한다."""

    return VectorRetriever(
        embedder=embedder,
        store=SearchReader(session),
    )


def get_hybrid_retriever(
    bm25_retriever: BM25Retriever | None = Depends(get_bm25_retriever),
    session: AsyncSession = Depends(get_db_session),
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> HybridRetriever | None:
    """공유 BM25와 요청별 VectorRetriever를 결합한다."""

    if bm25_retriever is None:
        return None
    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=VectorRetriever(
            embedder=embedder,
            store=SearchReader(session),
        ),
    )


def get_chat_service(
    retriever: HybridRetriever | None = Depends(get_hybrid_retriever),
    generation_service: GenerationService = Depends(get_generation_service),
    query_rewrite_service: QueryRewriteService = Depends(
        get_query_rewrite_service
    ),
    log_store: RagLogStore = Depends(get_rag_log_store),
    session: AsyncSession = Depends(get_db_session),
    embedder: OpenAIEmbedder = Depends(get_embedder),
    index_version_id: int | None = Depends(get_index_version_id),
    corpus_registry: CorpusRegistry | None = Depends(get_corpus_registry),
) -> ChatService:
    """요청별 검색·로그와 공유 Generation·Query Rewrite 서비스를 연결한다.

    검색과 로그가 같은 session을 공유하므로 트랜잭션 경계를 ChatService가 직접
    관리한다 (RagLogStore는 commit하지 않는다).
    """

    kwargs = {}
    if corpus_registry is not None:
        def build_for_group(group_id: int):
            return _build_group_retriever(
                corpus_registry, session, embedder, group_id
            )

        kwargs["retriever_factory"] = build_for_group
    return ChatService(
        retriever=retriever,
        generation_service=generation_service,
        query_rewrite_service=query_rewrite_service,
        log_store=log_store,
        session=session,
        index_version_id=index_version_id,
        profile_status=ChatProfileRevisionStatus.PUBLISHED,
        **kwargs,
    )


def get_testing_chat_service(
    generation_service: GenerationService = Depends(get_generation_service),
    query_rewrite_service: QueryRewriteService = Depends(
        get_query_rewrite_service
    ),
    session: AsyncSession = Depends(get_db_session),
    embedder: OpenAIEmbedder = Depends(get_embedder),
    corpus_registry: CorpusRegistry = Depends(get_corpus_registry),
) -> ChatService:
    """Internal test endpoint's ChatService.

    Profile resolution happens inside ChatService once the TESTING revision and
    its document group are known, so this dependency never accepts a group or
    revision id from the request.
    """

    return ChatService(
        retriever=None,
        generation_service=generation_service,
        query_rewrite_service=query_rewrite_service,
        log_store=RagLogStore(session),
        session=session,
        index_version_id=None,
        profile_status=ChatProfileRevisionStatus.TESTING,
        retriever_factory=lambda group_id: _build_group_retriever(
            corpus_registry, session, embedder, group_id
        ),
    )


def build_chat_service(
    *,
    session: AsyncSession,
    corpus_state: CorpusState,
    embedder: OpenAIEmbedder,
    generation_service: GenerationService,
    query_rewrite_service: QueryRewriteService,
    profile_status: ChatProfileRevisionStatus = ChatProfileRevisionStatus.PUBLISHED,
    corpus_registry: CorpusRegistry | None = None,
) -> ChatService:
    """요청 의존성 밖에서 ChatService를 조립한다.

    응답 스트림보다 오래 사는 파이프라인은 request-scoped session을 쓸 수 없어서,
    호출자가 직접 소유한 session을 받아 검색·로그 계층에 연결한다. 공유 부품인
    BM25 corpus, Embedder, GenerationService, QueryRewriteService는 인자로 받는다.
    """

    kwargs = {}
    if corpus_registry is not None:
        kwargs["retriever_factory"] = lambda group_id: _build_group_retriever(
            corpus_registry, session, embedder, group_id
        )
    use_legacy = corpus_registry is None or corpus_registry.is_legacy_compat
    legacy_index_version_id = corpus_state.index_version_id if use_legacy else None
    legacy_bm25_retriever = corpus_state.get_retriever() if use_legacy else None
    legacy_retriever = (
        None
        if legacy_bm25_retriever is None
        else HybridRetriever(
            bm25_retriever=legacy_bm25_retriever,
            vector_retriever=VectorRetriever(
                embedder=embedder,
                store=SearchReader(session),
            ),
        )
    )
    return ChatService(
        retriever=legacy_retriever,
        generation_service=generation_service,
        query_rewrite_service=query_rewrite_service,
        log_store=RagLogStore(session),
        session=session,
        index_version_id=legacy_index_version_id,
        profile_status=profile_status,
        **kwargs,
    )
