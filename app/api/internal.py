"""운영용 corpus 상태 조회와 재적재 endpoint를 제공한다."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from app.chat.dependencies import get_testing_chat_service
from app.chat.schema import ChatRequest, ChatResponse
from app.chat.service import ChatService
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db_session
from app.retrieval.corpus_state import CorpusRegistry, CorpusSnapshot, CorpusState
from app.chat.dependencies import get_corpus_registry, get_corpus_state
from app.retrieval.search_reader import ActiveIndexNotFoundError, SearchReader


router = APIRouter(prefix="/internal", tags=["internal"])
test_chat_router = APIRouter(prefix="/api/internal", tags=["internal"])


class CorpusStatusResponse(BaseModel):
    """corpus 적재 상태."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    loaded: bool
    chunk_count: int = Field(alias="chunkCount")
    document_count: int = Field(alias="documentCount")
    loaded_at: Optional[datetime] = Field(alias="loadedAt")
    source: str

    @classmethod
    def from_snapshot(cls, snapshot: CorpusSnapshot) -> "CorpusStatusResponse":
        return cls(
            loaded=snapshot.loaded,
            chunkCount=snapshot.chunk_count,
            documentCount=snapshot.document_count,
            loadedAt=snapshot.loaded_at,
            source=snapshot.source,
        )


@router.get("/corpus", summary="검색 corpus 적재 상태 확인")
async def read_corpus_status(
    corpus_state: CorpusState = Depends(get_corpus_state),
    corpus_registry: CorpusRegistry | None = Depends(get_corpus_registry),
    document_group_id: int | None = Query(default=None, alias="documentGroupId"),
) -> CorpusStatusResponse:
    if corpus_registry is not None and not corpus_registry.is_legacy_compat:
        if document_group_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="documentGroupId가 필요합니다.",
            )
        snapshot = corpus_registry.snapshot(document_group_id)
    else:
        snapshot = corpus_state.snapshot()
    return CorpusStatusResponse.from_snapshot(snapshot)


@router.post("/corpus/reload", summary="검색 corpus 재적재")
async def reload_corpus(
    corpus_state: CorpusState = Depends(get_corpus_state),
    session: AsyncSession = Depends(get_db_session),
    corpus_registry: CorpusRegistry | None = Depends(get_corpus_registry),
    document_group_id: int | None = Query(default=None, alias="documentGroupId"),
) -> CorpusStatusResponse:
    """ACTIVE index의 Chunk로 BM25 인덱스를 교체한다."""

    try:
        if (
            corpus_registry is not None
            and not corpus_registry.is_legacy_compat
            and document_group_id is None
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="documentGroupId가 필요합니다.",
            )
        reader = SearchReader(session, document_group_id=document_group_id)
        chunks = await reader.load_active_chunks()
        if (
            document_group_id is not None
            and corpus_registry is not None
            and not corpus_registry.is_legacy_compat
        ):
            snapshot = corpus_registry.replace(document_group_id, chunks)
        else:
            snapshot = corpus_state.replace(chunks)
    except (ActiveIndexNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return CorpusStatusResponse.from_snapshot(snapshot)


@test_chat_router.post(
    "/test-chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="TESTING chat profile로 답변 생성 (internal)",
)
async def test_chat(
    request: ChatRequest,
    service: ChatService = Depends(get_testing_chat_service),
) -> ChatResponse:
    """Run the normal ChatService against the HELP_CHATBOT TESTING revision.

    This route intentionally exposes no group_id or revision_id input.  It is
    an internal operational hook; authentication is left to the deployment
    boundary because this repository has no auth middleware.
    """

    return await service.answer_question(
        request.question,
        request.conversation_id,
    )
