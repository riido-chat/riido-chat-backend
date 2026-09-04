"""운영용 corpus 상태 조회와 재적재 endpoint를 제공한다."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db_session
from app.retrieval.corpus_state import CorpusSnapshot, CorpusState
from app.chat.dependencies import get_corpus_state
from app.retrieval.pgvector_store import ActiveIndexNotFoundError, PgVectorStore


router = APIRouter(prefix="/internal", tags=["internal"])


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
) -> CorpusStatusResponse:
    return CorpusStatusResponse.from_snapshot(corpus_state.snapshot())


@router.post("/corpus/reload", summary="검색 corpus 재적재")
async def reload_corpus(
    corpus_state: CorpusState = Depends(get_corpus_state),
    session: AsyncSession = Depends(get_db_session),
) -> CorpusStatusResponse:
    """ACTIVE index의 Chunk로 BM25 인덱스를 교체한다."""

    try:
        chunks = await PgVectorStore(session).load_active_chunks()
        snapshot = corpus_state.replace(chunks)
    except (ActiveIndexNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return CorpusStatusResponse.from_snapshot(snapshot)
