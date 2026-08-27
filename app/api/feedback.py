"""답변 피드백 등록, 변경, 해제 endpoint를 제공한다."""

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.feedback_schema import (
    ChatFeedbackErrorCode,
    ChatFeedbackRating,
    FeedbackErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from app.database.models import FeedbackRating
from app.database.session import get_db_session
from app.rag.dependencies import get_rag_log_store
from app.rag.log_store import RagLogStore


router = APIRouter(tags=["feedback"])

FEEDBACK_PATH = "/api/chat/{rag_run_id}/feedback"
RAG_RUN_NOT_FOUND_MESSAGE = "존재하지 않는 답변입니다."
FEEDBACK_NOT_ALLOWED_MESSAGE = "평가할 수 없는 답변입니다."

FEEDBACK_ERROR_RESPONSES = {
    status.HTTP_404_NOT_FOUND: {
        "model": FeedbackErrorResponse,
        "description": "존재하지 않는 ragRunId",
    },
    status.HTTP_409_CONFLICT: {
        "model": FeedbackErrorResponse,
        "description": "완료 또는 보류가 아닌 답변",
    },
}


def rag_run_not_found_response() -> FeedbackErrorResponse:
    """존재하지 않는 ragRunId로 요청했을 때의 응답을 만든다."""

    return FeedbackErrorResponse(
        code=ChatFeedbackErrorCode.NOT_FOUND,
        message=RAG_RUN_NOT_FOUND_MESSAGE,
    )


def feedback_not_allowed_response() -> FeedbackErrorResponse:
    """완료나 보류가 아닌 답변에 평가를 시도했을 때의 응답을 만든다."""

    return FeedbackErrorResponse(
        code=ChatFeedbackErrorCode.FEEDBACK_NOT_ALLOWED,
        message=FEEDBACK_NOT_ALLOWED_MESSAGE,
    )


@router.put(
    FEEDBACK_PATH,
    response_model=FeedbackResponse,
    status_code=status.HTTP_200_OK,
    responses=FEEDBACK_ERROR_RESPONSES,
    summary="답변 평가 등록과 변경",
)
async def put_feedback(
    rag_run_id: uuid.UUID,
    request: FeedbackRequest,
    log_store: RagLogStore = Depends(get_rag_log_store),
    session: AsyncSession = Depends(get_db_session),
) -> FeedbackResponse:
    """평가를 등록하거나 반대 값으로 바꾼다. 같은 값 재전송은 그대로 통과한다."""

    feedback = await log_store.set_feedback(
        rag_run_id,
        rating=FeedbackRating(request.rating.value),
    )
    rating = ChatFeedbackRating(feedback.rating.value)
    await session.commit()

    return FeedbackResponse(rag_run_id=rag_run_id, rating=rating)


@router.delete(
    FEEDBACK_PATH,
    response_model=FeedbackResponse,
    status_code=status.HTTP_200_OK,
    responses=FEEDBACK_ERROR_RESPONSES,
    summary="답변 평가 해제",
)
async def delete_feedback(
    rag_run_id: uuid.UUID,
    log_store: RagLogStore = Depends(get_rag_log_store),
    session: AsyncSession = Depends(get_db_session),
) -> FeedbackResponse:
    """평가를 지운다. 이미 없어도 같은 응답을 돌려준다."""

    await log_store.clear_feedback(rag_run_id)
    await session.commit()

    return FeedbackResponse(rag_run_id=rag_run_id, rating=None)
