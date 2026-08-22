"""Chat API MVP endpoint를 제공한다."""

from fastapi import APIRouter, Depends, Response, status

from app.api.chat_schema import (
    ChatErrorResponse,
    ChatRequest,
    ChatResponse,
)
from app.rag.chat_service import ChatService
from app.rag.dependencies import get_chat_service


router = APIRouter(tags=["chat"])


@router.post(
    "/api/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ChatErrorResponse,
            "description": "답변 생성 중 기술 오류",
        }
    },
    summary="이용가이드 기반 답변 생성",
)
async def chat(
    request: ChatRequest,
    response: Response,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """질문을 ChatService에 전달하고 결과 상태에 맞는 HTTP 응답을 반환한다."""

    result = await service.answer_question(request.question)
    if isinstance(result, ChatErrorResponse):
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return result
