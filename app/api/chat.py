"""Chat API MVP endpoint를 제공한다."""

from fastapi import APIRouter, Depends, Response, status

from app.api.chat_schema import (
    ChatError,
    ChatErrorCode,
    ChatErrorResponse,
    ChatRequest,
    ChatResponse,
    ChatResponseStatus,
)
from app.rag.chat_service import ChatService
from app.rag.dependencies import get_chat_service


router = APIRouter(tags=["chat"])

CORPUS_UNAVAILABLE_MESSAGE = "검색 데이터가 아직 준비되지 않았습니다."


def corpus_unavailable_response() -> ChatErrorResponse:
    """corpus 미적재로 답변할 수 없을 때의 응답을 만든다.

    의존성 해석 단계에서 막히므로 rag_run이 없고 두 식별자는 null이다.
    """

    return ChatErrorResponse(
        status=ChatResponseStatus.ERROR,
        conversation_id=None,
        rag_run_id=None,
        answer=None,
        error=ChatError(
            code=ChatErrorCode.SERVICE_UNAVAILABLE,
            message=CORPUS_UNAVAILABLE_MESSAGE,
        ),
        citations=[],
    )


@router.post(
    "/api/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ChatErrorResponse,
            "description": "답변 생성 중 기술 오류",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ChatErrorResponse,
            "description": "이어갈 수 없는 conversationId",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ChatErrorResponse,
            "description": "검색 corpus 미적재",
        },
    },
    summary="이용가이드 기반 답변 생성",
)
async def chat(
    request: ChatRequest,
    response: Response,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """질문을 ChatService에 전달하고 결과 상태에 맞는 HTTP 응답을 반환한다."""

    result = await service.answer_question(
        request.question,
        request.conversation_id,
    )
    if isinstance(result, ChatErrorResponse):
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return result
