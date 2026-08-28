"""Chat API MVP endpoint를 제공한다."""

from typing import Union

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import StreamingResponse

from app.api.chat_schema import (
    ChatError,
    ChatErrorCode,
    ChatErrorResponse,
    ChatRequest,
    ChatResponse,
    ChatResponseStatus,
)
from app.api.chat_stream import start_chat_stream, wants_event_stream
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
        status.HTTP_409_CONFLICT: {
            "model": ChatErrorResponse,
            "description": "같은 대화의 이전 질문 처리 중",
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
    http_request: Request,
    service: ChatService = Depends(get_chat_service),
) -> Union[ChatResponse, StreamingResponse]:
    """질문을 ChatService에 전달하고 결과 상태에 맞는 HTTP 응답을 반환한다.

    Accept에 `text/event-stream`을 명시한 요청만 진행 상태 SSE로 분기한다.
    턴 생성 전에 끝나면 스트림을 열지 않고 기존 동기 오류 응답을 그대로 쓴다.
    """

    if wants_event_stream(http_request.headers.get("accept")):
        stream = await start_chat_stream(
            http_request,
            question=request.question,
            conversation_id=request.conversation_id,
        )
        if isinstance(stream, StreamingResponse):
            return stream
        if stream.error is not None:
            raise stream.error
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return stream.response

    result = await service.answer_question(
        request.question,
        request.conversation_id,
    )
    if isinstance(result, ChatErrorResponse):
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return result
