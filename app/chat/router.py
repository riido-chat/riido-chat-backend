"""Chat API MVP endpoint를 제공한다."""

import asyncio
import uuid
from typing import Optional, Union

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import StreamingResponse

from app.chat.schema import (
    ChatErrorCode,
    ChatErrorResponse,
    ChatRequest,
    ChatResponse,
    ChatResponseStatus,
)
from app.chat.stream import start_chat_stream, wants_event_stream
from app.chat.service import ChatService, chat_error_response
from app.chat.dependencies import get_chat_service


router = APIRouter(tags=["chat"])

async def _wait_for_disconnect(request: Request) -> None:
    """클라이언트가 일반 HTTP 요청 연결을 끊을 때까지 기다린다."""

    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


async def _answer_until_disconnect(
    request: Request,
    service: ChatService,
    question: str,
    conversation_id: Optional[uuid.UUID],
) -> Optional[ChatResponse]:
    """답변 완료와 클라이언트 연결 종료 중 먼저 발생하는 쪽을 처리한다."""

    answer_task = asyncio.create_task(
        service.answer_question(question, conversation_id)
    )
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request))

    try:
        done, _ = await asyncio.wait(
            (answer_task, disconnect_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        # 완료와 연결 종료가 같이 관측되면 이미 확정된 답변을 우선한다.
        if answer_task in done:
            return await answer_task

        await disconnect_task
        answer_task.cancel()
        await asyncio.gather(answer_task, return_exceptions=True)
        return None
    finally:
        for task in (answer_task, disconnect_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(
            answer_task,
            disconnect_task,
            return_exceptions=True,
        )


def corpus_unavailable_response() -> ChatErrorResponse:
    """corpus 미적재로 답변할 수 없을 때의 응답을 만든다.

    의존성 해석 단계에서 막히므로 rag_run이 없고 두 식별자는 null이다.
    """

    return chat_error_response(ChatErrorCode.SERVICE_UNAVAILABLE)


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
) -> Union[ChatResponse, Response]:
    """질문을 ChatService에 전달하고 결과 상태에 맞는 HTTP 응답을 반환한다.

    Accept에 `text/event-stream`을 명시한 요청만 진행 상태 SSE로 분기한다.
    턴 생성 전에 끝나면 스트림을 열지 않고 기존 동기 오류 응답을 그대로 쓴다. (현재 SSE는 보류)
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

    result = await _answer_until_disconnect(
        http_request,
        service,
        request.question,
        request.conversation_id,
    )
    if result is None:
        # 클라이언트가 이미 연결을 끊어 실제 응답은 전송되지 않는다.
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    if isinstance(result, ChatErrorResponse):
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return result
