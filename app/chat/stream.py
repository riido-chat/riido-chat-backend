"""진행 상태를 SSE로 중계하는 전달 계층.

답변 본문을 스트리밍하지 않는다. Citation 검증 게이트 때문에 본문은 검증
후에만 확정되므로, 파이프라인 진행 단계만 중계하고 최종 답변은 터미널
이벤트로 한 번에 보낸다.

전달 계약은 `ordered / at-most-once / best-effort / no replay`다.

- ordered      : 단일 producer 코루틴 + 단일 Queue라 put 순서가 곧 get 순서다.
- at-most-once : 각 이벤트를 정확히 한 번 put하고 재전송하지 않는다.
- best-effort  : 소비자가 사라져도 put이 막히거나 실패하지 않는다(maxsize 미지정).
- no replay    : Queue는 요청 단위 지역 객체이며 어디에도 저장하지 않는다.

SSE를 걷어내야 하면 이 파일과 코어의 훅 인자만 제거하면 된다.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from fastapi import Request
from fastapi.responses import StreamingResponse

from app.chat.schema import ChatResponse
from app.core.task_registry import register_pipeline_task
from app.database.session import get_session_factory
from app.chat.service import ChatService, _internal_error_response
from app.retrieval.corpus_state import CorpusState
from app.chat.dependencies import build_chat_service
from app.answering.service import GenerationService
from app.chat.progress import ProgressStage
from app.chat.query_rewrite import QueryRewriteService
from app.retrieval.embedding import OpenAIEmbedder


logger = logging.getLogger(__name__)

SSE_MEDIA_TYPE = "text/event-stream"
SSE_ACCEPT_TOKEN = "text/event-stream"

EVENT_RUN = "run"
EVENT_STAGE = "stage"
EVENT_RESULT = "result"
EVENT_ERROR = "error"


# ----------------------------------------------------------------------
# 이벤트
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RunEvent:
    """턴 식별자. 스트림이 열렸다면 반드시 첫 이벤트다."""

    conversation_id: uuid.UUID
    rag_run_id: uuid.UUID


@dataclass(frozen=True)
class StageEvent:
    """진행 단계. 실제 실행 지점에서만 발생한다."""

    stage: ProgressStage


@dataclass(frozen=True)
class ResultEvent:
    """터미널. 파이프라인이 정상 마감한 결과이며 status=ERROR도 여기로 온다."""

    response: ChatResponse


@dataclass(frozen=True)
class ErrorEvent:
    """전달 계층 파손 전용 최후 수단. 파이프라인 오류는 ResultEvent로 간다."""

    response: ChatResponse


@dataclass(frozen=True)
class TurnStartFailed:
    """턴 생성 전에 끝난 경우. 스트림을 열지 않고 기존 HTTP 오류로 응답한다.

    `error`가 None이면 `response`가 반드시 채워진다.
    """

    error: Optional[BaseException] = None
    response: Optional[ChatResponse] = None


class _Sentinel:
    """스트림 종료 신호. 와이어에는 나가지 않는다."""


SENTINEL = _Sentinel()

StreamEvent = Union[RunEvent, StageEvent, ResultEvent, ErrorEvent]


def wants_event_stream(accept: Optional[str]) -> bool:
    """Accept에 SSE가 명시된 요청만 스트림으로 분기한다. `*/*`는 해당 없다."""

    return SSE_ACCEPT_TOKEN in (accept or "")


def encode_event(event: StreamEvent) -> bytes:
    """이벤트를 SSE 와이어 형식으로 만든다."""

    name, payload = _event_payload(event)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {data}\n\n".encode("utf-8")


def _event_payload(event: StreamEvent) -> "tuple":
    if isinstance(event, RunEvent):
        return EVENT_RUN, {
            "conversationId": str(event.conversation_id),
            "ragRunId": str(event.rag_run_id),
        }
    if isinstance(event, StageEvent):
        return EVENT_STAGE, {"progressStage": event.stage.value}
    if isinstance(event, ResultEvent):
        return EVENT_RESULT, _dump_response(event.response)
    if isinstance(event, ErrorEvent):
        return EVENT_ERROR, _dump_response(event.response)
    raise ValueError(f"지원하지 않는 이벤트입니다: {event!r}")


def _dump_response(response: ChatResponse) -> Dict[str, Any]:
    """터미널 payload는 동기 응답 DTO 직렬화를 그대로 재사용한다."""

    return response.model_dump(mode="json", by_alias=True)


# ----------------------------------------------------------------------
# producer — 세션을 직접 소유하고 파이프라인을 완주시킨다
# ----------------------------------------------------------------------


@asynccontextmanager
async def _chat_service_scope(
    corpus_state: CorpusState,
    embedder: OpenAIEmbedder,
    generation_service: GenerationService,
    query_rewrite_service: QueryRewriteService,
) -> AsyncIterator[ChatService]:
    """producer가 소유하는 session으로 ChatService를 만든다.

    응답 스트림이 먼저 끝나도 session이 살아 있어야 파이프라인이 완주한다.
    """

    async with get_session_factory()() as session:
        yield build_chat_service(
            session=session,
            corpus_state=corpus_state,
            embedder=embedder,
            generation_service=generation_service,
            query_rewrite_service=query_rewrite_service,
        )


async def produce_turn(
    queue: "asyncio.Queue",
    *,
    question: str,
    conversation_id: Optional[uuid.UUID],
    corpus_state: CorpusState,
    embedder: OpenAIEmbedder,
    generation_service: GenerationService,
    query_rewrite_service: QueryRewriteService,
) -> None:
    """파이프라인을 끝까지 실행하며 이벤트를 Queue에 넣는다.

    클라이언트가 끊겨도 취소하지 않고 터미널까지 완주한다. 어떤 경로로 끝나도
    finally에서 sentinel을 넣는다.
    """

    started = _TurnIdentity()
    try:
        async with _chat_service_scope(
            corpus_state,
            embedder,
            generation_service,
            query_rewrite_service,
        ) as service:

            async def on_turn_started(
                conv_id: uuid.UUID,
                run_id: uuid.UUID,
            ) -> None:
                started.set(conv_id, run_id)
                await queue.put(RunEvent(conv_id, run_id))

            async def on_progress_stage(stage: ProgressStage) -> None:
                await queue.put(StageEvent(stage))

            response = await service.answer_question(
                question,
                conversation_id,
                on_turn_started=on_turn_started,
                on_progress_stage=on_progress_stage,
            )

        if started.ready:
            await queue.put(ResultEvent(response))
        else:
            # 턴 생성 전에 끝났다. 스트림을 열지 않고 기존 HTTP 오류로 돌려준다.
            await queue.put(TurnStartFailed(response=response))
    except BaseException as error:  # noqa: BLE001 - 어떤 예외든 스트림을 마감해야 한다
        if started.ready:
            logger.exception(
                "진행 상태 중계 중 오류가 발생했습니다: rag_run_id=%s",
                started.rag_run_id,
            )
            await queue.put(
                ErrorEvent(
                    _internal_error_response(
                        started.conversation_id,
                        started.rag_run_id,
                    )
                )
            )
        else:
            await queue.put(TurnStartFailed(error=error))
    finally:
        await queue.put(SENTINEL)


class _TurnIdentity:
    """turn started 훅이 실행됐는지와 그때의 식별자를 담는다."""

    def __init__(self) -> None:
        self.conversation_id: Optional[uuid.UUID] = None
        self.rag_run_id: Optional[uuid.UUID] = None
        self.ready = False

    def set(self, conversation_id: uuid.UUID, rag_run_id: uuid.UUID) -> None:
        self.conversation_id = conversation_id
        self.rag_run_id = rag_run_id
        self.ready = True


# ----------------------------------------------------------------------
# 스트림
# ----------------------------------------------------------------------


async def event_stream(
    first_event: StreamEvent,
    queue: "asyncio.Queue",
) -> AsyncIterator[bytes]:
    """이미 꺼낸 첫 이벤트부터 sentinel까지 SSE로 내보낸다."""

    yield encode_event(first_event)
    while True:
        item = await queue.get()
        if isinstance(item, _Sentinel):
            return
        yield encode_event(item)


async def start_chat_stream(
    request: Request,
    *,
    question: str,
    conversation_id: Optional[uuid.UUID],
) -> Union[StreamingResponse, TurnStartFailed]:
    """파이프라인 task를 띄우고 첫 이벤트를 확인한 뒤 스트림을 연다.

    첫 이벤트가 RunEvent가 아니면 스트림을 만들지 않는다. 따라서 스트림이
    열렸다는 사실 자체가 "첫 이벤트는 run"을 보장한다.
    """

    queue: "asyncio.Queue" = asyncio.Queue()
    task = asyncio.create_task(
        produce_turn(
            queue,
            question=question,
            conversation_id=conversation_id,
            corpus_state=request.app.state.corpus_state,
            embedder=request.app.state.embedder,
            generation_service=request.app.state.generation_service,
            query_rewrite_service=request.app.state.query_rewrite_service,
        )
    )
    register_pipeline_task(request.app, task)

    first = await queue.get()
    if isinstance(first, RunEvent):
        return StreamingResponse(
            event_stream(first, queue),
            media_type=SSE_MEDIA_TYPE,
        )

    if isinstance(first, TurnStartFailed):
        return first
    # producer가 아무 이벤트도 만들지 못한 경우(첫 항목이 sentinel)다.
    return TurnStartFailed(response=_internal_error_response())
