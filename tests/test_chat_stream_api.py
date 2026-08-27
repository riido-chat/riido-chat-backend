import json
import unittest
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat_schema import (
    ChatAnswer,
    ChatCitation,
    ChatCompletedResponse,
    ChatError,
    ChatErrorCode,
    ChatErrorResponse,
    ChatResponseStatus,
    ChatWithheld,
    ChatWithheldReasonCode,
    ChatWithheldResponse,
)
from app.main import create_app
from app.rag.chat_service import ChatService, ConversationNotFoundError
from app.rag.corpus_state import CorpusNotLoadedError
from app.rag.dependencies import get_chat_service
from app.rag.progress import ProgressStage


CONVERSATION_ID = uuid.UUID("6b401388-b1ca-410a-9430-dd9beee85460")
RAG_RUN_ID = uuid.UUID("d49dc6fb-25f1-4782-a1db-659fe1c55892")

SSE_HEADERS = {"Accept": "text/event-stream"}
ALL_STAGES = (
    ProgressStage.RETRIEVING,
    ProgressStage.GENERATING,
    ProgressStage.VALIDATING,
)


@asynccontextmanager
async def test_lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


def completed_response() -> ChatCompletedResponse:
    return ChatCompletedResponse(
        status=ChatResponseStatus.COMPLETED,
        conversation_id=CONVERSATION_ID,
        rag_run_id=RAG_RUN_ID,
        answer=ChatAnswer(answer_markdown="멤버를 초대할 수 있습니다. [1]"),
        citations=[
            ChatCitation(
                citation_number=1,
                document_title="멤버 관리",
                section_path=["워크스페이스", "멤버 초대"],
                source_url="https://docs.riido.io/member/invite",
            )
        ],
    )


def withheld_response() -> ChatWithheldResponse:
    return ChatWithheldResponse(
        status=ChatResponseStatus.WITHHELD,
        conversation_id=CONVERSATION_ID,
        rag_run_id=RAG_RUN_ID,
        answer=None,
        withheld=ChatWithheld(
            reason_code=ChatWithheldReasonCode.OUT_OF_SCOPE,
            message="해당 질문은 이용가이드 범위를 벗어나 답변할 수 없습니다.",
        ),
        citations=[],
    )


def error_response(with_ids: bool = True) -> ChatErrorResponse:
    return ChatErrorResponse(
        status=ChatResponseStatus.ERROR,
        conversation_id=CONVERSATION_ID if with_ids else None,
        rag_run_id=RAG_RUN_ID if with_ids else None,
        answer=None,
        error=ChatError(
            code=ChatErrorCode.INTERNAL_ERROR,
            message="답변을 생성하는 중 오류가 발생했습니다.",
        ),
        citations=[],
    )


class ChatStreamApiTest(unittest.TestCase):
    """POST /api/chat의 진행 상태 SSE 분기를 검증한다."""

    def setUp(self) -> None:
        self.sync_service = AsyncMock(spec=ChatService)
        self.stream_service = AsyncMock(spec=ChatService)

        with patch("app.main.lifespan", test_lifespan):
            self.app = create_app()

        self.app.dependency_overrides[get_chat_service] = lambda: self.sync_service
        self.app.state.corpus_state = object()
        self.app.state.embedder = object()
        self.app.state.generation_service = object()

        @asynccontextmanager
        async def scope(*_args, **_kwargs):
            yield self.stream_service

        patcher = patch("app.api.chat_stream._chat_service_scope", scope)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()

    # ------------------------------------------------------------------
    # 정상 시퀀스
    # ------------------------------------------------------------------

    def test_completed_emits_run_stages_then_result(self) -> None:
        self._pipeline(stages=ALL_STAGES, result=completed_response())

        events = self._stream()

        self.assertEqual(
            ["run", "stage", "stage", "stage", "result"],
            [name for name, _ in events],
        )
        self.assertEqual(
            {
                "conversationId": str(CONVERSATION_ID),
                "ragRunId": str(RAG_RUN_ID),
            },
            events[0][1],
        )
        self.assertEqual(
            ["RETRIEVING", "GENERATING", "VALIDATING"],
            [payload["progressStage"] for _, payload in events[1:4]],
        )
        self.assertEqual("COMPLETED", events[4][1]["status"])

    def test_first_event_is_always_run(self) -> None:
        scenarios = {
            "정상": (ALL_STAGES, completed_response(), None),
            "검색 실패": ((ProgressStage.RETRIEVING,), error_response(), None),
            "보류": (ALL_STAGES[:2], withheld_response(), None),
            "전달 계층 파손": (ALL_STAGES[:1], None, RuntimeError("boom")),
        }
        for name, (stages, result, error) in scenarios.items():
            with self.subTest(scenario=name):
                self._pipeline(stages=stages, result=result, error=error)
                events = self._stream()
                self.assertEqual("run", events[0][0])
                self.assertIn("ragRunId", events[0][1])

    def test_withheld_skips_validating_stage(self) -> None:
        self._pipeline(stages=ALL_STAGES[:2], result=withheld_response())

        events = self._stream()

        self.assertEqual(
            ["run", "stage", "stage", "result"],
            [name for name, _ in events],
        )
        self.assertNotIn(
            "VALIDATING",
            [payload.get("progressStage") for _, payload in events],
        )
        self.assertEqual("WITHHELD", events[-1][1]["status"])

    def test_terminal_payload_matches_sync_response(self) -> None:
        expected = completed_response()
        self._pipeline(stages=ALL_STAGES, result=expected)

        events = self._stream()

        self.assertEqual(
            expected.model_dump(mode="json", by_alias=True),
            events[-1][1],
        )

    def test_pipeline_error_arrives_as_result_not_error_event(self) -> None:
        self._pipeline(stages=ALL_STAGES[:2], result=error_response())

        events = self._stream()

        self.assertEqual("result", events[-1][0])
        self.assertEqual("ERROR", events[-1][1]["status"])

    def test_delivery_failure_emits_error_event(self) -> None:
        self._pipeline(
            stages=ALL_STAGES[:1],
            result=None,
            error=RuntimeError("전달 계층 파손"),
        )

        with TestClient(self.app) as client:
            response = client.post(
                "/api/chat", json={"question": "질문"}, headers=SSE_HEADERS
            )

        events = self._parse(response.text)
        self.assertEqual(200, response.status_code)
        self.assertEqual("error", events[-1][0])
        self.assertEqual("INTERNAL_ERROR", events[-1][1]["error"]["code"])

    # ------------------------------------------------------------------
    # 동기 경로 무영향
    # ------------------------------------------------------------------

    def test_missing_accept_returns_sync_json(self) -> None:
        expected = completed_response()
        self.sync_service.answer_question.return_value = expected

        with TestClient(self.app) as client:
            response = client.post("/api/chat", json={"question": "질문"})

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            expected.model_dump(mode="json", by_alias=True),
            response.json(),
        )
        self.assertNotIn("text/event-stream", response.headers["content-type"])
        self.stream_service.answer_question.assert_not_awaited()
        self.sync_service.answer_question.assert_awaited_once_with("질문", None)

    def test_wildcard_accept_returns_sync_json(self) -> None:
        self.sync_service.answer_question.return_value = completed_response()

        with TestClient(self.app) as client:
            response = client.post(
                "/api/chat", json={"question": "질문"}, headers={"Accept": "*/*"}
            )

        self.assertEqual("COMPLETED", response.json()["status"])
        self.assertNotIn("text/event-stream", response.headers["content-type"])
        self.stream_service.answer_question.assert_not_awaited()

    # ------------------------------------------------------------------
    # 턴 생성 전 실패 — 스트림 미개시
    # ------------------------------------------------------------------

    def test_corpus_not_loaded_returns_503_for_sse_request(self) -> None:
        def raise_corpus_error() -> ChatService:
            raise CorpusNotLoadedError("corpus가 적재되지 않았습니다.")

        self.app.dependency_overrides[get_chat_service] = raise_corpus_error

        with TestClient(self.app) as client:
            response = client.post(
                "/api/chat", json={"question": "질문"}, headers=SSE_HEADERS
            )

        self.assertEqual(503, response.status_code)
        self.assertEqual("SERVICE_UNAVAILABLE", response.json()["error"]["code"])
        self.assertNotIn("text/event-stream", response.headers["content-type"])

    def test_conversation_not_found_returns_404_for_sse_request(self) -> None:
        self.stream_service.answer_question.side_effect = ConversationNotFoundError(
            "이어갈 수 없는 대화입니다."
        )

        with TestClient(self.app) as client:
            response = client.post(
                "/api/chat", json={"question": "질문"}, headers=SSE_HEADERS
            )

        self.assertEqual(404, response.status_code)
        self.assertEqual("NOT_FOUND", response.json()["error"]["code"])

    def test_turn_start_failure_returns_500_for_sse_request(self) -> None:
        self.stream_service.answer_question.return_value = error_response(
            with_ids=False
        )

        with TestClient(self.app) as client:
            response = client.post(
                "/api/chat", json={"question": "질문"}, headers=SSE_HEADERS
            )

        body = response.json()
        self.assertEqual(500, response.status_code)
        self.assertEqual("ERROR", body["status"])
        self.assertIsNone(body["conversationId"])
        self.assertIsNone(body["ragRunId"])

    # ------------------------------------------------------------------

    def _pipeline(self, *, stages, result, error=None) -> None:
        """훅을 호출한 뒤 결과를 돌려주는 가짜 파이프라인을 세운다."""

        async def answer(_question, _conversation_id=None, *, on_turn_started=None,
                         on_progress_stage=None):
            await on_turn_started(CONVERSATION_ID, RAG_RUN_ID)
            for stage in stages:
                await on_progress_stage(stage)
            if error is not None:
                raise error
            return result

        self.stream_service.answer_question.side_effect = answer

    def _stream(self):
        with TestClient(self.app) as client:
            with client.stream(
                "POST", "/api/chat", json={"question": "질문"}, headers=SSE_HEADERS
            ) as response:
                self.assertEqual(200, response.status_code)
                self.assertIn(
                    "text/event-stream", response.headers["content-type"]
                )
                return self._parse("".join(response.iter_text()))

    @staticmethod
    def _parse(body: str):
        events = []
        name = None
        for line in body.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line.split(":", 1)[1])))
        return events


if __name__ == "__main__":
    unittest.main()
