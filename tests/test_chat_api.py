import asyncio
import unittest
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.answering.models import CitationSourceKind
from app.chat.schema import (
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
from app.chat.router import _answer_until_disconnect
from app.main import create_app
from app.chat.service import (
    ChatService,
    ConversationNotFoundError,
    chat_error_response,
)
from app.retrieval.corpus_state import CorpusNotLoadedError
from app.chat.dependencies import get_chat_service
from app.chat.log_store import ConversationBusyError


CONVERSATION_ID = "8f4b2c1a-9d3e-4f7a-b6c5-2e8d9a0f1b3c"
RAG_RUN_ID = "c7a91e42-5b8f-4d2c-a1e6-9f0b3d7c8e5a"


@asynccontextmanager
async def test_lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


class ChatApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = AsyncMock(spec=ChatService)

        with patch("app.main.lifespan", test_lifespan):
            self.app = create_app()

        self.app.dependency_overrides[get_chat_service] = lambda: self.service

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()

    def test_completed_returns_200_and_contract_body(self) -> None:
        self.service.answer_question.return_value = ChatCompletedResponse(
            status=ChatResponseStatus.COMPLETED,
            conversation_id=uuid.UUID(CONVERSATION_ID),
            rag_run_id=uuid.UUID(RAG_RUN_ID),
            answer=ChatAnswer(answer_markdown="멤버를 초대할 수 있습니다. [1]"),
            citations=[
                ChatCitation(
                    citation_number=1,
                    document_title="멤버 관리",
                    section_path=["워크스페이스", "멤버 초대"],
                    source_url="https://docs.riido.io/member/invite",
                    source_kind=CitationSourceKind.GITBOOK,
                )
            ],
        )

        response = self._post(
            {"question": "  멤버를 어떻게 초대하나요?  "}
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "status": "COMPLETED",
                "conversationId": CONVERSATION_ID,
                "ragRunId": RAG_RUN_ID,
                "answer": {
                    "answerMarkdown": "멤버를 초대할 수 있습니다. [1]",
                },
                "citations": [
                    {
                        "citationNumber": 1,
                        "documentTitle": "멤버 관리",
                        "sectionPath": ["워크스페이스", "멤버 초대"],
                        "sourceUrl": "https://docs.riido.io/member/invite",
                        "sourceKind": "GITBOOK",
                    }
                ],
            },
            response.json(),
        )
        self.service.answer_question.assert_awaited_once_with(
            "멤버를 어떻게 초대하나요?",
            None,
        )

    def test_withheld_returns_200(self) -> None:
        self.service.answer_question.return_value = ChatWithheldResponse(
            status=ChatResponseStatus.WITHHELD,
            conversation_id=uuid.UUID(CONVERSATION_ID),
            rag_run_id=uuid.UUID(RAG_RUN_ID),
            answer=None,
            withheld=ChatWithheld(
                reason_code=ChatWithheldReasonCode.INSUFFICIENT_EVIDENCE,
                message="이용가이드에서 질문에 답할 충분한 근거를 찾지 못했습니다.",
            ),
            citations=[],
        )

        response = self._post({"question": "알 수 없는 기능을 알려주세요."})

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "status": "WITHHELD",
                "conversationId": CONVERSATION_ID,
                "ragRunId": RAG_RUN_ID,
                "answer": None,
                "withheld": {
                    "reasonCode": "INSUFFICIENT_EVIDENCE",
                    "message": (
                        "이용가이드에서 질문에 답할 충분한 근거를 찾지 못했습니다."
                    ),
                },
                "citations": [],
            },
            response.json(),
        )

    def test_error_returns_500(self) -> None:
        self.service.answer_question.return_value = ChatErrorResponse(
            status=ChatResponseStatus.ERROR,
            conversation_id=uuid.UUID(CONVERSATION_ID),
            rag_run_id=uuid.UUID(RAG_RUN_ID),
            answer=None,
            error=ChatError(
                code=ChatErrorCode.INTERNAL_ERROR,
                message="답변을 생성하는 중 오류가 발생했습니다.",
                retryable=False,
            ),
            citations=[],
        )

        response = self._post({"question": "질문"})

        self.assertEqual(500, response.status_code)
        self.assertEqual(
            {
                "status": "ERROR",
                "conversationId": CONVERSATION_ID,
                "ragRunId": RAG_RUN_ID,
                "answer": None,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "답변을 생성하는 중 오류가 발생했습니다.",
                    "retryable": False,
                },
                "citations": [],
            },
            response.json(),
        )

    def test_pipeline_error_codes_return_500_with_retry_policy(self) -> None:
        cases = (
            (
                ChatErrorCode.UPSTREAM_ERROR,
                True,
                "AI 서비스 연결이 원활하지 않습니다. "
                "잠시 후 다시 시도해주세요.",
            ),
            (
                ChatErrorCode.MODEL_OUTPUT_INVALID,
                True,
                "AI 응답을 처리하지 못했습니다. "
                "잠시 후 다시 시도해주세요.",
            ),
            (
                ChatErrorCode.CITATION_VALIDATION_ERROR,
                False,
                "답변 출처를 검증하는 중 오류가 발생했습니다.",
            ),
            (
                ChatErrorCode.INTERNAL_ERROR,
                False,
                "답변을 생성하는 중 오류가 발생했습니다.",
            ),
        )

        for error_code, retryable, message in cases:
            with self.subTest(error_code=error_code):
                self.service.reset_mock()
                self.service.answer_question.return_value = chat_error_response(
                    error_code,
                    uuid.UUID(CONVERSATION_ID),
                    uuid.UUID(RAG_RUN_ID),
                )

                response = self._post({"question": "질문"})

                self.assertEqual(500, response.status_code)
                self.assertEqual(error_code.value, response.json()["error"]["code"])
                self.assertEqual(
                    retryable,
                    response.json()["error"]["retryable"],
                )
                self.assertEqual(message, response.json()["error"]["message"])

    def test_corpus_unavailable_returns_503_without_identifiers(self) -> None:
        def raise_corpus_error() -> ChatService:
            raise CorpusNotLoadedError("corpus가 적재되지 않았습니다.")

        self.app.dependency_overrides[get_chat_service] = raise_corpus_error

        response = self._post({"question": "질문"})

        self.assertEqual(503, response.status_code)
        self.assertEqual(
            {
                "code": "SERVICE_UNAVAILABLE",
                "message": "검색 데이터가 아직 준비되지 않았습니다.",
                "retryable": False,
            },
            response.json()["error"],
        )
        self.assertIsNone(response.json()["conversationId"])
        self.assertIsNone(response.json()["ragRunId"])

    def test_empty_question_returns_422(self) -> None:
        response = self._post({"question": ""})

        self.assertEqual(422, response.status_code)
        self.service.answer_question.assert_not_awaited()

    def test_whitespace_only_question_returns_422(self) -> None:
        response = self._post({"question": " \n\t "})

        self.assertEqual(422, response.status_code)
        self.service.answer_question.assert_not_awaited()

    def test_question_over_4000_characters_returns_422(self) -> None:
        response = self._post({"question": "가" * 4001})

        self.assertEqual(422, response.status_code)
        self.service.answer_question.assert_not_awaited()

    def test_extra_request_field_returns_422(self) -> None:
        response = self._post({"question": "질문", "clientKey": "anonymous"})

        self.assertEqual(422, response.status_code)
        self.service.answer_question.assert_not_awaited()

    def test_forwards_conversation_id_to_the_service(self) -> None:
        self.service.answer_question.return_value = ChatWithheldResponse(
            status=ChatResponseStatus.WITHHELD,
            conversation_id=uuid.UUID(CONVERSATION_ID),
            rag_run_id=uuid.UUID(RAG_RUN_ID),
            answer=None,
            withheld=ChatWithheld(
                reason_code=ChatWithheldReasonCode.OUT_OF_SCOPE,
                message="범위를 벗어난 질문입니다.",
            ),
            citations=[],
        )

        response = self._post(
            {"question": "질문", "conversationId": CONVERSATION_ID}
        )

        self.assertEqual(200, response.status_code)
        self.service.answer_question.assert_awaited_once_with(
            "질문",
            uuid.UUID(CONVERSATION_ID),
        )

    def test_unusable_conversation_returns_404_without_identifiers(self) -> None:
        self.service.answer_question.side_effect = ConversationNotFoundError(
            "이어갈 수 없는 대화입니다."
        )

        response = self._post(
            {"question": "질문", "conversationId": CONVERSATION_ID}
        )

        self.assertEqual(404, response.status_code)
        body = response.json()
        self.assertEqual("ERROR", body["status"])
        self.assertEqual("NOT_FOUND", body["error"]["code"])
        self.assertFalse(body["error"]["retryable"])
        self.assertIsNone(body["conversationId"])
        self.assertIsNone(body["ragRunId"])

    def test_busy_conversation_returns_409_with_requested_conversation_id(
        self,
    ) -> None:
        conversation_id = uuid.UUID(CONVERSATION_ID)
        self.service.answer_question.side_effect = ConversationBusyError(
            conversation_id
        )

        response = self._post(
            {"question": "질문", "conversationId": CONVERSATION_ID}
        )

        self.assertEqual(409, response.status_code)
        self.assertEqual("application/json", response.headers["content-type"])
        self.assertEqual(
            {
                "status": "ERROR",
                "conversationId": CONVERSATION_ID,
                "ragRunId": None,
                "answer": None,
                "error": {
                    "code": "CONVERSATION_BUSY",
                    "message": (
                        "이 대화의 이전 질문을 처리 중입니다. "
                        "잠시 후 다시 시도해주세요."
                    ),
                    "retryable": False,
                },
                "citations": [],
            },
            response.json(),
        )

    def test_openapi_documents_conversation_busy_response(self) -> None:
        response = self.app.openapi()["paths"]["/api/chat"]["post"]["responses"]

        self.assertIn("409", response)

    def test_openapi_requires_retryable_and_lists_chat_error_codes(self) -> None:
        schemas = self.app.openapi()["components"]["schemas"]

        self.assertIn("retryable", schemas["ChatError"]["required"])
        self.assertEqual(
            {code.value for code in ChatErrorCode},
            set(schemas["ChatErrorCode"]["enum"]),
        )

    def _post(self, payload: dict[str, str]):
        with TestClient(self.app) as client:
            return client.post("/api/chat", json=payload)


class ChatDisconnectTest(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_cancels_answer_task(self) -> None:
        service = AsyncMock(spec=ChatService)
        answer_started = asyncio.Event()
        answer_cancelled = asyncio.Event()
        messages: asyncio.Queue = asyncio.Queue()

        async def answer(_question, _conversation_id=None):
            answer_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                answer_cancelled.set()

        async def receive():
            return await messages.get()

        service.answer_question.side_effect = answer
        request = Request({"type": "http"}, receive=receive)
        result_task = asyncio.create_task(
            _answer_until_disconnect(request, service, "질문", None)
        )

        await answer_started.wait()
        await messages.put({"type": "http.disconnect"})

        self.assertIsNone(await result_task)
        self.assertTrue(answer_cancelled.is_set())
        service.answer_question.assert_awaited_once_with("질문", None)


if __name__ == "__main__":
    unittest.main()
