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
from app.rag.dependencies import get_chat_service
from app.rag.log_store import ConversationBusyError


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
                },
                "citations": [],
            },
            response.json(),
        )

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
                },
                "citations": [],
            },
            response.json(),
        )

    def test_openapi_documents_conversation_busy_response(self) -> None:
        response = self.app.openapi()["paths"]["/api/chat"]["post"]["responses"]

        self.assertIn("409", response)

    def _post(self, payload: dict[str, str]):
        with TestClient(self.app) as client:
            return client.post("/api/chat", json=payload)


if __name__ == "__main__":
    unittest.main()
