import unittest
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
from app.rag.chat_service import ChatService
from app.rag.dependencies import get_chat_service


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
            "멤버를 어떻게 초대하나요?"
        )

    def test_withheld_returns_200(self) -> None:
        self.service.answer_question.return_value = ChatWithheldResponse(
            status=ChatResponseStatus.WITHHELD,
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

    def test_extra_request_field_returns_422(self) -> None:
        response = self._post(
            {
                "question": "질문",
                "conversationId": "6abcc9de-f92d-4f26-8ca8-576fdde882c7",
            }
        )

        self.assertEqual(422, response.status_code)
        self.service.answer_question.assert_not_awaited()

    def _post(self, payload: dict[str, str]):
        with TestClient(self.app) as client:
            return client.post("/api/chat", json=payload)


if __name__ == "__main__":
    unittest.main()
