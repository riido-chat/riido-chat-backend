import unittest
import uuid

from pydantic import TypeAdapter, ValidationError

from app.api.chat_schema import (
    ChatAnswer,
    ChatCitation,
    ChatCompletedResponse,
    ChatError,
    ChatErrorCode,
    ChatErrorResponse,
    ChatRequest,
    ChatResponse,
    ChatResponseStatus,
    ChatWithheld,
    ChatWithheldReasonCode,
    ChatWithheldResponse,
)
from generation.models import FinalGenerationResult


class ChatRequestTest(unittest.TestCase):
    def test_strips_question_whitespace(self) -> None:
        request = ChatRequest(question="  멤버를 어떻게 초대하나요?  ")

        self.assertEqual("멤버를 어떻게 초대하나요?", request.question)

    def test_rejects_empty_question(self) -> None:
        for question in ("", "   ", "\n\t"):
            with self.subTest(question=question):
                with self.assertRaises(ValidationError):
                    ChatRequest(question=question)

    def test_accepts_optional_conversation_id(self) -> None:
        conversation_id = "6abcc9de-f92d-4f26-8ca8-576fdde882c7"

        request = ChatRequest(question="질문", conversationId=conversation_id)

        self.assertEqual(uuid.UUID(conversation_id), request.conversation_id)
        self.assertIsNone(ChatRequest(question="질문").conversation_id)

    def test_rejects_fields_outside_the_mvp_request_contract(self) -> None:
        with self.assertRaises(ValidationError):
            ChatRequest(question="질문", clientKey="anonymous")


CONVERSATION_ID = "8f4b2c1a-9d3e-4f7a-b6c5-2e8d9a0f1b3c"
RAG_RUN_ID = "c7a91e42-5b8f-4d2c-a1e6-9f0b3d7c8e5a"


class ChatResponseTest(unittest.TestCase):
    response_adapter = TypeAdapter(ChatResponse)

    def test_serializes_completed_response_with_camel_case_fields(self) -> None:
        response = ChatCompletedResponse(
            status=ChatResponseStatus.COMPLETED,
            conversation_id=uuid.UUID(CONVERSATION_ID),
            rag_run_id=uuid.UUID(RAG_RUN_ID),
            answer=ChatAnswer(answer_markdown="초대할 수 있습니다. [1]"),
            citations=[
                ChatCitation(
                    citation_number=1,
                    document_title="멤버 관리",
                    section_path=["워크스페이스", "멤버 초대"],
                    source_url="https://docs.riido.io/member/invite",
                )
            ],
        )

        self.assertEqual(
            {
                "status": "COMPLETED",
                "conversationId": CONVERSATION_ID,
                "ragRunId": RAG_RUN_ID,
                "answer": {"answerMarkdown": "초대할 수 있습니다. [1]"},
                "citations": [
                    {
                        "citationNumber": 1,
                        "documentTitle": "멤버 관리",
                        "sectionPath": ["워크스페이스", "멤버 초대"],
                        "sourceUrl": "https://docs.riido.io/member/invite",
                    }
                ],
            },
            response.model_dump(mode="json", by_alias=True),
        )

    def test_validates_completed_response(self) -> None:
        response = self.response_adapter.validate_python(
            {
                "status": "COMPLETED",
                "conversationId": CONVERSATION_ID,
                "ragRunId": RAG_RUN_ID,
                "answer": {"answerMarkdown": "완료된 답변입니다. [1]"},
                "citations": [
                    {
                        "citationNumber": 1,
                        "documentTitle": "문서",
                        "sectionPath": ["문서", "섹션"],
                        "sourceUrl": "https://docs.riido.io/guide",
                    }
                ],
            }
        )

        self.assertIsInstance(response, ChatCompletedResponse)

    def test_validates_all_withheld_reasons(self) -> None:
        for reason in ChatWithheldReasonCode:
            with self.subTest(reason=reason):
                response = self.response_adapter.validate_python(
                    {
                        "status": "WITHHELD",
                        "conversationId": CONVERSATION_ID,
                        "ragRunId": RAG_RUN_ID,
                        "answer": None,
                        "withheld": {
                            "reasonCode": reason.value,
                            "message": "답변을 제공하지 않습니다.",
                        },
                        "citations": [],
                    }
                )

                self.assertIsInstance(response, ChatWithheldResponse)
                self.assertEqual(reason, response.withheld.reason_code)

    def test_validates_error_response(self) -> None:
        response = self.response_adapter.validate_python(
            {
                "status": "ERROR",
                "conversationId": CONVERSATION_ID,
                "ragRunId": RAG_RUN_ID,
                "answer": None,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": (
                        "답변을 생성하는 중 오류가 발생했습니다."
                    ),
                },
                "citations": [],
            }
        )

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(ChatErrorCode.INTERNAL_ERROR, response.error.code)
        self.assertEqual(uuid.UUID(RAG_RUN_ID), response.rag_run_id)

    def test_allows_error_response_without_identifiers(self) -> None:
        response = self.response_adapter.validate_python(
            {
                "status": "ERROR",
                "conversationId": None,
                "ragRunId": None,
                "answer": None,
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "검색 데이터가 아직 준비되지 않았습니다.",
                },
                "citations": [],
            }
        )

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertIsNone(response.conversation_id)
        self.assertIsNone(response.rag_run_id)

    def test_enforces_state_specific_response_shape(self) -> None:
        invalid_cases = (
            {
                "status": "COMPLETED",
                "conversationId": CONVERSATION_ID,
                "ragRunId": RAG_RUN_ID,
                "answer": {"answerMarkdown": "출처가 없는 답변"},
                "citations": [],
            },
            {
                "status": "COMPLETED",
                "answer": {"answerMarkdown": "식별자가 없는 답변 [1]"},
                "citations": [
                    {
                        "citationNumber": 1,
                        "documentTitle": "문서",
                        "sectionPath": ["문서", "섹션"],
                        "sourceUrl": "https://docs.riido.io/guide",
                    }
                ],
            },
            {
                "status": "WITHHELD",
                "conversationId": CONVERSATION_ID,
                "ragRunId": RAG_RUN_ID,
                "answer": None,
                "withheld": {
                    "reasonCode": "INSUFFICIENT_EVIDENCE",
                    "message": "답변을 제공하지 않습니다.",
                },
                "citations": [
                    {
                        "citationNumber": 1,
                        "documentTitle": "문서",
                        "sectionPath": ["문서", "섹션"],
                        "sourceUrl": "https://docs.riido.io/guide",
                    }
                ],
            },
            {
                "status": "ERROR",
                "answer": None,
                "error": {
                    "code": "UPSTREAM_ERROR",
                    "message": "내부 오류",
                },
                "citations": [],
            },
        )

        for payload in invalid_cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    self.response_adapter.validate_python(payload)

    def test_keeps_http_dto_separate_from_internal_generation_result(self) -> None:
        self.assertIsNot(ChatCompletedResponse, FinalGenerationResult)
        self.assertNotIn("summary", ChatAnswer.model_fields)
        self.assertNotIn("content_markdown", ChatAnswer.model_fields)

    def test_constructs_withheld_and_error_models_from_python_field_names(self) -> None:
        withheld = ChatWithheld(
            reason_code=ChatWithheldReasonCode.OUT_OF_SCOPE,
            message="범위를 벗어난 질문입니다.",
        )
        error = ChatError(
            code=ChatErrorCode.INTERNAL_ERROR,
            message="답변을 생성하는 중 오류가 발생했습니다.",
        )

        self.assertEqual("OUT_OF_SCOPE", withheld.reason_code.value)
        self.assertEqual("INTERNAL_ERROR", error.code.value)


if __name__ == "__main__":
    unittest.main()
