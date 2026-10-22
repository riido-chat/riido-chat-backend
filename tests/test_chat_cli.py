import json
import unittest

import httpx

from scripts.chat_cli import ChatCli, run_interactive


CONVERSATION_ID = "8f4b2c1a-9d3e-4f7a-b6c5-2e8d9a0f1b3c"
RAG_RUN_ID = "c7a91e42-5b8f-4d2c-a1e6-9f0b3d7c8e5a"
ENDPOINT = "http://testserver/api/chat"


class ChatCliTest(unittest.TestCase):
    def test_reuses_returned_conversation_id_for_next_question(self) -> None:
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "status": "COMPLETED",
                    "conversationId": CONVERSATION_ID,
                    "ragRunId": RAG_RUN_ID,
                    "answer": {"answerMarkdown": "답변 [1]"},
                    "citations": [
                        {
                            "citationNumber": 1,
                            "documentTitle": "멤버 관리",
                            "sectionPath": ["워크스페이스", "멤버 초대"],
                            "sourceUrl": "https://docs.example.com/invite",
                        }
                    ],
                },
            )

        output = []
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            chat = ChatCli(ENDPOINT, client, output.append)
            chat.ask("멤버를 어떻게 초대해?")
            chat.ask("그럼 삭제는?")

        self.assertEqual(
            {"question": "멤버를 어떻게 초대해?"},
            requests[0],
        )
        self.assertEqual(
            {
                "question": "그럼 삭제는?",
                "conversationId": CONVERSATION_ID,
            },
            requests[1],
        )
        self.assertIn("답변 [1]", output)
        self.assertIn("[1] 멤버 관리 > 워크스페이스 > 멤버 초대", output)

    def test_not_found_clears_unusable_conversation(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={
                    "status": "ERROR",
                    "conversationId": None,
                    "ragRunId": None,
                    "answer": None,
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "이어갈 수 없는 대화입니다.",
                    },
                    "citations": [],
                },
            )

        output = []
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            chat = ChatCli(ENDPOINT, client, output.append)
            chat.conversation_id = CONVERSATION_ID
            chat.ask("이전 질문 이어서")

        self.assertIsNone(chat.conversation_id)
        self.assertIn(
            "이어갈 수 없는 대화라서 다음 질문부터 새 대화로 시작해.",
            output,
        )

    def test_commands_show_status_and_reset_without_api_request(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            self.fail("명령어 처리 중에는 API를 호출하면 안 됩니다.")

        inputs = iter(["/status", "/new", "/status", "/quit"])
        output = []
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            chat = ChatCli(ENDPOINT, client, output.append)
            chat.conversation_id = CONVERSATION_ID
            run_interactive(chat, lambda _: next(inputs), output.append)

        self.assertIn(f"현재 conversationId: {CONVERSATION_ID}", output)
        self.assertIn("새 대화를 시작할게.", output)
        self.assertIn("현재 대화: 새 대화 (첫 질문 대기 중)", output)

    def test_connection_error_is_rendered_without_losing_conversation(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        output = []
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            chat = ChatCli(ENDPOINT, client, output.append)
            chat.conversation_id = CONVERSATION_ID
            chat.ask("질문")

        self.assertEqual(CONVERSATION_ID, chat.conversation_id)
        self.assertTrue(
            any("서버에 연결할 수 없어" in line for line in output)
        )


if __name__ == "__main__":
    unittest.main()
