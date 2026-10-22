"""채팅 요청과 응답 계약(JSON 스키마)을 스냅샷으로 고정한다.

프론트는 이 계약만 쓴다. /api/chat 과 /api/internal/test-chat 은 서비스가 달라도 같은 요청과
응답 DTO(ChatResponse 3종)를 쓰고, 서비스별 답변 형식 차이는 answerMarkdown 내용으로만 나타낸다.
결과를 tests/fixtures/chat_contract_schema.json 과 바이트 단위로 비교한다.

계약을 일부러 바꾼 PR 에서만 `UPDATE_CHAT_CONTRACT_SNAPSHOT=1` 환경변수를 켜고 이 테스트를 돌려
fixture 를 다시 만든다. 프론트 변경 없이는 바꾸지 않는다.
"""

import json
import os
import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from pydantic import TypeAdapter

from app.chat.schema import ChatRequest, ChatResponse
from app.main import create_app


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "chat_contract_schema.json"
UPDATE_ENV = "UPDATE_CHAT_CONTRACT_SNAPSHOT"
CHAT_PATHS = ("/api/chat", "/api/internal/test-chat")


@asynccontextmanager
async def _no_lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


def build_snapshot() -> dict:
    return {
        "request": ChatRequest.model_json_schema(by_alias=True, mode="validation"),
        "response": TypeAdapter(ChatResponse).json_schema(
            by_alias=True, mode="serialization"
        ),
    }


def serialize(snapshot: dict) -> str:
    return json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


class ChatContractSnapshotTest(unittest.TestCase):
    def test_schema_matches_fixture_byte_for_byte(self) -> None:
        serialized = serialize(build_snapshot())
        if os.environ.get(UPDATE_ENV) == "1":
            FIXTURE_PATH.write_text(serialized, "utf-8")
        self.assertTrue(
            FIXTURE_PATH.exists(),
            f"{FIXTURE_PATH.name} 가 없습니다. {UPDATE_ENV}=1 로 이 테스트를 돌려 만드세요.",
        )
        self.assertEqual(
            FIXTURE_PATH.read_text("utf-8"),
            serialized,
            "채팅 요청이나 응답 계약이 달라졌습니다. 프론트 변경 없이는 바꾸지 않습니다.",
        )

    def test_both_chat_addresses_share_the_contract(self) -> None:
        with patch("app.main.lifespan", _no_lifespan):
            openapi = create_app().openapi()

        operations = [openapi["paths"][path]["post"] for path in CHAT_PATHS]
        request_schemas = [
            operation["requestBody"]["content"]["application/json"]["schema"]
            for operation in operations
        ]
        # FastAPI 가 경로 이름으로 붙이는 title 만 다르다.
        response_schemas = [
            {
                key: value
                for key, value in operation["responses"]["200"]["content"][
                    "application/json"
                ]["schema"].items()
                if key != "title"
            }
            for operation in operations
        ]
        self.assertEqual(request_schemas[0], request_schemas[1])
        self.assertEqual(response_schemas[0], response_schemas[1])
        # 요청에 서비스, 프로필 버전, 문서 그룹을 고르는 칸이 없다.
        self.assertEqual(
            {"question", "conversationId"},
            set(build_snapshot()["request"]["properties"]),
        )


if __name__ == "__main__":
    unittest.main()
