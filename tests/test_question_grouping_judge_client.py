import json
import unittest
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, Mock, patch

import httpx
from openai import APIStatusError, APITimeoutError

from app.question_grouping.constants import (
    JUDGE_MAX_OUTPUT_TOKENS,
    JUDGE_MODEL,
    JUDGE_OUTPUT_SCHEMA_NAME,
    JUDGE_PROMPT_VERSION,
    JUDGE_PROVIDER,
    JUDGE_REASONING_EFFORT,
)
from app.question_grouping.judge_client import (
    QuestionJudgeClient,
    build_judge_request,
    safe_judge_error_message,
)
from app.question_grouping.models import JudgeFailureKind
from app.question_grouping.prompt_v7_2 import JUDGE_INSTRUCTIONS, JUDGE_OUTPUT_SCHEMA


QUESTION = "비밀 질문: 구독을 해지하면 데이터가 남나요?"
PAYLOAD = {"policyVersion": JUDGE_PROMPT_VERSION, "question": QUESTION, "candidates": []}
OUTPUT = json.dumps({"decision": "SEPARATE"})


def _usage(
    input_tokens: int = 5000,
    output_tokens: int = 300,
    cached: Optional[int] = 4000,
    reasoning: Optional[int] = 160,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )


def _response(
    output_text: Any = OUTPUT,
    *,
    status: str = "completed",
    usage: Any = None,
    incomplete_details: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        output_text=output_text,
        status=status,
        usage=_usage() if usage is None else usage,
        incomplete_details=incomplete_details,
    )


def _status_error(status_code: int, body: Any = None) -> APIStatusError:
    request = httpx.Request("POST", "https://api.openai.com")
    return APIStatusError(
        f"error echoing {QUESTION}",
        response=httpx.Response(status_code, request=request),
        body=body,
    )


def _client(*effects: Any) -> Mock:
    client = Mock()
    client.responses.create = AsyncMock(side_effect=list(effects))
    return client


class JudgeRequestTest(unittest.TestCase):
    def test_request_uses_strict_schema_low_reasoning_and_no_store(self) -> None:
        request = build_judge_request(PAYLOAD)

        self.assertEqual(JUDGE_MODEL, request["model"])
        self.assertFalse(request["store"])
        self.assertEqual(JUDGE_INSTRUCTIONS, request["instructions"])
        self.assertEqual(json.dumps(PAYLOAD, ensure_ascii=False), request["input"])
        self.assertEqual(
            {
                "type": "json_schema",
                "name": JUDGE_OUTPUT_SCHEMA_NAME,
                "strict": True,
                "schema": JUDGE_OUTPUT_SCHEMA,
            },
            request["text"]["format"],
        )
        self.assertEqual({"effort": JUDGE_REASONING_EFFORT}, request["reasoning"])
        self.assertEqual(JUDGE_MAX_OUTPUT_TOKENS, request["max_output_tokens"])

    def test_default_client_disables_sdk_retries(self) -> None:
        with patch("app.question_grouping.judge_client.get_settings") as settings, patch(
            "app.question_grouping.judge_client.AsyncOpenAI"
        ) as async_openai:
            settings.return_value.openai_api_key = "sk-test-key-000000"

            QuestionJudgeClient()

        kwargs = async_openai.call_args.kwargs
        self.assertEqual(0, kwargs["max_retries"])
        self.assertIn("timeout", kwargs)

    def test_default_client_requires_api_key(self) -> None:
        with patch("app.question_grouping.judge_client.get_settings") as settings:
            settings.return_value.openai_api_key = None

            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                QuestionJudgeClient()


class JudgeCallTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        sleep_patcher = patch(
            "app.question_grouping.judge_client.asyncio.sleep", new=AsyncMock()
        )
        self.sleep = sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    async def test_success_returns_output_text_and_usage(self) -> None:
        client = _client(_response())
        before_model_call = AsyncMock()

        call = await QuestionJudgeClient(client=client).judge(
            PAYLOAD, before_model_call=before_model_call
        )

        before_model_call.assert_awaited_once_with(
            JUDGE_PROVIDER, JUDGE_MODEL, JUDGE_PROMPT_VERSION
        )
        client.responses.create.assert_awaited_once_with(**build_judge_request(PAYLOAD))
        self.assertTrue(call.succeeded)
        self.assertEqual(OUTPUT, call.output_text)
        trace = call.trace
        self.assertTrue(trace.succeeded)
        self.assertEqual(JUDGE_PROVIDER, trace.provider)
        self.assertEqual(JUDGE_MODEL, trace.model_name)
        self.assertEqual(JUDGE_PROMPT_VERSION, trace.prompt_version)
        self.assertEqual(0, trace.retry_count)
        self.assertEqual(
            (5000, 300, 4000, 160),
            (
                trace.input_tokens,
                trace.output_tokens,
                trace.cached_input_tokens,
                trace.reasoning_tokens,
            ),
        )
        self.assertIsNone(trace.error_message)

    async def test_checkpoint_failure_prevents_the_call(self) -> None:
        client = _client(_response())
        before_model_call = AsyncMock(side_effect=RuntimeError("db down"))

        with self.assertRaisesRegex(RuntimeError, "db down"):
            await QuestionJudgeClient(client=client).judge(
                PAYLOAD, before_model_call=before_model_call
            )

        client.responses.create.assert_not_awaited()

    async def test_usage_without_details_keeps_breakdown_empty(self) -> None:
        usage = SimpleNamespace(input_tokens=10, output_tokens=2)
        client = _client(_response(usage=usage))

        call = await QuestionJudgeClient(client=client).judge(PAYLOAD)

        self.assertEqual(10, call.trace.input_tokens)
        self.assertIsNone(call.trace.cached_input_tokens)
        self.assertIsNone(call.trace.reasoning_tokens)

    async def test_retries_transient_error_once_and_sums_usage(self) -> None:
        for error in (
            APITimeoutError(httpx.Request("POST", "https://api.openai.com")),
            _status_error(429),
            _status_error(503),
        ):
            with self.subTest(error=type(error).__name__):
                client = _client(error, _response())

                call = await QuestionJudgeClient(client=client).judge(PAYLOAD)

                self.assertTrue(call.succeeded)
                self.assertEqual(2, client.responses.create.await_count)
                self.assertEqual(1, call.trace.retry_count)
                self.assertEqual(5000, call.trace.input_tokens)

    async def test_gives_up_after_second_transient_failure(self) -> None:
        client = _client(_status_error(500), _status_error(502))

        call = await QuestionJudgeClient(client=client).judge(PAYLOAD)

        self.assertFalse(call.succeeded)
        self.assertEqual(2, client.responses.create.await_count)
        self.assertEqual(JudgeFailureKind.API_ERROR, call.failure.kind)
        self.assertEqual("OpenAI 판별 호출 실패: HTTP 502", call.failure.safe_message)
        self.assertEqual(1, call.trace.retry_count)
        self.assertFalse(call.trace.succeeded)
        self.assertEqual(call.failure.safe_message, call.trace.error_message)
        self.assertIsNone(call.output_text)

    async def test_does_not_retry_permanent_error(self) -> None:
        client = _client(_status_error(400, body={"code": "invalid_json_schema"}))

        call = await QuestionJudgeClient(client=client).judge(PAYLOAD)

        self.assertEqual(1, client.responses.create.await_count)
        self.assertEqual(0, call.trace.retry_count)
        self.assertEqual(JudgeFailureKind.API_ERROR, call.failure.kind)
        self.sleep.assert_not_awaited()

    async def test_unexpected_exception_becomes_internal_failure(self) -> None:
        client = _client(KeyError(QUESTION))

        call = await QuestionJudgeClient(client=client).judge(PAYLOAD)

        self.assertEqual(JudgeFailureKind.INTERNAL_ERROR, call.failure.kind)
        self.assertEqual("판별 호출 중 오류: KeyError", call.failure.safe_message)
        self.assertEqual(1, client.responses.create.await_count)

    async def test_incomplete_response_is_a_failure_with_usage(self) -> None:
        client = _client(
            _response(
                output_text="",
                status="incomplete",
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            )
        )

        call = await QuestionJudgeClient(client=client).judge(PAYLOAD)

        self.assertEqual(JudgeFailureKind.INCOMPLETE_RESPONSE, call.failure.kind)
        self.assertEqual(
            "판별 응답이 완료되지 않았습니다: status=incomplete, reason=max_output_tokens",
            call.failure.safe_message,
        )
        self.assertEqual(300, call.trace.output_tokens)
        self.assertEqual(1, client.responses.create.await_count)

    async def test_malformed_output_is_returned_for_normalization(self) -> None:
        for text in ("", "{broken"):
            with self.subTest(text=text):
                client = _client(_response(output_text=text))

                call = await QuestionJudgeClient(client=client).judge(PAYLOAD)

                self.assertTrue(call.succeeded)
                self.assertEqual(text, call.output_text)
                self.assertEqual(1, client.responses.create.await_count)

    async def test_error_messages_never_contain_question_or_output(self) -> None:
        cases = (
            _client(_status_error(400), _response()),
            _client(KeyError(QUESTION)),
            _client(
                _response(
                    output_text=QUESTION,
                    status=QUESTION,
                    incomplete_details=SimpleNamespace(reason=QUESTION),
                )
            ),
        )
        for client in cases:
            call = await QuestionJudgeClient(client=client).judge(PAYLOAD)

            self.assertNotIn(QUESTION, call.failure.safe_message)
            self.assertNotIn("비밀", call.trace.error_message)

    def test_safe_message_keeps_only_status_like_codes(self) -> None:
        error = _status_error(429, body={"code": "rate_limit_exceeded"})
        noisy = _status_error(400, body={"code": QUESTION})

        self.assertEqual(
            "OpenAI 판별 호출 실패: HTTP 429 (rate_limit_exceeded)",
            safe_judge_error_message(error),
        )
        self.assertEqual(
            "OpenAI 판별 호출 실패: HTTP 400 (unknown)",
            safe_judge_error_message(noisy),
        )


if __name__ == "__main__":
    unittest.main()
