"""판별 모델(OpenAI Responses API) 호출.

- strict json_schema, reasoning low, store False. SDK 재시도는 끄고 일시 오류만 1회
  재시도한다. 출력 형식 문제는 재시도하지 않고 decision 정규화에 넘긴다.
- 실패를 예외로 던지지 않고 JudgeCall 에 담는다. 판별 호출 실패는 fail-open 이다.
- 오류 메시지에는 HTTP 상태, 예외 이름, API 상태 코드만 남긴다. 질문과 payload,
  모델 출력은 판별 행과 model_calls 어디에도 오류 문구로 들어가지 않는다.
"""

import asyncio
import json
import re
import time
from typing import Any, Mapping, Optional

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from app.core.config import get_settings
from app.core.model_trace import BeforeModelCallHook, ModelCallTrace
from app.core.openai_error import is_transient_openai_error
from app.core.openai_usage import cached_input_tokens as usage_cached_input_tokens
from app.core.openai_usage import reasoning_tokens as usage_reasoning_tokens
from app.question_grouping.constants import (
    JUDGE_MAX_ATTEMPTS,
    JUDGE_MAX_OUTPUT_TOKENS,
    JUDGE_MODEL,
    JUDGE_OUTPUT_SCHEMA_NAME,
    JUDGE_PROMPT_VERSION,
    JUDGE_PROVIDER,
    JUDGE_REASONING_EFFORT,
    JUDGE_RETRY_DELAY_SECONDS,
    JUDGE_TIMEOUT_SECONDS,
)
from app.question_grouping.models import JudgeCall, JudgeFailure, JudgeFailureKind
from app.question_grouping.prompt_v7_2 import JUDGE_INSTRUCTIONS, JUDGE_OUTPUT_SCHEMA


# API 상태·사유 값만 통과시켜 모델이 만든 문자열이 오류 문구에 섞이지 않게 한다.
_STATUS_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")


def _status_code(value: object) -> str:
    if isinstance(value, str) and _STATUS_CODE_PATTERN.match(value):
        return value
    return "unknown"


def _add_token_count(total: Optional[int], value: Optional[int]) -> Optional[int]:
    if value is None:
        return total
    return (total or 0) + value


def safe_judge_error_message(error: BaseException) -> str:
    """판별 호출 오류를 질문과 응답 본문 없이 설명한다."""

    if isinstance(error, APIStatusError):
        code = getattr(error, "code", None)
        suffix = f" ({_status_code(code)})" if code else ""
        return f"OpenAI 판별 호출 실패: HTTP {error.status_code}{suffix}"
    if isinstance(error, APIConnectionError):
        return f"OpenAI 판별 호출 연결 실패: {type(error).__name__}"
    return f"판별 호출 중 오류: {type(error).__name__}"


def build_judge_request(payload: Mapping[str, Any]) -> dict:
    """responses.create 인자. PoC classify_structured 와 같은 요청 형태다."""

    return {
        "model": JUDGE_MODEL,
        "store": False,
        "instructions": JUDGE_INSTRUCTIONS,
        "input": json.dumps(payload, ensure_ascii=False),
        "text": {
            "format": {
                "type": "json_schema",
                "name": JUDGE_OUTPUT_SCHEMA_NAME,
                "strict": True,
                "schema": JUDGE_OUTPUT_SCHEMA,
            }
        },
        "reasoning": {"effort": JUDGE_REASONING_EFFORT},
        "max_output_tokens": JUDGE_MAX_OUTPUT_TOKENS,
    }


class QuestionJudgeClient:
    """세부 문제와 문서 후보를 한 번에 판별한다."""

    def __init__(self, client: Optional[AsyncOpenAI] = None) -> None:
        if client is None:
            api_key = get_settings().openai_api_key
            if not api_key:
                raise ValueError("OPENAI_API_KEY 환경변수가 필요합니다.")
            client = AsyncOpenAI(
                api_key=api_key,
                max_retries=0,
                timeout=JUDGE_TIMEOUT_SECONDS,
            )
        self._client = client

    @property
    def provider(self) -> str:
        return JUDGE_PROVIDER

    @property
    def model_name(self) -> str:
        return JUDGE_MODEL

    @property
    def prompt_version(self) -> str:
        return JUDGE_PROMPT_VERSION

    async def judge(
        self,
        payload: Mapping[str, Any],
        *,
        before_model_call: Optional[BeforeModelCallHook] = None,
    ) -> JudgeCall:
        """논리적 호출 한 건으로 최대 JUDGE_MAX_ATTEMPTS 번 시도한다."""

        request = build_judge_request(payload)
        if before_model_call is not None:
            await before_model_call(JUDGE_PROVIDER, JUDGE_MODEL, JUDGE_PROMPT_VERSION)

        started = time.perf_counter()
        usage_totals = _UsageTotals()
        attempt = 0
        while True:
            try:
                response = await self._client.responses.create(**request)
            except Exception as error:  # noqa: BLE001 - 실패는 JudgeCall 로 올린다
                can_retry = attempt + 1 < JUDGE_MAX_ATTEMPTS
                if can_retry and is_transient_openai_error(error):
                    attempt += 1
                    await asyncio.sleep(JUDGE_RETRY_DELAY_SECONDS)
                    continue
                kind = (
                    JudgeFailureKind.API_ERROR
                    if isinstance(error, (APIStatusError, APIConnectionError))
                    else JudgeFailureKind.INTERNAL_ERROR
                )
                return self._failed_call(
                    started,
                    attempt,
                    usage_totals,
                    JudgeFailure(
                        kind=kind, safe_message=safe_judge_error_message(error)
                    ),
                )

            usage_totals.add(getattr(response, "usage", None))
            status = getattr(response, "status", None)
            if status is not None and status != "completed":
                details = getattr(response, "incomplete_details", None)
                reason = _status_code(getattr(details, "reason", None))
                return self._failed_call(
                    started,
                    attempt,
                    usage_totals,
                    JudgeFailure(
                        kind=JudgeFailureKind.INCOMPLETE_RESPONSE,
                        safe_message=(
                            "판별 응답이 완료되지 않았습니다: "
                            f"status={_status_code(status)}, reason={reason}"
                        ),
                    ),
                )

            output_text = getattr(response, "output_text", None)
            return JudgeCall(
                trace=self._trace(started, attempt, usage_totals),
                output_text=output_text if isinstance(output_text, str) else None,
            )

    def _failed_call(
        self,
        started: float,
        attempt: int,
        usage_totals: "_UsageTotals",
        failure: JudgeFailure,
    ) -> JudgeCall:
        return JudgeCall(
            trace=self._trace(started, attempt, usage_totals, failure=failure),
            failure=failure,
        )

    @staticmethod
    def _trace(
        started: float,
        attempt: int,
        usage_totals: "_UsageTotals",
        *,
        failure: Optional[JudgeFailure] = None,
    ) -> ModelCallTrace:
        return ModelCallTrace(
            provider=JUDGE_PROVIDER,
            model_name=JUDGE_MODEL,
            succeeded=failure is None,
            latency_ms=int((time.perf_counter() - started) * 1000),
            retry_count=attempt,
            input_tokens=usage_totals.input_tokens,
            output_tokens=usage_totals.output_tokens,
            cached_input_tokens=usage_totals.cached_input_tokens,
            reasoning_tokens=usage_totals.reasoning_tokens,
            prompt_version=JUDGE_PROMPT_VERSION,
            error_message=None if failure is None else failure.safe_message,
        )


class _UsageTotals:
    """시도별 usage 를 합산한다. 내역이 한 번도 없으면 None 으로 둔다."""

    def __init__(self) -> None:
        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self.cached_input_tokens: Optional[int] = None
        self.reasoning_tokens: Optional[int] = None

    def add(self, usage: object) -> None:
        if usage is None:
            return
        self.input_tokens = _add_token_count(
            self.input_tokens, getattr(usage, "input_tokens", None)
        )
        self.output_tokens = _add_token_count(
            self.output_tokens, getattr(usage, "output_tokens", None)
        )
        self.cached_input_tokens = _add_token_count(
            self.cached_input_tokens, usage_cached_input_tokens(usage)
        )
        self.reasoning_tokens = _add_token_count(
            self.reasoning_tokens, usage_reasoning_tokens(usage)
        )
