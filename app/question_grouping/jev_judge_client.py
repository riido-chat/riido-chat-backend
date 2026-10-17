"""Jev judge adapter using the established TypeSafe choice contract.

Network access and credentials stay in an injected transport. This module builds
the known request, applies the established retry/choice rules, and converts the
choice result to the JSON shape consumed by ``build_turn_judgment``.
"""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Optional

from app.core.model_trace import BeforeModelCallHook, ModelCallTrace
from app.core.stage_settings import StageSettings
from app.question_grouping.models import JudgeCall, JudgeFailure, JudgeFailureKind

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_PROVIDER = "typesafe"
JEV_MODEL = "jev-latest"
JEV_PROMPT_VERSION = "jev-choice-set-p"
JEV_THRESHOLD = 0.95
JEV_QUESTION_NAME = "세부문제"
JEV_NONE_KEY = "_none"
JEV_NONE_TEXT = "위 어느 세부 문제에도 해당하지 않음"
JEV_MAX_ATTEMPTS = 2
JEV_RETRY_DELAY_SECONDS = 2.0
JEV_TIMEOUT_SECONDS = 90.0
JEV_RULES_PATH = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "judge_eval"
    / "jev_rules_set_p.json"
)
JEV_RULES: Mapping[str, Any] = json.loads(JEV_RULES_PATH.read_text(encoding="utf-8"))

JevTransport = Callable[[dict[str, Any]], Awaitable[Mapping[str, Any]]]


def jev_criteria_text(subproblem: Mapping[str, Any]) -> str:
    """Render candidate criteria exactly as evaluation bundle P does."""

    inclusion = " ".join((subproblem.get("inclusionCriteria") or {}).values())
    exclusion = " ".join((subproblem.get("exclusionCriteria") or {}).values())
    text = f"{subproblem['name']} / 포함: {inclusion} / 제외: {exclusion}".replace(
        "\n", " "
    )
    canonical = subproblem.get("canonicalAnswer")
    if canonical:
        text += (
            " / 정본(사용자에게 그대로 전송될 답변): "
            f"{canonical['contentMarkdown']}"
        )
        rules = canonical.get("applicabilityRules") or []
        if rules:
            text += f" / 정본이 다루지 않는 것: {'; '.join(rules)}"
    return text


def build_jev_request(
    payload: Mapping[str, Any], *, model_name: str = JEV_MODEL
) -> dict[str, Any]:
    """Translate the production judge payload to the established Jev choice request."""

    criteria: dict[str, str] = {}
    for candidate in payload.get("candidates") or []:
        subproblem = candidate["subproblem"]
        criteria[subproblem["id"]] = jev_criteria_text(subproblem)
    criteria[JEV_NONE_KEY] = JEV_NONE_TEXT
    return {
        "model": model_name,
        "state": {"질문": payload["question"]},
        "questions": {
            JEV_QUESTION_NAME: {
                "type": "choice",
                "instructions": JEV_RULES,
                "criteria": criteria,
            }
        },
    }


def _transient(outcome: Mapping[str, Any]) -> bool:
    status = outcome.get("httpStatus")
    return not outcome.get("ok") and (
        status == 429 or (isinstance(status, int) and status >= 500)
    )


def _normalized_output(choice: Any, confidence: Any, presented: set[str]) -> str:
    """Convert Jev's choice primitive to the R7 normalized-judge input shape."""

    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or (
            choice != JEV_NONE_KEY
            and (not isinstance(choice, str) or choice not in presented)
        )
    ):
        if isinstance(choice, str) and choice and choice != JEV_NONE_KEY:
            # Keep R7's established unknown-key repair and audit behavior.
            output = {"decision": "CONNECT", "subproblemId": choice}
            if isinstance(confidence, (int, float)) and not isinstance(
                confidence, bool
            ):
                output["confidence"] = confidence
            return json.dumps(output, ensure_ascii=False)
        return "{}"

    if confidence < JEV_THRESHOLD:
        output = {"decision": "UNCLASSIFIED", "confidence": confidence}
    elif choice == JEV_NONE_KEY:
        output = {"decision": "SEPARATE", "confidence": confidence}
    else:
        output = {
            "decision": "CONNECT",
            "subproblemId": choice,
            "confidence": confidence,
        }
    return json.dumps(output, ensure_ascii=False)


class JevJudgeClient:
    """Production judge-client interface backed by an injected Jev transport."""

    def __init__(
        self,
        transport: JevTransport,
        *,
        retry_delay: float = JEV_RETRY_DELAY_SECONDS,
    ) -> None:
        self._transport = transport
        self._retry_delay = retry_delay

    @property
    def provider(self) -> str:
        return JEV_PROVIDER

    @property
    def model_name(self) -> str:
        return JEV_MODEL

    @property
    def prompt_version(self) -> str:
        return JEV_PROMPT_VERSION

    async def judge(
        self,
        payload: Mapping[str, Any],
        *,
        before_model_call: Optional[BeforeModelCallHook] = None,
        stage: Optional[StageSettings] = None,
    ) -> JudgeCall:
        model_name = self.model_name if stage is None else stage.model_name
        prompt_version = self.prompt_version if stage is None else stage.prompt_version
        if before_model_call is not None:
            await before_model_call(self.provider, model_name, prompt_version)

        request = build_jev_request(payload, model_name=model_name)
        started = time.perf_counter()
        attempts = 1
        try:
            outcome = await self._transport(request)
            if _transient(outcome) and attempts < JEV_MAX_ATTEMPTS:
                await asyncio.sleep(self._retry_delay)
                attempts += 1
                outcome = await self._transport(request)
        except Exception as error:  # noqa: BLE001 - preserve judge failure contract
            failure = JudgeFailure(
                kind=JudgeFailureKind.INTERNAL_ERROR,
                safe_message=f"Jev 판별 호출 중 오류: {type(error).__name__}",
            )
            return self._failed(started, attempts, failure, model_name, prompt_version)

        if not outcome.get("ok"):
            status = outcome.get("httpStatus")
            safe_status = status if isinstance(status, int) else "unknown"
            failure = JudgeFailure(
                kind=JudgeFailureKind.API_ERROR,
                safe_message=f"Jev 호출 실패: HTTP {safe_status}",
            )
            return self._failed(started, attempts, failure, model_name, prompt_version)

        response = outcome.get("body") or {}
        if not isinstance(response, Mapping):
            response = {}
        usage_body = response.get("usage") or {}
        if not isinstance(usage_body, Mapping):
            usage_body = {}
        answers = response.get("answers") or {}
        if not isinstance(answers, Mapping):
            answers = {}
        answer = answers.get(JEV_QUESTION_NAME) or {}
        if not isinstance(answer, Mapping):
            answer = {}
        output_text = _normalized_output(
            answer.get("choice"),
            answer.get("confidence"),
            {
                candidate["subproblem"]["id"]
                for candidate in payload.get("candidates") or []
            },
        )
        trace = ModelCallTrace(
            provider=self.provider,
            model_name=model_name,
            succeeded=True,
            latency_ms=int((time.perf_counter() - started) * 1000),
            retry_count=attempts - 1,
            input_tokens=usage_body.get("input_tokens"),
            output_tokens=usage_body.get("output_tokens"),
            prompt_version=prompt_version,
        )
        return JudgeCall(trace=trace, output_text=output_text)

    def _failed(
        self,
        started: float,
        attempts: int,
        failure: JudgeFailure,
        model_name: str,
        prompt_version: Optional[str],
    ) -> JudgeCall:
        return JudgeCall(
            trace=ModelCallTrace(
                provider=self.provider,
                model_name=model_name,
                succeeded=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
                retry_count=attempts - 1,
                prompt_version=prompt_version,
                error_message=failure.safe_message,
            ),
            failure=failure,
        )
