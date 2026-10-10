"""판별 모델 어댑터.

두 어댑터 모두 PreparedTurn.payload_bytes 하나만 입력으로 받는다.

- luna: 운영 QuestionJudgeClient(app/question_grouping/judge_client.py)에 payload 를 넘기고, 응답은
  운영 build_turn_judgment(decision.py, R7 보정)로 결정을 만든다. OpenAI 클라이언트는 주입한다.
  responses.create 의 input 문자열이 payload_bytes 와 같은지 호출 전에 확인한다.
- Jev: 2026-10-01 탐색 실행(jev-judge-run-20261001)과 같은 요청이다. choice 프리미티브, 모델
  jev-latest, 지시문은 묶음 P 규칙 11키(jev_rules_set_p.json, 원본 Claude outputs/jev-smoke-20260925
  run_set_mno.py 8키와 run_set_p.py 3키), 후보마다 `이름 / 포함: ... / 제외: ...` 에 정본 본문과
  적용 제외 규칙을 붙이고 `_none` 선택지를 더한다. 그룹 이름과 문서 후보는 넘기지 않는다(묶음 P 와 같음).
  신뢰도 0.95 이상이면 고른 키로 CONNECT, `_none` 이면 SEPARATE, 0.95 미만이면 UNCLASSIFIED 다.
  전송 함수는 주입한다.

실제 전송(OpenAI, TypeSafe)은 cli.py 의 --live 에서만 만든다. 시험은 가짜 클라이언트만 쓴다.
"""

import asyncio
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

from app.question_grouping.constants import (
    JUDGE_MAX_ATTEMPTS,
    JUDGE_MAX_OUTPUT_TOKENS,
    JUDGE_MODEL,
    JUDGE_PROMPT_VERSION,
    JUDGE_PROVIDER,
    JUDGE_REASONING_EFFORT,
)
from app.question_grouping.decision import build_turn_judgment
from app.question_grouping.judge_client import QuestionJudgeClient, build_judge_request
from evaluation.judge_eval.inputs import PreparedTurn, request_sha256

JEV_RULES_PATH = Path(__file__).with_name("jev_rules_set_p.json")
JEV_RULES: Dict[str, Any] = json.loads(JEV_RULES_PATH.read_text(encoding="utf-8"))
JEV_RULES_SHA256 = request_sha256(JEV_RULES)
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_PROMPT_VERSION = "jev-choice-set-p"
JEV_THRESHOLD = 0.95
JEV_QUESTION_NAME = "세부문제"
JEV_NONE_KEY = "_none"
JEV_NONE_TEXT = "위 어느 세부 문제에도 해당하지 않음"
# 첫 시도 1회 + 429, 5xx 일 때 2초 뒤 1회 (run_set_mno.py jev_call 과 같음)
JEV_MAX_ATTEMPTS = 2
JEV_RETRY_DELAY_SECONDS = 2.0
JEV_TIMEOUT_SECONDS = 90.0


@dataclass(frozen=True)
class Variant:
    name: str
    provider: str
    model: str
    prompt_version: str
    reasoning_effort: Optional[str]
    max_attempts: int
    settings: Dict[str, Any] = field(default_factory=dict)

    def stage_model(self) -> Dict[str, Any]:
        """eval-result.schema.json models.judge"""

        return {
            "provider": self.provider,
            "model": self.model,
            "promptVersion": self.prompt_version,
            "reasoningEffort": self.reasoning_effort,
        }


VARIANTS: Dict[str, Variant] = {
    "luna": Variant(
        name="luna",
        provider=JUDGE_PROVIDER,
        model=JUDGE_MODEL,
        prompt_version=JUDGE_PROMPT_VERSION,
        reasoning_effort=JUDGE_REASONING_EFFORT,
        max_attempts=JUDGE_MAX_ATTEMPTS,
        settings={
            "client": "app.question_grouping.judge_client.QuestionJudgeClient",
            "normalization": "app.question_grouping.decision.build_turn_judgment",
            "maxOutputTokens": JUDGE_MAX_OUTPUT_TOKENS,
            "maxAttempts": JUDGE_MAX_ATTEMPTS,
        },
    ),
    "jev": Variant(
        name="jev",
        provider="typesafe",
        model=JEV_MODEL,
        prompt_version=JEV_PROMPT_VERSION,
        reasoning_effort=None,
        max_attempts=JEV_MAX_ATTEMPTS,
        settings={
            "endpoint": JEV_ENDPOINT,
            "primitive": "choice",
            "threshold": JEV_THRESHOLD,
            "rulesFile": "evaluation/judge_eval/jev_rules_set_p.json",
            "rulesSha256": JEV_RULES_SHA256,
            "decisionMapping": {
                "CONNECT": f"confidence >= {JEV_THRESHOLD} 이고 choice 가 세부 문제 키",
                "SEPARATE": f"confidence >= {JEV_THRESHOLD} 이고 choice 가 {JEV_NONE_KEY}",
                "UNCLASSIFIED": f"confidence < {JEV_THRESHOLD}",
            },
            "notPassed": "그룹 이름, 문서 후보 (묶음 P 와 같음)",
            "maxAttempts": JEV_MAX_ATTEMPTS,
        },
    ),
}


def variant_by_name(name: str) -> Variant:
    try:
        return VARIANTS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"모르는 판별 모델입니다: {name} (가능: {', '.join(VARIANTS)})") from exc


@dataclass(frozen=True)
class Usage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None

    def to_json(self) -> Dict[str, Optional[int]]:
        return {
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
            "cachedInputTokens": self.cached_input_tokens,
            "reasoningTokens": self.reasoning_tokens,
        }


@dataclass(frozen=True)
class JudgeOutcome:
    """판별 한 턴. error 가 있으면 호출 실패(결정 없음)다. 원출력에는 입력 문장을 담지 않는다."""

    decision: Optional[str]
    subproblem_key: Optional[str] = None
    confidence: Optional[float] = None
    invalid_reason: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: Optional[int] = None
    usage: Usage = Usage()
    attempts: int = 1


# ---------------------------------------------------------------- luna


class LunaJudge:
    variant = VARIANTS["luna"]

    def __init__(self, openai_client: Any = None) -> None:
        # openai_client 가 None 이면 QuestionJudgeClient 가 설정의 OPENAI_API_KEY 로 만든다 (--live 전용).
        self._client = QuestionJudgeClient(client=openai_client)

    async def judge(self, turn: PreparedTurn) -> JudgeOutcome:
        assert turn.payload_bytes is not None and turn.presentation is not None
        payload = json.loads(turn.payload_bytes)
        if build_judge_request(payload)["input"].encode("utf-8") != turn.payload_bytes:
            raise RuntimeError(f"{turn.key}: luna 요청 input 이 준비한 payload 와 다릅니다")
        call = await self._client.judge(payload)
        trace = call.trace
        usage = Usage(
            input_tokens=trace.input_tokens,
            output_tokens=trace.output_tokens,
            cached_input_tokens=trace.cached_input_tokens,
            reasoning_tokens=trace.reasoning_tokens,
        )
        attempts = (trace.retry_count or 0) + 1
        if call.failure is not None:
            return JudgeOutcome(
                decision=None,
                error=f"{call.failure.kind.value}: {call.failure.safe_message}",
                latency_ms=trace.latency_ms,
                usage=usage,
                attempts=attempts,
                raw={"failure": call.failure.kind.value},
            )
        judgment = build_turn_judgment(call, turn.presentation)
        normalized = judgment.normalized
        return JudgeOutcome(
            decision=judgment.decision.value,
            subproblem_key=None if judgment.subproblem is None else judgment.subproblem.key,
            confidence=None if normalized is None else normalized.confidence,
            invalid_reason=None if normalized is None else normalized.invalid_reason,
            raw={
                "output": None if normalized is None else normalized.raw_output,
                "normalization": None if normalized is None else normalized.normalization_judgment_input(),
            },
            latency_ms=trace.latency_ms,
            usage=usage,
            attempts=attempts,
        )


# ---------------------------------------------------------------- Jev


def jev_criteria_text(subproblem: Mapping[str, Any]) -> str:
    inclusion = " ".join((subproblem.get("inclusionCriteria") or {}).values())
    exclusion = " ".join((subproblem.get("exclusionCriteria") or {}).values())
    text = f"{subproblem['name']} / 포함: {inclusion} / 제외: {exclusion}".replace("\n", " ")
    canonical = subproblem.get("canonicalAnswer")
    if canonical:
        text += f" / 정본(사용자에게 그대로 전송될 답변): {canonical['contentMarkdown']}"
        rules = canonical.get("applicabilityRules") or []
        if rules:
            text += f" / 정본이 다루지 않는 것: {'; '.join(rules)}"
    return text


def build_jev_request(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """luna payload 에서 Jev 요청 본문을 만든다. 후보는 payload 제시 순서 그대로다."""

    criteria: Dict[str, str] = {}
    for candidate in payload.get("candidates") or []:
        subproblem = candidate["subproblem"]
        criteria[subproblem["id"]] = jev_criteria_text(subproblem)
    criteria[JEV_NONE_KEY] = JEV_NONE_TEXT
    return {
        "model": JEV_MODEL,
        "state": {"질문": payload["question"]},
        "questions": {JEV_QUESTION_NAME: {"type": "choice", "instructions": JEV_RULES, "criteria": criteria}},
    }


JevTransport = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


def _transient(outcome: Mapping[str, Any]) -> bool:
    status = outcome.get("httpStatus")
    return not outcome.get("ok") and (status == 429 or (isinstance(status, int) and status >= 500))


class JevJudge:
    variant = VARIANTS["jev"]

    def __init__(self, transport: JevTransport, *, retry_delay: float = JEV_RETRY_DELAY_SECONDS) -> None:
        self._transport = transport
        self._retry_delay = retry_delay

    async def judge(self, turn: PreparedTurn) -> JudgeOutcome:
        assert turn.payload_bytes is not None
        payload = json.loads(turn.payload_bytes)
        body = build_jev_request(payload)
        started = time.perf_counter()
        attempts = 1
        outcome = await self._transport(body)
        if _transient(outcome) and attempts < JEV_MAX_ATTEMPTS:
            await asyncio.sleep(self._retry_delay)
            attempts += 1
            outcome = await self._transport(body)
        latency = int((time.perf_counter() - started) * 1000)
        if not outcome.get("ok"):
            return JudgeOutcome(
                decision=None,
                error=f"Jev 호출 실패: HTTP {outcome.get('httpStatus')}",
                latency_ms=latency,
                attempts=attempts,
                raw={"httpStatus": outcome.get("httpStatus")},
            )
        response = outcome.get("body") or {}
        usage_body = response.get("usage") or {}
        usage = Usage(input_tokens=usage_body.get("input_tokens"), output_tokens=usage_body.get("output_tokens"))
        answer = (response.get("answers") or {}).get(JEV_QUESTION_NAME) or {}
        choice = answer.get("choice")
        confidence = answer.get("confidence")
        raw = {"model": response.get("model"), "answer": answer}
        presented = set(turn.presented_keys)
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or (
            choice != JEV_NONE_KEY and choice not in presented
        ):
            invalid = "UNKNOWN_SUBPROBLEM_KEY" if isinstance(choice, str) and choice else "UNPARSEABLE_OUTPUT"
            return JudgeOutcome(
                decision="UNCLASSIFIED", invalid_reason=invalid, raw=raw, latency_ms=latency, usage=usage, attempts=attempts
            )
        if confidence < JEV_THRESHOLD:
            decision, key = "UNCLASSIFIED", None
        elif choice == JEV_NONE_KEY:
            decision, key = "SEPARATE", None
        else:
            decision, key = "CONNECT", choice
        return JudgeOutcome(
            decision=decision,
            subproblem_key=key,
            confidence=float(confidence),
            raw=raw,
            latency_ms=latency,
            usage=usage,
            attempts=attempts,
        )


def urllib_jev_transport(api_key: str, *, timeout: float = JEV_TIMEOUT_SECONDS) -> JevTransport:
    """TypeSafe systemone 실제 전송 (--live 전용). 키는 출력하지 않고 오류에는 상태만 남긴다."""

    def post(body: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            JEV_ENDPOINT,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return {"ok": True, "httpStatus": response.status, "body": json.loads(response.read().decode("utf-8"))}
        except urllib.error.HTTPError as error:
            return {"ok": False, "httpStatus": error.code}
        except Exception as error:  # noqa: BLE001 - 요청 본문이 메시지에 실리지 않게 종류만 남긴다
            return {"ok": False, "httpStatus": None, "error": type(error).__name__}

    async def transport(body: Dict[str, Any]) -> Dict[str, Any]:
        return await asyncio.to_thread(post, body)

    return transport


# ---------------------------------------------------------------- 단가와 추정


@dataclass(frozen=True)
class Prices:
    """USD / 100만 토큰. None 은 모름."""

    input: Optional[float]
    output: Optional[float]
    cached_input: Optional[float] = None


# luna 단가: evaluation/run_generation_model_comparison.py CANDIDATE_CONFIGS 의 gpt-5.6-luna (0.20, 1.20).
# 캐시 입력 단가는 저장소에 없다 (#204 작업 목록 "평가 단가표에 캐시 입력 단가" 대기).
# Jev 단가는 저장소에 없다. 인자로 받는다.
DEFAULT_PRICES: Dict[str, Prices] = {
    "luna": Prices(input=0.20, output=1.20, cached_input=None),
    "jev": Prices(input=None, output=None, cached_input=None),
}

# 문자 수 → 토큰 추정 비율과 턴당 출력 토큰. 2026-10-01 탐색 실행의 A층 159턴 실측 토큰
# (Claude outputs/version-history-20260928/luna-judge-run 의 ids-only 결과, luna 평균 입력 5,300.5 출력 373.7,
# Jev 평균 입력 3,073.1)과 이 모듈이 보충 파일 없이 재구성한 같은 턴의 요청 문자 수로 맞췄다. 추정용이며 보고 수치가 아니다.
CHARS_PER_TOKEN = {"luna": 3.38, "jev": 2.02}
MEAN_OUTPUT_TOKENS = {"luna": 374, "jev": 0}


def request_chars(variant: str, turn: PreparedTurn) -> int:
    assert turn.payload_bytes is not None
    payload = json.loads(turn.payload_bytes)
    if variant == "luna":
        request = build_judge_request(payload)
        schema = json.dumps(request["text"]["format"]["schema"], ensure_ascii=False)
        return len(request["instructions"]) + len(request["input"]) + len(schema)
    return len(json.dumps(build_jev_request(payload), ensure_ascii=False))


def turn_cost(variant: str, usage: Usage, prices: Prices) -> Dict[str, Optional[float]]:
    """(cachedUsd, upperUsd). 단가를 모르면 None. 캐시 단가를 모르면 cached 는 upper 와 같다."""

    if prices.input is None:
        return {"cachedUsd": None, "upperUsd": None}
    input_tokens = usage.input_tokens or 0
    output_tokens = usage.output_tokens or 0
    output_cost = output_tokens * (prices.output or 0.0) / 1e6
    upper = input_tokens * prices.input / 1e6 + output_cost
    cached_tokens = usage.cached_input_tokens or 0
    if prices.cached_input is None or not cached_tokens:
        cached = upper
    else:
        cached = (input_tokens - cached_tokens) * prices.input / 1e6 + cached_tokens * prices.cached_input / 1e6 + output_cost
    return {"cachedUsd": cached, "upperUsd": upper}


def estimate(variant: Variant, prepared: List[PreparedTurn], prices: Prices) -> Dict[str, Any]:
    judged = [turn for turn in prepared if turn.judged]
    ratio = CHARS_PER_TOKEN[variant.name]
    input_tokens = [int(-(-request_chars(variant.name, turn) // ratio)) for turn in judged]
    total_input = sum(input_tokens)
    mean_output = MEAN_OUTPUT_TOKENS[variant.name]
    max_output = JUDGE_MAX_OUTPUT_TOKENS if variant.name == "luna" else 0
    result: Dict[str, Any] = {
        "variant": variant.name,
        "model": variant.model,
        "turns": len(prepared),
        "calls": len(judged),
        "callsUpperWithRetries": len(judged) * variant.max_attempts,
        "skippedTurns": len(prepared) - len(judged),
        "estimatedInputTokens": total_input,
        "meanInputTokensPerCall": round(total_input / len(judged), 1) if judged else 0,
        "assumedOutputTokensPerCall": mean_output,
        "charsPerToken": ratio,
        "prices": {"input": prices.input, "output": prices.output, "cachedInput": prices.cached_input},
    }
    if prices.input is None:
        result["estimatedUsd"] = None
        result["upperUsd"] = None
        result["note"] = "단가를 모름. --input-price 로 준다"
        return result
    expected = (total_input * prices.input + len(judged) * mean_output * (prices.output or 0.0)) / 1e6
    upper = variant.max_attempts * (total_input * prices.input + len(judged) * max_output * (prices.output or 0.0)) / 1e6
    result["estimatedUsd"] = round(expected, 6)
    result["upperUsd"] = round(upper, 6)
    result["note"] = "upperUsd 는 캐시 할인 없음, 모든 호출 재시도, 출력 상한 토큰을 가정"
    return result
