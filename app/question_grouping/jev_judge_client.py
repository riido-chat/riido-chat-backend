"""Jev judge adapter using the established TypeSafe choice contract.

Network access and credentials stay in an injected transport. This module builds
the known request, applies the established retry/choice rules, and converts the
choice result to the JSON shape consumed by ``build_turn_judgment``.
"""

import asyncio
import copy
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from app.core.model_trace import (
    BeforeModelCallHook,
    ModelCallTrace,
    ModelRequestEvidence,
    canonical_request_sha256,
)
from app.prompts import registry as prompt_registry
from app.core.stage_settings import StageSettings
from app.question_grouping.models import JudgeCall, JudgeFailure, JudgeFailureKind
from app.question_grouping.constants import DOCUMENT_CANDIDATE_TOP_K

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_PROVIDER = "typesafe"
JEV_MODEL = "jev-latest"
JEV_LEGACY_PROMPT_VERSION = "jev-choice-set-p"
JEV_PROMPT_VERSION = "jev-cache-choice-context-v1"
JEV_INPUT_TRANSFORM_VERSION = "jev-choice-context-v1"
JEV_THRESHOLD = 0.95
JEV_QUESTION_NAME = "세부문제"
JEV_NONE_KEY = "_none"
JEV_NONE_TEXT = "위 어느 세부 문제에도 해당하지 않음"
JEV_MAX_ATTEMPTS = 2
JEV_MAX_CANDIDATES = 5
JEV_RETRY_DELAY_SECONDS = 2.0
JEV_TIMEOUT_SECONDS = 90.0
JEV_RULES_PATH = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "judge_eval"
    / "jev_rules_set_p.json"
)
JEV_CONTEXT_PROMPT = prompt_registry.get("judge", JEV_PROMPT_VERSION)
JEV_CONTEXT_INSTRUCTIONS: Mapping[str, Any] = {"state_context": JEV_CONTEXT_PROMPT}
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TOP_LEVEL_KEYS = {
    "policyVersion", "question", "contextSummary", "candidateCount",
    "candidates", "documentCandidateCount", "documentCandidates",
}
_SUBPROBLEM_KEYS = {
    "id", "name", "inclusionCriteria", "exclusionCriteria", "canonicalAnswer",
}
_CANONICAL_KEYS = {"contentMarkdown", "applicabilityRules"}
_DOCUMENT_KEYS = {"id", "title", "parentPath", "headings"}
_CANDIDATE_KEYS = {"rank", "group", "subproblem"}


@lru_cache(maxsize=1)
def _legacy_jev_rules() -> Mapping[str, Any]:
    """Read the frozen P rules only when the explicitly legacy API is selected."""

    return json.loads(JEV_RULES_PATH.read_text(encoding="utf-8"))


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
    """Build the context request while retaining the common input verbatim in state."""

    common_input = copy.deepcopy(dict(payload))
    _validate_common_input(common_input)

    criteria: dict[str, str] = {}
    for candidate in common_input["candidates"]:
        subproblem = candidate["subproblem"]
        criteria[subproblem["id"]] = (
            f"{subproblem['name']} (state.classifierInput.candidates ID: "
            f"{subproblem['id']})"
        )
    criteria[JEV_NONE_KEY] = JEV_NONE_TEXT
    return {
        "model": model_name,
        "state": {"classifierInput": common_input},
        "questions": {
            JEV_QUESTION_NAME: {
                "type": "choice",
                "instructions": copy.deepcopy(dict(JEV_CONTEXT_INSTRUCTIONS)),
                "criteria": criteria,
            }
        },
    }


def build_jev_request_legacy(
    payload: Mapping[str, Any], *, model_name: str = JEV_MODEL
) -> dict[str, Any]:
    """Reproduce the original P choice request for explicitly labeled legacy use."""

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
                "instructions": copy.deepcopy(dict(_legacy_jev_rules())),
                "criteria": criteria,
            }
        },
    }


def _validate_common_input(payload: Mapping[str, Any]) -> set[str]:
    """Validate the exact production common-payload shape before hooks or transport."""

    if set(payload) != _TOP_LEVEL_KEYS:
        raise ValueError("invalid common input shape")
    if not isinstance(payload.get("question"), str) or not payload["question"]:
        raise ValueError("invalid common input shape")
    if not isinstance(payload.get("policyVersion"), str) or (
        payload.get("contextSummary") is not None
        and not isinstance(payload.get("contextSummary"), str)
    ):
        raise ValueError("invalid common input shape")
    candidates = payload.get("candidates")
    documents = payload.get("documentCandidates")
    if not isinstance(candidates, list) or not isinstance(documents, list):
        raise ValueError("invalid common input shape")
    candidate_count = payload.get("candidateCount")
    document_count = payload.get("documentCandidateCount")
    if (
        not isinstance(candidate_count, int)
        or isinstance(candidate_count, bool)
        or len(candidates) > JEV_MAX_CANDIDATES
        or candidate_count != len(candidates)
    ):
        raise ValueError("invalid candidate count")
    if (
        not isinstance(document_count, int)
        or isinstance(document_count, bool)
        or len(documents) > DOCUMENT_CANDIDATE_TOP_K
        or document_count != len(documents)
    ):
        raise ValueError("invalid document count")
    candidate_ids: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or set(candidate) != _CANDIDATE_KEYS:
            raise ValueError("invalid candidate shape")
        if (
            not isinstance(candidate.get("rank"), int)
            or isinstance(candidate.get("rank"), bool)
            or candidate.get("rank") != len(candidate_ids) + 1
        ):
            raise ValueError("invalid candidate shape")
        group = candidate.get("group")
        subproblem = candidate.get("subproblem")
        if not isinstance(group, Mapping) or set(group) - {"id", "name"}:
            raise ValueError("invalid candidate shape")
        if not isinstance(group.get("id"), str) or (
            "name" in group and not isinstance(group["name"], str)
        ):
            raise ValueError("invalid candidate shape")
        if not isinstance(subproblem, Mapping) or set(subproblem) - _SUBPROBLEM_KEYS:
            raise ValueError("invalid candidate shape")
        candidate_id = subproblem.get("id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("invalid candidate identifier")
        candidate_ids.append(candidate_id)
        if not isinstance(subproblem.get("name"), str):
            raise ValueError("invalid candidate shape")
        for key in ("inclusionCriteria", "exclusionCriteria"):
            rules = subproblem.get(key)
            if not isinstance(rules, Mapping) or any(
                not isinstance(label, str)
                or not re.fullmatch(
                    rf"{'I' if key == 'inclusionCriteria' else 'E'}[1-9][0-9]*",
                    label,
                )
                or not isinstance(text, str)
                for label, text in rules.items()
            ):
                raise ValueError("invalid candidate shape")
        canonical = subproblem.get("canonicalAnswer")
        if canonical is not None and (
            not isinstance(canonical, Mapping)
            or set(canonical) - _CANONICAL_KEYS
            or not isinstance(canonical.get("contentMarkdown"), str)
            or not isinstance(canonical.get("applicabilityRules", []), list)
            or any(not isinstance(rule, str) for rule in canonical.get("applicabilityRules", []))
        ):
            raise ValueError("invalid candidate shape")
    if any(
        not isinstance(item, Mapping) or set(item) != _DOCUMENT_KEYS
        for item in documents
    ):
        raise ValueError("invalid document candidate shape")
    document_ids = [doc["id"] for doc in documents]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("invalid document candidate identifiers")
    if any(
        not isinstance(doc.get("id"), str)
        or not isinstance(doc.get("title"), str)
        or not isinstance(doc.get("parentPath"), str)
        or not isinstance(doc.get("headings"), list)
        or any(not isinstance(heading, str) for heading in doc.get("headings", []))
        for doc in documents
    ):
        raise ValueError("invalid document candidate shape")
    if not 0 <= len(candidates) <= JEV_MAX_CANDIDATES:
        raise ValueError("invalid candidate count")
    if JEV_NONE_KEY in candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("invalid candidate identifiers")
    try:
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("common input is not JSON serializable") from None
    return set(candidate_ids)


def _request_evidence(payload: Mapping[str, Any], request: Mapping[str, Any]) -> ModelRequestEvidence:
    return ModelRequestEvidence(
        common_input_sha256=canonical_request_sha256(payload),
        request_sha256=canonical_request_sha256(request),
        input_transform_version=JEV_INPUT_TRANSFORM_VERSION,
        instructions_sha256=canonical_request_sha256(request["questions"][JEV_QUESTION_NAME]["instructions"]),
    )


def _safe_response_model(value: object) -> Optional[str]:
    return value if isinstance(value, str) and _SAFE_IDENTIFIER.fullmatch(value) else None


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
        input_mode: str = "context",
    ) -> None:
        if input_mode not in {"context", "p-legacy"}:
            raise ValueError("unsupported Jev input mode")
        self._transport = transport
        self._retry_delay = retry_delay
        self._input_mode = input_mode

    @property
    def provider(self) -> str:
        return JEV_PROVIDER

    @property
    def model_name(self) -> str:
        return JEV_MODEL

    @property
    def prompt_version(self) -> str:
        return (
            JEV_PROMPT_VERSION
            if self._input_mode == "context"
            else JEV_LEGACY_PROMPT_VERSION
        )

    async def judge(
        self,
        payload: Mapping[str, Any],
        *,
        before_model_call: Optional[BeforeModelCallHook] = None,
        stage: Optional[StageSettings] = None,
    ) -> JudgeCall:
        model_name = self.model_name if stage is None else stage.model_name
        prompt_version = self.prompt_version if stage is None else stage.prompt_version
        started = time.perf_counter()
        invalid_stage = stage is not None and (
            self._input_mode != "context"
            or stage.prompt_version != JEV_PROMPT_VERSION
            or stage.instructions != JEV_CONTEXT_PROMPT
            or bool(dict(stage.params))
            or not isinstance(stage.model_name, str)
            or not _SAFE_IDENTIFIER.fullmatch(stage.model_name)
        )
        if invalid_stage:
            return self._failed(
                started,
                0,
                JudgeFailure(
                    kind=JudgeFailureKind.INTERNAL_ERROR,
                    safe_message="Jev 단계 설정이 context prompt 계약과 일치하지 않습니다.",
                ),
                model_name,
                prompt_version,
            )
        try:
            request = (
                build_jev_request(payload, model_name=model_name)
                if self._input_mode == "context"
                else build_jev_request_legacy(payload, model_name=model_name)
            )
            evidence = (
                _request_evidence(request["state"]["classifierInput"], request)
                if self._input_mode == "context"
                else None
            )
        except (TypeError, ValueError, KeyError):
            return self._failed(
                started,
                0,
                JudgeFailure(
                    kind=JudgeFailureKind.INTERNAL_ERROR,
                    safe_message="Jev 입력 계약을 확인하지 못했습니다.",
                ),
                model_name,
                prompt_version,
            )
        presented = frozenset(
            key
            for key in request["questions"][JEV_QUESTION_NAME]["criteria"]
            if key != JEV_NONE_KEY
        )
        if before_model_call is not None:
            await before_model_call(self.provider, model_name, prompt_version)

        attempts = 1
        try:
            outcome = await self._transport(copy.deepcopy(request))
            if _transient(outcome) and attempts < JEV_MAX_ATTEMPTS:
                await asyncio.sleep(self._retry_delay)
                attempts += 1
                outcome = await self._transport(copy.deepcopy(request))
        except Exception as error:  # noqa: BLE001 - preserve judge failure contract
            failure = JudgeFailure(
                kind=JudgeFailureKind.INTERNAL_ERROR,
                safe_message=f"Jev 판별 호출 중 오류: {type(error).__name__}",
            )
            return self._failed(started, attempts, failure, model_name, prompt_version, evidence)

        if not outcome.get("ok"):
            status = outcome.get("httpStatus")
            safe_status = status if isinstance(status, int) else "unknown"
            failure = JudgeFailure(
                kind=JudgeFailureKind.API_ERROR,
                safe_message=f"Jev 호출 실패: HTTP {safe_status}",
            )
            return self._failed(started, attempts, failure, model_name, prompt_version, evidence)

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
            presented,
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
            request_evidence=(
                ModelRequestEvidence(
                    common_input_sha256=evidence.common_input_sha256,
                    request_sha256=evidence.request_sha256,
                    input_transform_version=evidence.input_transform_version,
                    instructions_sha256=evidence.instructions_sha256,
                    response_model_name=_safe_response_model(response.get("model")),
                )
                if evidence is not None
                else None
            ),
        )
        return JudgeCall(trace=trace, output_text=output_text)

    def _failed(
        self,
        started: float,
        attempts: int,
        failure: JudgeFailure,
        model_name: str,
        prompt_version: Optional[str],
        request_evidence: Optional[ModelRequestEvidence] = None,
    ) -> JudgeCall:
        return JudgeCall(
            trace=ModelCallTrace(
                provider=self.provider,
                model_name=model_name,
                succeeded=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
                retry_count=max(0, attempts - 1),
                prompt_version=prompt_version,
                error_message=failure.safe_message,
                request_evidence=request_evidence,
            ),
            failure=failure,
        )
