"""후속 질문의 문맥 선택과 독립 검색 질의 생성을 담당한다."""

import json
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence, Tuple

from openai import APIError, AsyncOpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from app import prompts
from app.core.config import get_settings
from app.core.model_trace import BeforeModelCallHook, ModelCallTrace
from app.core.openai_error import is_transient_openai_error
from app.core.openai_usage import cached_input_tokens as usage_cached_input_tokens
from app.core.openai_usage import reasoning_tokens as usage_reasoning_tokens
from app.core.stage_settings import StageSettings
from app.answering.models import FinalWithheldReason


OPENAI_QUERY_REWRITE_PROVIDER = "openai"
OPENAI_QUERY_REWRITE_MODEL = "gpt-5.4-mini"
QUERY_REWRITE_PROMPT_VERSION = "v10"
QUERY_REWRITE_TIMEOUT_SECONDS = 30.0
QUERY_REWRITE_MAX_OUTPUT_TOKENS = 512
MAX_QUERY_REWRITE_ATTEMPTS = 2
MAX_QUERY_REWRITE_TURNS = 5
MAX_QUERY_LENGTH = 4000
CONTEXT_SNAPSHOT_SCHEMA_VERSION = "v2"
UNRESOLVED_REFERENCE_REPLACEMENTS = {
    "해당 버튼": "버튼",
    "그 버튼": "버튼",
    "해당 메뉴": "메뉴",
    "그 메뉴": "메뉴",
    "해당 설정": "설정",
    "그 설정": "설정",
}
PRIOR_CONTEXT_REFERENCE_MARKERS = (
    "그것",
    "그거",
    "그건",
    "그중",
    "그 ",
    "해당",
    "앞서",
    "앞에서",
    "이렇게",
    "이 방식",
)

MODEL_OUTPUT_INVALID_ERROR_CODE = "MODEL_OUTPUT_INVALID"
UPSTREAM_ERROR_CODE = "UPSTREAM_ERROR"
INTERNAL_ERROR_CODE = "INTERNAL_ERROR"

# 지시문 본문은 app/prompts/rewrite/<version>.md 에 있다. 상수 이름은 기존 import 를 위한 별칭이다.
QUERY_REWRITE_PROMPT_V4 = prompts.get("rewrite", QUERY_REWRITE_PROMPT_VERSION)
# 프로필 버전 구성 행의 params 가 비었을 때 보내는 요청 설정.
QUERY_REWRITE_DEFAULT_PARAMS = {"max_output_tokens": QUERY_REWRITE_MAX_OUTPUT_TOKENS}
# 프로필 버전을 받지 않은 호출(평가 도구, 서비스 없는 조립)이 쓰는 코드 기본값.
CODE_QUERY_REWRITE_STAGE = StageSettings(
    model_name=OPENAI_QUERY_REWRITE_MODEL,
    prompt_version=QUERY_REWRITE_PROMPT_VERSION,
    instructions=QUERY_REWRITE_PROMPT_V4,
    params=QUERY_REWRITE_DEFAULT_PARAMS,
)


class QueryRewriteDecision(str, Enum):
    """현재 질문의 문맥 사용 여부와 해석 결과."""

    NEW_TOPIC = "NEW_TOPIC"
    FOLLOW_UP_RESOLVED = "FOLLOW_UP_RESOLVED"
    FOLLOW_UP_UNRESOLVED = "FOLLOW_UP_UNRESOLVED"


class QueryRewriteTurnStatus(str, Enum):
    """Query Rewrite 후보로 사용할 수 있는 이전 턴 상태."""

    COMPLETED = "COMPLETED"
    WITHHELD = "WITHHELD"


class QueryRewriteCandidateTurn(BaseModel):
    """모델 입력과 context snapshot이 공유하는 이전 턴 값."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    rag_run_id: uuid.UUID = Field(alias="ragRunId")
    turn_no: int = Field(alias="turnNo", ge=1)
    status: QueryRewriteTurnStatus
    user_query: str = Field(alias="userQuery", min_length=1, max_length=MAX_QUERY_LENGTH)
    resolved_query: Optional[str] = Field(
        default=None,
        alias="resolvedQuery",
        max_length=MAX_QUERY_LENGTH,
    )
    answer_content: Optional[str] = Field(alias="answerContent")
    withheld_reason_code: Optional[FinalWithheldReason] = Field(
        alias="withheldReasonCode"
    )

    @field_validator("user_query")
    @classmethod
    def validate_user_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("후보 턴의 userQuery는 비어 있을 수 없습니다.")
        return value

    @field_validator("resolved_query", mode="before")
    @classmethod
    def strip_resolved_query(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("resolved_query")
    @classmethod
    def reject_blank_resolved_query(cls, value: Optional[str]) -> Optional[str]:
        if value == "":
            raise ValueError("후보 턴의 resolvedQuery는 공백일 수 없습니다.")
        return value

    @model_validator(mode="after")
    def validate_status_fields(self) -> "QueryRewriteCandidateTurn":
        if self.status == QueryRewriteTurnStatus.COMPLETED:
            if self.answer_content is None or not self.answer_content.strip():
                raise ValueError("COMPLETED 후보에는 answerContent가 필요합니다.")
            if self.withheld_reason_code is not None:
                raise ValueError(
                    "COMPLETED 후보에는 withheldReasonCode를 사용할 수 없습니다."
                )
            return self

        if self.answer_content is not None:
            raise ValueError("WITHHELD 후보의 answerContent는 null이어야 합니다.")
        if self.withheld_reason_code is None:
            raise ValueError("WITHHELD 후보에는 withheldReasonCode가 필요합니다.")
        return self


class QueryRewriteOutput(BaseModel):
    """OpenAI Structured Output으로 전달받는 분류와 독립 질문."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    decision: QueryRewriteDecision
    selected_turn_no: Optional[StrictInt] = Field(alias="selectedTurnNo")
    context_phrase: Optional[str] = Field(
        alias="contextPhrase",
        max_length=MAX_QUERY_LENGTH,
    )
    resolved_query: Optional[str] = Field(
        alias="resolvedQuery",
        max_length=MAX_QUERY_LENGTH,
    )

    @field_validator("context_phrase", "resolved_query", mode="before")
    @classmethod
    def strip_resolved_query(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_decision_fields(self) -> "QueryRewriteOutput":
        if self.decision == QueryRewriteDecision.NEW_TOPIC:
            if (
                self.selected_turn_no is not None
                or self.context_phrase is not None
                or self.resolved_query is not None
            ):
                raise ValueError(
                    "NEW_TOPIC은 null selectedTurnNo, contextPhrase와 "
                    "resolvedQuery가 필요합니다."
                )
            return self

        if self.decision == QueryRewriteDecision.FOLLOW_UP_RESOLVED:
            if self.selected_turn_no is None or self.selected_turn_no < 1:
                raise ValueError(
                    "FOLLOW_UP_RESOLVED는 양의 selectedTurnNo가 필요합니다."
                )
            if self.context_phrase is None or not self.context_phrase:
                raise ValueError(
                    "FOLLOW_UP_RESOLVED는 공백이 아닌 contextPhrase가 필요합니다."
                )
            if self.resolved_query is None or not self.resolved_query:
                raise ValueError(
                    "FOLLOW_UP_RESOLVED는 공백이 아닌 resolvedQuery가 필요합니다."
                )
            return self

        if (
            self.selected_turn_no is not None
            or self.context_phrase is not None
            or self.resolved_query is not None
        ):
            raise ValueError(
                "FOLLOW_UP_UNRESOLVED는 null selectedTurnNo, contextPhrase와 "
                "resolvedQuery가 필요합니다."
            )
        return self


@dataclass(frozen=True)
class QueryResolution:
    """후보 턴까지 검증한 최종 Query Rewrite 결과."""

    decision: QueryRewriteDecision
    resolved_query: Optional[str]
    selected_turns: Tuple[QueryRewriteCandidateTurn, ...]

    @property
    def context_turn_count(self) -> int:
        return len(self.selected_turns)

    @property
    def should_retrieve(self) -> bool:
        return self.resolved_query is not None


@dataclass(frozen=True)
class QueryRewriteCall:
    """논리적 Query Rewrite 호출의 결과와 기록용 관측값."""

    trace: ModelCallTrace
    resolution: Optional[QueryResolution] = None
    fallback_reason: Optional[str] = None
    error_code: Optional[str] = None
    error: Optional[Exception] = None


class QueryRewriteOutputInvalidError(ValueError):
    """구조화 출력 또는 후보 선택 계약을 위반했을 때 발생한다."""


class QueryRewriteUpstreamError(RuntimeError):
    """Responses API가 완료되지 않은 상태로 끝났을 때 발생한다."""


def _normalize_user_query(user_query: str) -> str:
    if not isinstance(user_query, str):
        raise TypeError("user_query는 문자열이어야 합니다.")

    normalized = user_query.strip()
    if not normalized:
        raise ValueError("user_query는 비어 있을 수 없습니다.")
    if len(normalized) > MAX_QUERY_LENGTH:
        raise ValueError(f"user_query는 최대 {MAX_QUERY_LENGTH}자까지 허용합니다.")
    return normalized


def _ordered_candidates(
    candidates: Sequence[QueryRewriteCandidateTurn],
) -> Tuple[QueryRewriteCandidateTurn, ...]:
    candidate_turns = tuple(candidates)
    if len(candidate_turns) > MAX_QUERY_REWRITE_TURNS:
        raise ValueError(
            f"Query Rewrite 후보는 최대 {MAX_QUERY_REWRITE_TURNS}턴까지 허용합니다."
        )

    turn_nos = [candidate.turn_no for candidate in candidate_turns]
    if len(turn_nos) != len(set(turn_nos)):
        raise ValueError("Query Rewrite 후보 turn_no는 중복될 수 없습니다.")

    rag_run_ids = [candidate.rag_run_id for candidate in candidate_turns]
    if len(rag_run_ids) != len(set(rag_run_ids)):
        raise ValueError("Query Rewrite 후보 rag_run_id는 중복될 수 없습니다.")

    return tuple(sorted(candidate_turns, key=lambda candidate: candidate.turn_no))


def _candidate_input(candidate: QueryRewriteCandidateTurn) -> dict:
    value = {
        "turnNo": candidate.turn_no,
        "status": candidate.status.value,
        "userQuery": candidate.user_query,
        "resolvedQuery": candidate.resolved_query,
    }
    if candidate.status == QueryRewriteTurnStatus.COMPLETED:
        value["answerContent"] = candidate.answer_content
    else:
        value["withheldReasonCode"] = candidate.withheld_reason_code.value
    return value


def _prepare_query_rewrite_input(
    user_query: str,
    candidates: Sequence[QueryRewriteCandidateTurn],
) -> Tuple[str, Tuple[QueryRewriteCandidateTurn, ...], str]:
    normalized_query = _normalize_user_query(user_query)
    ordered_candidates = _ordered_candidates(candidates)
    payload = {
        "currentUserQuery": normalized_query,
        "candidateTurns": [
            _candidate_input(candidate) for candidate in ordered_candidates
        ],
    }
    return (
        normalized_query,
        ordered_candidates,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )


def build_query_rewrite_input(
    user_query: str,
    candidates: Sequence[QueryRewriteCandidateTurn],
) -> str:
    """현재 질문과 최근 5턴 및 가장 최근 턴을 JSON 데이터로 직렬화한다."""

    return _prepare_query_rewrite_input(user_query, candidates)[2]


def _validated_output(value: object) -> QueryRewriteOutput:
    if value is None:
        raise QueryRewriteOutputInvalidError(
            "OpenAI Query Rewrite 응답에 Structured Output이 없습니다."
        )

    raw_value = (
        value.model_dump(mode="json", by_alias=True)
        if isinstance(value, QueryRewriteOutput)
        else value
    )
    try:
        return QueryRewriteOutput.model_validate(raw_value)
    except ValidationError as error:
        raise QueryRewriteOutputInvalidError(
            "OpenAI Query Rewrite 응답이 4필드 계약에 맞지 않습니다."
        ) from error


def _normalize_reference_phrases(value: str) -> str:
    normalized = value
    for phrase, replacement in UNRESOLVED_REFERENCE_REPLACEMENTS.items():
        normalized = normalized.replace(phrase, replacement)
    return normalized


def _preserve_previous_scope_for_ui_reference(
    normalized_query: str,
    previous_query: str,
    resolved_query: str,
) -> str:
    if not any(
        phrase in normalized_query
        for phrase in UNRESOLVED_REFERENCE_REPLACEMENTS
    ):
        return resolved_query

    normalized_current = _normalize_reference_phrases(normalized_query)
    if previous_query in resolved_query:
        return resolved_query

    merged_query = f"{previous_query} {normalized_current}"
    if len(merged_query) <= MAX_QUERY_LENGTH:
        return merged_query
    return resolved_query


def _resolve_query_rewrite_output(
    normalized_query: str,
    candidates: Tuple[QueryRewriteCandidateTurn, ...],
    output: object,
) -> QueryResolution:
    parsed = _validated_output(output)
    decision = parsed.decision
    resolved_query = parsed.resolved_query
    if (
        parsed.decision == QueryRewriteDecision.FOLLOW_UP_RESOLVED
        and not candidates
    ):
        raise QueryRewriteOutputInvalidError(
            "FOLLOW_UP_RESOLVED에는 바로 이전의 유효한 턴이 필요합니다."
        )

    if parsed.decision == QueryRewriteDecision.FOLLOW_UP_RESOLVED:
        candidate_by_turn_no = {
            candidate.turn_no: candidate for candidate in candidates
        }
        selected_turn = candidate_by_turn_no.get(parsed.selected_turn_no)
        if selected_turn is None:
            raise QueryRewriteOutputInvalidError(
                "selectedTurnNo가 최근 후보 턴에 포함되지 않습니다."
            )
        previous_query = selected_turn.resolved_query or selected_turn.user_query
        context_phrase = parsed.context_phrase or ""
        answer_content = selected_turn.answer_content or ""
        if (
            context_phrase not in previous_query
            and context_phrase not in answer_content
        ):
            raise QueryRewriteOutputInvalidError(
                "contextPhrase가 선택한 턴의 질의나 답변에 그대로 존재하지 않습니다."
            )
        if context_phrase not in (parsed.resolved_query or ""):
            raise QueryRewriteOutputInvalidError(
                "resolvedQuery가 contextPhrase를 그대로 보존하지 않았습니다."
            )
        resolved_query = _normalize_reference_phrases(resolved_query or "")
        resolved_query = _preserve_previous_scope_for_ui_reference(
            normalized_query,
            previous_query,
            resolved_query,
        )

        if (
            context_phrase in normalized_query
            and not any(
                marker in normalized_query
                for marker in PRIOR_CONTEXT_REFERENCE_MARKERS
            )
        ):
            decision = QueryRewriteDecision.NEW_TOPIC
            resolved_query = normalized_query

    selected_turns = (
        tuple(
            candidate
            for candidate in candidates
            if candidate.turn_no == parsed.selected_turn_no
        )
        if decision == QueryRewriteDecision.FOLLOW_UP_RESOLVED
        else ()
    )

    if decision == QueryRewriteDecision.NEW_TOPIC:
        resolved_query = normalized_query

    return QueryResolution(
        decision=decision,
        resolved_query=resolved_query,
        selected_turns=selected_turns,
    )


def resolve_query_rewrite_output(
    user_query: str,
    candidates: Sequence[QueryRewriteCandidateTurn],
    output: object,
) -> QueryResolution:
    """구조화 응답을 후보와 대조하고 선택 턴을 시간순으로 확정한다."""

    normalized_query = _normalize_user_query(user_query)
    ordered_candidates = _ordered_candidates(candidates)
    return _resolve_query_rewrite_output(
        normalized_query,
        ordered_candidates,
        output,
    )


def build_context_snapshot(
    resolution: QueryResolution,
) -> Optional[dict[str, Any]]:
    """실제로 선택된 이전 턴만 감사 가능한 v2 snapshot으로 만든다."""

    if not resolution.selected_turns:
        return None

    return {
        "schemaVersion": CONTEXT_SNAPSHOT_SCHEMA_VERSION,
        "selectedTurns": [
            turn.model_dump(mode="json", by_alias=True)
            for turn in resolution.selected_turns
        ],
    }


def _query_rewrite_trace(
    started: float,
    *,
    stage: StageSettings,
    retry_count: int,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cached_input_tokens: Optional[int] = None,
    reasoning_tokens: Optional[int] = None,
    error: Optional[Exception] = None,
) -> ModelCallTrace:
    return ModelCallTrace(
        provider=OPENAI_QUERY_REWRITE_PROVIDER,
        model_name=stage.model_name,
        succeeded=error is None,
        latency_ms=int((time.perf_counter() - started) * 1000),
        retry_count=retry_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        reasoning_tokens=reasoning_tokens,
        prompt_version=stage.prompt_version,
        error_message=None if error is None else str(error),
    )


def _add_token_count(total: Optional[int], value: Optional[int]) -> Optional[int]:
    if value is None:
        return total
    return (total or 0) + value


def _validate_response_status(response: object) -> None:
    status = getattr(response, "status", None)
    if status == "completed":
        return

    incomplete_details = getattr(response, "incomplete_details", None)
    incomplete_reason = getattr(incomplete_details, "reason", None)
    if status == "incomplete" and incomplete_reason == "max_output_tokens":
        raise QueryRewriteOutputInvalidError(
            "OpenAI Query Rewrite 응답이 출력 한도 안에서 완료되지 않았습니다."
        )

    raise QueryRewriteUpstreamError(
        "OpenAI Query Rewrite 응답이 완료되지 않았습니다: "
        f"status={status}, reason={incomplete_reason}"
    )


def _to_output_invalid_error(
    error: Exception,
) -> Optional[QueryRewriteOutputInvalidError]:
    if isinstance(error, QueryRewriteOutputInvalidError):
        return error
    if isinstance(error, ValidationError):
        return QueryRewriteOutputInvalidError(
            "OpenAI Query Rewrite 응답이 구조화 출력 계약에 맞지 않습니다."
        )
    return None


class QueryRewriteService:
    """OpenAI Responses API로 후속 질문을 분류하고 검색 질의를 확정한다."""

    def __init__(self, client: Optional[AsyncOpenAI] = None) -> None:
        if client is None:
            api_key = get_settings().openai_api_key
            if not api_key:
                raise ValueError("OPENAI_API_KEY 환경변수가 필요합니다.")
            client = AsyncOpenAI(
                api_key=api_key,
                max_retries=0,
                timeout=QUERY_REWRITE_TIMEOUT_SECONDS,
            )

        self._client = client

    @property
    def model_name(self) -> str:
        """실제 query rewrite 호출에 사용하는 모델 이름."""

        return OPENAI_QUERY_REWRITE_MODEL

    @property
    def prompt_version(self) -> str:
        """model_calls에 기록하는 query rewrite 프롬프트 버전."""

        return QUERY_REWRITE_PROMPT_VERSION

    async def rewrite(
        self,
        user_query: str,
        candidates: Sequence[QueryRewriteCandidateTurn],
        *,
        before_model_call: Optional[BeforeModelCallHook] = None,
        stage: Optional[StageSettings] = None,
    ) -> QueryRewriteCall:
        """논리적 호출 한 건으로 최대 두 번 시도해 검색 질의를 확정한다.

        stage 는 턴의 프로필 버전이 고른 모델, 프롬프트 버전, 요청 설정이다. 없으면 코드 기본값이다.
        """

        stage = CODE_QUERY_REWRITE_STAGE if stage is None else stage
        normalized_query, ordered_candidates, query_input = (
            _prepare_query_rewrite_input(user_query, candidates)
        )
        if before_model_call is not None:
            await before_model_call(
                OPENAI_QUERY_REWRITE_PROVIDER,
                stage.model_name,
                stage.prompt_version,
            )

        started = time.perf_counter()
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None
        cached_input_tokens: Optional[int] = None
        reasoning_tokens: Optional[int] = None
        for attempt in range(MAX_QUERY_REWRITE_ATTEMPTS):
            try:
                response = await self._client.responses.parse(
                    model=stage.model_name,
                    instructions=stage.instructions,
                    input=query_input,
                    text_format=QueryRewriteOutput,
                    **stage.request_params(),
                )
                usage = getattr(response, "usage", None)
                input_tokens = _add_token_count(
                    input_tokens,
                    getattr(usage, "input_tokens", None),
                )
                output_tokens = _add_token_count(
                    output_tokens,
                    getattr(usage, "output_tokens", None),
                )
                cached_input_tokens = _add_token_count(
                    cached_input_tokens,
                    usage_cached_input_tokens(usage),
                )
                reasoning_tokens = _add_token_count(
                    reasoning_tokens,
                    usage_reasoning_tokens(usage),
                )
                _validate_response_status(response)
                resolution = _resolve_query_rewrite_output(
                    normalized_query,
                    ordered_candidates,
                    response.output_parsed,
                )
                return QueryRewriteCall(
                    trace=_query_rewrite_trace(
                        started,
                        stage=stage,
                        retry_count=attempt,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cached_input_tokens=cached_input_tokens,
                        reasoning_tokens=reasoning_tokens,
                    ),
                    resolution=resolution,
                )
            except Exception as error:
                output_error = _to_output_invalid_error(error)
                if output_error is not None:
                    if attempt == 0:
                        continue
                    return QueryRewriteCall(
                        trace=_query_rewrite_trace(
                            started,
                            stage=stage,
                            retry_count=attempt,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cached_input_tokens=cached_input_tokens,
                            reasoning_tokens=reasoning_tokens,
                            error=output_error,
                        ),
                        resolution=QueryResolution(
                            decision=QueryRewriteDecision.FOLLOW_UP_UNRESOLVED,
                            resolved_query=None,
                            selected_turns=(),
                        ),
                        fallback_reason=str(output_error),
                    )

                if isinstance(error, APIError):
                    if is_transient_openai_error(error):
                        if attempt == 0:
                            continue
                        return QueryRewriteCall(
                            trace=_query_rewrite_trace(
                                started,
                                stage=stage,
                                retry_count=attempt,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                cached_input_tokens=cached_input_tokens,
                                reasoning_tokens=reasoning_tokens,
                                error=error,
                            ),
                            error_code=UPSTREAM_ERROR_CODE,
                            error=error,
                        )
                    return QueryRewriteCall(
                        trace=_query_rewrite_trace(
                            started,
                            stage=stage,
                            retry_count=attempt,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cached_input_tokens=cached_input_tokens,
                            reasoning_tokens=reasoning_tokens,
                            error=error,
                        ),
                        error_code=INTERNAL_ERROR_CODE,
                        error=error,
                    )

                if isinstance(error, QueryRewriteUpstreamError):
                    return QueryRewriteCall(
                        trace=_query_rewrite_trace(
                            started,
                            stage=stage,
                            retry_count=attempt,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cached_input_tokens=cached_input_tokens,
                            reasoning_tokens=reasoning_tokens,
                            error=error,
                        ),
                        error_code=UPSTREAM_ERROR_CODE,
                        error=error,
                    )
                return QueryRewriteCall(
                    trace=_query_rewrite_trace(
                        started,
                        stage=stage,
                        retry_count=attempt,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cached_input_tokens=cached_input_tokens,
                        reasoning_tokens=reasoning_tokens,
                        error=error,
                    ),
                    error_code=INTERNAL_ERROR_CODE,
                    error=error,
                )

        raise RuntimeError("OpenAI Query Rewrite 호출이 완료되지 않았습니다.")
