"""모델 호출 한 건의 관측값을 담는다.

model_calls 기록에 필요한 값만 담고 DB Enum(ExecutionStatus)에는 의존하지 않는다.
retrieval과 generation 계층이 이 객체를 만들어 올리고, 저장 계층 매핑은
ChatService가 담당한다.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Optional


BeforeModelCallHook = Callable[[str, str, Optional[str]], Awaitable[None]]
"""외부 모델 호출 직전에 provider, model, prompt version을 전달하는 hook."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def canonical_request_sha256(value: object) -> str:
    """Hash a logical JSON value using the shared stable encoding."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ModelRequestEvidence:
    """Hashes and safe identifiers describing one logical model request."""

    common_input_sha256: str
    request_sha256: str
    input_transform_version: str
    instructions_sha256: str
    response_model_name: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in ("common_input_sha256", "request_sha256", "instructions_sha256"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if not isinstance(self.input_transform_version, str) or not _SAFE_IDENTIFIER.fullmatch(
            self.input_transform_version
        ):
            raise ValueError("input_transform_version must be a safe identifier")
        if self.response_model_name is not None and (
            not isinstance(self.response_model_name, str)
            or not _SAFE_IDENTIFIER.fullmatch(self.response_model_name)
        ):
            raise ValueError("response_model_name must be a safe identifier or None")


@dataclass(frozen=True)
class ModelCallTrace:
    """외부 모델 호출 한 건의 결과와 비용 관측값.

    재시도는 논리적 호출 1건으로 보므로 retry_count에 재시도 횟수를 담고
    latency_ms는 재시도를 포함한 총 소요다.

    input_tokens와 output_tokens는 총량이다. cached_input_tokens는 입력 중
    프롬프트 캐시에서 온 몫, reasoning_tokens는 출력 중 숨은 추론에 쓴 몫으로
    각 총량에 포함된 부분값이다. 제공자가 내역을 주지 않으면 None 으로 둔다.
    다른 제공자를 붙일 때도 총량에 캐시와 추론 몫이 포함되도록 맞춰 넣는다.
    """

    provider: str
    model_name: str
    succeeded: bool
    latency_ms: int
    retry_count: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    prompt_version: Optional[str] = None
    error_message: Optional[str] = None
    request_evidence: Optional[ModelRequestEvidence] = None
