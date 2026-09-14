"""모델 호출 한 건의 관측값을 담는다.

model_calls 기록에 필요한 값만 담고 DB Enum(ExecutionStatus)에는 의존하지 않는다.
retrieval과 generation 계층이 이 객체를 만들어 올리고, 저장 계층 매핑은
ChatService가 담당한다.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Optional


BeforeModelCallHook = Callable[[str, str, Optional[str]], Awaitable[None]]
"""외부 모델 호출 직전에 provider, model, prompt version을 전달하는 hook."""


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
