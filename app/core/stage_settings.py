"""모델 호출 한 단계의 설정(모델, 프롬프트 버전, 지시문, 요청 설정).

프로필 버전의 구성 행이 단계마다 이 값을 정하고(app/profiles/stage_settings.py), 재작성, 판별,
생성 클라이언트는 호출마다 받은 값으로 요청을 만든다. 값을 받지 않은 호출은 각 클라이언트의
코드 기본값을 쓴다. 이 module 은 app 의 다른 패키지를 import 하지 않는다.
"""

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class StageSettings:
    """한 단계의 모델 호출 설정.

    instructions 는 모델에 실제로 보내는 지시문 전체다(재시도 단계는 처음 지시문과 재시도 덧붙임을
    이은 문자열). params 는 요청에 그대로 싣는 결과 영향 설정(reasoning, max_output_tokens)이다.
    """

    model_name: str
    prompt_version: str
    instructions: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 호출마다 같은 값을 보내도록 받은 값을 복사해 읽기 전용으로 둔다.
        object.__setattr__(
            self, "params", MappingProxyType(copy.deepcopy(dict(self.params)))
        )

    def request_params(self) -> Dict[str, Any]:
        """요청 인자로 펼칠 설정의 복사본. 호출부가 고쳐도 이 값은 바뀌지 않는다."""

        return copy.deepcopy(dict(self.params))
