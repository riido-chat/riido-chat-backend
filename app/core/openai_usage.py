"""OpenAI Responses API 사용량에서 캐시 입력과 추론 출력 몫을 꺼낸다."""

from typing import Optional


def cached_input_tokens(usage: object) -> Optional[int]:
    """입력 토큰 중 프롬프트 캐시에서 온 몫. 내역이 없으면 None 이다."""

    details = getattr(usage, "input_tokens_details", None)
    return getattr(details, "cached_tokens", None)


def reasoning_tokens(usage: object) -> Optional[int]:
    """출력 토큰 중 숨은 추론에 쓴 몫. 내역이 없으면 None 이다."""

    details = getattr(usage, "output_tokens_details", None)
    return getattr(details, "reasoning_tokens", None)
