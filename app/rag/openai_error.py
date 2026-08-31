"""OpenAI 호출 오류의 일시적 실패 여부를 판단한다."""

from openai import APIConnectionError, APIStatusError


def is_transient_openai_error(error: Exception) -> bool:
    """연결 문제, timeout, rate limit과 서버 오류만 일시적 실패로 본다."""

    if isinstance(error, APIConnectionError):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in (408, 409, 429) or error.status_code >= 500
    return False
