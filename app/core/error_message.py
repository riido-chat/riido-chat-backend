"""실행 이력에 남기는 오류 메시지를 안전한 형태로 다듬는다."""

import re


API_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{8,}")
MAX_ERROR_MESSAGE_LENGTH = 4000


def sanitize_error_message(error: Exception) -> str:
    """API key를 가리고 저장 가능한 길이로 자른다.

    라이브러리가 오류 메시지에 인증 헤더를 그대로 담는 경우가 있다.
    """

    message = API_KEY_PATTERN.sub("sk-***REDACTED***", str(error))
    return message[:MAX_ERROR_MESSAGE_LENGTH]
