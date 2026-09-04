"""여러 도메인이 공유하는 내용 해시 계산."""

import hashlib


def sha256_hex(value: str) -> str:
    """문자열을 UTF-8로 인코딩해 SHA-256 hex digest를 만든다."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
