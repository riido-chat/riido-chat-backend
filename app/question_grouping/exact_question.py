"""첫 턴 질문 로그 정확 일치 키를 만드는 순수 함수."""

import hashlib
import re
import unicodedata


_WHITESPACE = re.compile(r"\s+")


def normalize_exact_question(question: str) -> str:
    """NFKC 정규화 뒤 양끝 공백을 제거하고 연속 공백을 하나로 줄인다.

    어미, 구두점, 대소문자와 문장 내부의 의미 있는 문자는 바꾸지 않는다.
    """

    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", question).strip())


def exact_question_hash(question: str) -> str:
    """정규화한 질문의 sha256 hex. ``rag_runs.query_hash`` 조회 키로 쓴다."""

    return hashlib.sha256(normalize_exact_question(question).encode("utf-8")).hexdigest()
