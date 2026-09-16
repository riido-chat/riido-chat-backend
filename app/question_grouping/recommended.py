"""추천·과거 질문의 정확 일치 키를 만드는 순수 함수."""

import re
import unicodedata


_WHITESPACE = re.compile(r"\s+")


def normalize_exact_question(question: str) -> str:
    """NFKC 정규화 뒤 양끝 공백을 제거하고 연속 공백을 하나로 줄인다.

    어미, 구두점, 대소문자와 문장 내부의 의미 있는 문자는 바꾸지 않는다.
    따라서 추천 질문 원문은 별도로 보존하고 이 값만 조회 유일 키로 사용한다.
    """

    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", question).strip())


# Existing seed callers keep the public name while the mapping is now shared with
# historical questions.
normalize_recommended_question = normalize_exact_question
