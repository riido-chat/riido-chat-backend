"""질문 로그 정확 일치 키를 만드는 순수 함수."""

import hashlib
import re
import unicodedata
import uuid
from typing import Any, Mapping, Optional, Tuple


_WHITESPACE = re.compile(r"\s+")
JUDGMENT_INPUT_SUBPROBLEM_CANDIDATES_FIELD = "subproblemCandidates"


def normalize_exact_question(question: str) -> str:
    """NFKC 정규화 뒤 양끝 공백을 제거하고 연속 공백을 하나로 줄인다.

    어미, 구두점, 대소문자와 문장 내부의 의미 있는 문자는 바꾸지 않는다.
    """

    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", question).strip())


def exact_question_hash(question: str) -> str:
    """정규화한 질문의 sha256 hex. ``rag_runs.query_hash`` 조회 키로 쓴다."""

    return hashlib.sha256(normalize_exact_question(question).encode("utf-8")).hexdigest()


def presented_canonical_answer(
    presented_items: Any,
    subproblem_id: uuid.UUID,
) -> Tuple[bool, Optional[uuid.UUID]]:
    """Return the recorded canonical answer shown for a subproblem, if known."""

    if not isinstance(presented_items, list):
        return False, None
    target = str(subproblem_id)
    for item in presented_items:
        if not isinstance(item, Mapping) or item.get("subproblemId") != target:
            continue
        if "canonicalAnswerId" not in item:
            return False, None
        value = item["canonicalAnswerId"]
        if value is None:
            return True, None
        try:
            return True, uuid.UUID(str(value))
        except ValueError:
            return False, None
    return False, None
