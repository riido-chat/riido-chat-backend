"""정본 본문과 인용 행이 서로 맞는지 검사한다.

정본 본문은 [1], [2] 번호로 인용을 가리킨다. FE 가 번호와 출처를 잇도록 본문 번호
집합과 인용 행 citation_order 집합이 같아야 하고, 인용은 하나 이상이어야 한다.
코드 블록과 인라인 코드 안의 [n] 은 인용이 아니다. 시드 스크립트가 저장 전에 쓴다.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import FrozenSet, Optional, Sequence, Tuple

from app.answering.service import (
    INTERNAL_SOURCE_REFERENCE_PATTERN,
    UnverifiableAnswerError,
    strip_code_regions,
    validate_answer_content,
)


# 인라인 링크 [1](...) 는 인용 번호로 보지 않는다. 붙어 있는 [1][2] 는 인용 두 개다.
CITATION_NUMBER_PATTERN = re.compile(r"\[([1-9][0-9]*)\](?!\()")

ERROR_NO_CITATIONS = "NO_CITATIONS"
ERROR_NO_CITATION_MARKERS = "NO_CITATION_MARKERS"
ERROR_DUPLICATE_CITATION_ORDER = "DUPLICATE_CITATION_ORDER"
ERROR_MARKER_WITHOUT_CITATION = "MARKER_WITHOUT_CITATION"
ERROR_CITATION_NOT_REFERENCED = "CITATION_NOT_REFERENCED"

# Known script contamination observed in Korean canonical answers. This deliberately
# targets only Bengali and Georgian letters instead of imposing a Korean-only
# character allowlist. The Unicode names identify these scripts without rejecting
# ordinary Hangul, Latin extensions, symbols, or emoji.
_SUSPECT_SCRIPT_NAMES = ("BENGALI", "GEORGIAN")
_SUSPECT_SCRIPT_LABELS = {"BENGALI": "벵골", "GEORGIAN": "조지아"}


def _mixed_suspect_scripts(content_markdown: str) -> Tuple[str, ...]:
    """Return suspect script names in a Korean-context body, excluding code."""

    text = strip_code_regions(content_markdown)
    hangul_count = 0
    script_counts = {script: 0 for script in _SUSPECT_SCRIPT_NAMES}
    for char in text:
        name = unicodedata.name(char, "")
        if "HANGUL" in name and unicodedata.category(char).startswith("L"):
            hangul_count += 1
        elif unicodedata.category(char).startswith("L"):
            for script in _SUSPECT_SCRIPT_NAMES:
                if script in name:
                    script_counts[script] += 1

    # Any Hangul outside code marks Korean context; bodies with no Hangul remain
    # unaffected. This narrow check can still flag a legitimate Bengali or
    # Georgian quotation inside Korean prose.
    if not hangul_count:
        return ()
    return tuple(script for script, count in script_counts.items() if count)


@dataclass(frozen=True)
class CanonicalCitationCheck:
    """정본 본문과 인용 번호 대조 결과. errors 가 비면 통과다."""

    body_numbers: FrozenSet[int]
    citation_orders: FrozenSet[int]
    errors: Tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def extract_citation_numbers(content_markdown: str) -> FrozenSet[int]:
    """코드 영역 밖 본문의 [n] 인용 번호 집합."""

    text = strip_code_regions(content_markdown)
    return frozenset(
        int(match.group(1)) for match in CITATION_NUMBER_PATTERN.finditer(text)
    )


def check_canonical_citations(
    content_markdown: str,
    citation_orders: Sequence[int],
) -> CanonicalCitationCheck:
    """본문 번호 집합 = 인용 번호 집합, 인용 1개 이상, 인용 번호 중복 없음."""

    body_numbers = extract_citation_numbers(content_markdown)
    orders = frozenset(citation_orders)
    errors = []
    if not citation_orders:
        errors.append(ERROR_NO_CITATIONS)
    if len(orders) != len(citation_orders):
        errors.append(ERROR_DUPLICATE_CITATION_ORDER)
    if not body_numbers:
        errors.append(ERROR_NO_CITATION_MARKERS)
    if body_numbers - orders:
        errors.append(ERROR_MARKER_WITHOUT_CITATION)
    if orders - body_numbers:
        errors.append(ERROR_CITATION_NOT_REFERENCED)
    return CanonicalCitationCheck(
        body_numbers=body_numbers,
        citation_orders=orders,
        errors=tuple(errors),
    )


def check_canonical_content(content_markdown: str) -> Optional[str]:
    """정본 본문이 생성 답변과 같은 본문 규칙을 지키는지 본다. 위반 사유, 통과면 None.

    정본은 SERVED 턴에서 답변 본문으로 그대로 나가므로 링크, HTML, 내부 SOURCE 식별자를
    생성 답변처럼 막는다. 본문의 [n] 인용 번호는 생성 경로의 [SOURCE_n] 표시와 같은 자리라
    검사 전에 표시로 바꿔 지운다(붙은 [1][2] 를 참조형 링크로 오인하지 않게).
    """

    if not content_markdown or not content_markdown.strip():
        return "정본 본문이 비어 있습니다."
    if INTERNAL_SOURCE_REFERENCE_PATTERN.search(content_markdown):
        return "정본 본문에 내부 Source 식별자가 있습니다."
    as_markers = CITATION_NUMBER_PATTERN.sub(
        lambda match: f"[SOURCE_{match.group(1)}]", content_markdown
    )
    try:
        validate_answer_content(as_markers)
    except UnverifiableAnswerError as error:
        return str(error)
    mixed_scripts = _mixed_suspect_scripts(content_markdown)
    if mixed_scripts:
        labels = ", ".join(_SUSPECT_SCRIPT_LABELS[script] for script in mixed_scripts)
        return f"정본 본문에 한국어 맥락의 {labels} 문자가 포함되어 있습니다."
    return None
