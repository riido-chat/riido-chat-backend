"""정본 캐시 게이트. 모델을 부르지 않는 순수 판단이다.

outcome 우선순위(R16, 먼저 걸리는 것을 기록한다):

1. 판별 실패 → FAILED
2. CONNECT 아님 → REJECTED[CLASSIFICATION_NOT_CONNECTED]
3. 세부 문제·정본·버전 검사 실패 → REJECTED[사유들]
4. 세부 문제 SHADOW → SHADOW
5. 세부 문제 UNUSED/STOPPED → REJECTED[SUBPROBLEM_UNUSED | SUBPROBLEM_STOPPED]
6. 프로필 semantic_cache_enabled 꺼짐 → GROUP_DISABLED
7. SERVED

3단계는 한 단계 안의 실패 사유를 모두 모은다. 세부 문제가 APPROVED 가 아니거나
정본이 없으면 그 사유 하나로 끝낸다. 정본 유효 기간(valid_from/to)은 보지 않는다(R17).
입력은 게이트 직전에 다시 읽은 값이며 조회는 호출자가 한다.
"""

import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from app.database.models import (
    CacheAttemptOutcome,
    ClassificationDecision,
    QuestionSubproblemServingState,
    QuestionSubproblemStatus,
)
from app.question_grouping.constants import (
    REJECT_CANONICAL_ANSWER_NOT_FOUND,
    REJECT_CANONICAL_CITATION_MISSING,
    REJECT_CITED_DOCUMENT_NOT_INDEXED,
    REJECT_CITED_SECTION_CHANGED,
    REJECT_CLASSIFICATION_NOT_CONNECTED,
    REJECT_SUBPROBLEM_NOT_APPROVED,
    REJECT_SUBPROBLEM_STOPPED,
    REJECT_SUBPROBLEM_UNUSED,
    REJECT_SUBPROBLEM_VERSION_MISMATCH,
)
from app.question_grouping.models import (
    CanonicalCitationSnapshot,
    CitationIndexContext,
    CitationResolution,
    GateCanonicalAnswer,
    GateResult,
    GateSubproblemState,
    IndexedSection,
    TurnJudgment,
)


# ---------------------------------------------------------------------------
# R17 인용 절 변경 검사
# ---------------------------------------------------------------------------


def resolve_citation(
    citation: CanonicalCitationSnapshot,
    context: CitationIndexContext,
) -> CitationResolution:
    """정본 인용 한 건을 이번 턴 색인의 절에 대응시킨다.

    1. 인용 문서 판이 턴 색인에 있고 인용 청크가 턴 색인의 청킹 설정 절이면 그대로 쓴다.
       문서 판은 같아도 청킹 설정이 바뀌어 청크가 절 목록에 없으면 2단계로 간다.
    2. 턴 색인에 든 같은 문서의 판에서 신원 해시와 내용 해시가 모두 같은 절.
    3. 내용 해시만 같은 절. 여럿이면 옛 node_order 와 가장 가까운 것, 같으면 앞 절.
    4. 실패. 문서가 턴 색인에 없으면 CITED_DOCUMENT_NOT_INDEXED, 아니면 CITED_SECTION_CHANGED.
    """

    if context.indexed_document_version_id is None:
        return CitationResolution(
            citation_order=citation.citation_order,
            rejection_reason=REJECT_CITED_DOCUMENT_NOT_INDEXED,
        )

    sections = tuple(
        section
        for section in context.sections
        if section.document_version_id == context.indexed_document_version_id
    )
    if context.indexed_document_version_id == citation.document_version_id:
        for section in sections:
            if section.chunk_id == citation.chunk_id:
                return _resolved(citation, section, step=1)

    if citation.node_identity_hash is not None:
        for section in sections:
            if (
                section.node_identity_hash == citation.node_identity_hash
                and section.content_hash == citation.content_hash
            ):
                return _resolved(citation, section, step=2)

    same_content = [
        section for section in sections if section.content_hash == citation.content_hash
    ]
    if same_content:
        closest = min(
            same_content,
            key=lambda section: (
                abs(section.node_order - citation.node_order),
                section.node_order,
            ),
        )
        return _resolved(citation, closest, step=3)

    return CitationResolution(
        citation_order=citation.citation_order,
        rejection_reason=REJECT_CITED_SECTION_CHANGED,
    )


def _resolved(
    citation: CanonicalCitationSnapshot,
    section: IndexedSection,
    *,
    step: int,
) -> CitationResolution:
    return CitationResolution(
        citation_order=citation.citation_order,
        section=section,
        step=step,
    )


def resolve_citations(
    citations: Sequence[CanonicalCitationSnapshot],
    contexts_by_source_id: Mapping[int, CitationIndexContext],
) -> Tuple[CitationResolution, ...]:
    """정본 인용 전체를 citation_order 순으로 해석한다. 문서 맥락이 없으면 색인 밖이다."""

    empty = CitationIndexContext(indexed_document_version_id=None)
    return tuple(
        resolve_citation(
            citation,
            contexts_by_source_id.get(citation.document_source_id, empty),
        )
        for citation in sorted(citations, key=lambda item: item.citation_order)
    )


# ---------------------------------------------------------------------------
# R16 게이트
# ---------------------------------------------------------------------------


def _dedupe(values: Iterable[str]) -> Tuple[str, ...]:
    seen: List[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return tuple(seen)


def _rejected(*reasons: str) -> GateResult:
    return GateResult(
        outcome=CacheAttemptOutcome.REJECTED,
        rejection_reasons=_dedupe(reasons),
    )


def evaluate_cache_gate(
    judgment: TurnJudgment,
    *,
    subproblem: Optional[GateSubproblemState],
    canonical_answer: Optional[GateCanonicalAnswer],
    citation_resolutions: Sequence[CitationResolution],
    semantic_cache_enabled: bool,
) -> GateResult:
    """판별 결과와 다시 읽은 세부 문제, 정본, 인용 해석으로 캐시 시도 결과를 정한다.

    CONNECT 가 아니면 subproblem 이하 입력은 보지 않는다. SERVED 는 인용 해석 결과를
    citation_order 순으로 served_citations 에 담는다.
    """

    if judgment.failed:
        return GateResult(outcome=CacheAttemptOutcome.FAILED)
    if (
        judgment.decision != ClassificationDecision.CONNECT
        or judgment.subproblem is None
    ):
        return _rejected(REJECT_CLASSIFICATION_NOT_CONNECTED)

    if (
        subproblem is None
        or subproblem.subproblem_id != judgment.subproblem.subproblem_id
        or subproblem.status != QuestionSubproblemStatus.APPROVED
    ):
        return _rejected(REJECT_SUBPROBLEM_NOT_APPROVED)
    if canonical_answer is None:
        return _rejected(REJECT_CANONICAL_ANSWER_NOT_FOUND)

    reasons: List[str] = []
    if canonical_answer.subproblem_version != subproblem.current_version:
        reasons.append(REJECT_SUBPROBLEM_VERSION_MISMATCH)
    resolutions = tuple(
        sorted(citation_resolutions, key=lambda item: item.citation_order)
    )
    if not resolutions:
        reasons.append(REJECT_CANONICAL_CITATION_MISSING)
    reasons.extend(
        resolution.rejection_reason or REJECT_CITED_SECTION_CHANGED
        for resolution in resolutions
        if not resolution.passed
    )
    if reasons:
        return _rejected(*reasons)

    canonical_answer_id = canonical_answer.canonical_answer_id
    if subproblem.serving_state == QuestionSubproblemServingState.SHADOW:
        return GateResult(
            outcome=CacheAttemptOutcome.SHADOW,
            canonical_answer_id=canonical_answer_id,
        )
    if subproblem.serving_state == QuestionSubproblemServingState.UNUSED:
        return _rejected(REJECT_SUBPROBLEM_UNUSED)
    if subproblem.serving_state != QuestionSubproblemServingState.SERVING:
        return _rejected(REJECT_SUBPROBLEM_STOPPED)
    if not semantic_cache_enabled:
        return GateResult(
            outcome=CacheAttemptOutcome.GROUP_DISABLED,
            canonical_answer_id=canonical_answer_id,
        )
    return GateResult(
        outcome=CacheAttemptOutcome.SERVED,
        canonical_answer_id=canonical_answer_id,
        served_citations=resolutions,
    )


def gate_judgment_input(
    result: GateResult,
    *,
    citation_resolutions: Sequence[CitationResolution] = (),
    canonical_answer_id: Optional[uuid.UUID] = None,
) -> Dict[str, Any]:
    """judgment_input.gate 칸. 판별 행을 쓰기 전에 게이트를 평가해 한 번에 쓴다(결정 A).

    question_cache_attempts 는 REJECTED 의 정본 id 를 담지 못하므로, 게이트가 읽은 정본 id
    (canonical_answer_id)와 인용 해석 전체를 여기 남긴다. 인자를 비우면 결과의 값을 쓴다.
    """

    resolutions = tuple(citation_resolutions) or tuple(result.served_citations)
    answer_id = canonical_answer_id or result.canonical_answer_id
    return {
        "outcome": result.outcome.value,
        "canonicalAnswerId": None if answer_id is None else str(answer_id),
        "rejectionReasons": list(result.rejection_reasons),
        "citationResolutions": [
            resolution.to_judgment_input()
            for resolution in sorted(resolutions, key=lambda item: item.citation_order)
        ],
    }
