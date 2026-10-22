"""질문 연결 행의 문제 그룹 귀속을 정한다(인용 우선, 설계기록 3절 X).

| 순위 | 턴 상황 | attribution_source |
| --- | --- | --- |
| 1 | CONNECT | SUBPROBLEM |
| 2 | CONNECT 아님, 인용 하나 이상을 달고 COMPLETED | CITATION(턴 끝에 덮어쓴다) |
| 3 | CONNECT 아님, 답이 없음, 판별 문서 MATCHED | DOCUMENT |
| 4 | CONNECT 아님, 답이 없음, 판별 NONE 또는 판별 실패 | NONE |

판별 시점에는 턴이 어떻게 끝날지 모르므로 1, 3, 4순위 값으로 먼저 쓰고 턴 끝에서
2순위만 같은 행에 덮어쓴다. 캐시가 서빙한 턴은 CONNECT 라 덮어쓰지 않는다.
"""

import uuid
from typing import Dict, Optional, Sequence

from app.database.models import (
    AnswerStatus,
    AttributionSource,
    ClassificationDecision,
    QuestionProblemGroupKind,
)
from app.question_grouping.models import (
    AttributionTarget,
    CitedDocument,
    PresentedDocument,
    PresentedSubproblem,
)


def subproblem_attribution(problem_group_id: uuid.UUID) -> AttributionTarget:
    return AttributionTarget(
        attribution_source=AttributionSource.SUBPROBLEM,
        group_kind=QuestionProblemGroupKind.DOCUMENT,
        problem_group_id=problem_group_id,
    )


def document_attribution(
    document_source_id: int,
    source: AttributionSource = AttributionSource.DOCUMENT,
) -> AttributionTarget:
    return AttributionTarget(
        attribution_source=source,
        group_kind=QuestionProblemGroupKind.DOCUMENT,
        document_source_id=document_source_id,
    )


def no_document_attribution() -> AttributionTarget:
    return AttributionTarget(
        attribution_source=AttributionSource.NONE,
        group_kind=QuestionProblemGroupKind.NO_DOCUMENT,
    )


def initial_attribution(
    decision: ClassificationDecision,
    *,
    subproblem: Optional[PresentedSubproblem] = None,
    document: Optional[PresentedDocument] = None,
) -> AttributionTarget:
    """판별 결과로 쓰는 귀속. CONNECT→SUBPROBLEM, MATCHED→DOCUMENT, 그 외→NONE."""

    if decision == ClassificationDecision.CONNECT:
        if subproblem is None:
            raise ValueError("CONNECT 귀속에는 세부 문제가 필요합니다.")
        return subproblem_attribution(subproblem.problem_group_id)
    if document is not None:
        return document_attribution(document.document_source_id)
    return no_document_attribution()


def should_attribute_by_citation(
    decision: ClassificationDecision,
    answer_status: AnswerStatus,
    citation_count: int,
) -> bool:
    """턴 끝 CITATION 덮어쓰기 대상인가. 판별 실패(UNCLASSIFIED)도 포함한다."""

    return (
        decision != ClassificationDecision.CONNECT
        and answer_status == AnswerStatus.COMPLETED
        and citation_count > 0
    )


def most_cited_document_source_id(
    citations: Sequence[CitedDocument],
) -> Optional[int]:
    """인용 행 수가 가장 많은 문서. 동수면 그중 가장 앞 번호로 인용된 문서다.

    "동수면 인용 [1] 의 문서" 규칙을 동수 문서 안에서 가장 작은 citation_order 를
    가진 문서로 읽는다. [1] 의 문서가 동수 안에 있으면 같은 결과이고, [1] 의 문서가
    동수 밖(인용 수가 적음)이면 최다 인용 원칙을 먼저 지킨다.
    """

    if not citations:
        return None
    counts: Dict[int, int] = {}
    first_order: Dict[int, int] = {}
    for citation in citations:
        source_id = citation.document_source_id
        counts[source_id] = counts.get(source_id, 0) + 1
        previous = first_order.get(source_id)
        if previous is None or citation.citation_order < previous:
            first_order[source_id] = citation.citation_order
    return min(
        counts,
        key=lambda source_id: (-counts[source_id], first_order[source_id]),
    )


def citation_attribution(
    citations: Sequence[CitedDocument],
) -> Optional[AttributionTarget]:
    """턴 끝 인용 귀속 대상. 인용이 없으면 None 이다."""

    source_id = most_cited_document_source_id(citations)
    if source_id is None:
        return None
    return document_attribution(source_id, AttributionSource.CITATION)
