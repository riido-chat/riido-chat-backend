"""세부 문제 후보 검색. 질문 벡터와 포함 기준 임베딩의 코사인으로 top_k 를 고른다.

임베딩 문장(SUBPROBLEM_EMBEDDING_TEXT_VERSION = name-inclusion-v1)은 이름 한 줄과
포함 기준 한 줄씩이다. 제외 기준은 판정에만 쓰고 임베딩에는 넣지 않는다. 시드 스크립트가
이 함수로 문장을 만들어 question_subproblem_revisions.inclusion_embedding 을 계산한다.

순위는 DB 가 아니라 이 모듈의 순수 함수로 매긴다.

- 카탈로그는 문서 그룹당 수백 건 이하다. 세부 문제 개정 벡터에는 ANN 색인이 없어
  pgvector 로 보내도 전수 비교이고, 카탈로그 조회 한 번에 이미 벡터를 읽는다.
- 건너뛴 임베딩 건수, 동점 순서(key, id), 차원 검사를 DB 없이 단위 테스트로 고정한다.
- 질문 벡터는 호출자가 넘긴다. 검색의 retrieval_query 가 resolved_query 와 다르면 검색
  벡터를 쓰면 안 되므로 여기서 어느 벡터인지 가정하지 않는다.

카탈로그가 수천 건으로 커져 벡터 전송이 턴 지연에 보이면 같은 정렬 규칙으로 pgvector
``<=>`` 조회를 더한다.
"""

import math
import operator
from typing import List, Optional, Sequence, Tuple

from app.question_grouping.constants import (
    SUBPROBLEM_CANDIDATE_TOP_K,
    SUBPROBLEM_EMBEDDING_TEXT_VERSION,
)
from app.question_grouping.models import SubproblemCandidate, SubproblemCatalogItem


SUPPORTED_EMBEDDING_TEXT_VERSIONS = (SUBPROBLEM_EMBEDDING_TEXT_VERSION,)


def build_subproblem_embedding_text(
    name: str,
    inclusion_criteria: Sequence[str],
    *,
    text_version: str = SUBPROBLEM_EMBEDDING_TEXT_VERSION,
) -> str:
    """세부 문제 포함 기준 임베딩 입력 문장.

    name-inclusion-v1: 이름 한 줄 뒤에 포함 기준을 한 줄에 하나씩 둔다. 앞뒤 공백과 빈
    기준은 버린다. 기준은 payload.split_criteria 로 나눈 목록을 받는다.
    """

    if text_version not in SUPPORTED_EMBEDDING_TEXT_VERSIONS:
        raise ValueError(f"지원하지 않는 세부 문제 임베딩 문장 판입니다: {text_version}")
    title = (name or "").strip()
    if not title:
        raise ValueError("세부 문제 이름이 비어 있습니다.")
    criteria = [criterion.strip() for criterion in inclusion_criteria]
    criteria = [criterion for criterion in criteria if criterion]
    if not criteria:
        raise ValueError("세부 문제 포함 기준이 비어 있습니다.")
    return "\n".join([title, *criteria])


def subproblem_embedding_text(item: SubproblemCatalogItem) -> str:
    return build_subproblem_embedding_text(item.name, item.inclusion_criteria)


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(map(operator.mul, vector, vector)))


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
    *,
    left_norm: Optional[float] = None,
) -> float:
    """두 벡터의 코사인. 한쪽 크기가 0 이면 0.0 이다."""

    if not left or len(left) != len(right):
        raise ValueError("코사인 비교 벡터의 차원이 일치해야 합니다.")
    left_size = _norm(left) if left_norm is None else left_norm
    right_size = _norm(right)
    if not left_size or not right_size:
        return 0.0
    return sum(map(operator.mul, left, right)) / (left_size * right_size)


def rank_subproblems(
    query_embedding: Sequence[float],
    items: Sequence[SubproblemCatalogItem],
    *,
    top_k: int = SUBPROBLEM_CANDIDATE_TOP_K,
) -> Tuple[SubproblemCandidate, ...]:
    """카탈로그 항목을 질문 벡터와의 코사인 내림차순으로 세워 top_k 를 고른다.

    동점은 (key, subproblem_id 문자열) 순이다. key 는 문서 그룹 안에서 유일하다고
    가정하지 않는다. 임베딩이 없는 항목은 카탈로그 조회에서 빠져야 하므로 오류다.
    """

    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 합니다.")
    if not query_embedding:
        raise ValueError("질문 벡터가 비어 있습니다.")
    query = [float(value) for value in query_embedding]
    query_norm = _norm(query)

    scored: List[Tuple[float, SubproblemCatalogItem]] = []
    for item in items:
        if not item.inclusion_embedding:
            raise ValueError(f"세부 문제 포함 기준 임베딩이 없습니다: {item.subproblem_id}")
        scored.append(
            (
                cosine_similarity(query, item.inclusion_embedding, left_norm=query_norm),
                item,
            )
        )
    ranked = sorted(
        scored,
        key=lambda pair: (-pair[0], pair[1].key, str(pair[1].subproblem_id)),
    )[:top_k]
    return tuple(
        SubproblemCandidate(item=item, similarity=similarity, retrieval_rank=rank)
        for rank, (similarity, item) in enumerate(ranked, 1)
    )
