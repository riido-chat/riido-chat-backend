"""판별 호출에 보내는 신뢰하지 않는 데이터 payload 를 만든다.

PoC prompt.build_classifier_input(v7.2: number_criteria, include_canonical_answers,
document_candidates) 을 옮겼다. 운영에서 달라진 점은 식별자(R5)와 섞기 seed 뿐이다.

- 세부 문제 id 는 question_subproblems.key, 그룹 id 는 문서 키, 문서 후보 id 는 D1..Dn.
- 순서는 rag_run_id 기반 seed 로 섞고 세부 문제와 문서는 스트림을 나눈다. 한쪽 후보 수가
  바뀌어도 다른 쪽 순서가 흔들리지 않는다.
- 유사도, 검색 점수, 검색 순위는 payload 에 넣지 않는다. rank 는 제시 순서다.
- PoC 의 taxonomyVersion 은 운영에 대응 개념이 없어 넣지 않는다. 판마다 세부 문제 판은
  judgment_input 에 남는다. contextSummary 는 판별이 대화 이력을 받지 않으므로 null 이다.
"""

import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.question_grouping.constants import (
    DOCUMENT_CANDIDATE_TOP_K,
    DOCUMENT_ID_PREFIX,
    DOCUMENT_SHUFFLE_STREAM,
    JUDGE_PROMPT_VERSION,
    SUBPROBLEM_CANDIDATE_TOP_K,
    SUBPROBLEM_SHUFFLE_STREAM,
)
from app.question_grouping.models import (
    DocumentCandidate,
    JudgePresentation,
    PresentedDocument,
    PresentedSubproblem,
    SubproblemCandidate,
)


def split_criteria(text: Optional[str]) -> Tuple[str, ...]:
    """DB text 칸의 기준 문장을 한 줄에 하나씩 나눈다.

    PoC console_catalog.split_criteria 와 같은 규칙이다. 줄 앞뒤의 공백과 목록 기호
    (-, *)를 걷어내고 빈 줄은 버린다. 저장 형식(줄바꿈 구분)은 시드 스크립트가 맞춘다.
    """

    if not text:
        return ()
    lines = [line.strip(" -*\t") for line in str(text).replace("\r", "").split("\n")]
    criteria = tuple(line for line in lines if line)
    return criteria or (str(text).strip(),)


def numbered_rules(prefix: str, rules: Sequence[str]) -> Dict[str, str]:
    """{"I1": 첫 기준, "I2": 둘째 기준, ...}. 판별 응답이 가리키는 번호다."""

    return {f"{prefix}{index}": text for index, text in enumerate(rules, 1)}


def presentation_seed(rag_run_id: object) -> str:
    """턴의 섞기 seed. 같은 턴을 다시 판별해도 같은 순서가 나온다."""

    return str(rag_run_id)


def stream_seed(seed: str, stream: str) -> str:
    return f"{seed}:{stream}"


def shuffle_rng(seed: str, stream: str) -> random.Random:
    """스트림별 난수 생성기. 문자열 seed 라 Python 판이 달라도 순서가 같다."""

    return random.Random(stream_seed(seed, stream))


def _group_payload(candidate: SubproblemCandidate) -> Dict[str, Any]:
    item = candidate.item
    payload: Dict[str, Any] = {"id": item.document_key}
    if item.document_title and item.document_title != item.name:
        payload["name"] = item.document_title
    return payload


def _subproblem_payload(candidate: SubproblemCandidate) -> Dict[str, Any]:
    item = candidate.item
    payload: Dict[str, Any] = {
        "id": item.key,
        "name": item.name,
        "inclusionCriteria": numbered_rules("I", item.inclusion_criteria),
        "exclusionCriteria": numbered_rules("E", item.exclusion_criteria),
    }
    canonical = item.canonical_answer
    if canonical is not None:
        canonical_payload: Dict[str, Any] = {
            "contentMarkdown": canonical.content_markdown
        }
        if canonical.applicability_rules:
            canonical_payload["applicabilityRules"] = list(
                canonical.applicability_rules
            )
        payload["canonicalAnswer"] = canonical_payload
    return payload


def _validate_subproblem_candidates(
    candidates: Sequence[SubproblemCandidate],
) -> None:
    if len(candidates) > SUBPROBLEM_CANDIDATE_TOP_K:
        raise ValueError(
            f"세부 문제 후보는 {SUBPROBLEM_CANDIDATE_TOP_K}개 이하여야 합니다: "
            f"{len(candidates)}"
        )
    keys = [candidate.item.key for candidate in candidates]
    if any(not key for key in keys):
        raise ValueError("세부 문제 key 가 비어 있습니다.")
    if len(set(keys)) != len(keys):
        raise ValueError("세부 문제 후보 key 가 중복되었습니다.")
    ids = [candidate.item.subproblem_id for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("세부 문제 후보 id 가 중복되었습니다.")


def _validate_document_candidates(candidates: Sequence[DocumentCandidate]) -> None:
    if len(candidates) > DOCUMENT_CANDIDATE_TOP_K:
        raise ValueError(
            f"문서 후보는 {DOCUMENT_CANDIDATE_TOP_K}개 이하여야 합니다: "
            f"{len(candidates)}"
        )
    sources = [candidate.outline.document_source_id for candidate in candidates]
    if len(set(sources)) != len(sources):
        raise ValueError("문서 후보 문서가 중복되었습니다.")


def build_judge_presentation(
    resolved_query: str,
    subproblem_candidates: Sequence[SubproblemCandidate],
    document_candidates: Sequence[DocumentCandidate],
    *,
    seed: str,
) -> JudgePresentation:
    """후보를 검색 순위대로 세운 뒤 스트림별로 섞어 payload 와 제시 목록을 만든다."""

    question = resolved_query.strip()
    if not question:
        raise ValueError("판별할 질문이 비어 있습니다.")
    _validate_subproblem_candidates(subproblem_candidates)
    _validate_document_candidates(document_candidates)

    subproblems: List[SubproblemCandidate] = sorted(
        subproblem_candidates, key=lambda candidate: candidate.retrieval_rank
    )
    shuffle_rng(seed, SUBPROBLEM_SHUFFLE_STREAM).shuffle(subproblems)
    documents: List[DocumentCandidate] = sorted(
        document_candidates, key=lambda candidate: candidate.retrieval_rank
    )
    shuffle_rng(seed, DOCUMENT_SHUFFLE_STREAM).shuffle(documents)

    candidate_payloads = []
    presented_subproblems = []
    for order, candidate in enumerate(subproblems, 1):
        item = candidate.item
        candidate_payloads.append(
            {
                "rank": order,
                "group": _group_payload(candidate),
                "subproblem": _subproblem_payload(candidate),
            }
        )
        presented_subproblems.append(
            PresentedSubproblem(
                key=item.key,
                subproblem_id=item.subproblem_id,
                subproblem_version=item.current_version,
                problem_group_id=item.problem_group_id,
                document_source_id=item.document_source_id,
                document_key=item.document_key,
                canonical_answer_id=(
                    None
                    if item.canonical_answer is None
                    else item.canonical_answer.canonical_answer_id
                ),
                similarity=candidate.similarity,
                retrieval_rank=candidate.retrieval_rank,
                presented_order=order,
                inclusion_count=len(item.inclusion_criteria),
                exclusion_count=len(item.exclusion_criteria),
            )
        )

    document_payloads = []
    presented_documents = []
    for order, candidate in enumerate(documents, 1):
        outline = candidate.outline
        document_id = f"{DOCUMENT_ID_PREFIX}{order}"
        document_payloads.append(
            {
                "id": document_id,
                "title": outline.title,
                "parentPath": outline.parent_path,
                "headings": list(outline.headings),
            }
        )
        presented_documents.append(
            PresentedDocument(
                id=document_id,
                document_source_id=outline.document_source_id,
                document_version_id=outline.document_version_id,
                document_key=outline.document_key,
                title=outline.title,
                parent_path=outline.parent_path,
                score=candidate.score,
                retrieval_rank=candidate.retrieval_rank,
                presented_order=order,
            )
        )

    payload: Dict[str, Any] = {
        "policyVersion": JUDGE_PROMPT_VERSION,
        "question": question,
        "contextSummary": None,
        "candidateCount": len(candidate_payloads),
        "candidates": candidate_payloads,
        "documentCandidateCount": len(document_payloads),
        "documentCandidates": document_payloads,
    }
    return JudgePresentation(
        payload=payload,
        subproblems=tuple(presented_subproblems),
        documents=tuple(presented_documents),
        subproblem_seed=stream_seed(seed, SUBPROBLEM_SHUFFLE_STREAM),
        document_seed=stream_seed(seed, DOCUMENT_SHUFFLE_STREAM),
    )


def presentation_judgment_input(
    presentation: JudgePresentation,
    *,
    document_retrieval: Mapping[str, Any],
) -> Dict[str, Any]:
    """judgment_input 의 subproblemCandidates 와 documentCandidates 블록."""

    return {
        "subproblemCandidates": {
            "seed": presentation.subproblem_seed,
            "items": [item.to_judgment_input() for item in presentation.subproblems],
        },
        "documentCandidates": {
            "seed": presentation.document_seed,
            "retrieval": dict(document_retrieval),
            "items": [item.to_judgment_input() for item in presentation.documents],
        },
    }
