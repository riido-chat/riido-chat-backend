"""판별에 보여줄 문서 후보를 검색 결과에서 고르고 outline 을 만든다.

문서 순위는 추가 호출 없이 이번 턴 하이브리드 검색의 검색기별 후보를 다시 쓴다.

1. BM25 top10 과 vector top10 의 청크 합집합에 RRF(k=60) 점수를 매긴다. 앱의
   fuse_rrf_results 와 같은 점수이고, 융합 top5 로 자르기 전의 전체 합집합이다.
2. 청크를 (점수 내림차순, chunk_id) 로 세우고 문서 판마다 최고 점수를 문서 점수로 한다.
3. 문서를 (점수 내림차순, 그 문서 청크가 처음 나온 위치) 로 세워 top5 를 고른다.

PoC 측정(scratchpad v7run/build_candidates.py)과 같은 규칙이며, 동점 청크의 순서만
PoC 의 청크 목록 위치 대신 앱 규칙인 chunk_id 를 쓴다.

outline 은 제목, 상위 경로, 절 제목이다. 상위 경로는 문서 키의 마지막 "/" 앞부분이고
절 제목은 H2 와 "H2 > H3" 이다(H1 은 문서 제목이라 뺀다).
"""

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from app.document.section_parser import find_headings
from app.question_grouping.constants import DOCUMENT_CANDIDATE_TOP_K, HEADING_SEPARATOR
from app.question_grouping.models import (
    DocumentCandidate,
    DocumentOutline,
    RankedDocument,
)
from app.retrieval.hybrid_retriever import CANDIDATE_K, RRF_RANK_CONSTANT
from app.retrieval.models import HybridSearchCall, RetrievalResult


def document_retrieval_settings() -> Dict[str, int]:
    """judgment_input.documentCandidates.retrieval 에 남기는 검색 조건."""

    return {"dense": CANDIDATE_K, "bm25": CANDIDATE_K, "rrfK": RRF_RANK_CONSTANT}


def _ranks_by_chunk(
    results: Iterable[RetrievalResult],
    document_versions: Dict[int, int],
) -> Dict[int, int]:
    ranks: Dict[int, int] = {}
    for result in results:
        chunk = result.chunk
        if chunk.chunk_id is None or chunk.document_version_id is None:
            raise ValueError("문서 후보에는 DB 식별자가 있는 Chunk가 필요합니다.")
        known = document_versions.setdefault(chunk.chunk_id, chunk.document_version_id)
        if known != chunk.document_version_id:
            raise ValueError("동일 Chunk PK의 문서 판이 일치하지 않습니다.")
        ranks.setdefault(chunk.chunk_id, result.rank)
    return ranks


def rank_documents(
    bm25_results: Sequence[RetrievalResult],
    vector_results: Sequence[RetrievalResult],
    *,
    top_k: int = DOCUMENT_CANDIDATE_TOP_K,
    rrf_k: int = RRF_RANK_CONSTANT,
) -> Tuple[RankedDocument, ...]:
    """검색기별 후보 청크를 RRF 로 합쳐 문서 판 순위 top_k 를 만든다."""

    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    document_versions: Dict[int, int] = {}
    bm25_ranks = _ranks_by_chunk(bm25_results, document_versions)
    vector_ranks = _ranks_by_chunk(vector_results, document_versions)

    def fused(chunk_id: int) -> float:
        score = 0.0
        if chunk_id in bm25_ranks:
            score += 1.0 / (rrf_k + bm25_ranks[chunk_id])
        if chunk_id in vector_ranks:
            score += 1.0 / (rrf_k + vector_ranks[chunk_id])
        return score

    ordered_chunks = sorted(
        document_versions, key=lambda chunk_id: (-fused(chunk_id), chunk_id)
    )
    # 점수 내림차순이므로 문서 판마다 처음 나온 청크가 최고 점수이자 첫 위치다.
    best_score: Dict[int, float] = {}
    best_chunk: Dict[int, int] = {}
    first_position: Dict[int, int] = {}
    for position, chunk_id in enumerate(ordered_chunks):
        version_id = document_versions[chunk_id]
        if version_id not in best_score:
            best_score[version_id] = fused(chunk_id)
            best_chunk[version_id] = chunk_id
            first_position[version_id] = position

    ranked = sorted(
        best_score,
        key=lambda version_id: (-best_score[version_id], first_position[version_id]),
    )[:top_k]
    return tuple(
        RankedDocument(
            document_version_id=version_id,
            score=best_score[version_id],
            retrieval_rank=rank,
            best_chunk_id=best_chunk[version_id],
        )
        for rank, version_id in enumerate(ranked, 1)
    )


def rank_documents_from_search(
    search: HybridSearchCall,
    *,
    top_k: int = DOCUMENT_CANDIDATE_TOP_K,
) -> Tuple[RankedDocument, ...]:
    """하이브리드 검색 한 번의 검색기별 후보 전체로 문서 순위를 만든다."""

    return rank_documents(search.bm25_results, search.vector_results, top_k=top_k)


def parent_path_from_document_key(document_key: str) -> str:
    """문서 키의 마지막 "/" 앞부분. 폴더가 없으면 빈 문자열이다."""

    head, _, _ = document_key.rpartition("/")
    return head


def headings_from_markdown(markdown: str) -> Tuple[str, ...]:
    """문서 Markdown 의 H2 와 "H2 > H3" 제목을 문서 순서대로 만든다.

    PoC document_candidates.parse_headings 와 같다. 빈 제목과 H1 은 빼고, 첫 H2 앞의
    H3 는 부모가 없어 그대로 둔다.
    """

    headings: List[str] = []
    current_h2: Optional[str] = None
    for _, level, title in find_headings(markdown.splitlines()):
        if not title or level == 1:
            continue
        if level == 2:
            current_h2 = title
            headings.append(title)
        else:
            headings.append(
                f"{current_h2}{HEADING_SEPARATOR}{title}" if current_h2 else title
            )
    return tuple(headings)


def headings_from_sections(
    sections: Sequence[Tuple[Sequence[str], str]],
) -> Tuple[str, ...]:
    """content_nodes 로 저장한 절에서 H2 와 "H2 > H3" 제목을 만든다.

    sections 는 node_order 순서의 ``(section_path, normalized_content)`` 이다. 절 하나가
    H2 하나이고 section_path 는 ``(문서 제목, H2 제목)`` 이며 H3 는 본문 안에 남아 있다.
    문서 제목만 있는 머리말 절은 제목을 더하지 않는다.

    알려진 차이: 첫 H2 앞에 H3 가 있으면 파서가 첫 H3 를 머리말 절 제목으로 올리므로
    그 뒤 H3 가 "첫 H3 > H3" 로 나온다. Markdown 기준(headings_from_markdown)은 따로 둔다.
    """

    headings: List[str] = []
    for section_path, body in sections:
        local_path = tuple(section_path[1:])
        parent = local_path[-1] if local_path else None
        if parent:
            headings.append(parent)
        for _, level, title in find_headings((body or "").splitlines()):
            if level != 3 or not title:
                continue
            headings.append(f"{parent}{HEADING_SEPARATOR}{title}" if parent else title)
    return tuple(headings)


def build_document_outline(
    *,
    document_source_id: int,
    document_version_id: int,
    document_key: str,
    title: str,
    headings: Sequence[str],
) -> DocumentOutline:
    if not title.strip():
        raise ValueError("문서 후보 제목이 비어 있습니다.")
    return DocumentOutline(
        document_source_id=document_source_id,
        document_version_id=document_version_id,
        document_key=document_key,
        title=title.strip(),
        parent_path=parent_path_from_document_key(document_key),
        headings=tuple(headings),
    )


def attach_outlines(
    ranked: Sequence[RankedDocument],
    outlines_by_version_id: Mapping[int, DocumentOutline],
) -> Tuple[DocumentCandidate, ...]:
    """문서 순위에 outline 을 붙인다. 같은 색인에서 읽으므로 빠지면 데이터 오류다."""

    missing = [
        item.document_version_id
        for item in ranked
        if item.document_version_id not in outlines_by_version_id
    ]
    if missing:
        raise ValueError(f"문서 후보 outline 을 찾지 못했습니다: {missing}")
    return tuple(
        DocumentCandidate(
            outline=outlines_by_version_id[item.document_version_id],
            score=item.score,
            retrieval_rank=item.retrieval_rank,
        )
        for item in ranked
    )
