"""동결 판별 입력으로 운영 판별 payload 를 다시 만든다.

운영 코드(app.question_grouping.payload.build_judge_presentation, judge_client.build_judge_request)를
그대로 부른다. 동결 행의 seed 와 검색 순위로 섞기를 재현하고, 제시 순서가 동결 기록과 다르면 멈춘다.
판별 요청 해시가 동결 payloadSha256 과 같으면 2026-10-01 판별 실행의 입력과 바이트까지 같다는 뜻이다.

모델에 주는 입력은 턴마다 한 번 만든 payload_bytes 하나다. luna 는 이것을 responses.create 의 input
문자열로 그대로 보내고, Jev 는 같은 바이트를 읽어 요청을 만든다. 두 모델이 받는 판별 입력(질의, 후보,
제시 순서, 기준, 정본)은 이 바이트에서만 나온다.

동결 1.0.0 의 판별 입력과 카탈로그에는 문서 후보의 parentPath, headings 와 정본 적용 제외 규칙이 없다.
이 값은 manifest frozen.files 에 judge-payload-supplement-<날짜>.json 이 등록돼 있으면 거기서 읽는다:

    {"documentOutlines": {"<documentVersionId>": {"parentPath": "...", "headings": ["..."]}},
     "applicabilityRules": {"<세부 문제 key>": ["..."]}}

보충 파일이 없으면 parentPath 는 문서 키에서 계산하고 headings 와 적용 제외 규칙은 비운다. 이때 요청
해시는 동결 기록과 맞지 않으므로 payload 확인(strict)에서 실행을 멈춘다.
"""

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.database.models import QuestionSubproblemServingState
from app.question_grouping.document_candidates import parent_path_from_document_key
from app.question_grouping.judge_client import build_judge_request
from app.question_grouping.models import (
    CatalogCanonicalAnswer,
    DocumentCandidate,
    DocumentOutline,
    JudgePresentation,
    SubproblemCandidate,
    SubproblemCatalogItem,
)
from app.question_grouping.payload import build_judge_presentation, split_criteria
from evaluation.judge_eval.golden import GoldenError, TurnKey, sha256_bytes

# payload 에 들어가지 않는 UUID 칸은 key 에서 결정적으로 만든다.
_UUID_NAMESPACE = uuid.UUID("5b0c7a52-6d3e-4f1e-9a51-0e6a0b1f2c33")


def request_sha256(request: Mapping[str, Any]) -> str:
    """freeze_judge_inputs.payload_sha256 과 같은 직렬화 (키 정렬, 공백 없는 구분자, ensure_ascii=False)."""

    text = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(text.encode("utf-8"))


@dataclass(frozen=True)
class PreparedTurn:
    item_id: str
    layer: str
    turn_no: int
    skip_reason: Optional[str] = None
    payload_bytes: Optional[bytes] = None
    presentation: Optional[JudgePresentation] = None
    frozen_payload_sha256: Optional[str] = None
    request_sha256: Optional[str] = None
    presented_keys: Tuple[str, ...] = ()

    @property
    def key(self) -> TurnKey:
        return (self.item_id, self.turn_no)

    @property
    def judged(self) -> bool:
        return self.payload_bytes is not None

    @property
    def input_sha256(self) -> Optional[str]:
        return None if self.payload_bytes is None else sha256_bytes(self.payload_bytes)

    @property
    def payload_verified(self) -> bool:
        return self.request_sha256 is not None and self.request_sha256 == self.frozen_payload_sha256


def _item(candidate: Mapping[str, Any], entry: Mapping[str, Any], rules: Sequence[str]) -> SubproblemCatalogItem:
    key = candidate["key"]
    canonical = entry.get("canonical")
    canonical_answer = None
    if canonical and canonical.get("text") is not None:
        canonical_id = candidate.get("canonicalAnswerId") or canonical.get("id")
        canonical_answer = CatalogCanonicalAnswer(
            canonical_answer_id=uuid.UUID(canonical_id) if canonical_id else uuid.uuid5(_UUID_NAMESPACE, f"canon:{key}"),
            content_markdown=canonical["text"],
            applicability_rules=tuple(rules),
        )
    return SubproblemCatalogItem(
        subproblem_id=uuid.uuid5(_UUID_NAMESPACE, f"subproblem:{key}"),
        key=key,
        name=entry["name"],
        inclusion_criteria=split_criteria("\n".join(entry.get("inclusionCriteria") or [])),
        exclusion_criteria=split_criteria("\n".join(entry.get("exclusionCriteria") or [])),
        current_version=int(candidate.get("subproblemVersion") or entry.get("currentRevision") or 1),
        problem_group_id=uuid.uuid5(_UUID_NAMESPACE, f"group:{candidate.get('documentKey')}"),
        document_source_id=None,
        document_key=candidate.get("documentKey") or "",
        serving_state=QuestionSubproblemServingState(entry["servingState"]),
        document_title=entry.get("documentTitle"),
        canonical_answer=canonical_answer,
    )


def _documents(row: Mapping[str, Any], outlines: Mapping[str, Any]) -> List[DocumentCandidate]:
    documents = []
    for index, doc in enumerate(row.get("documentCandidates") or [], start=1):
        outline = outlines.get(str(doc["documentVersionId"])) or {}
        documents.append(
            DocumentCandidate(
                outline=DocumentOutline(
                    document_source_id=index,
                    document_version_id=int(doc["documentVersionId"]),
                    document_key=doc["documentKey"],
                    title=doc["title"],
                    parent_path=outline.get("parentPath", parent_path_from_document_key(doc["documentKey"])),
                    headings=tuple(outline.get("headings") or ()),
                ),
                score=float(doc.get("score") or 0.0),
                retrieval_rank=int(doc["retrievalRank"]),
            )
        )
    return documents


def prepare_turn(
    item_id: str,
    layer: str,
    turn_no: int,
    row: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
    supplement: Optional[Mapping[str, Any]] = None,
) -> PreparedTurn:
    if not row.get("payloadSha256"):
        return PreparedTurn(
            item_id=item_id,
            layer=layer,
            turn_no=turn_no,
            skip_reason=row.get("notJudgedReason") or "동결 입력에 판별 payload 없음",
        )
    supplement = supplement or {}
    rules_by_key = supplement.get("applicabilityRules") or {}
    subproblems = []
    for candidate in row["subproblemCandidates"]:
        entry = catalog.get(candidate["key"])
        if entry is None:
            raise GoldenError(f"{item_id} 턴 {turn_no}: 카탈로그에 없는 후보 {candidate['key']}")
        subproblems.append(
            SubproblemCandidate(
                item=_item(candidate, entry, rules_by_key.get(candidate["key"]) or ()),
                similarity=float(candidate.get("similarity") or 0.0),
                retrieval_rank=int(candidate["retrievalRank"]),
            )
        )
    presentation = build_judge_presentation(
        row["resolvedQuery"],
        subproblems,
        _documents(row, supplement.get("documentOutlines") or {}),
        seed=row["seed"],
    )
    presented = tuple(item.key for item in presentation.subproblems)
    if list(presented) != list(row.get("presentedSubproblemKeys") or []) or [
        doc.document_key for doc in presentation.documents
    ] != [doc["documentKey"] for doc in sorted(row.get("documentCandidates") or [], key=lambda d: d["presentedOrder"])]:
        raise GoldenError(f"{item_id} 턴 {turn_no}: 후보 제시 순서가 동결 기록과 다릅니다 (섞기 코드가 바뀜)")
    request = build_judge_request(presentation.payload)
    return PreparedTurn(
        item_id=item_id,
        layer=layer,
        turn_no=turn_no,
        payload_bytes=request["input"].encode("utf-8"),
        presentation=presentation,
        frozen_payload_sha256=row["payloadSha256"],
        request_sha256=request_sha256(request),
        presented_keys=presented,
    )


def prepare_turns(
    turns: Sequence[Tuple[str, str, int]],
    rows: Mapping[TurnKey, Mapping[str, Any]],
    catalog: Mapping[str, Mapping[str, Any]],
    supplement: Optional[Mapping[str, Any]] = None,
) -> List[PreparedTurn]:
    return [
        prepare_turn(item_id, layer, turn_no, rows[(item_id, turn_no)], catalog, supplement)
        for item_id, layer, turn_no in turns
    ]


def presented_keys_by_turn(prepared: Sequence[PreparedTurn]) -> Dict[TurnKey, Tuple[str, ...]]:
    return {turn.key: turn.presented_keys for turn in prepared}
