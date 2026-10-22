"""판별 응답을 R7 규칙으로 보정하고 턴 판별 결과를 만든다.

무효는 두 경우뿐이다. (a) 빈 출력이거나 결정을 읽을 수 없는 출력, (b) CONNECT 인데
제시하지 않은 세부 문제 key. 나머지 형식 위반은 보정하고 위반 사실만 남긴다.

- CONNECT 의 문서 칸은 무시하고 세부 문제의 문서를 쓴다(위반은 기록).
- CONNECT 의 groupId 가 세부 문제의 문서 키와 다르면 세부 문제의 그룹으로 고친다.
- 비연결 결정의 MATCHED 문서 id 가 제시 목록에 없으면 NONE 으로 고친다.
- 기준 항목(matchedCriteria, conflictingCriteria) 형식 위반은 개수만 센다.

(a) 의 "읽을 수 없는 출력" 은 JSON 객체가 아니거나 decision 값이 세 결정 중 하나가
아닌 경우로 본다. strict 구조화 출력에서는 사실상 빈 출력이나 잘린 출력일 때만 생긴다.
"""

import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Tuple

from app.database.models import ClassificationDecision
from app.question_grouping.attribution import initial_attribution
from app.question_grouping.constants import (
    INVALID_EMPTY_OUTPUT,
    INVALID_UNKNOWN_SUBPROBLEM_KEY,
    INVALID_UNPARSEABLE_OUTPUT,
)
from app.question_grouping.models import (
    DOCUMENT_RATIONALE_CODES,
    DocumentDecision,
    JudgeCall,
    JudgeFailure,
    JudgeFailureKind,
    JudgePresentation,
    NormalizedJudgment,
    PresentedDocument,
    TurnJudgment,
)


COMPACT_CONFLICT_CODES = ("OUTCOME", "ANSWER", "QUALIFIER", "OBJECT", "TIE")
_MATCHED_ENTRY = re.compile(r"^I([1-9][0-9]*)$")
_CONFLICTING_ENTRY = re.compile(
    r"^(?P<id>[^\s:]+):(?:(?P<code>"
    + "|".join(COMPACT_CONFLICT_CODES)
    + r")|E(?P<rule>[1-9][0-9]*))$"
)

# 문서 칸 위반 코드. judgment_input.normalization.documentFieldViolations 에 남긴다.
DOC_VIOLATION_DECISION = "DOCUMENT_DECISION"
DOC_VIOLATION_CANDIDATE_ID = "DOCUMENT_CANDIDATE_ID"
DOC_VIOLATION_RATIONALE_CODE = "DOCUMENT_RATIONALE_CODE"
# 비연결 결정이 세부 문제 칸을 채운 위반
SUBPROBLEM_VIOLATION_GROUP_ID = "GROUP_ID"
SUBPROBLEM_VIOLATION_SUBPROBLEM_ID = "SUBPROBLEM_ID"


def parse_judge_output(
    output_text: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """출력 문자열을 JSON 객체로 읽는다. ``(객체, 무효 사유)`` 를 돌려준다."""

    if output_text is None or not output_text.strip():
        return None, INVALID_EMPTY_OUTPUT
    try:
        value = json.loads(output_text)
    except (TypeError, ValueError):
        return None, INVALID_UNPARSEABLE_OUTPUT
    if not isinstance(value, dict):
        return None, INVALID_UNPARSEABLE_OUTPUT
    return value, None


def _decision_of(raw: Mapping[str, Any]) -> Optional[ClassificationDecision]:
    value = raw.get("decision")
    if not isinstance(value, str):
        return None
    try:
        return ClassificationDecision(value)
    except ValueError:
        return None


def _confidence_of(raw: Mapping[str, Any]) -> Optional[float]:
    value = raw.get("confidence")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if math.isnan(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _optional_string(raw: Mapping[str, Any], field: str) -> Optional[str]:
    value = raw.get(field)
    return value if isinstance(value, str) and value else None


def _document_decision_of(raw: Mapping[str, Any]) -> Optional[DocumentDecision]:
    value = raw.get("documentDecision")
    if not isinstance(value, str):
        return None
    try:
        return DocumentDecision(value)
    except ValueError:
        return None


def criteria_report(
    raw: Mapping[str, Any],
    presentation: JudgePresentation,
) -> Dict[str, Any]:
    """v7.2 기준 항목 형식을 센다. 판별을 무효로 만들지 않는다.

    PoC decision.compact_criteria_report 와 같은 규칙이며 세부 문제 id 는 key 다.
    malformed 는 문법에 맞지 않는 항목, unknownReference 는 문법은 맞지만 제시하지 않은
    세부 문제나 없는 기준 번호를 가리킨 항목이다. 남기는 값은 번호와 코드뿐이다.
    """

    shown = presentation.subproblem_by_key()
    chosen = None
    chosen_key = raw.get("subproblemId")
    if raw.get("decision") == ClassificationDecision.CONNECT.value and isinstance(
        chosen_key, str
    ):
        chosen = shown.get(chosen_key)

    def entries(field: str) -> List[Any]:
        value = raw.get(field)
        return list(value) if isinstance(value, list) else []

    matched_kept: List[str] = []
    matched = {"entries": 0, "malformed": 0, "unknownReference": 0}
    for entry in entries("matchedCriteria"):
        matched["entries"] += 1
        match = _MATCHED_ENTRY.match(entry.strip()) if isinstance(entry, str) else None
        if not match:
            matched["malformed"] += 1
            continue
        if chosen is not None and int(match.group(1)) > chosen.inclusion_count:
            matched["unknownReference"] += 1
            continue
        matched_kept.append(entry.strip())

    conflicting_kept: List[str] = []
    codes: Counter = Counter()
    conflicting = {"entries": 0, "malformed": 0, "unknownReference": 0}
    for entry in entries("conflictingCriteria"):
        conflicting["entries"] += 1
        match = (
            _CONFLICTING_ENTRY.match(entry.strip()) if isinstance(entry, str) else None
        )
        if not match:
            conflicting["malformed"] += 1
            continue
        target = shown.get(match.group("id"))
        rule = match.group("rule")
        if target is None or (rule is not None and int(rule) > target.exclusion_count):
            conflicting["unknownReference"] += 1
            continue
        conflicting_kept.append(entry.strip())
        codes["E" if rule is not None else match.group("code")] += 1

    return {
        "matchedCriteria": matched,
        "conflictingCriteria": conflicting,
        "violations": sum(
            block["malformed"] + block["unknownReference"]
            for block in (matched, conflicting)
        ),
        "matched": matched_kept,
        "conflicting": conflicting_kept,
        "conflictCodes": dict(sorted(codes.items())),
    }


def _connect_document_violations(raw: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if raw.get("documentDecision") != DocumentDecision.SUBPROBLEM_DOCUMENT.value:
        violations.append(DOC_VIOLATION_DECISION)
    if raw.get("documentCandidateId") is not None:
        violations.append(DOC_VIOLATION_CANDIDATE_ID)
    if raw.get("documentRationaleCode") is not None:
        violations.append(DOC_VIOLATION_RATIONALE_CODE)
    return tuple(violations)


def _separate_document(
    raw: Mapping[str, Any],
    presentation: JudgePresentation,
) -> Tuple[DocumentDecision, Optional[PresentedDocument], Tuple[str, ...], bool]:
    """비연결 결정의 문서 칸. ``(결정, 문서, 위반, 목록 밖 id 여부)``."""

    document_decision = _document_decision_of(raw)
    candidate_id = raw.get("documentCandidateId")
    rationale = raw.get("documentRationaleCode")
    violations: List[str] = []

    if document_decision is DocumentDecision.MATCHED:
        if rationale not in DOCUMENT_RATIONALE_CODES[DocumentDecision.MATCHED]:
            violations.append(DOC_VIOLATION_RATIONALE_CODE)
        document = (
            presentation.document_by_id().get(candidate_id)
            if isinstance(candidate_id, str)
            else None
        )
        if document is None:
            violations.append(DOC_VIOLATION_CANDIDATE_ID)
            return DocumentDecision.NONE, None, tuple(violations), True
        return DocumentDecision.MATCHED, document, tuple(violations), False

    if document_decision is not DocumentDecision.NONE:
        # SUBPROBLEM_DOCUMENT 나 알 수 없는 값은 비연결에서 쓸 수 없어 NONE 으로 둔다.
        violations.append(DOC_VIOLATION_DECISION)
    if candidate_id is not None:
        violations.append(DOC_VIOLATION_CANDIDATE_ID)
    if document_decision is DocumentDecision.NONE and (
        rationale not in DOCUMENT_RATIONALE_CODES[DocumentDecision.NONE]
    ):
        violations.append(DOC_VIOLATION_RATIONALE_CODE)
    return DocumentDecision.NONE, None, tuple(violations), False


def normalize_judgment(
    output_text: Optional[str],
    presentation: JudgePresentation,
) -> NormalizedJudgment:
    """판별 출력 문자열을 제시 목록과 대조해 R7 규칙으로 보정한다."""

    raw, invalid_reason = parse_judge_output(output_text)
    if raw is None:
        return NormalizedJudgment(
            decision=ClassificationDecision.UNCLASSIFIED,
            invalid_reason=invalid_reason,
        )

    report = criteria_report(raw, presentation)
    decision = _decision_of(raw)
    if decision is None:
        return NormalizedJudgment(
            decision=ClassificationDecision.UNCLASSIFIED,
            raw_output=raw,
            invalid_reason=INVALID_UNPARSEABLE_OUTPUT,
            criteria_report=report,
        )

    common: Dict[str, Any] = {
        "confidence": _confidence_of(raw),
        "rationale_code": _optional_string(raw, "rationaleCode"),
        "ambiguity_reason": _optional_string(raw, "ambiguityReason"),
        "raw_output": raw,
        "criteria_report": report,
    }

    if decision is ClassificationDecision.CONNECT:
        key = raw.get("subproblemId")
        subproblem = (
            presentation.subproblem_by_key().get(key) if isinstance(key, str) else None
        )
        if subproblem is None:
            return NormalizedJudgment(
                decision=ClassificationDecision.UNCLASSIFIED,
                invalid_reason=INVALID_UNKNOWN_SUBPROBLEM_KEY,
                **common,
            )
        return NormalizedJudgment(
            decision=ClassificationDecision.CONNECT,
            subproblem=subproblem,
            document_decision=DocumentDecision.SUBPROBLEM_DOCUMENT,
            document_fields_ignored=True,
            document_field_violations=_connect_document_violations(raw),
            group_id_mismatch=raw.get("groupId") != subproblem.document_key,
            **common,
        )

    subproblem_violations = []
    if raw.get("groupId") is not None:
        subproblem_violations.append(SUBPROBLEM_VIOLATION_GROUP_ID)
    if raw.get("subproblemId") is not None:
        subproblem_violations.append(SUBPROBLEM_VIOLATION_SUBPROBLEM_ID)
    document_decision, document, violations, unknown = _separate_document(
        raw, presentation
    )
    return NormalizedJudgment(
        decision=decision,
        document_decision=document_decision,
        document=document,
        document_field_violations=violations,
        subproblem_field_violations=tuple(subproblem_violations),
        unknown_document_id=unknown,
        **common,
    )


def failed_turn_judgment(failure: JudgeFailure) -> TurnJudgment:
    """판별이 없는 턴. UNCLASSIFIED + NO_DOCUMENT/NONE 이다."""

    return TurnJudgment(
        decision=ClassificationDecision.UNCLASSIFIED,
        attribution=initial_attribution(ClassificationDecision.UNCLASSIFIED),
        failure=failure,
    )


def build_turn_judgment(
    call: JudgeCall,
    presentation: JudgePresentation,
) -> TurnJudgment:
    """판별 호출 결과를 정규화하고 초기 귀속을 붙인다.

    호출 실패와 R7 무효는 모두 판별 실패로 다룬다(캐시 시도 FAILED). 무효일 때도
    정규화 결과를 남겨 judgment_input 에 원 출력과 무효 사유를 적을 수 있게 한다.
    """

    if call.failure is not None:
        return failed_turn_judgment(call.failure)

    normalized = normalize_judgment(call.output_text, presentation)
    if not normalized.valid:
        failed = failed_turn_judgment(
            JudgeFailure(
                kind=JudgeFailureKind.INVALID_OUTPUT,
                safe_message=f"판별 응답이 계약에 맞지 않습니다: {normalized.invalid_reason}",
            )
        )
        return TurnJudgment(
            decision=failed.decision,
            attribution=failed.attribution,
            normalized=normalized,
            failure=failed.failure,
        )

    return TurnJudgment(
        decision=normalized.decision,
        attribution=initial_attribution(
            normalized.decision,
            subproblem=normalized.subproblem,
            document=normalized.document,
        ),
        subproblem=normalized.subproblem,
        normalized=normalized,
    )
