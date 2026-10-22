"""판별 프롬프트 v7.2 지시문과 strict 출력 스키마.

지시문 본문은 app/prompts/judge/question-grouping-v7-2.md 에 있다. PoC
evaluation/question_grouping_poc/prompts/classifier_v7_2.md 를 앞뒤 공백만 걷어낸 원문
그대로다(PoC 도 같은 방식으로 읽어 보냈다). 한 글자라도 바꾸면 v7.2 측정 조건이 깨지므로
고치지 않는다. 바꿀 때는 새 버전 파일을 만든다. 운영 코드는 evaluation/ 을 읽지 않는다.

출력 스키마는 PoC openai_client.CLASSIFIER_SCHEMA_V7_2 와 속성 순서까지 같다. strict
구조화 출력은 조건부 널을 표현하지 못하므로 결정 사이의 조건은 decision.py 가 보정한다.
"""

from typing import Any, Dict

from app import prompts
from app.question_grouping.constants import JUDGE_PROMPT_VERSION


JUDGE_INSTRUCTIONS_SHA256 = "4e6fb86c0a8a0fb9210e3c4e3bbc1c385a88a887908f437982369316ace1b902"

JUDGE_INSTRUCTIONS = prompts.get("judge", JUDGE_PROMPT_VERSION)

JUDGE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["CONNECT", "SEPARATE", "UNCLASSIFIED"]},
        "groupId": {"type": ["string", "null"]},
        "subproblemId": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationaleCode": {"type": "string"},
        "matchedCriteria": {"type": "array", "items": {"type": "string"}},
        "conflictingCriteria": {"type": "array", "items": {"type": "string"}},
        "ambiguityReason": {"type": ["string", "null"]},
        "documentDecision": {
            "type": "string",
            "enum": ["MATCHED", "NONE", "SUBPROBLEM_DOCUMENT"],
        },
        "documentCandidateId": {"type": ["string", "null"]},
        "documentRationaleCode": {
            "anyOf": [
                {
                    "type": "string",
                    "enum": [
                        "ASK_COVERED",
                        "TOPIC_ONLY",
                        "TANGENTIAL_OVERLAP",
                        "NO_CANDIDATE_COVERS",
                        "TOO_VAGUE_TO_PLACE",
                    ],
                },
                {"type": "null"},
            ]
        },
    },
    "required": [
        "decision",
        "groupId",
        "subproblemId",
        "confidence",
        "rationaleCode",
        "matchedCriteria",
        "conflictingCriteria",
        "ambiguityReason",
        "documentDecision",
        "documentCandidateId",
        "documentRationaleCode",
    ],
}
