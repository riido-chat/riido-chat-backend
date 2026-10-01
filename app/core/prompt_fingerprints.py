"""모델에 실제로 보내는 프롬프트와 출력 스키마의 지문을 계산한다.

지문 표는 같은 폴더의 prompt_fingerprints.json 이다. 키는 `<component>@<version>`,
값은 sha256 hex 다. 운영 코드는 이 모듈을 부르지 않는다. 테스트
(tests/test_prompt_fingerprints.py)가 현재 판의 지문이 표와 같은지 확인하고,
CI(scripts/check_prompt_fingerprints.py)가 기존 키가 바뀌거나 지워지지 않았는지 확인한다.

정규화 규칙:

- 지시문(instructions): 상수 문자열 그대로를 UTF-8 로 인코딩한 sha256. 공백이나 줄바꿈을
  다듬지 않는다. 판별 지시문의 기존 JUDGE_INSTRUCTIONS_SHA256 과 같은 방식이다.
- 출력 스키마: 요청의 text.format 객체 전체(type, name, strict, schema)를
  json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False) 로 직렬화한
  문자열의 sha256. Pydantic 모델을 넘기는 호출(responses.parse)은 OpenAI SDK 가 실제로
  보내는 형태(type_to_text_format_param)로 바꾼 뒤 계산한다. 그래서 openai 나 pydantic
  업그레이드로 보내는 스키마가 달라져도 지문이 바뀐다.
- 묶음 판(generation@vNN): GENERATION_COMPOSITE_PARTS 순서대로 `<키>=<지문>` 줄을
  "\n" 으로 이은 문자열(끝 줄바꿈 없음)의 sha256. 하위 프롬프트나 스키마가 바뀌었는데
  묶음 판을 올리지 않으면 지문이 달라져 테스트가 실패한다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from openai.lib._parsing._responses import type_to_text_format_param

from app.answering import generator
from app.answering.models import GenerationResult, GenerationSourcePlan
from app.chat import query_rewrite
from app.chat.query_rewrite import QueryRewriteOutput
from app.question_grouping import constants as judge_constants
from app.question_grouping import prompt_v7_2
from app.question_grouping.judge_client import build_judge_request


FINGERPRINT_TABLE_PATH = Path(__file__).with_name("prompt_fingerprints.json")

# 묶음 판 generation@vNN 을 이루는 하위 키. 순서를 바꾸면 묶음 지문이 바뀐다.
GENERATION_COMPOSITE_PARTS: Tuple[str, ...] = (
    "generation.source_planning",
    "generation.source_planning_repair",
    "generation.source_planning.output_schema",
    "generation.answer",
    "generation.answer_repair",
    "generation.answer.output_schema",
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    """키 정렬, 여분 공백 없음, 비ASCII 그대로인 JSON 문자열."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def text_format_of(text_format: type) -> Dict[str, Any]:
    """responses.parse(text_format=Model) 가 요청에 싣는 text.format 객체."""

    return dict(type_to_text_format_param(text_format))


def composite_fingerprint(parts: Mapping[str, str]) -> str:
    """하위 키와 지문을 GENERATION_COMPOSITE_PARTS 순서로 묶은 지문."""

    lines: List[str] = []
    for component in GENERATION_COMPOSITE_PARTS:
        matches = [key for key in parts if key.split("@", 1)[0] == component]
        if len(matches) != 1:
            raise ValueError(f"묶음 판의 하위 키가 정확히 하나가 아닙니다: {component}")
        key = matches[0]
        lines.append(f"{key}={parts[key]}")
    return sha256_text("\n".join(lines))


def current_fingerprints() -> Dict[str, str]:
    """현재 코드가 쓰는 판의 키와 실제로 보내는 내용의 지문."""

    planning = generator.SOURCE_PLANNING_PROMPT_VERSION
    planning_repair = generator.SOURCE_PLANNING_REPAIR_PROMPT_VERSION
    answer = generator.ANSWER_PROMPT_VERSION
    answer_repair = generator.ANSWER_REPAIR_PROMPT_VERSION
    generation_parts = {
        f"generation.source_planning@{planning}": sha256_text(
            generator.SOURCE_PLANNING_PROMPT_V11
        ),
        f"generation.source_planning_repair@{planning_repair}": sha256_text(
            generator.SOURCE_PLANNING_REPAIR_PROMPT_V11
        ),
        # 재시도 호출도 같은 스키마를 쓰므로 스키마 키는 처음 판에만 둔다.
        f"generation.source_planning.output_schema@{planning}": sha256_json(
            text_format_of(GenerationSourcePlan)
        ),
        f"generation.answer@{answer}": sha256_text(generator.ANSWER_PROMPT_V17),
        f"generation.answer_repair@{answer_repair}": sha256_text(
            generator.ANSWER_REPAIR_PROMPT_V17
        ),
        f"generation.answer.output_schema@{answer}": sha256_json(
            text_format_of(GenerationResult)
        ),
    }

    rewrite = query_rewrite.QUERY_REWRITE_PROMPT_VERSION
    judge = judge_constants.JUDGE_PROMPT_VERSION
    judge_format = build_judge_request({})["text"]["format"]
    return {
        **generation_parts,
        f"generation@{generator.GENERATION_PROMPT_VERSION}": composite_fingerprint(
            generation_parts
        ),
        f"rewrite@{rewrite}": sha256_text(query_rewrite.QUERY_REWRITE_PROMPT_V4),
        f"rewrite.output_schema@{rewrite}": sha256_json(
            text_format_of(QueryRewriteOutput)
        ),
        f"judge@{judge}": sha256_text(prompt_v7_2.JUDGE_INSTRUCTIONS),
        f"judge.output_schema@{judge}": sha256_json(judge_format),
    }


def load_fingerprint_table(path: Path = FINGERPRINT_TABLE_PATH) -> Dict[str, str]:
    return dict(json.loads(path.read_text("utf-8"))["fingerprints"])
