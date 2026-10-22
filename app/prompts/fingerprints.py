"""모델에 실제로 보내는 프롬프트와 출력 스키마의 지문을 계산하는 규칙과 지문 표.

지문 표는 같은 폴더의 fingerprints.json 이다. 키는 `<component>@<version>`,
값은 sha256 hex 다. 운영 코드에서는 프로필 버전 검사(app/profiles/stage_settings.py)만 표와
계산 규칙을 읽어, 선택한 묶음 버전과 출력 스키마를 실행 중인 코드가 서빙할 수 있는지 본다. 테스트
(tests/test_prompt_fingerprints.py)가 현재 버전의 지문이 표와 같은지 확인하고,
tests/test_prompt_registry.py 가 버전 파일 본문이 표와 같은지 확인하며,
CI(scripts/check_prompt_fingerprints.py)가 기존 키가 바뀌거나 지워지지 않았는지 확인한다.

정규화 규칙:

- 지시문(instructions): 실제로 보내는 문자열 그대로를 UTF-8 로 인코딩한 sha256. 공백이나
  줄바꿈을 다듬지 않는다. 판별 지시문의 기존 JUDGE_INSTRUCTIONS_SHA256 과 같은 방식이다.
  재시도 지시문은 처음 지시문과 재시도 덧붙임을 이은 전체 문자열의 지문이다.
- 출력 스키마: 요청의 text.format 객체 전체(type, name, strict, schema)를
  json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False) 로 직렬화한
  문자열의 sha256. Pydantic 모델을 넘기는 호출(responses.parse)은 OpenAI SDK 가 실제로
  보내는 형태(type_to_text_format_param)로 바꾼 뒤 계산한다. 그래서 openai 나 pydantic
  업그레이드로 보내는 스키마가 달라져도 지문이 바뀐다.
- 묶음 버전(generation@vNN): GENERATION_COMPOSITE_PARTS 순서대로 `<키>=<지문>` 줄을
  "\\n" 으로 이은 문자열(끝 줄바꿈 없음)의 sha256. 하위 프롬프트나 스키마가 바뀌었는데
  묶음 버전을 올리지 않으면 지문이 달라져 테스트가 실패한다.

이 모듈은 app 의 다른 패키지를 import 하지 않는다. 현재 코드가 쓰는 버전과 출력 스키마를
모으는 일은 그 코드를 아는 테스트가 한다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from openai.lib._parsing._responses import type_to_text_format_param


FINGERPRINT_TABLE_PATH = Path(__file__).with_name("fingerprints.json")

# 묶음 버전 generation@vNN 을 이루는 하위 키. 순서를 바꾸면 묶음 지문이 바뀐다.
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
            raise ValueError(f"묶음 버전의 하위 키가 정확히 하나가 아닙니다: {component}")
        key = matches[0]
        lines.append(f"{key}={parts[key]}")
    return sha256_text("\n".join(lines))


def load_fingerprint_table(path: Path = FINGERPRINT_TABLE_PATH) -> Dict[str, str]:
    return dict(json.loads(path.read_text("utf-8"))["fingerprints"])
