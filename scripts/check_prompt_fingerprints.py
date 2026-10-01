#!/usr/bin/env python3
"""프롬프트 지문 표의 기존 키가 바뀌거나 지워진 변경을 막는다(추가만 허용).

표: app/core/prompt_fingerprints.json 의 "fingerprints" 객체. 키는 `<component>@<version>`,
값은 모델에 보내는 내용의 sha256 이다. 같은 판의 내용이 바뀌면 판을 올리고 새 키를 더해야
하므로, 기준 ref 시점의 키는 현재 작업 트리에서도 같은 값으로 남아 있어야 한다.

검사 규칙:

1. 현재 표가 읽히고, 모든 값이 64자리 소문자 hex 여야 한다.
2. 기준 ref 시점의 표에 있던 키가 현재 표에 없으면 실패한다.
3. 기준 ref 시점의 표에 있던 키의 값이 바뀌었으면 실패한다.
4. 기준 ref 시점에 표가 없으면(최초 도입) 1만 확인한다.

기준 ref 는 인자, PROMPT_FINGERPRINT_BASE_REF 환경변수, origin/develop 순으로 정하고,
그 ref 와 HEAD 의 merge-base 시점 표와 비교한다. 지문이 실제 코드와 같은지는
tests/test_prompt_fingerprints.py 가 확인한다. 표준 라이브러리만 사용하고 네트워크를 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
TABLE_FILE = "app/core/prompt_fingerprints.json"
DEFAULT_BASE_REF = "origin/develop"
BASE_REF_ENV = "PROMPT_FINGERPRINT_BASE_REF"

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class TableFormatError(ValueError):
    """지문 표를 읽을 수 없거나 형식이 틀렸을 때."""


def parse_table(source: str) -> Dict[str, str]:
    """표 원문에서 fingerprints 객체를 꺼낸다."""

    try:
        document = json.loads(source)
    except json.JSONDecodeError as error:
        raise TableFormatError(f"JSON 으로 읽지 못했습니다: {error}") from error
    if not isinstance(document, dict) or not isinstance(
        document.get("fingerprints"), dict
    ):
        raise TableFormatError('"fingerprints" 객체가 없습니다.')
    table = document["fingerprints"]
    for key, value in table.items():
        if not isinstance(value, str):
            raise TableFormatError(f"{key} 의 값이 문자열이 아닙니다.")
    return dict(table)


def evaluate(
    base_table: Optional[Mapping[str, str]],
    head_table: Mapping[str, str],
) -> List[str]:
    """검사 규칙을 적용해 실패 사유 목록을 돌려준다. 빈 목록이면 통과다.

    base_table 이 None 이면 기준 ref 에 표가 없는 것(최초 도입)으로 본다.
    """

    errors: List[str] = []
    for key, value in sorted(head_table.items()):
        if not _SHA256_HEX.match(value):
            errors.append(f"{key} 의 값이 64자리 소문자 sha256 hex 가 아닙니다: {value}")

    if base_table is None:
        return errors

    for key in sorted(base_table):
        if key not in head_table:
            errors.append(
                f"기존 키 {key} 가 지워졌습니다. 옛 판의 지문은 이력으로 남겨 두세요."
            )
        elif head_table[key] != base_table[key]:
            errors.append(
                f"기존 키 {key} 의 지문이 바뀌었습니다 "
                f"({base_table[key][:12]} -> {head_table[key][:12]}). "
                "내용을 바꿨다면 판을 올리고 새 키를 추가하세요."
            )
    return errors


def added_keys(
    base_table: Optional[Mapping[str, str]],
    head_table: Mapping[str, str],
) -> List[str]:
    if base_table is None:
        return sorted(head_table)
    return sorted(set(head_table) - set(base_table))


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def base_table_source(base_ref: str, root: Path = ROOT) -> Optional[str]:
    """base_ref 와 HEAD 의 merge-base 시점 표 원문. 그 시점에 파일이 없으면 None."""

    merge_base = _git(root, "merge-base", base_ref, "HEAD").strip()
    try:
        return _git(root, "show", f"{merge_base}:{TABLE_FILE}")
    except subprocess.CalledProcessError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="프롬프트 지문 표가 추가만 되었는지 확인합니다."
    )
    parser.add_argument(
        "base_ref",
        nargs="?",
        default=os.environ.get(BASE_REF_ENV) or DEFAULT_BASE_REF,
        help=f"비교 기준 ref (기본: ${BASE_REF_ENV} 또는 {DEFAULT_BASE_REF})",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None, root: Path = ROOT) -> int:
    args = build_parser().parse_args(argv)
    base_ref = args.base_ref

    try:
        base_source = base_table_source(base_ref, root)
    except subprocess.CalledProcessError as error:
        print(
            f"기준 ref '{base_ref}' 와 비교하지 못했습니다: {error.stderr.strip()}",
            file=sys.stderr,
        )
        return 1

    try:
        head_table = parse_table((root / TABLE_FILE).read_text("utf-8"))
        base_table = None if base_source is None else parse_table(base_source)
    except (OSError, TableFormatError) as error:
        print(f"실패: {TABLE_FILE} 를 읽지 못했습니다: {error}", file=sys.stderr)
        return 1

    errors = evaluate(base_table, head_table)

    print(f"기준 ref: {base_ref}")
    base_count = "표 없음" if base_table is None else f"{len(base_table)}개"
    print(f"기준 키: {base_count} / 현재 키: {len(head_table)}개")
    new_keys = added_keys(base_table, head_table)
    print(f"추가된 키: {len(new_keys)}개")
    for key in new_keys:
        print(f"  - {key}")

    if errors:
        for message in errors:
            print(f"실패: {message}", file=sys.stderr)
        return 1
    print("통과: 프롬프트 지문 표의 기존 키가 그대로입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
