#!/usr/bin/env python3
"""동작 관련 파일이 바뀌었는데 시스템 버전이 그대로인 변경을 막는다.

감시 경로(응답 동작에 영향을 주는 코드):

- app/answering/                 생성 모델·프롬프트·근거 규칙
- app/chat/query_rewrite.py      재작성 모델·프롬프트
- app/question_grouping/         판별 모델·프롬프트·payload·캐시 게이트
- app/retrieval/                 임베딩 모델·임베딩 문장·검색 파라미터
- app/document/clean.py          정제 규칙(검색 텍스트)
- app/document/section_parser.py Section 경계
- app/document/chunker.py        Chunk 구성
- app/document/chunking_config.py 청킹 설정 판

tests/, docs/, evaluation/ 과 수집·색인 실행 흐름(app/indexing/, 그 밖의 app/document/)은
감시하지 않는다.

검사 규칙:

1. 항상: 현재 SYSTEM_VERSION 이 MAJOR.MINOR.PATCH 형식이고
   evaluation/SYSTEM_VERSIONS.md 버전 목록에 같은 버전의 행이 있어야 한다.
2. 감시 경로가 바뀌었으면: app/core/system_version.py 도 바뀌었고,
   새 버전이 기준 ref 의 버전보다 커야 한다.

기준 ref 는 인자, SYSTEM_VERSION_BASE_REF 환경변수, origin/develop 순으로 정한다.
표준 라이브러리만 사용하고 네트워크를 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = "app/core/system_version.py"
VERSIONS_DOC = "evaluation/SYSTEM_VERSIONS.md"
DEFAULT_BASE_REF = "origin/develop"
BASE_REF_ENV = "SYSTEM_VERSION_BASE_REF"

# 끝이 "/" 면 디렉터리 전체, 아니면 파일 하나를 감시한다.
WATCHED_PATHS: Tuple[str, ...] = (
    "app/answering/",
    "app/chat/query_rewrite.py",
    "app/question_grouping/",
    "app/retrieval/",
    "app/document/clean.py",
    "app/document/section_parser.py",
    "app/document/chunker.py",
    "app/document/chunking_config.py",
)

_SEMVER_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_VERSION_ASSIGN_PATTERN = re.compile(
    r"""^SYSTEM_VERSION\s*=\s*["']([^"']*)["']\s*$""",
    re.MULTILINE,
)

Semver = Tuple[int, int, int]


def parse_semver(value: str) -> Optional[Semver]:
    """MAJOR.MINOR.PATCH 문자열을 정수 튜플로 바꾼다. 형식이 틀리면 None."""

    match = _SEMVER_PATTERN.match(value.strip())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def is_version_greater(new: str, old: str) -> bool:
    """new 가 old 보다 큰 semver 인지 확인한다. 둘 중 하나라도 형식이 틀리면 False."""

    new_parsed = parse_semver(new)
    old_parsed = parse_semver(old)
    if new_parsed is None or old_parsed is None:
        return False
    return new_parsed > old_parsed


def read_system_version(source: str) -> Optional[str]:
    """system_version.py 원문에서 SYSTEM_VERSION 값을 꺼낸다."""

    match = _VERSION_ASSIGN_PATTERN.search(source)
    return match.group(1) if match else None


def parse_version_rows(markdown: str) -> List[str]:
    """SYSTEM_VERSIONS.md 표에서 첫 칸이 semver 인 행의 버전만 모은다.

    버전 규칙 표(MAJOR/MINOR/PATCH)와 헤더·구분선은 첫 칸이 semver 가 아니라 빠진다.
    """

    versions: List[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        first_cell = stripped.strip("|").split("|", 1)[0].strip()
        if parse_semver(first_cell) is not None:
            versions.append(first_cell)
    return versions


def is_watched(path: str) -> bool:
    return any(
        path.startswith(watched) if watched.endswith("/") else path == watched
        for watched in WATCHED_PATHS
    )


def watched_changes(changed_files: Sequence[str]) -> List[str]:
    return sorted({path for path in changed_files if is_watched(path)})


def evaluate(
    changed_files: Sequence[str],
    base_version: Optional[str],
    head_version: Optional[str],
    documented_versions: Sequence[str],
) -> List[str]:
    """검사 규칙을 적용해 실패 사유 목록을 돌려준다. 빈 목록이면 통과다.

    base_version 이 None 이면 기준 ref 에 버전 파일이 없는 것(최초 도입)으로 본다.
    """

    errors: List[str] = []

    if head_version is None:
        errors.append(f"{VERSION_FILE} 에서 SYSTEM_VERSION 을 찾지 못했습니다.")
        return errors
    if parse_semver(head_version) is None:
        errors.append(
            f"SYSTEM_VERSION '{head_version}' 이 MAJOR.MINOR.PATCH 형식이 아닙니다."
        )
        return errors
    if head_version not in documented_versions:
        errors.append(
            f"{VERSIONS_DOC} 버전 목록에 {head_version} 행이 없습니다."
        )

    watched = watched_changes(changed_files)
    if not watched:
        return errors

    changed_list = "\n".join(f"  - {path}" for path in watched)
    if VERSION_FILE not in changed_files:
        errors.append(
            "동작 관련 파일이 바뀌었는데 시스템 버전을 올리지 않았습니다. "
            f"{VERSION_FILE} 의 SYSTEM_VERSION 을 올리고 {VERSIONS_DOC} 에 행을 추가하세요.\n"
            f"바뀐 감시 파일:\n{changed_list}"
        )
    elif base_version is not None and not is_version_greater(
        head_version, base_version
    ):
        errors.append(
            f"SYSTEM_VERSION 이 기준 버전 {base_version} 보다 커야 합니다 "
            f"(현재 {head_version}).\n바뀐 감시 파일:\n{changed_list}"
        )
    return errors


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def changed_files_since(base_ref: str) -> List[str]:
    output = _git("diff", "--name-only", f"{base_ref}...HEAD")
    return [line for line in output.splitlines() if line.strip()]


def base_system_version(base_ref: str) -> Optional[str]:
    """base_ref 와 HEAD 의 merge-base 시점 버전. 그 시점에 파일이 없으면 None."""

    merge_base = _git("merge-base", base_ref, "HEAD").strip()
    try:
        source = _git("show", f"{merge_base}:{VERSION_FILE}")
    except subprocess.CalledProcessError:
        return None
    return read_system_version(source)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="동작 관련 변경에 시스템 버전 갱신과 기록 행이 있는지 확인합니다."
    )
    parser.add_argument(
        "base_ref",
        nargs="?",
        default=os.environ.get(BASE_REF_ENV) or DEFAULT_BASE_REF,
        help=f"비교 기준 ref (기본: ${BASE_REF_ENV} 또는 {DEFAULT_BASE_REF})",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    base_ref = args.base_ref

    try:
        changed_files = changed_files_since(base_ref)
        base_version = base_system_version(base_ref)
    except subprocess.CalledProcessError as error:
        print(
            f"기준 ref '{base_ref}' 와 비교하지 못했습니다: {error.stderr.strip()}",
            file=sys.stderr,
        )
        return 1

    head_version = read_system_version((ROOT / VERSION_FILE).read_text("utf-8"))
    documented_versions = parse_version_rows(
        (ROOT / VERSIONS_DOC).read_text("utf-8")
    )
    errors = evaluate(changed_files, base_version, head_version, documented_versions)

    print(f"기준 ref: {base_ref}")
    print(f"기준 버전: {base_version or '없음'} / 현재 버전: {head_version}")
    watched = watched_changes(changed_files)
    print(f"바뀐 감시 파일: {len(watched)}개")
    for path in watched:
        print(f"  - {path}")

    if errors:
        for message in errors:
            print(f"실패: {message}", file=sys.stderr)
        return 1
    print("통과: 시스템 버전 기록이 변경 사항과 맞습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
