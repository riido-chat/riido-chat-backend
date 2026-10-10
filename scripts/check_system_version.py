#!/usr/bin/env python3
"""동작 관련 파일이 바뀌었는데 시스템 버전이 그대로인 변경을 막는다.

감시 경로(응답 동작에 영향을 주는 코드):

- app/answering/                 생성 모델, 프롬프트, 근거 규칙
- app/chat/query_rewrite.py      재작성 모델, 프롬프트
- app/prompts/                   프롬프트 본문 버전 파일, 지문 표
- app/question_grouping/         판별 모델, 프롬프트, payload, 캐시 게이트
- app/retrieval/                 임베딩 모델, 임베딩 문장, 검색 파라미터
- app/document/clean.py          정제 규칙(검색 텍스트)
- app/document/section_parser.py Section 경계
- app/document/chunker.py        Chunk 구성
- app/document/chunking_config.py 청킹 설정 판

tests/, docs/, evaluation/ 과 수집과 색인 실행 흐름(app/indexing/, 그 밖의 app/document/)은
감시하지 않는다.

검사 규칙:

1. 항상: 현재 SYSTEM_VERSION 이 MAJOR.MINOR.PATCH 형식이고
   evaluation/SYSTEM_VERSIONS.md 버전 목록에 같은 버전의 행이 있어야 한다.
2. 감시 경로가 바뀌었으면: app/core/system_version.py 도 바뀌었고,
   새 버전이 기준 ref 의 버전보다 커야 한다.
3. 동작 동일성 기록이 이번 변경에 포함되면 기준 스냅샷 무수정, 파일 지문, 합성 동작 검증을
   모두 통과한 변경만 버전 상승 예외로 인정한다.

기준 ref 는 인자, SYSTEM_VERSION_BASE_REF 환경변수, origin/develop 순으로 정한다.
검사 자체는 표준 라이브러리만 사용한다. 동작 동일성 예외 검증에는 앱 의존성이 필요하다.
네트워크와 실제 모델 API 를 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = "app/core/system_version.py"
VERSIONS_DOC = "evaluation/SYSTEM_VERSIONS.md"
EQUIVALENCE_FILE = "evaluation/system_version_equivalence.json"
EQUIVALENCE_TESTS = ("tests.test_help_chatbot_identity", "tests.test_chat_contract_snapshot")
EQUIVALENCE_SNAPSHOTS = (
    "tests/fixtures/help_chatbot_identity.json",
    "tests/fixtures/chat_contract_schema.json",
)
DEFAULT_BASE_REF = "origin/develop"
BASE_REF_ENV = "SYSTEM_VERSION_BASE_REF"

# 끝이 "/" 면 디렉터리 전체, 아니면 파일 하나를 감시한다.
WATCHED_PATHS: Tuple[str, ...] = (
    "app/answering/",
    "app/chat/query_rewrite.py",
    "app/prompts/",
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

    버전 규칙 표(MAJOR/MINOR/PATCH)와 헤더와 구분선은 첫 칸이 semver 가 아니라 빠진다.
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
    equivalent_changes: Sequence[str] = (),
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

    watched = watched_changes(set(changed_files) - set(equivalent_changes))
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



def equivalent_changes(base_ref: str, changed_files: Sequence[str], head_version: Optional[str]) -> List[str]:
    """이번 PR 에 추가하거나 갱신한 동작 동일성 기록만 사용한다.

    기준 커밋, 버전, 변경 파일 지문, 무수정 스냅샷을 확인하고 고정된 두 검증을 실행한다.
    이후 PR 은 기록이 diff 에 없거나 변경 파일 지문이 달라 같은 예외를 재사용하지 못한다.
    실제 모델 API 를 부르지 않는 합성 시나리오 검증이다.
    """

    if EQUIVALENCE_FILE not in changed_files:
        return []
    record = json.loads((ROOT / EQUIVALENCE_FILE).read_text("utf-8"))
    base_commit = _git("merge-base", base_ref, "HEAD").strip()
    if record.get("baseCommit") != base_commit or record.get("systemVersion") != head_version:
        raise ValueError("동작 동일성 기록의 기준 커밋이나 시스템 버전이 다릅니다.")
    if base_system_version(base_ref) != head_version:
        raise ValueError("동작 동일성 예외는 기준 시스템 버전을 유지할 때만 허용합니다.")
    runtime = {path for path in changed_files if path.startswith("app/") or path == "scripts/deploy.sh"}
    proof_files = {name.replace(".", "/") + ".py" for name in EQUIVALENCE_TESTS}
    files = record.get("files", {})
    if set(files) != runtime | proof_files:
        raise ValueError("동작 동일성 기록의 변경 파일 목록이 실제 변경과 다릅니다.")
    for path, digest in files.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"동작 동일성 기록의 파일 지문이 다릅니다: {path}")
    for path in EQUIVALENCE_SNAPSHOTS:
        previous = _git("show", f"{base_commit}:{path}").encode("utf-8")
        current = (ROOT / path).read_bytes()
        if current != previous or hashlib.sha256(current).hexdigest() != record.get("snapshots", {}).get(path):
            raise ValueError(f"기준 동작 스냅샷이 바뀌었습니다: {path}")
    env = dict(os.environ)
    for name in ("UPDATE_IDENTITY_SNAPSHOT", "UPDATE_CHAT_CONTRACT_SNAPSHOT"):
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-m", "unittest", *EQUIVALENCE_TESTS], cwd=ROOT, env=env,
    )
    if result.returncode:
        raise ValueError("동작 동일성 또는 응답 계약 검증이 실패했습니다.")
    return sorted(runtime)


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
    try:
        equivalent = equivalent_changes(base_ref, changed_files, head_version)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"실패: 동작 동일성 예외 검증: {error}", file=sys.stderr)
        return 1
    errors = evaluate(changed_files, base_version, head_version, documented_versions, equivalent)
    if equivalent:
        print("동작 동일성 예외: 파일 지문, 기준 스냅샷, 동작과 응답 계약 검증 통과")

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
