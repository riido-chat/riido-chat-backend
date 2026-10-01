"""서버 빌드 식별자. 턴 행(rag_runs.build_version)에 남겨 어떤 코드가 답했는지 추적한다.

값은 프로세스 시작 때 BUILD_VERSION 환경변수에서 한 번 읽는다. 배포 이미지는
Dockerfile 의 BUILD_VERSION 빌드 인자로 받고, deploy 워크플로가 커밋 SHA(GITHUB_SHA)를
넘긴다. 로컬 실행이나 인자 없이 만든 이미지는 unknown 이다.
"""

import os
from typing import Mapping


BUILD_VERSION_ENV = "BUILD_VERSION"
UNKNOWN_BUILD_VERSION = "unknown"
# rag_runs.build_version 칸 길이.
MAX_BUILD_VERSION_LENGTH = 100


def read_build_version(environ: Mapping[str, str] = os.environ) -> str:
    """환경변수의 빌드 식별자. 비었거나 공백뿐이면 unknown, 칸보다 길면 자른다."""

    value = (environ.get(BUILD_VERSION_ENV) or "").strip()
    if not value:
        return UNKNOWN_BUILD_VERSION
    return value[:MAX_BUILD_VERSION_LENGTH]


BUILD_VERSION = read_build_version()
