"""프롬프트 버전 파일을 읽는 등록부.

본문은 `app/prompts/<component>/<version>.md` 에 둔다. component 와 version 은 지문 표
(fingerprints.json)의 키 `<component>@<version>` 과 같은 이름을 쓴다. 파일은 추가만 하고
기존 파일을 고치거나 지우지 않는다. 내용을 바꾸려면 새 버전 파일을 만들고 지문 표에 새 키를
더한다. 기존 파일이 바뀌면 tests/test_prompt_registry.py 가 표의 지문과 달라 실패한다.

파일 바이트를 UTF-8 로 그대로 읽는다. 줄바꿈 변환이나 앞뒤 공백 정리를 하지 않으므로
파일 끝 줄바꿈 유무도 본문의 일부다.

이 모듈은 app 의 다른 패키지를 import 하지 않는다.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple


PROMPTS_DIR = Path(__file__).resolve().parent
PROMPT_FILE_SUFFIX = ".md"

_COMPONENT = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class PromptNotFoundError(LookupError):
    """등록부에 없는 component 나 version 을 찾았을 때."""


def _component_dir(component: str) -> Path:
    if not _COMPONENT.match(component):
        raise PromptNotFoundError(f"프롬프트 component 이름 형식이 아닙니다: {component!r}")
    return PROMPTS_DIR / component


@lru_cache(maxsize=None)
def get(component: str, version: str) -> str:
    """component 의 version 본문. 같은 인자는 한 번만 읽는다."""

    if not _VERSION.match(version):
        raise PromptNotFoundError(f"프롬프트 버전 이름 형식이 아닙니다: {version!r}")
    path = _component_dir(component) / f"{version}{PROMPT_FILE_SUFFIX}"
    if not path.is_file():
        raise PromptNotFoundError(f"등록되지 않은 프롬프트입니다: {component}@{version}")
    return path.read_bytes().decode("utf-8")


def versions(component: str) -> Tuple[str, ...]:
    """component 에 등록된 버전 이름. 파일 이름 순서이며 출시 순서를 뜻하지 않는다."""

    directory = _component_dir(component)
    if not directory.is_dir():
        raise PromptNotFoundError(f"등록되지 않은 프롬프트 component 입니다: {component}")
    return tuple(
        sorted(
            path.name[: -len(PROMPT_FILE_SUFFIX)]
            for path in directory.iterdir()
            if path.is_file() and path.name.endswith(PROMPT_FILE_SUFFIX)
        )
    )


def components() -> Tuple[str, ...]:
    """버전 파일이 하나 이상 있는 component 이름."""

    found: List[str] = []
    for directory in sorted(PROMPTS_DIR.iterdir()):
        if not directory.is_dir() or not _COMPONENT.match(directory.name):
            continue
        if any(path.name.endswith(PROMPT_FILE_SUFFIX) for path in directory.iterdir()):
            found.append(directory.name)
    return tuple(found)
