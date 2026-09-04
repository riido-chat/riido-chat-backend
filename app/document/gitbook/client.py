"""docs.riido.io 의 페이지 목록과 본문을 읽는다.

CLI 수집 스크립트와 같은 규칙을 쓰되 파일이 아니라 값으로 돌려준다.
재탐색이 이 함수를 쓴다.
"""

from dataclasses import dataclass
from typing import List

import requests

from app.document.gitbook.list_urls import LLMS_TXT_URL, parse_pages


LIST_TIMEOUT_SECONDS = 10
PAGE_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class GitBookPage:
    """목록에서 읽은 페이지 한 건."""

    title: str
    url: str
    category: str


class GitBookListError(RuntimeError):
    """페이지 목록을 읽지 못했을 때 발생한다."""


def list_pages() -> List[GitBookPage]:
    """llms.txt 에서 문서 목록을 읽는다.

    목록 조회 실패는 배치 전체의 실패이므로 예외로 올린다.
    """

    try:
        response = requests.get(LLMS_TXT_URL, timeout=LIST_TIMEOUT_SECONDS)
        response.raise_for_status()
        pages = parse_pages(response.text)
    except Exception as error:
        raise GitBookListError(str(error)) from error

    if not pages:
        raise GitBookListError("페이지 목록이 비어 있습니다.")

    return [
        GitBookPage(
            title=page["title"],
            url=page["url"],
            category=page["category"],
        )
        for page in pages
    ]


def fetch_page(url: str) -> str:
    """페이지 하나의 Markdown 원문을 읽는다.

    한 페이지의 실패는 배치를 멈추지 않으므로 호출한 쪽이 잡는다.
    """

    response = requests.get(url, timeout=PAGE_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text
