"""docs.riido.io 의 페이지 목록과 본문을 읽는다.

CLI 수집 스크립트와 같은 규칙을 쓰되 파일이 아니라 값으로 돌려준다.
재탐색이 이 함수를 쓴다.
"""

import time
from dataclasses import dataclass
from typing import List

import requests

from app.document.document_key import normalize_gitbook_root_url
from app.document.gitbook.list_urls import parse_pages


LIST_TIMEOUT_SECONDS = 10
PAGE_TIMEOUT_SECONDS = 15
# 페이지 조회는 멱등한 GET 이라 한 번은 다시 시도한다.
PAGE_RETRY_COUNT = 1
PAGE_RETRY_DELAY_SECONDS = 1.0


@dataclass(frozen=True)
class GitBookPage:
    """목록에서 읽은 페이지 한 건."""

    title: str
    url: str
    category: str


class GitBookListError(RuntimeError):
    """페이지 목록을 읽지 못했을 때 발생한다."""


def list_pages(root_url: str) -> List[GitBookPage]:
    """루트 URL의 llms.txt 에서 문서 목록을 읽는다.

    목록 조회 실패는 배치 전체의 전제가 무너진 것이므로 예외로 올린다.
    """

    root = normalize_gitbook_root_url(root_url)
    try:
        response = requests.get(
            f"{root}/llms.txt",
            timeout=LIST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        pages = parse_pages(response.text)
    except Exception as error:
        raise GitBookListError(str(error)) from error

    if not pages:
        raise GitBookListError("페이지 목록이 비어 있습니다.")

    outside = [page for page in pages if not page["url"].startswith(f"{root}/")]
    if outside:
        raise GitBookListError(
            f"루트 밖 페이지가 {len(outside)}건 있습니다: {outside[0]['url']}"
        )

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

    일시적인 네트워크 실패가 잦아 한 번은 다시 시도한다. 그래도 실패하면
    배치를 멈추지 않고 그 페이지의 실행에만 기록하도록 예외를 올린다.
    """

    last_error = None
    for attempt in range(PAGE_RETRY_COUNT + 1):
        try:
            response = requests.get(url, timeout=PAGE_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.text
        except Exception as error:
            last_error = error
            if attempt < PAGE_RETRY_COUNT:
                time.sleep(PAGE_RETRY_DELAY_SECONDS)

    raise last_error
