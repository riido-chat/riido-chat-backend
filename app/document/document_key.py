"""문서 그룹 안에서 문서를 식별하는 키와 출처 상수를 한곳에서 정의한다.

- source_type 값과 콘솔 업로드 문서의 canonical_uri 스킴은 이 module만 정의한다.
- 문서명 정규화는 migration 20260904_06의 backfill과 업로드 접수가 함께 사용한다.
- GitBook 문서 키 추출은 같은 migration의 backfill SQL과 같은 규칙을 사용한다.
"""

import re
import unicodedata


# 1차 문서 그룹. 2차에서 그룹이 늘어나면 호출자가 group_key를 전달한다.
DEFAULT_DOCUMENT_GROUP_KEY = "HELP_CHATBOT"
DEFAULT_DOCUMENT_GROUP_NAME = "도움말 챗봇 이용가이드"
DEFAULT_DOCUMENT_GROUP_CONSUMER_KEY = "HELP_CHATBOT"

# source_type은 출처만 나타낸다. 문서 형식은 mime_type과 parser_name이 담당한다.
SOURCE_TYPE_GITBOOK = "GITBOOK"
SOURCE_TYPE_UPLOAD = "UPLOAD"

# 콘솔 업로드 문서의 원문 위치자. 사용자에게 노출하지 않는 내부 스킴이다.
CONSOLE_URI_SCHEME = "riido-doc"

UPLOAD_DOCUMENT_KEY_PREFIX = "upload/"
DEFAULT_GITBOOK_ROOT_URL = "https://docs.riido.io"
GITBOOK_DOCUMENT_URL_SUFFIX = re.compile(r"\.md$")

_WHITESPACE = re.compile(r"\s+")
_DISALLOWED_CHARACTERS = re.compile(r"[^0-9a-z가-힣\-]")
_REPEATED_HYPHENS = re.compile(r"-{2,}")


def normalize_document_title(title: str) -> str:
    """콘솔 문서명을 문서 키 조각으로 정규화한다.

    NFC 정규화, 앞뒤 공백 제거, 영문 소문자화, 공백의 hyphen 치환,
    연속 hyphen 축약을 거치고 영문 소문자, 숫자, 한글, hyphen만 남긴다.
    """

    normalized = unicodedata.normalize("NFC", title).strip().lower()
    normalized = _WHITESPACE.sub("-", normalized)
    normalized = _DISALLOWED_CHARACTERS.sub("", normalized)
    normalized = _REPEATED_HYPHENS.sub("-", normalized).strip("-")
    if not normalized:
        raise ValueError("문서명을 정규화한 결과가 비어 있습니다.")
    return normalized


def build_upload_document_key(title: str) -> str:
    """콘솔 업로드 문서의 그룹 내 문서 키를 만든다."""

    return f"{UPLOAD_DOCUMENT_KEY_PREFIX}{normalize_document_title(title)}"


def normalize_gitbook_root_url(root_url: str) -> str:
    """GitBook 루트 URL의 끝 슬래시를 떼어 비교 가능한 형태로 만든다."""

    normalized = root_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("GitBook 루트 URL이 비어 있습니다.")
    return normalized


def build_gitbook_document_key(
    canonical_uri: str,
    root_url: str = DEFAULT_GITBOOK_ROOT_URL,
) -> str:
    """GitBook 페이지 URL에서 그룹 내 문서 키(루트 기준 상대 경로)를 만든다."""

    root = normalize_gitbook_root_url(root_url)
    path = canonical_uri
    if path.startswith(f"{root}/"):
        path = path[len(root) + 1 :]
    return GITBOOK_DOCUMENT_URL_SUFFIX.sub("", path)


def build_console_canonical_uri(group_key: str, document_key: str) -> str:
    """콘솔 업로드 문서의 서버 생성 canonical_uri를 만든다."""

    return f"{CONSOLE_URI_SCHEME}://{group_key}/{document_key}"
