"""llms.txt에서 가이드 문서 URL 목록과 설명을 추출한다."""

import json
import re
from pathlib import Path

import requests

LLMS_TXT_URL = "https://docs.riido.io/llms.txt"
OUTPUT_PATH = Path("data/pages.json")

# - [제목](URL): 설명
LINE_PATTERN = re.compile(r"^-\s*\[(.+?)\]\((https?://[^)]+)\):\s*(.*)$", re.M)


def parse_pages(text: str) -> list[dict]:
    pages = []
    for order, (title, url, summary) in enumerate(LINE_PATTERN.findall(text), 1):
        category = url.replace("https://docs.riido.io/", "").split("/")[0]
        pages.append({
            "order": order,
            "title": title.strip(),
            "url": url.strip(),
            "summary": summary.strip(),
            "category": category,
        })
    return pages


def main() -> None:
    response = requests.get(LLMS_TXT_URL, timeout=10)
    response.raise_for_status()

    pages = parse_pages(response.text)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"페이지 수: {len(pages)}")
    print(f"URL 중복: {len(pages) - len({p['url'] for p in pages})}")
    print(f"카테고리: {sorted({p['category'] for p in pages})}")


if __name__ == "__main__":
    main()