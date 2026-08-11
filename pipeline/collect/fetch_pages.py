"""pages.json의 URL을 순회하여 마크다운 원문을 수집한다."""

import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import requests

PAGES_PATH = Path("data/pages.json")
RAW_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/manifest.json")
FAILED_PATH = Path("data/failed.json")
REQUEST_DELAY = 1.0


def to_local_path(url: str) -> Path:
    path = urlparse(url).path.lstrip("/")
    path = unicodedata.normalize("NFC", path)
    path = re.sub(r"[^\w\-./가-힣]", "-", path)
    return RAW_DIR / path


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    pages = json.loads(PAGES_PATH.read_text(encoding="utf-8"))
    manifest, failed = [], []

    for i, page in enumerate(pages, 1):
        url = page["url"]
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            text = response.text
        except Exception as e:
            failed.append({"url": url, "error": str(e)})
            print(f"[{i}/{len(pages)}] 실패: {url}")
            continue

        local_path = to_local_path(url)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(text, encoding="utf-8")

        manifest.append({
            "doc_id": sha256(url)[:12],
            "url": url,
            "path": str(local_path.relative_to("data")),
            "title": page["title"],
            "category": page["category"],
            "order": page["order"],
            "sha256": sha256(text),
            "bytes": len(text.encode("utf-8")),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        print(f"[{i}/{len(pages)}] {page['title']}")
        time.sleep(REQUEST_DELAY)

    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    FAILED_PATH.write_text(
        json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n수집 성공: {len(manifest)} / 실패: {len(failed)}")

    by_hash = {}
    for m in manifest:
        by_hash.setdefault(m["sha256"], []).append(m["path"])
    dups = [paths for paths in by_hash.values() if len(paths) > 1]
    print(f"중복 페이지: {len(dups)}건")
    for paths in dups:
        print(f"  {paths}")


if __name__ == "__main__":
    main()