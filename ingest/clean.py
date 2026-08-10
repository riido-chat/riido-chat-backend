"""수집 원본(L0)을 정제해 검색에 사용할 텍스트(L1)를 생성한다."""

import json
import re
from collections import defaultdict
from pathlib import Path

MANIFEST_PATH = Path("data/manifest.json")
DATA_DIR = Path("data")
CLEAN_DIR = Path("data/clean")
CLEAN_MANIFEST_PATH = Path("data/clean_manifest.json")

BOILERPLATE_PREFIX = "> For the complete documentation index"
SITE_ROOT = "https://docs.riido.io"


def strip_boilerplate(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith(BOILERPLATE_PREFIX):
        lines = lines[1:]
    return "\n".join(lines).lstrip()


def extract_images(text: str) -> tuple[str, list[str]]:
    """figure 블록을 본문에서 제거하고 파일 ID만 회수한다. 캡션은 본문에 남긴다."""
    images = []

    def replace(match: re.Match) -> str:
        block = match.group(0)
        images.extend(re.findall(r'src="([^"]+)"', block))
        caption = re.search(r"<figcaption>(.*?)</figcaption>", block, re.S)
        if caption:
            inner = re.sub(r"<[^>]+>", "", caption.group(1)).strip()
            if inner:
                return inner
        return ""

    text = re.sub(r"<figure>.*?</figure>", replace, text, flags=re.S)
    return text, images


def html_table_to_text(text: str) -> str:
    """HTML 표를 '헤더: 값' 형태의 행 단위 텍스트로 바꾼다."""

    def convert(match: re.Match) -> str:
        table = match.group(0)
        if 'data-view="cards"' in table:
            return ""

        rows = re.findall(r"<tr>(.*?)</tr>", table, re.S)
        if not rows:
            return ""

        def cells(row: str, tag: str) -> list[str]:
            raw = re.findall(rf"<{tag}[^>]*>(.*?)</{tag}>", row, re.S)
            return [re.sub(r"<[^>]+>", "", c).strip() for c in raw]

        headers = cells(rows[0], "th")
        lines = []
        for row in rows[1:]:
            values = cells(row, "td")
            if not any(values):
                continue
            if headers and len(headers) == len(values):
                pairs = [f"{h}: {v}" for h, v in zip(headers, values) if v]
                lines.append(" / ".join(pairs))
            else:
                lines.append(" / ".join(v for v in values if v))
        return "\n".join(lines)

    return re.sub(r"<table.*?</table>", convert, text, flags=re.S)


def strip_gitbook_syntax(text: str) -> str:
    # tab title은 속성 안에 있어 태그를 지우면 사라지므로 본문으로 꺼낸다
    text = re.sub(r'\{%\s*tab title="(.*?)"\s*%\}', r"**\1**", text)
    text = re.sub(r"\{%.*?%\}", "", text, flags=re.S)
    return text


def normalize_links(text: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\(broken://[^)]*\)", r"\1", text)
    text = re.sub(r"\]\((/[^)]*)\)", rf"]({SITE_ROOT}\1)", text)
    return text


def tidy(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    results = []

    for entry in manifest:
        text = (DATA_DIR / entry["path"]).read_text(encoding="utf-8")

        text = strip_boilerplate(text)
        text, images = extract_images(text)
        text = html_table_to_text(text)
        text = strip_gitbook_syntax(text)
        text = normalize_links(text)
        text = tidy(text)

        out_path = CLEAN_DIR / Path(entry["path"]).relative_to("raw")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")

        results.append({
            **{k: entry[k] for k in ("doc_id", "url", "title", "category", "order")},
            "path": str(out_path.relative_to("data")),
            "images": images,
            "bytes": len(text.encode("utf-8")),
        })

    # 정제 후 본문이 같은 페이지를 canonical 1건으로 묶는다
    by_text = defaultdict(list)
    for r in results:
        body = (DATA_DIR / r["path"]).read_text(encoding="utf-8")
        by_text[body].append(r)

    canonical = []
    for group in by_text.values():
        group.sort(key=lambda r: r["order"])
        head = dict(group[0])
        head["source_urls"] = [r["url"] for r in group]
        canonical.append(head)
    canonical.sort(key=lambda r: r["order"])

    CLEAN_MANIFEST_PATH.write_text(
        json.dumps(canonical, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    before = sum(e["bytes"] for e in manifest)
    after = sum(r["bytes"] for r in results)
    print(f"정제 완료: {len(results)}건 → canonical {len(canonical)}건")
    print(f"용량: {before:,} → {after:,} bytes ({after / before:.0%})")
    for r in canonical:
        if len(r["source_urls"]) > 1:
            print(f"중복 통합: {r['title']} ← {len(r['source_urls'])}개 URL")


if __name__ == "__main__":
    main()