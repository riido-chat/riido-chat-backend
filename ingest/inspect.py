"""수집 원본을 분석해 정제·청킹 전략 결정용 통계를 산출한다."""

import json
import re
from collections import Counter
from pathlib import Path

MANIFEST_PATH = Path("data/manifest.json")
DATA_DIR = Path("data")
STATS_PATH = Path("data/stats.json")

GITBOOK_TAGS = ["hint", "tabs", "tab", "stepper", "step", "columns", "column", "embed"]


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    headers = Counter()
    gitbook = Counter()
    lengths = []
    images = empty_alt = md_tables = html_tables = details = 0
    first_lines = Counter()

    for entry in manifest:
        text = (DATA_DIR / entry["path"]).read_text(encoding="utf-8")
        lengths.append(len(text))

        for level in re.findall(r"^(#{1,4}) ", text, re.M):
            headers[len(level)] += 1

        for tag in GITBOOK_TAGS:
            gitbook[tag] += len(re.findall(r"\{%\s*" + tag + r"\b", text))

        alts = re.findall(r'<img[^>]*alt="(.*?)"', text)
        images += len(alts)
        empty_alt += sum(1 for a in alts if not a.strip())

        md_tables += len(re.findall(r"^\|.+\|$", text, re.M))
        html_tables += len(re.findall(r"<table", text))
        details += len(re.findall(r"<details>", text))

        for line in text.splitlines()[:3]:
            if line.strip():
                first_lines[line.strip()[:60]] += 1

    stats = {
        "pages": len(manifest),
        "length": {
            "avg": sum(lengths) // len(lengths),
            "min": min(lengths),
            "max": max(lengths),
        },
        "headers": dict(sorted(headers.items())),
        "images": {"total": images, "empty_alt": empty_alt},
        "tables": {"markdown_rows": md_tables, "html": html_tables},
        "details_blocks": details,
        "gitbook_syntax": {k: v for k, v in sorted(gitbook.items()) if v},
        "repeated_first_lines": {k: v for k, v in first_lines.items() if v > 1},
    }

    STATS_PATH.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()