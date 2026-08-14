"""clean manifest와 Markdown을 NormalizedDocument로 읽는다."""

import json
from pathlib import Path
from typing import List, Union

from pipeline.document.models import NormalizedDocument


DEFAULT_MANIFEST_PATH = Path("data/clean_manifest.json")
REQUIRED_FIELDS = ("doc_id", "title", "url", "path")


def load_normalized_documents(
    manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
) -> List[NormalizedDocument]:
    """manifest에 명시된 정제 Markdown을 순서대로 읽는다."""

    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"clean manifest를 찾을 수 없습니다: {manifest_path}")

    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"clean manifest가 올바른 JSON이 아닙니다: {manifest_path}") from exc

    if not isinstance(entries, list):
        raise ValueError("clean manifest의 최상위 값은 배열이어야 합니다.")

    documents = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest entry #{index}는 객체여야 합니다.")

        missing_fields = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing_fields:
            fields = ", ".join(missing_fields)
            raise ValueError(f"manifest entry #{index}에 필수 필드가 없습니다: {fields}")

        markdown_path = manifest_path.parent / entry["path"]
        if not markdown_path.is_file():
            raise FileNotFoundError(
                f"manifest entry #{index}의 Markdown을 찾을 수 없습니다: {markdown_path}"
            )

        content = markdown_path.read_text(encoding="utf-8")
        documents.append(
            NormalizedDocument(
                document_id=entry["doc_id"],
                title=entry["title"],
                source_url=entry["url"],
                category=entry.get("category"),
                content=content,
            )
        )

    return documents
