"""옛 import 경로 호환용. 지문 계산 규칙과 지문 표는 app/prompts/fingerprints.py 로 옮겼다.

새 코드는 app.prompts.fingerprints 를 import 한다. 이 모듈은 상위 패키지(answering, chat,
question_grouping)를 import 하지 않으며, 기존 테스트가 쓰는 이름만 다시 내보낸다.
"""

from app.prompts.fingerprints import (
    FINGERPRINT_TABLE_PATH,
    GENERATION_COMPOSITE_PARTS,
    canonical_json,
    composite_fingerprint,
    load_fingerprint_table,
    sha256_json,
    sha256_text,
    text_format_of,
)

__all__ = [
    "FINGERPRINT_TABLE_PATH",
    "GENERATION_COMPOSITE_PARTS",
    "canonical_json",
    "composite_fingerprint",
    "load_fingerprint_table",
    "sha256_json",
    "sha256_text",
    "text_format_of",
]
