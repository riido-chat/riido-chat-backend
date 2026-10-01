"""현재 판의 프롬프트와 출력 스키마가 지문 표(app/core/prompt_fingerprints.json)와 같은지 고정한다.

프롬프트나 스키마를 고쳤으면 기존 키의 값을 바꾸지 말고, 판(버전 상수)을 올린 뒤 새 키를 표에 추가한다.
옛 키는 이력으로 남기며 지금 코드로 다시 계산할 수 있을 필요가 없다.
"""

import re
import unittest

from app.answering.generator import GENERATION_PROMPT_VERSION
from app.core.prompt_fingerprints import (
    FINGERPRINT_TABLE_PATH,
    GENERATION_COMPOSITE_PARTS,
    canonical_json,
    composite_fingerprint,
    current_fingerprints,
    load_fingerprint_table,
)
from app.question_grouping.constants import JUDGE_PROMPT_VERSION
from app.question_grouping.prompt_v7_2 import JUDGE_INSTRUCTIONS_SHA256


_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_KEY = re.compile(r"^[a-z_.]+@[A-Za-z0-9_.-]+$")


class PromptFingerprintTableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.table = load_fingerprint_table()
        self.current = current_fingerprints()

    def test_every_current_version_is_in_table(self) -> None:
        for key in sorted(self.current):
            with self.subTest(key=key):
                self.assertIn(
                    key,
                    self.table,
                    f"{key} 가 {FINGERPRINT_TABLE_PATH.name} 에 없습니다. 판을 올렸다면 "
                    "새 키와 지문을 표에 추가하세요(기존 키는 그대로 둡니다).",
                )

    def test_current_text_matches_table_hash(self) -> None:
        for key, digest in sorted(self.current.items()):
            if key not in self.table:
                continue
            with self.subTest(key=key):
                self.assertEqual(
                    self.table[key],
                    digest,
                    f"{key} 의 실제 내용이 표의 지문과 다릅니다. 프롬프트나 스키마를 고쳤다면 "
                    "해당 판 상수(묶음 판 GENERATION_PROMPT_VERSION 포함)를 올리고 새 키를 "
                    "추가하세요. 기존 키의 값은 바꾸지 않습니다. 코드를 고치지 않았는데 스키마 "
                    "지문만 다르면 openai 나 pydantic 업그레이드로 보내는 스키마가 바뀐 것입니다.",
                )

    def test_table_entries_are_well_formed(self) -> None:
        for key, digest in self.table.items():
            with self.subTest(key=key):
                self.assertRegex(key, _KEY)
                self.assertRegex(digest, _SHA256_HEX)

    def test_judge_entry_reuses_existing_instructions_hash(self) -> None:
        self.assertEqual(
            JUDGE_INSTRUCTIONS_SHA256,
            self.table[f"judge@{JUDGE_PROMPT_VERSION}"],
        )

    def test_generation_composite_is_built_from_table_entries(self) -> None:
        parts = {
            key: digest
            for key, digest in self.current.items()
            if key.split("@", 1)[0] in GENERATION_COMPOSITE_PARTS
        }
        self.assertEqual(len(GENERATION_COMPOSITE_PARTS), len(parts))
        self.assertEqual(
            self.table[f"generation@{GENERATION_PROMPT_VERSION}"],
            composite_fingerprint({key: self.table[key] for key in parts}),
        )


class CompositeFingerprintTest(unittest.TestCase):
    def _parts(self, **overrides: str) -> dict:
        parts = {f"{component}@v1": "0" * 64 for component in GENERATION_COMPOSITE_PARTS}
        parts.update(overrides)
        return parts

    def test_sub_prompt_change_changes_composite(self) -> None:
        base = composite_fingerprint(self._parts())
        changed = composite_fingerprint(
            self._parts(**{"generation.answer@v1": "1" * 64})
        )

        self.assertNotEqual(base, changed)

    def test_sub_version_change_changes_composite(self) -> None:
        parts = self._parts()
        renamed = dict(parts)
        renamed["generation.answer@v2"] = renamed.pop("generation.answer@v1")

        self.assertNotEqual(composite_fingerprint(parts), composite_fingerprint(renamed))

    def test_missing_part_is_rejected(self) -> None:
        parts = self._parts()
        del parts["generation.answer_repair@v1"]

        with self.assertRaisesRegex(ValueError, "generation.answer_repair"):
            composite_fingerprint(parts)

    def test_canonical_json_sorts_keys_without_whitespace(self) -> None:
        self.assertEqual(
            '{"a":[1,"한"],"b":null}',
            canonical_json({"b": None, "a": [1, "한"]}),
        )


if __name__ == "__main__":
    unittest.main()
