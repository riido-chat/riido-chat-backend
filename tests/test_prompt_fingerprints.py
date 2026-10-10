"""현재 버전의 프롬프트와 출력 스키마가 지문 표(app/prompts/fingerprints.json)와 같은지 고정한다.

프롬프트나 스키마를 고쳤으면 기존 키의 값을 바꾸지 말고, 버전 상수를 올린 뒤 새 키를 표에 추가한다.
옛 키는 이력으로 남기며 지금 코드로 다시 계산할 수 있을 필요가 없다.

현재 코드가 쓰는 버전과 실제로 보내는 지시문, 출력 스키마는 호출부(answering, chat,
question_grouping)가 알고 있으므로 그 조합(current_fingerprints)은 여기 둔다.
app/prompts 는 상위 패키지를 import 하지 않는다.
"""

import re
import unittest
from typing import Dict

from app.answering import generator
from app.answering.generator import GENERATION_PROMPT_VERSION
from app.answering.models import GenerationResult, GenerationSourcePlan
from app.chat import query_rewrite
from app.chat.query_rewrite import QueryRewriteOutput
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
from app.question_grouping import prompt_v7_2
from app.question_grouping.constants import JUDGE_PROMPT_VERSION
from app.question_grouping.judge_client import build_judge_request
from app.question_grouping.prompt_v7_2 import JUDGE_INSTRUCTIONS_SHA256


_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_KEY = re.compile(r"^[a-z_.]+@[A-Za-z0-9_.-]+$")


def current_fingerprints() -> Dict[str, str]:
    """현재 코드가 쓰는 버전의 키와 실제로 보내는 내용의 지문."""

    planning = generator.SOURCE_PLANNING_PROMPT_VERSION
    planning_repair = generator.SOURCE_PLANNING_REPAIR_PROMPT_VERSION
    answer = generator.ANSWER_PROMPT_VERSION
    answer_repair = generator.ANSWER_REPAIR_PROMPT_VERSION
    generation_parts = {
        f"generation.source_planning@{planning}": sha256_text(
            generator.SOURCE_PLANNING_PROMPT_V11
        ),
        f"generation.source_planning_repair@{planning_repair}": sha256_text(
            generator.SOURCE_PLANNING_REPAIR_PROMPT_V11
        ),
        # 재시도 호출도 같은 스키마를 쓰므로 스키마 키는 처음 버전에만 둔다.
        f"generation.source_planning.output_schema@{planning}": sha256_json(
            text_format_of(GenerationSourcePlan)
        ),
        f"generation.answer@{answer}": sha256_text(generator.ANSWER_PROMPT_V17),
        f"generation.answer_repair@{answer_repair}": sha256_text(
            generator.ANSWER_REPAIR_PROMPT_V17
        ),
        f"generation.answer.output_schema@{answer}": sha256_json(
            text_format_of(GenerationResult)
        ),
    }

    rewrite = query_rewrite.QUERY_REWRITE_PROMPT_VERSION
    judge = JUDGE_PROMPT_VERSION
    judge_format = build_judge_request({})["text"]["format"]
    return {
        **generation_parts,
        f"generation@{GENERATION_PROMPT_VERSION}": composite_fingerprint(
            generation_parts
        ),
        f"rewrite@{rewrite}": sha256_text(query_rewrite.QUERY_REWRITE_PROMPT_V4),
        f"rewrite.output_schema@{rewrite}": sha256_json(
            text_format_of(QueryRewriteOutput)
        ),
        f"judge@{judge}": sha256_text(prompt_v7_2.JUDGE_INSTRUCTIONS),
        f"judge.output_schema@{judge}": sha256_json(judge_format),
    }


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
                    f"{key} 가 {FINGERPRINT_TABLE_PATH.name} 에 없습니다. 버전을 올렸다면 "
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
                    "해당 버전 상수(묶음 버전 GENERATION_PROMPT_VERSION 포함)를 올리고 새 키를 "
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
