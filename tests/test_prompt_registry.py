"""프롬프트 버전 파일 등록부(app/prompts)가 지문 표와 같은 본문을 돌려주는지 고정한다.

버전 파일은 추가만 한다. 기존 파일의 한 바이트라도 바뀌면 표의 지문과 달라 이 테스트가 실패한다.
재시도 지시문 파일은 덧붙임만 담고, 지문은 처음 지시문과 이은 전체 문자열로 계산한다.
"""

import ast
import re
import unittest
from pathlib import Path
from typing import Iterator, Tuple

from app import prompts
from app.answering import generator
from app.chat import query_rewrite
from app.prompts.fingerprints import load_fingerprint_table, sha256_text
from app.prompts.registry import PROMPTS_DIR, PromptNotFoundError
from app.question_grouping import prompt_v7_2
from app.question_grouping.constants import JUDGE_PROMPT_VERSION


ROOT = Path(__file__).resolve().parents[1]

# 지문 표 키가 아닌 버전 파일. 여러 지시문이 같은 문장으로 담는 문단이라 모델에 따로 보내지 않는다.
SHARED_SECTION_COMPONENTS = frozenset({"generation.procedure_evidence_rules"})
# 재시도 덧붙임 component 와 그 앞에 오는 처음 지시문 component.
REPAIR_BASE_COMPONENTS = {
    "generation.source_planning_repair": "generation.source_planning",
    "generation.answer_repair": "generation.answer",
}
_REPAIR_VERSION = re.compile(r"^(?P<base>.+)-repair-\d+$")
# app/prompts 와 app/core 가 import 하면 안 되는 상위 패키지.
UPPER_PACKAGES = ("app.answering", "app.chat", "app.question_grouping")


def _registered() -> Iterator[Tuple[str, str]]:
    for component in prompts.components():
        for version in prompts.versions(component):
            yield component, version


def _registered_text(component: str, version: str) -> str:
    """지문 표가 가리키는 실제 전송 문자열. 재시도 지시문은 처음 지시문과 잇는다."""

    text = prompts.get(component, version)
    base_component = REPAIR_BASE_COMPONENTS.get(component)
    if base_component is None:
        return text
    match = _REPAIR_VERSION.match(version)
    if match is None:
        raise AssertionError(f"재시도 버전 이름이 <처음 버전>-repair-N 형식이 아닙니다: {version}")
    base = prompts.get(base_component, match.group("base"))
    return generator.with_repair_instructions(base, text)


class PromptRegistryTableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.table = load_fingerprint_table()

    def test_every_registered_prompt_matches_table_hash(self) -> None:
        for component, version in _registered():
            if component in SHARED_SECTION_COMPONENTS:
                continue
            key = f"{component}@{version}"
            with self.subTest(key=key):
                self.assertIn(key, self.table, f"{key} 의 지문을 표에 추가하세요.")
                self.assertEqual(
                    self.table[key],
                    sha256_text(_registered_text(component, version)),
                    f"{key} 버전 파일이 표의 지문과 다릅니다. 기존 버전 파일은 고치지 말고 "
                    "새 버전 파일과 새 키를 추가하세요.",
                )

    def test_every_table_prompt_key_has_a_file(self) -> None:
        for key in sorted(self.table):
            component, version = key.split("@", 1)
            if component.endswith(".output_schema") or component == "generation":
                continue
            with self.subTest(key=key):
                self.assertIn(version, prompts.versions(component))

    def test_registers_the_current_twelve_keys(self) -> None:
        self.assertEqual(12, len(self.table))
        self.assertEqual(
            [
                ("generation.answer", "v23"),
                ("generation.answer_repair", "v23-repair-1"),
                ("generation.procedure_evidence_rules", "v1"),
                ("generation.source_planning", "v34"),
                ("generation.source_planning_repair", "v34-repair-1"),
                ("judge", "jev-cache-choice-context-v1"),
                ("judge", "question-grouping-v7-2"),
                ("rewrite", "v10"),
            ],
            list(_registered()),
        )


class CurrentPromptAliasTest(unittest.TestCase):
    """호출부의 기존 상수 이름이 현재 버전 파일 본문을 가리키는지 확인한다."""

    def test_generation_aliases_read_current_versions(self) -> None:
        self.assertEqual(
            prompts.get("generation.source_planning", generator.SOURCE_PLANNING_PROMPT_VERSION),
            generator.SOURCE_PLANNING_PROMPT_V11,
        )
        self.assertEqual(
            prompts.get("generation.answer", generator.ANSWER_PROMPT_VERSION),
            generator.ANSWER_PROMPT_V17,
        )
        self.assertEqual(
            _registered_text(
                "generation.source_planning_repair",
                generator.SOURCE_PLANNING_REPAIR_PROMPT_VERSION,
            ),
            generator.SOURCE_PLANNING_REPAIR_PROMPT_V11,
        )
        self.assertEqual(
            _registered_text(
                "generation.answer_repair", generator.ANSWER_REPAIR_PROMPT_VERSION
            ),
            generator.ANSWER_REPAIR_PROMPT_V17,
        )

    def test_repair_instructions_extend_first_instructions(self) -> None:
        for base, repair in (
            (generator.SOURCE_PLANNING_PROMPT_V11, generator.SOURCE_PLANNING_REPAIR_PROMPT_V11),
            (generator.ANSWER_PROMPT_V17, generator.ANSWER_REPAIR_PROMPT_V17),
        ):
            with self.subTest(repair=repair[len(base):][:40]):
                self.assertTrue(repair.startswith(base + "\n\n## Backend "))

    def test_rewrite_and_judge_aliases_read_current_versions(self) -> None:
        self.assertEqual(
            prompts.get("rewrite", query_rewrite.QUERY_REWRITE_PROMPT_VERSION),
            query_rewrite.QUERY_REWRITE_PROMPT_V4,
        )
        self.assertEqual(
            prompts.get("judge", JUDGE_PROMPT_VERSION), prompt_v7_2.JUDGE_INSTRUCTIONS
        )
        self.assertEqual(
            prompt_v7_2.JUDGE_INSTRUCTIONS_SHA256,
            sha256_text(prompt_v7_2.JUDGE_INSTRUCTIONS),
        )


class RegistryLookupTest(unittest.TestCase):
    def test_reads_file_bytes_without_normalizing(self) -> None:
        path = PROMPTS_DIR / "judge" / f"{JUDGE_PROMPT_VERSION}.md"

        self.assertEqual(path.read_bytes().decode("utf-8"), prompts.get("judge", JUDGE_PROMPT_VERSION))
        # 판별 지시문은 끝 줄바꿈이 없고, 생성 지시문은 끝 줄바꿈이 있다.
        self.assertFalse(prompts.get("judge", JUDGE_PROMPT_VERSION).endswith("\n"))
        self.assertTrue(prompts.get("generation.answer", "v23").endswith("\n"))

    def test_unknown_component_or_version_is_rejected(self) -> None:
        for component, version in (
            ("rewrite", "v999"),
            ("no_such_component", "v1"),
            ("../core", "v1"),
            ("rewrite", "../v10"),
            ("rewrite", ""),
        ):
            with self.subTest(component=component, version=version):
                with self.assertRaises(PromptNotFoundError):
                    prompts.get(component, version)

    def test_versions_of_unknown_component_is_rejected(self) -> None:
        with self.assertRaises(PromptNotFoundError):
            prompts.versions("no_such_component")


class PromptLayerImportTest(unittest.TestCase):
    """app/prompts 와 app/core 는 상위 패키지를 import 하지 않는다."""

    def _imported_modules(self, path: Path) -> Iterator[str]:
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                yield node.module
                for alias in node.names:
                    yield f"{node.module}.{alias.name}"

    def test_no_upper_package_imports(self) -> None:
        for package in ("app/prompts", "app/core"):
            for path in sorted((ROOT / package).rglob("*.py")):
                for module in self._imported_modules(path):
                    with self.subTest(path=str(path.relative_to(ROOT)), module=module):
                        self.assertFalse(
                            any(
                                module == upper or module.startswith(f"{upper}.")
                                for upper in UPPER_PACKAGES
                            )
                        )


if __name__ == "__main__":
    unittest.main()
