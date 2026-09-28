import unittest
from pathlib import Path

from app.core.system_version import SYSTEM_VERSION
from scripts.check_system_version import (
    VERSION_FILE,
    VERSIONS_DOC,
    evaluate,
    is_version_greater,
    parse_semver,
    parse_version_rows,
    read_system_version,
    watched_changes,
)


ROOT = Path(__file__).resolve().parents[1]
WATCHED_FILE = "app/answering/generator.py"


class SystemVersionTest(unittest.TestCase):
    def test_current_version_is_semver(self) -> None:
        self.assertIsNotNone(parse_semver(SYSTEM_VERSION))

    def test_versions_doc_has_current_version_row(self) -> None:
        markdown = (ROOT / VERSIONS_DOC).read_text("utf-8")

        self.assertIn(SYSTEM_VERSION, parse_version_rows(markdown))

    def test_script_reads_same_version_as_module(self) -> None:
        source = (ROOT / VERSION_FILE).read_text("utf-8")

        self.assertEqual(SYSTEM_VERSION, read_system_version(source))


class SemverTest(unittest.TestCase):
    def test_parse_semver_accepts_major_minor_patch(self) -> None:
        self.assertEqual((1, 2, 3), parse_semver("1.2.3"))
        self.assertEqual((0, 10, 0), parse_semver("0.10.0"))

    def test_parse_semver_rejects_invalid_forms(self) -> None:
        for value in ("1.2", "1.2.3.4", "v1.2.3", "01.2.3", "1.2.3-rc1", "", "a.b.c"):
            with self.subTest(value=value):
                self.assertIsNone(parse_semver(value))

    def test_is_version_greater_compares_numerically(self) -> None:
        self.assertTrue(is_version_greater("1.10.0", "1.9.0"))
        self.assertTrue(is_version_greater("2.0.0", "1.99.99"))
        self.assertTrue(is_version_greater("1.0.1", "1.0.0"))
        self.assertFalse(is_version_greater("1.0.0", "1.0.0"))
        self.assertFalse(is_version_greater("1.0.0", "1.0.1"))
        self.assertFalse(is_version_greater("bad", "1.0.0"))


class ParsingTest(unittest.TestCase):
    def test_read_system_version_finds_assignment(self) -> None:
        source = '"""설명."""\n\n\nSYSTEM_VERSION = "1.4.2"\n'

        self.assertEqual("1.4.2", read_system_version(source))
        self.assertIsNone(read_system_version("OTHER = '1.0.0'\n"))

    def test_parse_version_rows_skips_rule_table_and_header(self) -> None:
        markdown = "\n".join(
            [
                "| 구분 | 올리는 경우 |",
                "| --- | --- |",
                "| MAJOR | 모델 교체 |",
                "",
                "| 버전 | 날짜 | 바뀐 것 | 바뀐 단계 | 평가 기록 |",
                "| --- | --- | --- | --- | --- |",
                "| 1.0.0 | 2026-09-28 | 기준선 | 기준선 | 미평가 |",
                "| 1.1.0 | 2026-10-01 | 문턱 조정 | 판별 | 미평가 |",
            ]
        )

        self.assertEqual(["1.0.0", "1.1.0"], parse_version_rows(markdown))

    def test_watched_changes_matches_directories_and_files_only(self) -> None:
        changed = [
            "app/answering/generator.py",
            "app/chat/query_rewrite.py",
            "app/chat/service.py",
            "app/question_grouping/constants.py",
            "app/retrieval/embedding.py",
            "app/document/chunker.py",
            "app/document/recollect.py",
            "tests/test_generator.py",
            "docs/note.md",
        ]

        self.assertEqual(
            [
                "app/answering/generator.py",
                "app/chat/query_rewrite.py",
                "app/document/chunker.py",
                "app/question_grouping/constants.py",
                "app/retrieval/embedding.py",
            ],
            watched_changes(changed),
        )


class EvaluateTest(unittest.TestCase):
    def test_passes_without_watched_changes(self) -> None:
        errors = evaluate(["app/api/health.py"], "1.0.0", "1.0.0", ["1.0.0"])

        self.assertEqual([], errors)

    def test_fails_when_current_version_row_missing(self) -> None:
        errors = evaluate([], "1.0.0", "1.0.0", ["0.9.0"])

        self.assertEqual(1, len(errors))
        self.assertIn("1.0.0 행이 없습니다", errors[0])

    def test_fails_when_current_version_is_not_semver(self) -> None:
        errors = evaluate([], "1.0.0", "1.0", ["1.0"])

        self.assertEqual(1, len(errors))
        self.assertIn("형식이 아닙니다", errors[0])

    def test_fails_when_watched_changed_without_version_file(self) -> None:
        errors = evaluate([WATCHED_FILE], "1.0.0", "1.0.0", ["1.0.0"])

        self.assertEqual(1, len(errors))
        self.assertIn("시스템 버전을 올리지 않았습니다", errors[0])
        self.assertIn(WATCHED_FILE, errors[0])

    def test_fails_when_version_not_increased(self) -> None:
        errors = evaluate(
            [WATCHED_FILE, VERSION_FILE], "1.2.0", "1.1.9", ["1.1.9", "1.2.0"]
        )

        self.assertEqual(1, len(errors))
        self.assertIn("1.2.0 보다 커야 합니다", errors[0])

    def test_fails_when_bumped_version_row_missing(self) -> None:
        errors = evaluate([WATCHED_FILE, VERSION_FILE], "1.0.0", "1.1.0", ["1.0.0"])

        self.assertEqual(1, len(errors))
        self.assertIn("1.1.0 행이 없습니다", errors[0])

    def test_passes_when_version_bumped_and_recorded(self) -> None:
        errors = evaluate(
            [WATCHED_FILE, VERSION_FILE, VERSIONS_DOC],
            "1.0.0",
            "1.1.0",
            ["1.0.0", "1.1.0"],
        )

        self.assertEqual([], errors)

    def test_passes_when_version_file_introduced(self) -> None:
        errors = evaluate([WATCHED_FILE, VERSION_FILE], None, "1.0.0", ["1.0.0"])

        self.assertEqual([], errors)


if __name__ == "__main__":
    unittest.main()
