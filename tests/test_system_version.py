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


class EquivalenceGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        import hashlib
        import json
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts import check_system_version as checker

        self.checker = checker
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.changed = [WATCHED_FILE, checker.EQUIVALENCE_FILE]
        files = {WATCHED_FILE} | {name.replace(".", "/") + ".py" for name in checker.EQUIVALENCE_TESTS}
        for path in files | set(checker.EQUIVALENCE_SNAPSHOTS):
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}\n")
        self.record = {
            "baseCommit": "a" * 40,
            "systemVersion": "1.2.2",
            "files": {path: hashlib.sha256((self.root / path).read_bytes()).hexdigest() for path in files},
            "snapshots": {path: hashlib.sha256(b"{}\n").hexdigest() for path in checker.EQUIVALENCE_SNAPSHOTS},
        }
        record_file = self.root / checker.EQUIVALENCE_FILE
        record_file.parent.mkdir(parents=True)
        record_file.write_text(json.dumps(self.record))
        self.addCleanup(patch.stopall)
        patch.object(checker, "ROOT", self.root).start()
        patch.object(checker, "base_system_version", return_value="1.2.2").start()
        patch.object(checker, "_git", side_effect=lambda *args: "a" * 40 if args[0] == "merge-base" else "{}\n").start()
        self.run = patch.object(checker.subprocess, "run", return_value=SimpleNamespace(returncode=0)).start()

    def validate(self):
        return self.checker.equivalent_changes("origin/develop", self.changed, "1.2.2")

    def test_unchanged_snapshot_and_proven_files_allow_same_version(self):
        paths = self.validate()
        self.assertEqual([WATCHED_FILE], paths)
        self.assertEqual([], evaluate(self.changed, "1.2.2", "1.2.2", ["1.2.2"], paths))
        self.assertEqual(list(self.checker.EQUIVALENCE_TESTS), self.run.call_args.args[0][-2:])

    def test_old_record_cannot_exempt_another_pr(self):
        self.changed.remove(self.checker.EQUIVALENCE_FILE)
        self.assertEqual([], self.validate())
        self.assertTrue(evaluate(self.changed, "1.2.2", "1.2.2", ["1.2.2"]))
        self.run.assert_not_called()

    def test_new_runtime_file_is_not_implicitly_exempt(self):
        self.changed.append("app/chat/service.py")
        with self.assertRaisesRegex(ValueError, "파일 목록"):
            self.validate()
        self.run.assert_not_called()

    def test_file_change_after_proof_invalidates_exemption(self):
        (self.root / WATCHED_FILE).write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "파일 지문"):
            self.validate()

    def test_rewritten_snapshot_cannot_claim_equivalence(self):
        import hashlib
        import json
        path = self.checker.EQUIVALENCE_SNAPSHOTS[0]
        (self.root / path).write_text("changed\n")
        self.record["snapshots"][path] = hashlib.sha256(b"changed\n").hexdigest()
        (self.root / self.checker.EQUIVALENCE_FILE).write_text(json.dumps(self.record))
        with self.assertRaisesRegex(ValueError, "기준 동작 스냅샷"):
            self.validate()

    def test_failed_behavior_verification_refuses_exemption(self):
        self.run.return_value.returncode = 1
        with self.assertRaisesRegex(ValueError, "검증이 실패"):
            self.validate()

    def test_snapshot_update_flags_are_removed(self):
        from unittest.mock import patch
        with patch.dict(self.checker.os.environ, {"UPDATE_IDENTITY_SNAPSHOT": "1", "UPDATE_CHAT_CONTRACT_SNAPSHOT": "1"}):
            self.validate()
        env = self.run.call_args.kwargs["env"]
        self.assertNotIn("UPDATE_IDENTITY_SNAPSHOT", env)
        self.assertNotIn("UPDATE_CHAT_CONTRACT_SNAPSHOT", env)



if __name__ == "__main__":
    unittest.main()
