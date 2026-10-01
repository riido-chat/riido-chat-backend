import contextlib
import io
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

from scripts.check_prompt_fingerprints import (
    TABLE_FILE,
    TableFormatError,
    added_keys,
    evaluate,
    main,
    parse_table,
)


ROOT = Path(__file__).resolve().parents[1]
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _table_source(fingerprints: Dict[str, str]) -> str:
    return json.dumps({"description": "test", "fingerprints": fingerprints}, indent=2)


class ParseTableTest(unittest.TestCase):
    def test_reads_repository_table(self) -> None:
        table = parse_table((ROOT / TABLE_FILE).read_text("utf-8"))

        self.assertIn("generation@v40", table)
        self.assertEqual([], evaluate(None, table))

    def test_rejects_invalid_documents(self) -> None:
        for source in ("{", "[]", '{"fingerprints": []}', '{"fingerprints": {"k@v1": 1}}'):
            with self.subTest(source=source):
                with self.assertRaises(TableFormatError):
                    parse_table(source)


class EvaluateTest(unittest.TestCase):
    def test_passes_when_unchanged(self) -> None:
        table = {"rewrite@v1": HASH_A}

        self.assertEqual([], evaluate(table, dict(table)))

    def test_passes_when_key_added(self) -> None:
        base = {"rewrite@v1": HASH_A}
        head = {"rewrite@v1": HASH_A, "rewrite@v2": HASH_B}

        self.assertEqual([], evaluate(base, head))
        self.assertEqual(["rewrite@v2"], added_keys(base, head))

    def test_fails_when_existing_hash_changed(self) -> None:
        errors = evaluate({"rewrite@v1": HASH_A}, {"rewrite@v1": HASH_B})

        self.assertEqual(1, len(errors))
        self.assertIn("rewrite@v1 의 지문이 바뀌었습니다", errors[0])

    def test_fails_when_existing_key_removed(self) -> None:
        errors = evaluate(
            {"rewrite@v1": HASH_A, "rewrite@v2": HASH_B}, {"rewrite@v2": HASH_B}
        )

        self.assertEqual(1, len(errors))
        self.assertIn("rewrite@v1 가 지워졌습니다", errors[0])

    def test_fails_on_malformed_hash_even_without_base(self) -> None:
        errors = evaluate(None, {"rewrite@v1": "ABC"})

        self.assertEqual(1, len(errors))
        self.assertIn("sha256 hex", errors[0])

    def test_without_base_every_key_is_added(self) -> None:
        self.assertEqual(
            ["a@v1", "b@v1"], added_keys(None, {"b@v1": HASH_A, "a@v1": HASH_B})
        )


class MainWithGitRepositoryTest(unittest.TestCase):
    """임시 git 저장소에서 merge-base 표와 작업 트리 표를 비교하는 전체 흐름."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self._git("init", "-q", "-b", "base")
        self._write({"rewrite@v1": HASH_A})
        self._git("add", "-A")
        self._commit("기준 표")
        self._git("checkout", "-q", "-b", "work")

    def _git(self, *args: str) -> None:
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            cwd=self.root,
            check=True,
            capture_output=True,
        )

    def _commit(self, message: str) -> None:
        self._git("commit", "-q", "--allow-empty", "-m", message)

    def _write(self, fingerprints: Dict[str, str]) -> None:
        path = self.root / TABLE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_table_source(fingerprints), "utf-8")

    def _run(self, base_ref: str = "base") -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            return main([base_ref], root=self.root)

    def test_added_key_in_working_tree_passes(self) -> None:
        self._write({"rewrite@v1": HASH_A, "rewrite@v2": HASH_C})

        self.assertEqual(0, self._run())

    def test_changed_key_in_working_tree_fails(self) -> None:
        self._write({"rewrite@v1": HASH_C})

        self.assertEqual(1, self._run())

    def test_removed_key_in_working_tree_fails(self) -> None:
        self._write({"rewrite@v2": HASH_C})

        self.assertEqual(1, self._run())

    def test_compares_with_merge_base_not_base_tip(self) -> None:
        # 기준 브랜치가 나중에 키를 더해도 그 이전에 갈라진 작업 브랜치는 실패하지 않는다.
        self._git("checkout", "-q", "base")
        self._write({"rewrite@v1": HASH_A, "judge@v9": HASH_B})
        self._git("add", "-A")
        self._commit("기준 브랜치 키 추가")
        self._git("checkout", "-q", "work")

        self.assertEqual(0, self._run())

    def test_table_missing_at_base_passes(self) -> None:
        self._git("checkout", "-q", "--orphan", "empty")
        self._git("rm", "-rq", "--cached", ".")
        self._commit("빈 기준")
        self._git("branch", "-q", "-f", "base", "empty")
        self._git("checkout", "-q", "-B", "work", "empty")

        self.assertEqual(0, self._run())

    def test_unknown_base_ref_fails(self) -> None:
        self.assertEqual(1, self._run("no-such-ref"))


if __name__ == "__main__":
    unittest.main()
