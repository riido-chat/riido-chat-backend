"""Black-box checks for the group-aware ALL reindex shell flow."""

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reindex.sh"


class ReindexScriptTest(unittest.TestCase):
    def test_continues_after_group_failure_and_reloads_successful_group_by_id(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            docker = tmp / "docker"
            curl = tmp / "curl"
            docker.write_text("#!/bin/sh\nprintf '%s\\n' test-image\n")
            curl.write_text(
                """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

args = sys.argv[1:]
url = args[-1]
method = 'POST' if '-X' in args else 'GET'
if '-o' in args:
    output = Path(args[args.index('-o') + 1])
else:
    output = None
if method == 'GET':
    print(json.dumps({'groups': [
        {'groupId': 2, 'groupKey': 'HELP_CHATBOT_TEST'},
        {'groupId': 1, 'groupKey': 'HELP_CHATBOT'},
    ]}))
    raise SystemExit(0)
if '/reindex' in url and url.endswith('/1/reindex'):
    if output:
        output.write_text(json.dumps({'code': 'INTERNAL_ERROR'}))
    print('500', end='')
    raise SystemExit(0)
if '/reindex' in url:
    if output:
        output.write_text(json.dumps({'indexRunId': 9}))
    print('200', end='')
    raise SystemExit(0)
if '/corpus/reload?documentGroupId=2' in url:
    if output:
        output.write_text(json.dumps({'loaded': True}))
    print('200', end='')
    raise SystemExit(0)
print('404', end='')
raise SystemExit(0)
"""
            )
            for executable in (docker, curl):
                executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

            app_dir = tmp / "app"
            result = subprocess.run(
                ["bash", str(SCRIPT), "ALL"],
                cwd=ROOT,
                env={
                    **os.environ,
                    "APP_DIR": str(app_dir),
                    "DOCKER_BIN": str(docker),
                    "BASE_URL": "http://test",
                    "PATH": f"{tmp}:{os.environ.get('PATH', '')}",
                },
                capture_output=True,
                text=True,
            )

            self.assertEqual(1, result.returncode)
            self.assertIn("그룹 1 색인 실패", result.stdout)
            self.assertIn("그룹 2 색인 및 corpus reload 완료", result.stdout)
            self.assertIn("전체 2, 성공 1, 실패 1", result.stdout)

    def test_rejects_non_all_mode(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "GITBOOK"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(2, result.returncode)
        self.assertIn("ALL만 허용", result.stdout)


if __name__ == "__main__":
    unittest.main()
