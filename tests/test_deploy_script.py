"""Black-box checks for the image retention in scripts/deploy.sh.

docker, aws, curl, df 를 PATH 의 스텁으로 바꿔 실제 서버나 Docker 없이 실행한다.
"""

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy.sh"
ECR_REPO = "123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/riido-chat-api"

DOCKER_STUB = r'''#!/usr/bin/env python3
import json
import os
import sys

state_path = os.environ["DOCKER_STUB_STATE"]
with open(state_path) as handle:
    state = json.load(handle)
with open(os.environ["DOCKER_STUB_LOG"], "a") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\n")

args = sys.argv[1:]
command = " ".join(args[:2])


def save():
    with open(state_path, "w") as handle:
        json.dump(state, handle)


def positional(values):
    result, skip = [], False
    for value in values:
        if skip:
            skip = False
        elif value == "--format":
            skip = True
        elif not value.startswith("-"):
            result.append(value)
    return result


if "--password-stdin" in args:
    sys.stdin.read()

if command in state.get("fail", []):
    print("stub failure: " + command, file=sys.stderr)
    raise SystemExit(1)

if command == "image ls":
    for image in state["images"]:
        print("\t".join((image["id"], image["repo"], image["tag"])))
elif command == "image inspect":
    by_id = {image["id"]: image for image in state["images"]}
    for image_id in positional(args[2:]):
        if image_id not in by_id:
            print("no such image: " + image_id, file=sys.stderr)
            raise SystemExit(1)
        print(by_id[image_id]["created"] + " " + image_id)
elif command == "container ls":
    for container in state["containers"]:
        print(container["id"])
elif command == "container inspect":
    by_id = {container["id"]: container for container in state["containers"]}
    for container_id in positional(args[2:]):
        print(by_id[container_id]["image"])
elif command == "image rm":
    refs = positional(args[2:])
    failing = [ref for ref in refs if ref in state.get("fail_rm", [])]
    if failing:
        print("Error response from daemon: conflict: " + failing[0], file=sys.stderr)
        raise SystemExit(1)
    state["images"] = [
        image for image in state["images"]
        if image["id"] not in refs and image["repo"] + ":" + image["tag"] not in refs
    ]
    save()
elif command == "image prune":
    pass
elif command == "info --format":
    print(state.get("docker_root", "/var/lib/docker"))
elif args[:1] == ["compose"] and "pull" in args:
    state["images"].append(state["pulled_image"])
    save()
elif args[:1] == ["compose"] and "up" in args:
    for container in state["containers"]:
        if container["id"] == "api-container":
            container["image"] = state["pulled_image"]["id"]
    save()
'''

DF_STUB = """#!/bin/sh
echo "Filesystem 1024-blocks Used Available Capacity Mounted on"
echo "/dev/root 31457280 1000 ${DF_AVAIL_KB} 50% /"
"""

AWS_STUB = "#!/bin/sh\necho stub-token\n"
CURL_STUB = "#!/bin/sh\nexit 0\n"


def image_id(number: int) -> str:
    return f"sha256:{number:02d}" + "ab" * 31


def image(number: int, repo: str = ECR_REPO, tag: Optional[str] = None) -> Dict[str, str]:
    return {
        "id": image_id(number),
        "repo": repo,
        "tag": tag if tag is not None else f"commit{number}",
        "created": f"2026-09-{number:02d}T03:00:00.123456789Z",
    }




class DeployScriptHarness:
    def __init__(self, tmp: Path, state: Dict) -> None:
        self.tmp = tmp
        self.state_path = tmp / "state.json"
        self.log_path = tmp / "docker.log"
        self.state_path.write_text(json.dumps(state))
        self.log_path.write_text("")
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        for name, source in (
            ("docker", DOCKER_STUB),
            ("df", DF_STUB),
            ("aws", AWS_STUB),
            ("curl", CURL_STUB),
        ):
            path = bin_dir / name
            path.write_text(source)
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
        self.app_dir = tmp / "app"
        self.app_dir.mkdir()
        self.env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "APP_DIR": str(self.app_dir),
            "DOCKER_STUB_STATE": str(self.state_path),
            "DOCKER_STUB_LOG": str(self.log_path),
            "DF_AVAIL_KB": str(20 * 1024 * 1024),
        }

    def run_function(self, name: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "-c", f'source "$1" && {name}', "deploy-test", str(SCRIPT)],
            env={**self.env, **env},
            capture_output=True,
            text=True,
        )

    def run_script(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
        )

    def calls(self) -> List[List[str]]:
        return [json.loads(line) for line in self.log_path.read_text().splitlines()]

    def removed(self) -> List[List[str]]:
        return [call[2:] for call in self.calls() if call[:2] == ["image", "rm"]]

    def remaining_ids(self) -> List[str]:
        return sorted({entry["id"] for entry in json.loads(self.state_path.read_text())["images"]})


def base_state() -> Dict:
    # 앱 이미지 1~5(5가 최신), 실행 중인 앱은 2, caddy 와 저장소만 비슷한 이미지는 정리 대상이 아니다
    images = [image(number) for number in (1, 2, 3, 4, 5)]
    images.append({**image(1), "repo": "riido-chat-api", "tag": "manual"})
    images.append(image(20, repo="caddy", tag="2"))
    images.append(image(21, repo="other/riido-chat-api-tools", tag="latest"))
    return {
        "images": images,
        "containers": [
            {"id": "api-container", "image": image_id(2)},
            {"id": "caddy-container", "image": image_id(20)},
        ],
        "pulled_image": image(9),
    }


class PruneAppImagesTest(unittest.TestCase):
    def _harness(self, directory: str, state: Dict) -> DeployScriptHarness:
        return DeployScriptHarness(Path(directory), state)

    def test_keeps_running_image_and_newest_n_and_non_app_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            harness = self._harness(directory, base_state())

            result = harness.run_function("prune_app_images")

            self.assertEqual(0, result.returncode, result.stderr)
            # 이미지 1은 태그가 둘이므로 두 태그를 함께 지운다
            self.assertEqual(
                [[f"{ECR_REPO}:commit3"], [f"{ECR_REPO}:commit1", "riido-chat-api:manual"]],
                harness.removed(),
            )
            self.assertEqual(
                sorted([image_id(2), image_id(4), image_id(5), image_id(20), image_id(21)]),
                harness.remaining_ids(),
            )
            self.assertIn("삭제 2개, 실패 0개, 롤백용 유지 2개", result.stdout)

    def test_keep_count_is_configurable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            harness = self._harness(directory, base_state())

            result = harness.run_function("prune_app_images", KEEP_APP_IMAGES="0")

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(
                sorted([image_id(2), image_id(20), image_id(21)]),
                harness.remaining_ids(),
            )

    def test_single_remove_failure_does_not_abort(self) -> None:
        state = base_state()
        state["fail_rm"] = [f"{ECR_REPO}:commit3"]
        with tempfile.TemporaryDirectory() as directory:
            harness = self._harness(directory, state)

            result = harness.run_function("prune_app_images")

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn("앱 이미지 삭제 실패, 계속 진행합니다", result.stderr)
            self.assertEqual(2, len(harness.removed()))
            self.assertNotIn(image_id(1), harness.remaining_ids())
            self.assertIn(image_id(3), harness.remaining_ids())

    def test_skips_removal_when_container_list_fails(self) -> None:
        state = base_state()
        state["fail"] = ["container ls"]
        with tempfile.TemporaryDirectory() as directory:
            harness = self._harness(directory, state)

            result = harness.run_function("prune_app_images")

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn("컨테이너 목록을 읽지 못해", result.stderr)
            self.assertEqual([], harness.removed())

    def test_invalid_keep_count_skips_removal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            harness = self._harness(directory, base_state())

            result = harness.run_function("prune_app_images", KEEP_APP_IMAGES="two")

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual([], harness.removed())


class DeployFlowTest(unittest.TestCase):
    def test_prunes_before_pull_and_after_successful_deploy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            harness = DeployScriptHarness(Path(directory), base_state())

            result = harness.run_script(f"{ECR_REPO}:commit9")

            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            calls = harness.calls()
            pull_index = next(i for i, call in enumerate(calls) if call[:1] == ["compose"] and "pull" in call)
            first_rm = next(i for i, call in enumerate(calls) if call[:2] == ["image", "rm"])
            self.assertLess(first_rm, pull_index)
            self.assertIn(["image", "prune", "-f"], calls)
            self.assertFalse(any("-a" in call or "volume" in call or "network" in call for call in calls))
            # 배포 후: 새 이미지 9가 실행 중이고, 이전 이미지 중 최신 2개(5, 4)만 남는다
            self.assertEqual(
                sorted([image_id(4), image_id(5), image_id(9), image_id(20), image_id(21)]),
                harness.remaining_ids(),
            )

    def test_low_disk_space_fails_before_pull(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            harness = DeployScriptHarness(Path(directory), base_state())

            result = harness.run_script(f"{ECR_REPO}:commit9", DF_AVAIL_KB=str(1024 * 1024))

            self.assertNotEqual(0, result.returncode)
            self.assertIn("디스크 여유 공간 부족", result.stderr)
            self.assertIn("남은 공간 1024MB, 필요 2048MB", result.stderr)
            self.assertFalse(any(call[:1] == ["compose"] for call in harness.calls()))

    def test_free_space_threshold_is_configurable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            harness = DeployScriptHarness(Path(directory), base_state())

            result = harness.run_script(
                f"{ECR_REPO}:commit9",
                DF_AVAIL_KB=str(1024 * 1024),
                MIN_FREE_DISK_MB="512",
            )

            self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_unreadable_free_space_fails_before_pull(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            harness = DeployScriptHarness(Path(directory), base_state())

            result = harness.run_script(f"{ECR_REPO}:commit9", DF_AVAIL_KB="unknown")

            self.assertNotEqual(0, result.returncode)
            self.assertIn("디스크 여유 공간을 확인하지 못해", result.stderr)
            self.assertFalse(any(call[:1] == ["compose"] for call in harness.calls()))


if __name__ == "__main__":
    unittest.main()
