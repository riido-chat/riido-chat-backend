"""평가 등록부(riido-evals) 읽기.

- 골든셋 버전 확인: manifest goldenVersion 이 기대 버전과 같고, manifest 의 층별 문항 해시와
  frozen.files 해시, 크기가 모두 맞아야 연다. 하나라도 다르면 멈춘다.
- 묶음 선택은 등록부의 scripts/build_suites.py 를 그대로 실행해 얻는다(등록부의 문서화된 사용법).
  선택한 턴 수가 manifest suiteCounts 와 다르면 멈춘다.
- 보류 시험셋(holdoutLayers, holdout 묶음)은 holdout_allowed 없이 열지 않는다. B층 문항 파일과
  B층 동결 행은 이 관문을 지난 경우에만 읽는다. 동결 판별 입력 파일에서는 턴 ID 만 보고 고른
  줄만 JSON 으로 읽는다.
- 라벨과 동결 행에서 질문 문장은 꺼내지 않는다. 판별 질의(resolvedQuery)는 payload 재구성에만 쓴다.
"""

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

TurnKey = Tuple[str, int]

JUDGE_DECISIONS = ("CONNECT", "SEPARATE", "UNCLASSIFIED")
# labels.md 3절: 문항 review.labelSource 가 없으면 human 으로 본다.
DEFAULT_LABEL_SOURCE = "human"
# 판별 단계 확정 상태만 보고용 묶음에 들어간다 (rules.json requireConfirmed).
CONFIRMED = "confirmed"
CRITERIA_DIR = "criteria"
# 사전 등록 문서가 확정본인지 보는 줄. 초안은 `상태: 초안 ...` 이다.
CRITERIA_FINAL_PATTERN = re.compile(r"^\s*(?:[-*]\s*)?상태\s*:\s*확정", re.MULTILINE)
# freeze_judge_inputs.py 가 쓰는 줄 모양: {"id":"A-...","turnNo":1,...}
_FROZEN_ROW_HEAD = re.compile(r'^\{"id":"(?P<id>[^"]+)","turnNo":(?P<turn>[0-9]+),')
# 등록부의 git 상태를 볼 경로. evals/ 는 실행 기록과 결과가 쌓이는 곳이라 보지 않는다.
CONTENT_PATHS = ("golden", "schemas", "scripts", "criteria")


class GoldenError(Exception):
    """골든셋을 열 수 없거나 실행 조건이 맞지 않음."""


class HoldoutRefused(GoldenError):
    """보류 시험셋 관문을 지나지 못함."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


@dataclass(frozen=True)
class Label:
    decision: str
    keys: Tuple[str, ...]
    source: str
    status: str


@dataclass
class Golden:
    root: Path
    manifest: Dict[str, Any]
    rules: Dict[str, Any]
    version: str
    commit: Optional[str]
    dirty: bool
    judge_inputs: Path
    catalog: Path
    environment: Path
    supplement: Optional[Path] = None

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def suite_rule(self, suite: str) -> Dict[str, Any]:
        rule = (self.rules.get("suites") or {}).get(suite)
        if rule is None:
            raise GoldenError(f"rules.json 에 없는 묶음입니다: {suite}")
        if rule.get("stage") != "judge":
            raise GoldenError(f"{suite} 는 판별 묶음이 아닙니다 (stage={rule.get('stage')})")
        return rule

    def suite_layers(self, suite: str) -> List[str]:
        return list(self.suite_rule(suite).get("defaultLayers") or [])

    def is_holdout(self, suite: str) -> bool:
        rule = self.suite_rule(suite)
        holdout_layers = set(self.rules.get("holdoutLayers") or [])
        return bool(rule.get("holdout")) or bool(holdout_layers & set(rule.get("allowedLayers") or []))

    def is_reportable(self, suite: str) -> bool:
        return bool(self.suite_rule(suite).get("reportable"))


def _frozen_file(files: Mapping[str, Any], prefix: str, required: bool = True) -> Optional[str]:
    matches = sorted(name for name in files if Path(name).name.startswith(prefix))
    if len(matches) > 1:
        raise GoldenError(f"manifest frozen.files 에 {prefix} 파일이 둘 이상입니다: {matches}")
    if not matches:
        if required:
            raise GoldenError(f"manifest frozen.files 에 {prefix} 파일이 없습니다")
        return None
    return matches[0]


def open_golden(root: Path, expected_version: str) -> Golden:
    """골든셋 버전과 동결 해시를 확인하고 연다. 맞지 않으면 GoldenError."""

    root = Path(root).resolve()
    manifest_path = root / "golden" / "manifest.json"
    if not manifest_path.exists():
        raise GoldenError(f"평가 등록부가 아닙니다 (golden/manifest.json 없음): {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("goldenVersion")
    if version != expected_version:
        raise GoldenError(f"골든셋 버전이 다릅니다: manifest {version}, 기대 {expected_version}")

    problems: List[str] = []
    for layer, meta in sorted((manifest.get("layers") or {}).items()):
        path = root / meta["path"]
        if not path.exists():
            problems.append(f"{layer}층 문항 파일 없음 {meta['path']}")
        elif sha256_file(path) != meta.get("sha256"):
            problems.append(f"{layer}층 문항 해시 불일치 {meta['path']}")
    files = ((manifest.get("frozen") or {}).get("files")) or {}
    if not files:
        problems.append("manifest frozen.files 가 비어 있습니다")
    for relative, meta in sorted(files.items()):
        path = root / relative
        if not path.exists():
            problems.append(f"동결 파일 없음 {relative}")
            continue
        data = path.read_bytes()
        if sha256_bytes(data) != meta.get("sha256") or len(data) != meta.get("bytes"):
            problems.append(f"동결 파일 해시 또는 크기 불일치 {relative}")
    if problems:
        raise GoldenError("골든셋 동결 확인 실패: " + "; ".join(problems))

    rules_path = root / (manifest.get("suiteRules") or "golden/suites/rules.json")
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--", *CONTENT_PATHS)
    dirty = bool(status) if status is not None else True
    return Golden(
        root=root,
        manifest=manifest,
        rules=rules,
        version=version,
        commit=commit if commit and re.fullmatch(r"[0-9a-f]{40}", commit) else None,
        dirty=dirty,
        judge_inputs=root / _frozen_file(files, "judge-inputs-"),
        catalog=root / _frozen_file(files, "catalog-"),
        environment=root / _frozen_file(files, "environment-"),
        supplement=(
            root / name if (name := _frozen_file(files, "judge-payload-supplement-", required=False)) else None
        ),
    )


# ---------------------------------------------------------------- 묶음 선택


@dataclass
class Selection:
    suite: str
    layers: List[str]
    holdout: bool
    reportable: bool
    turns: List[Tuple[str, str, int]]  # (문항 ID, 층, 턴 번호) 묶음 파일 순서
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def keys(self) -> List[TurnKey]:
        return [(item_id, turn_no) for item_id, _, turn_no in self.turns]


def select_suite(golden: Golden, suite: str, *, holdout_allowed: bool = False) -> Selection:
    """등록부 build_suites.py 로 묶음 턴을 뽑는다. 보류 시험셋은 holdout_allowed 없이는 거부한다."""

    rule = golden.suite_rule(suite)
    holdout = golden.is_holdout(suite)
    if holdout and not holdout_allowed:
        raise HoldoutRefused(
            f"{suite} 는 보류 시험셋 묶음입니다. --holdout 과 확정해 커밋한 criteria/ 파일 없이는 열지 않습니다"
        )
    layers = golden.suite_layers(suite)
    script = golden.root / "scripts" / "build_suites.py"
    if not script.exists():
        raise GoldenError(f"등록부에 묶음 생성 스크립트가 없습니다: {script}")
    with tempfile.TemporaryDirectory(prefix="judge-suite-") as tmp:
        result = subprocess.run(
            [sys.executable, str(script), "--suite", suite, "--layers", ",".join(layers), "--out", tmp],
            cwd=str(golden.root),
            capture_output=True,
            text=True,
        )
        out = Path(tmp) / f"{suite}.json"
        if result.returncode != 0 or not out.exists():
            raise GoldenError(f"build_suites.py 실패 (종료 코드 {result.returncode}): {result.stderr.strip()[-300:]}")
        payload = json.loads(out.read_text(encoding="utf-8"))

    turns: List[Tuple[str, str, int]] = []
    for entry in payload.get("items") or []:
        if entry["layer"] not in layers:
            raise GoldenError(f"묶음 파일에 허용하지 않은 층이 있습니다: {entry['layer']}")
        turns.extend((entry["id"], entry["layer"], int(turn_no)) for turn_no in entry["turnNos"])
    expected = (golden.manifest.get("suiteCounts") or {}).get(suite) or {}
    for layer in layers:
        got = sum(1 for _, turn_layer, _ in turns if turn_layer == layer)
        want = (expected.get(layer) or {}).get("turns")
        if want is not None and got != want:
            raise GoldenError(f"{suite} {layer}층 턴 수가 manifest suiteCounts 와 다릅니다: {got} != {want}")
    meta = {key: payload.get(key) for key in ("kind", "reportable", "stage", "requireConfirmed")}
    for key in ("inPrompt", "inCatalog"):
        if key in payload:
            meta[key] = _relative_lists(payload[key], golden.root)
    return Selection(
        suite=suite,
        layers=layers,
        holdout=holdout,
        reportable=bool(rule.get("reportable")),
        turns=turns,
        meta=meta,
    )


def _relative_lists(value: Any, root: Path) -> Any:
    """묶음 파일의 목록 경로를 등록부 기준 상대 경로로 바꾼다 (로컬 절대 경로를 결과에 남기지 않음)."""

    if isinstance(value, dict):
        return {key: _relative_lists(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_relative_lists(item, root) for item in value]
    if isinstance(value, str) and value.startswith(str(root)):
        return Path(value).resolve().relative_to(root).as_posix()
    return value


# ---------------------------------------------------------------- 라벨, 동결 행, 카탈로그


def _judge_review(item: Mapping[str, Any], turn: Mapping[str, Any]) -> Tuple[str, str]:
    review = turn.get("review") if isinstance(turn.get("review"), dict) else {}
    entry = review.get("judge") if isinstance(review.get("judge"), dict) else None
    if entry is None:
        item_review = item.get("review") if isinstance(item.get("review"), dict) else {}
        return item_review.get("labelSource") or DEFAULT_LABEL_SOURCE, item_review.get("status") or "draft"
    return entry.get("source") or DEFAULT_LABEL_SOURCE, entry.get("status") or "draft"


def load_labels(golden: Golden, selection: Selection) -> Dict[TurnKey, Label]:
    """선택한 턴의 판별 라벨과 라벨 출처. 묶음 층의 문항 파일만 읽는다."""

    wanted = set(selection.keys)
    labels: Dict[TurnKey, Label] = {}
    for layer in selection.layers:
        path = golden.root / golden.manifest["layers"][layer]["path"]
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            for turn_no, turn in enumerate(item.get("turns") or [], start=1):
                key = (item["id"], turn_no)
                if key not in wanted:
                    continue
                decision = turn.get("expectedJudgeDecision")
                if decision not in JUDGE_DECISIONS:
                    raise GoldenError(f"{key} 판별 라벨이 없습니다")
                source, status = _judge_review(item, turn)
                labels[key] = Label(
                    decision=decision,
                    keys=tuple(turn.get("expectedSubproblemKeysAny") or ()),
                    source=source,
                    status=status,
                )
    missing = wanted - set(labels)
    if missing:
        raise GoldenError(f"묶음 턴의 라벨을 찾지 못했습니다: {sorted(missing)[:3]}")
    return labels


def load_frozen_rows(golden: Golden, keys: Sequence[TurnKey]) -> Dict[TurnKey, Dict[str, Any]]:
    """동결 판별 입력에서 고른 턴의 줄만 읽는다. 다른 턴(다른 층 포함)은 JSON 으로 읽지 않는다."""

    wanted = set(keys)
    rows: Dict[TurnKey, Dict[str, Any]] = {}
    with golden.judge_inputs.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            head = _FROZEN_ROW_HEAD.match(line)
            if head is None:
                raise GoldenError(f"{golden.judge_inputs.name}:{line_no} 줄 머리 모양이 예상과 다릅니다")
            key = (head.group("id"), int(head.group("turn")))
            if key in wanted:
                rows[key] = json.loads(line)
    missing = wanted - set(rows)
    if missing:
        raise GoldenError(f"동결 판별 입력에 없는 턴이 있습니다: {sorted(missing)[:3]}")
    return rows


def load_catalog(golden: Golden) -> Dict[str, Dict[str, Any]]:
    catalog = json.loads(golden.catalog.read_text(encoding="utf-8"))
    fingerprint = ((golden.manifest.get("frozen") or {}).get("catalogFingerprint") or {}).get("fingerprint")
    if catalog.get("meta", {}).get("fingerprint") != fingerprint:
        raise GoldenError("동결 카탈로그 지문이 manifest 와 다릅니다")
    return {entry["key"]: entry for entry in catalog["entries"]}


def load_environment(golden: Golden) -> Dict[str, Any]:
    return json.loads(golden.environment.read_text(encoding="utf-8"))


def load_supplement(golden: Golden) -> Optional[Dict[str, Any]]:
    if golden.supplement is None:
        return None
    return json.loads(golden.supplement.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 사전 등록 기준


def check_criteria(golden: Golden, relative: str, suite: str) -> str:
    """criteria/ 파일이 확정본이고 커밋돼 있으며 이 묶음과 골든셋 버전을 적었는지 본다."""

    path = (golden.root / relative).resolve()
    try:
        rel = path.relative_to(golden.root).as_posix()
    except ValueError as exc:
        raise GoldenError(f"기준 파일은 등록부 안에 있어야 합니다: {relative}") from exc
    if not rel.startswith(f"{CRITERIA_DIR}/") or not path.is_file():
        raise GoldenError(f"기준 파일은 {CRITERIA_DIR}/ 아래에 있어야 합니다: {relative}")
    if _git(golden.root, "ls-files", "--error-unmatch", rel) is None:
        raise GoldenError(f"기준 파일이 커밋돼 있지 않습니다: {rel}")
    if _git(golden.root, "status", "--porcelain", "--", rel):
        raise GoldenError(f"기준 파일에 커밋하지 않은 변경이 있습니다: {rel}")
    text = path.read_text(encoding="utf-8")
    if not CRITERIA_FINAL_PATTERN.search(text):
        raise GoldenError(f"기준 파일이 확정본이 아닙니다 (`상태: 확정` 줄 없음): {rel}")
    for needle in (suite, golden.version):
        if needle not in text:
            raise GoldenError(f"기준 파일에 {needle} 가 적혀 있지 않습니다: {rel}")
    return rel
