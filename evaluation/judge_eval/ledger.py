"""판별 측정 실행 기록부 (JSONL, 기본 위치 <등록부>/evals/run-ledger.jsonl).

- 실행은 모델을 부르기 전에 한 줄을 더한다. 중간에 멈춘 실행도 한 번으로 센다.
- 같은 (묶음, 모델, 골든셋 버전) 의 실행은 MAX_RUNS 번까지다. 1번이 주 결과다 (labels.md 6절 재실행 규칙).
- 보류 시험셋 사용 횟수는 baseline 줄의 이전 사용 수에 이 기록부의 보류 실행 수를 더해 센다.
  baseline 줄이 없으면 보류 시험셋 실행을 거부한다 (이전 사용 수를 0 으로 가정하지 않는다).
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.judge_eval.golden import GoldenError

MAX_RUNS = 2
DEFAULT_LEDGER = Path("evals") / "run-ledger.jsonl"


class RunCapReached(GoldenError):
    """같은 묶음, 모델, 골든셋 버전의 실행 상한에 닿음."""


@dataclass(frozen=True)
class RunSlot:
    run_no: int
    primary: bool
    holdout_run_no: Optional[int]


def read_ledger(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def holdout_uses(entries: List[Dict[str, Any]]) -> Optional[int]:
    """지금까지의 보류 시험셋 사용 수. baseline 줄이 없으면 None."""

    baselines = [entry for entry in entries if entry.get("kind") == "baseline"]
    if not baselines:
        return None
    prior = int(baselines[-1].get("holdoutUses", 0))
    return prior + sum(1 for entry in entries if entry.get("kind") == "run" and entry.get("holdout"))


def next_slot(path: Path, *, suite: str, model: str, golden_version: str, holdout: bool) -> RunSlot:
    entries = read_ledger(path)
    done = [
        entry
        for entry in entries
        if entry.get("kind") == "run"
        and entry.get("suite") == suite
        and entry.get("model") == model
        and entry.get("goldenVersion") == golden_version
    ]
    if len(done) >= MAX_RUNS:
        raise RunCapReached(
            f"{suite}, {model}, 골든셋 {golden_version} 실행이 이미 {len(done)}회입니다 (상한 {MAX_RUNS}). "
            "다시 재려면 골든셋 버전이나 측정 대상을 바꾼다"
        )
    holdout_run_no = None
    if holdout:
        used = holdout_uses(entries)
        if used is None:
            raise GoldenError(
                f"실행 기록부에 보류 시험셋 baseline 줄이 없습니다: {path}. 이전 사용 수를 먼저 기록한다"
            )
        holdout_run_no = used + 1
    run_no = len(done) + 1
    return RunSlot(run_no=run_no, primary=run_no == 1, holdout_run_no=holdout_run_no)


def append_run(
    path: Path,
    slot: RunSlot,
    *,
    suite: str,
    model: str,
    variant: str,
    golden_version: str,
    golden_commit: Optional[str],
    holdout: bool,
    criteria_path: Optional[str],
    output: str,
) -> Dict[str, Any]:
    entry = {
        "kind": "run",
        "suite": suite,
        "model": model,
        "variant": variant,
        "goldenVersion": golden_version,
        "goldenCommit": golden_commit,
        "runNo": slot.run_no,
        "primary": slot.primary,
        "holdout": holdout,
        "holdoutRunNo": slot.holdout_run_no,
        "criteriaPath": criteria_path,
        "output": output,
        "startedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
