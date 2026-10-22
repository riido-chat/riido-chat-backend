"""판별 측정 실행기 명령.

    # 호출 수와 비용 추정 (모델을 부르지 않고 기록부도 건드리지 않음)
    python -m evaluation.judge_eval run --suite judge-official-A --judge luna --golden-version 1.0.0 --estimate

    # 실제 측정 (--live 가 있어야 실제 API 를 부른다)
    python -m evaluation.judge_eval run --suite judge-official-A --judge claude-haiku --golden-version 1.0.0 \\
        --criteria criteria/<파일>.md --build-commit <40자리 SHA> --profile-key HELP_CHATBOT \\
        --profile-revision-id 12 --output evaluation/registry/evals/<실행 ID>/judge-official-A.claude.run1.json --live

    python -m evaluation.judge_eval run --suite judge-official-A --judge jev --golden-version 1.0.0 \\
        --criteria criteria/<파일>.md --build-commit <40자리 SHA> --profile-key HELP_CHATBOT \\
        --profile-revision-id 12 --input-price <USD/100만 토큰> \\
        --output evaluation/registry/evals/<실행 ID>/judge-official-A.jev.run1.json --live

    # 결과 다시 채점, 두 결과 짝 비교
    python -m evaluation.judge_eval score --result <결과.json>
    python -m evaluation.judge_eval compare --base <luna 결과.json> --candidate <Jev 결과.json>

안전 장치
- 골든셋 버전과 동결 해시가 manifest 와 다르면 멈춘다.
- 보류 시험셋 묶음(judge-official-B 등)은 --holdout 과 확정해 커밋한 criteria/ 파일이 모두 있어야 연다.
  추정과 채점도 같다. 보고용 묶음의 실제 측정은 확정한 criteria/ 파일이 있어야 한다.
- (묶음, 모델, 골든셋 버전) 마다 실행 2회까지. 1회차가 주 결과다. 실행은 모델을 부르기 전에 기록부에 남긴다.
- 입력 확인(strict, 기본): 동결 payloadSha256 과 요청 입력 해시가 다른 턴이 있거나
  운영 지시문 해시가 동결 환경과 다르면 멈춘다. `instructions` 는 동결 당시 별도 해시로 확인한다.
  record 는 입력 해시 불일치를 기록하고 진행한다. 사용 여부는 사전 등록 기준에 적는다.
- 결과 파일은 덮어쓰지 않는다. 쓰기 전에 등록부의 eval-result.schema.json 으로 검사한다.
"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evaluation.judge_eval import golden as golden_mod
from evaluation.judge_eval import ledger as ledger_mod
from evaluation.judge_eval.clients import (
    DEFAULT_PRICES,
    JudgeOutcome,
    Prices,
    Usage,
    Variant,
    estimate,
    turn_cost,
    variant_by_name,
)
from evaluation.judge_eval.golden import Golden, GoldenError, HoldoutRefused, Label, Selection, TurnKey
from evaluation.judge_eval.inputs import PreparedTurn, prepare_turns
from evaluation.judge_eval.scoring import TurnScore, compare, score_turn, summarize
from app.question_grouping.prompt_v7_2 import JUDGE_INSTRUCTIONS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = PROJECT_ROOT / "evaluation" / "registry"
RUNNER_NAME = "evaluation/judge_eval"
RUNNER_VERSION = "judge-eval-v2"
SCHEMA_PATH = Path("schemas") / "eval-result.schema.json"
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class RunConfig:
    suite: str
    judge: str
    golden_version: str
    registry: Path = DEFAULT_REGISTRY
    output: Optional[Path] = None
    ledger: Optional[Path] = None
    criteria: Optional[str] = None
    holdout: bool = False
    build_commit: Optional[str] = None
    profile_key: Optional[str] = None
    profile_revision_id: Optional[int] = None
    health_checked_at: Optional[str] = None
    payload_check: str = "strict"
    prices: Optional[Prices] = None
    concurrency: int = 4
    args: List[str] = field(default_factory=list)

    def ledger_path(self, golden: Golden) -> Path:
        return self.ledger if self.ledger is not None else golden.root / ledger_mod.DEFAULT_LEDGER

    def resolved_prices(self) -> Prices:
        return self.prices if self.prices is not None else DEFAULT_PRICES[self.judge]


@dataclass
class Prepared:
    golden: Golden
    selection: Selection
    labels: Dict[TurnKey, Label]
    turns: List[PreparedTurn]
    catalog: Dict[str, Dict[str, Any]]
    environment: Dict[str, Any]
    criteria_path: Optional[str]


# ---------------------------------------------------------------- 준비와 관문


def _gate_holdout_and_criteria(config: RunConfig, golden: Golden, *, live: bool) -> Optional[str]:
    holdout = golden.is_holdout(config.suite)
    if config.holdout and not holdout:
        raise GoldenError(f"{config.suite} 는 보류 시험셋 묶음이 아닙니다. --holdout 을 빼라")
    if holdout and not config.holdout:
        raise HoldoutRefused(f"{config.suite} 는 보류 시험셋 묶음입니다. --holdout 없이는 열지 않습니다")
    needs_criteria = holdout or (live and golden.is_reportable(config.suite))
    if needs_criteria and not config.criteria:
        raise (HoldoutRefused if holdout else GoldenError)(
            f"{config.suite} 는 확정해 커밋한 criteria/ 기준 파일이 있어야 합니다 (--criteria)"
        )
    if config.criteria:
        return golden_mod.check_criteria(golden, config.criteria, config.suite)
    return None


def prepare(config: RunConfig, *, live: bool) -> Prepared:
    golden = golden_mod.open_golden(config.registry, config.golden_version)
    criteria_path = _gate_holdout_and_criteria(config, golden, live=live)
    selection = golden_mod.select_suite(golden, config.suite, holdout_allowed=config.holdout)
    labels = golden_mod.load_labels(golden, selection)
    rows = golden_mod.load_frozen_rows(golden, selection.keys)
    catalog = golden_mod.load_catalog(golden)
    turns = prepare_turns(selection.turns, rows, catalog, golden_mod.load_supplement(golden))
    return Prepared(
        golden=golden,
        selection=selection,
        labels=labels,
        turns=turns,
        catalog=catalog,
        environment=golden_mod.load_environment(golden),
        criteria_path=criteria_path,
    )


def payload_status(turns: Sequence[PreparedTurn]) -> Dict[str, int]:
    judged = [turn for turn in turns if turn.judged]
    verified = sum(1 for turn in judged if turn.payload_verified)
    return {"judgedTurns": len(judged), "verified": verified, "unverified": len(judged) - verified}


def run_estimate(config: RunConfig) -> Dict[str, Any]:
    variant = variant_by_name(config.judge)
    prepared = prepare(config, live=False)
    result = estimate(variant, prepared.turns, config.resolved_prices())
    result.update(
        {
            "suite": config.suite,
            "goldenVersion": prepared.golden.version,
            "layers": prepared.selection.layers,
            "payloadCheck": payload_status(prepared.turns),
            "instructionsSha256Verified": (
                (prepared.environment.get("judge") or {}).get("instructionsSha256")
                == hashlib.sha256(JUDGE_INSTRUCTIONS.encode("utf-8")).hexdigest()
            ),
        }
    )
    return result


# ---------------------------------------------------------------- 실행


def _git_output(root: Path, *args: str) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _app_revision() -> Dict[str, Any]:
    revision = _git_output(PROJECT_ROOT, "rev-parse", "HEAD") or "0000000"
    status = _git_output(PROJECT_ROOT, "status", "--porcelain")
    return {"gitRevision": revision, "dirty": bool(status) if status is not None else True}


def _baseline_build_commit(environment: Dict[str, Any]) -> Optional[str]:
    """고정 입력을 만든 앱 저장소의 커밋을 반환한다."""

    commit = (environment.get("mainRepo") or {}).get("commit")
    return commit if isinstance(commit, str) and COMMIT_SHA.fullmatch(commit) else None


def load_schema_checker(golden: Golden):
    """등록부 scripts/validate.py 의 make_schema_checker 로 결과 스키마 검사 함수를 만든다."""

    schema_path = golden.root / SCHEMA_PATH
    validate_path = golden.root / "scripts" / "validate.py"
    if not schema_path.exists() or not validate_path.exists():
        raise GoldenError("등록부에 결과 스키마나 scripts/validate.py 가 없습니다")
    spec = importlib.util.spec_from_file_location("riido_evals_validate", validate_path)
    module = importlib.util.module_from_spec(spec)
    # validate.py 가 같은 폴더의 build_suites, stages 등을 import 한다. 다른 등록부와 섞이지 않게 끝나면 걷어낸다.
    before = set(sys.modules)
    sys.path.insert(0, str(validate_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        while str(validate_path.parent) in sys.path:
            sys.path.remove(str(validate_path.parent))
        for name in set(sys.modules) - before:
            if getattr(sys.modules[name], "__file__", "") and str(validate_path.parent) in str(sys.modules[name].__file__):
                del sys.modules[name]
    checker, _ = module.make_schema_checker(json.loads(schema_path.read_text(encoding="utf-8")))
    return checker


def schema_supports_identity_v2(golden: Golden) -> bool:
    schema = json.loads((golden.root / SCHEMA_PATH).read_text(encoding="utf-8"))
    version = schema.get("properties", {}).get("schemaVersion", {})
    if "2" not in version.get("enum", []):
        return False
    branch = next(
        (
            clause.get("then", {})
            for clause in schema.get("allOf", [])
            if isinstance(clause, dict)
            and clause.get("if", {}).get("properties", {}).get("schemaVersion", {}).get("const") == "2"
        ),
        {},
    )
    target = schema.get("properties", {}).get("target", {})
    required_target = {"buildCommit", "profileKey", "profileRevisionId"}
    if not required_target.issubset(target.get("required", [])):
        return False
    target_properties = target.get("properties", {})
    if any(name not in target_properties for name in required_target):
        return False
    if (
        target_properties["buildCommit"].get("type") != "string"
        or target_properties["buildCommit"].get("pattern") != r"^[0-9a-f]{40}$"
        or target_properties["profileKey"].get("type") != "string"
        or target_properties["profileKey"].get("minLength", 0) < 1
        or target_properties["profileRevisionId"].get("type") != "integer"
        or target_properties["profileRevisionId"].get("minimum", 0) < 1
    ):
        return False
    if "target" not in branch.get("required", []):
        return False
    branch_properties = branch.get("properties", {})
    if branch_properties.get("systemVersion", {}).get("type") != "null":
        return False
    fixed_inputs = branch_properties.get("fixedInputs", {})
    legacy_field = fixed_inputs.get("then", {}).get("properties", {}).get("baselineSystemVersion", {})
    guide_index = branch_properties.get("guideIndex", {})
    index_version = guide_index.get("properties", {}).get("indexVersionId", {})
    if (
        guide_index.get("type") != "object"
        or "indexVersionId" not in guide_index.get("required", [])
        or index_version.get("type") != "integer"
        or index_version.get("minimum", 0) < 1
    ):
        return False
    summary = branch_properties.get("summary", {})
    summary_inputs = summary.get("properties", {}).get("inputs", {})
    catalog_fingerprint = summary_inputs.get("properties", {}).get("catalogFingerprint", {})
    if (
        "inputs" not in summary.get("required", [])
        or "catalogFingerprint" not in summary_inputs.get("required", [])
        or catalog_fingerprint.get("type") != "string"
        or catalog_fingerprint.get("minLength", 0) < 1
    ):
        return False
    return (
        fixed_inputs.get("if", {}).get("type") == "object"
        and {"baselineBuildCommit", "path"}.issubset(fixed_inputs.get("then", {}).get("required", []))
        and legacy_field.get("type") == "null"
    )


def _display_path(path: Path, golden: Golden) -> str:
    """결과와 기록부에 남길 경로. 등록부나 앱 저장소 안이면 상대 경로, 밖이면 파일 이름만 남긴다."""

    resolved = Path(path).resolve()
    for base in (golden.root, PROJECT_ROOT):
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError:
            continue
    return resolved.name


def _raw_path(output: Path) -> Path:
    return output.with_name(output.stem + ".raw.jsonl")


def _check_live_preconditions(config: RunConfig, golden: Golden) -> None:
    if golden.commit is None or golden.dirty:
        raise GoldenError("등록부가 git 커밋 상태가 아니거나 golden, schemas, scripts, criteria 에 커밋하지 않은 변경이 있습니다")
    if not isinstance(config.build_commit, str) or not COMMIT_SHA.fullmatch(config.build_commit):
        raise GoldenError("--build-commit 에 서버가 보고한 40자리 소문자 커밋 SHA를 준다")
    if not isinstance(config.profile_key, str) or not config.profile_key.strip():
        raise GoldenError("--profile-key 에 서빙 중인 프로필 키를 준다")
    if (
        not isinstance(config.profile_revision_id, int)
        or isinstance(config.profile_revision_id, bool)
        or config.profile_revision_id < 1
    ):
        raise GoldenError("--profile-revision-id 에 서빙 중인 양의 프로필 버전 ID를 준다")
    environment = golden_mod.load_environment(golden)
    expected_instructions_sha256 = (environment.get("judge") or {}).get("instructionsSha256")
    actual_instructions_sha256 = hashlib.sha256(JUDGE_INSTRUCTIONS.encode("utf-8")).hexdigest()
    if expected_instructions_sha256 != actual_instructions_sha256:
        raise GoldenError("현재 판별 지시문이 동결 환경의 instructionsSha256 과 다릅니다")
    if _baseline_build_commit(environment) is None:
        raise GoldenError("동결 입력 환경에 기준 앱 커밋 SHA가 없습니다")
    index_version_id = (environment.get("index") or {}).get("indexVersionId")
    if (
        not isinstance(index_version_id, int)
        or isinstance(index_version_id, bool)
        or index_version_id < 1
    ):
        raise GoldenError("동결 입력 환경에 양의 기준 색인 버전 ID가 없습니다")
    catalog_fingerprint = (environment.get("catalog") or {}).get("fingerprint")
    if not isinstance(catalog_fingerprint, str) or not catalog_fingerprint.strip():
        raise GoldenError("동결 입력 환경에 기준 카탈로그 지문이 없습니다")
    if not schema_supports_identity_v2(golden):
        raise GoldenError("평가 등록부 결과 스키마가 schemaVersion 2를 지원하지 않습니다")
    if config.output is None:
        raise GoldenError("--output 이 필요합니다")
    if config.output.exists() or _raw_path(config.output).exists():
        raise GoldenError(f"결과 파일이 이미 있습니다. 덮어쓰지 않습니다: {config.output}")
    if config.resolved_prices().input is None:
        raise GoldenError(f"{config.judge} 단가를 모릅니다. --input-price 를 준다")


async def _judge_all(adapter: Any, turns: Sequence[PreparedTurn], concurrency: int) -> List[Optional[JudgeOutcome]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(turn: PreparedTurn) -> Optional[JudgeOutcome]:
        if not turn.judged:
            return None
        async with semaphore:
            return await adapter.judge(turn)

    return list(await asyncio.gather(*(one(turn) for turn in turns)))


def _score_all(prepared: Prepared, actual: Dict[TurnKey, Tuple[Optional[str], Optional[str], bool]]) -> List[TurnScore]:
    serving = {key: entry.get("servingState") for key, entry in prepared.catalog.items()}
    scores = []
    for turn in prepared.turns:
        decision, key, error = actual[turn.key]
        scores.append(
            score_turn(turn.key, turn.layer, prepared.labels[turn.key], turn.presented_keys, decision, key, error, serving)
        )
    return scores


def _latency(outcomes: Sequence[Optional[JudgeOutcome]]) -> Dict[str, Optional[float]]:
    values = sorted(o.latency_ms for o in outcomes if o is not None and o.latency_ms is not None)
    if not values:
        return {"n": 0, "meanMs": None, "p50Ms": None, "p95Ms": None}

    def pick(q: float) -> float:
        return float(values[min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))])

    return {"n": len(values), "meanMs": sum(values) / len(values), "p50Ms": pick(0.5), "p95Ms": pick(0.95)}


def _input_set_sha256(turns: Sequence[PreparedTurn]) -> str:
    joined = "\n".join(f"{t.item_id}\t{t.turn_no}\t{t.input_sha256 or ''}" for t in turns)
    return golden_mod.sha256_bytes(joined.encode("utf-8"))


async def execute_run(config: RunConfig, adapter: Optional[Any]) -> Dict[str, Any]:
    """관문을 모두 지나면 기록부에 실행을 남기고 판별을 돌린 뒤 결과를 쓴다. 결과 객체를 돌려준다."""

    variant: Variant = variant_by_name(config.judge)
    golden = golden_mod.open_golden(config.registry, config.golden_version)
    _gate_holdout_and_criteria(config, golden, live=True)
    _check_live_preconditions(config, golden)
    checker = load_schema_checker(golden)
    if adapter is None:
        adapter = _live_adapter(config.judge)
    assert adapter is not None
    if adapter.variant.name != variant.name:
        raise GoldenError(f"어댑터({adapter.variant.name})와 --judge({variant.name})가 다릅니다")
    ledger_path = config.ledger_path(golden)
    holdout = golden.is_holdout(config.suite)
    slot = ledger_mod.next_slot(
        ledger_path, suite=config.suite, model=variant.model, golden_version=golden.version, holdout=holdout
    )
    prepared = prepare(config, live=True)
    status = payload_status(prepared.turns)
    if config.payload_check == "strict" and status["unverified"]:
        raise GoldenError(
            f"재구성한 판별 요청 {status['unverified']}/{status['judgedTurns']}턴이 동결 payloadSha256 과 다릅니다. "
            "동결 입력과 보충 파일을 확인하거나, 사전 등록 기준에 적고 --payload-check record 로 돌린다"
        )
    assert config.output is not None
    ledger_mod.append_run(
        ledger_path,
        slot,
        suite=config.suite,
        model=variant.model,
        variant=variant.name,
        golden_version=golden.version,
        golden_commit=golden.commit,
        holdout=holdout,
        criteria_path=prepared.criteria_path,
        output=_display_path(config.output, golden),
    )

    executed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    outcomes = await _judge_all(adapter, prepared.turns, config.concurrency)
    finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    prices = config.resolved_prices()
    raw_path = _raw_path(config.output)
    raw_lines: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    actual: Dict[TurnKey, Tuple[Optional[str], Optional[str], bool]] = {}
    totals = {"inputTokens": 0, "outputTokens": 0, "cachedInputTokens": 0, "reasoningTokens": 0, "calls": 0}
    cached_usd = upper_usd = 0.0
    for index, (turn, outcome) in enumerate(zip(prepared.turns, outcomes)):
        ref = f"{raw_path.name}#/{index}"
        line: Dict[str, Any] = {
            "itemId": turn.item_id,
            "layer": turn.layer,
            "turnNo": turn.turn_no,
            "attempt": 1,
            "inputSha256": turn.input_sha256,
            "requestSha256": turn.request_sha256,
            "frozenPayloadSha256": turn.frozen_payload_sha256,
            "payloadVerified": turn.payload_verified if turn.judged else None,
        }
        entry: Dict[str, Any] = {
            "itemId": turn.item_id,
            "layer": turn.layer,
            "turnNo": turn.turn_no,
            "attempt": 1,
            "rawOutputRef": ref,
        }
        if outcome is None:
            line["skipReason"] = turn.skip_reason
            entry.update({"outcome": "SKIPPED", "skipReason": turn.skip_reason or "판별하지 않음"})
            actual[turn.key] = (None, None, False)
        else:
            cost = turn_cost(variant.name, outcome.usage, prices)
            decision = outcome.decision or "UNCLASSIFIED"
            line.update(
                {
                    "decision": outcome.decision,
                    "subproblemKey": outcome.subproblem_key,
                    "confidence": outcome.confidence,
                    "invalidReason": outcome.invalid_reason,
                    "error": outcome.error,
                    "usage": outcome.usage.to_json(),
                    "attempts": outcome.attempts,
                    "latencyMs": outcome.latency_ms,
                    "costUsd": cost,
                    "raw": outcome.raw,
                }
            )
            entry.update(
                {
                    "outcome": "ERROR" if outcome.error else "SCORED",
                    "actual": {"judgeDecision": decision, "subproblemKey": outcome.subproblem_key},
                }
            )
            if outcome.error:
                entry["error"] = outcome.error
            if outcome.latency_ms is not None:
                entry["latencyMs"] = outcome.latency_ms
            if cost["cachedUsd"] is not None:
                entry["costUsd"] = cost["cachedUsd"]
                cached_usd += cost["cachedUsd"]
                upper_usd += cost["upperUsd"]
            for name, value in outcome.usage.to_json().items():
                totals[name] += value or 0
            totals["calls"] += outcome.attempts
            actual[turn.key] = (decision, outcome.subproblem_key, bool(outcome.error))
        raw_lines.append(line)
        results.append(entry)

    scores = _score_all(prepared, actual)
    for entry, score in zip(results, scores):
        if score.scored:
            entry["scores"] = {"judge": score.judge_correct}

    env_index = prepared.environment.get("index") or {}
    result: Dict[str, Any] = {
        "schemaVersion": "2",
        "target": {
            "buildCommit": config.build_commit,
            "profileKey": config.profile_key,
            "profileRevisionId": config.profile_revision_id,
        },
        "goldenVersion": golden.version,
        "goldenCommit": golden.commit,
        "suite": config.suite,
        "layers": prepared.selection.layers,
        "holdoutRunNo": slot.holdout_run_no,
        "criteriaPath": prepared.criteria_path,
        "runNo": slot.run_no,
        "primary": slot.primary,
        "executedAt": executed_at,
        "finishedAt": finished_at,
        "runner": {"name": RUNNER_NAME, "version": RUNNER_VERSION, "args": list(config.args)},
        "app": _app_revision(),
        "guideIndex": {
            "groupKey": env_index.get("documentGroup"),
            "indexVersionId": env_index.get("indexVersionId"),
            "version": env_index.get("version"),
        },
        "models": {"judge": variant.stage_model()},
        "fixedInputs": {
            "baselineBuildCommit": _baseline_build_commit(prepared.environment),
            "path": golden.relative(golden.judge_inputs),
        },
        "repeat": 1,
        "cost": {"cachedUsd": round(cached_usd, 6), "upperUsd": round(upper_usd, 6)},
        "summary": {
            "judgeVariant": {"name": variant.name, "settings": variant.settings},
            "run": {"runNo": slot.run_no, "primary": slot.primary, "ledger": _display_path(ledger_path, golden)},
            "suiteSelection": dict(prepared.selection.meta, turns=len(prepared.turns)),
            "inputs": {
                "judgeInputsSha256": golden_mod.sha256_file(golden.judge_inputs),
                "catalogFingerprint": prepared.environment.get("catalog", {}).get("fingerprint"),
                "supplement": None if golden.supplement is None else golden.relative(golden.supplement),
                "payloadCheck": dict(payload_status(prepared.turns), mode=config.payload_check),
                "inputSetSha256": _input_set_sha256(prepared.turns),
            },
            "usage": totals,
            "latency": _latency(outcomes),
            "prices": {"input": prices.input, "output": prices.output, "cachedInput": prices.cached_input},
            "costNote": None if prices.cached_input is not None else "캐시 입력 단가를 몰라 cachedUsd 는 upperUsd 와 같다",
            "scoring": summarize(scores),
            "scoresJudge": "results[].scores.judge 는 판별기 층 정답 (후보 밖 정답 턴은 연결하지 않으면 정답)",
        },
        "results": results,
    }
    if config.health_checked_at:
        result["healthCheckedAt"] = config.health_checked_at
    errors = checker(result)
    if errors:
        raise GoldenError("결과가 eval-result.schema.json 에 맞지 않습니다: " + "; ".join(errors[:5]))
    config.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in raw_lines), encoding="utf-8")
    config.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


# ---------------------------------------------------------------- 다시 채점과 비교


def _load_result(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    raw_path = _raw_path(path)
    raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return result, raw


def rescore(registry: Path, path: Path, *, holdout: bool = False) -> Tuple[Dict[str, Any], Dict[TurnKey, TurnScore], List[Dict[str, Any]]]:
    result, raw = _load_result(path)
    golden = golden_mod.open_golden(registry, result["goldenVersion"])
    is_holdout = golden.is_holdout(result["suite"])
    if is_holdout and not holdout:
        raise HoldoutRefused(f"{result['suite']} 는 보류 시험셋 결과입니다. --holdout 없이는 라벨을 읽지 않습니다")
    turns = [(entry["itemId"], entry["layer"], entry["turnNo"]) for entry in result["results"]]
    selection = Selection(
        suite=result["suite"], layers=list(result["layers"]), holdout=is_holdout, reportable=False, turns=turns
    )
    labels = golden_mod.load_labels(golden, selection)
    rows = golden_mod.load_frozen_rows(golden, selection.keys)
    serving = {key: entry.get("servingState") for key, entry in golden_mod.load_catalog(golden).items()}
    scores: Dict[TurnKey, TurnScore] = {}
    for entry in result["results"]:
        key = (entry["itemId"], entry["turnNo"])
        actual = entry.get("actual") or {}
        decision = None if entry["outcome"] == "SKIPPED" else actual.get("judgeDecision")
        scores[key] = score_turn(
            key,
            entry["layer"],
            labels[key],
            rows[key].get("presentedSubproblemKeys") or [],
            decision,
            actual.get("subproblemKey"),
            entry["outcome"] == "ERROR",
            serving,
        )
    return result, scores, raw


def compare_results(registry: Path, base: Path, candidate: Path, *, holdout: bool = False) -> Dict[str, Any]:
    base_result, base_scores, base_raw = rescore(registry, base, holdout=holdout)
    cand_result, cand_scores, cand_raw = rescore(registry, candidate, holdout=holdout)
    for name in ("goldenVersion", "suite"):
        if base_result[name] != cand_result[name]:
            raise GoldenError(f"두 결과의 {name} 가 다릅니다: {base_result[name]} != {cand_result[name]}")
    if set(base_scores) != set(cand_scores):
        raise GoldenError("두 결과의 턴 구성이 다릅니다. 같은 턴끼리만 비교한다")
    base_inputs = {(line["itemId"], line["turnNo"]): line.get("inputSha256") for line in base_raw}
    cand_inputs = {(line["itemId"], line["turnNo"]): line.get("inputSha256") for line in cand_raw}
    differing = sorted(key for key in base_inputs if base_inputs[key] != cand_inputs.get(key))
    if differing:
        raise GoldenError(f"판별 입력이 다른 턴이 {len(differing)}개 있습니다. 같은 입력끼리만 비교한다")
    notes = []
    for label, result in (("base", base_result), ("candidate", cand_result)):
        if not result.get("primary", False):
            notes.append(f"{label} 는 주 결과(1회차)가 아니다. 진단용으로만 읽는다")
    return {
        "goldenVersion": base_result["goldenVersion"],
        "suite": base_result["suite"],
        "base": {"model": base_result["models"]["judge"], "runNo": base_result.get("runNo"), "primary": base_result.get("primary")},
        "candidate": {"model": cand_result["models"]["judge"], "runNo": cand_result.get("runNo"), "primary": cand_result.get("primary")},
        "inputsIdentical": True,
        "notes": notes,
        "paired": compare(base_scores, cand_scores),
        "baseSummary": summarize(list(base_scores.values())),
        "candidateSummary": summarize(list(cand_scores.values())),
    }


# ---------------------------------------------------------------- 명령줄


def _live_adapter(
    judge: str,
    *,
    max_attempts: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
) -> Any:
    from evaluation.judge_eval.clients import (
        ClaudeJudge,
        JevJudge,
        LunaJudge,
        urllib_claude_transport,
        urllib_jev_transport,
    )

    if judge == "luna":
        return LunaJudge(max_attempts=max_attempts)
    if judge == "jev":
        key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not key:
            raise GoldenError("TYPESAFE_API_KEY 환경 변수가 필요합니다")
        return JevJudge(urllib_jev_transport(key), max_attempts=max_attempts)
    if not judge.startswith("claude-"):
        raise GoldenError(f"지원하지 않는 판별 모델입니다: {judge}")
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise GoldenError("ANTHROPIC_API_KEY 환경 변수가 필요합니다")
    return ClaudeJudge(
        urllib_claude_transport(key),
        variant=variant_by_name(judge),
        max_attempts=max_attempts,
        max_output_tokens=max_output_tokens,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluation.judge_eval", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="판별 측정 실행 또는 추정")
    run.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    run.add_argument("--golden-version", required=True, help="기대 골든셋 버전 (manifest 와 다르면 멈춤)")
    run.add_argument("--suite", required=True)
    run.add_argument(
        "--judge", required=True,
        choices=sorted(("luna", "jev", "claude-haiku", "claude-sonnet", "claude-opus", "claude-fable")),
    )
    run.add_argument("--output", type=Path)
    run.add_argument("--ledger", type=Path, help="실행 기록부 (기본 <등록부>/evals/run-ledger.jsonl)")
    run.add_argument("--criteria", help="등록부 기준 상대 경로 criteria/<파일>.md")
    run.add_argument("--holdout", action="store_true", help="보류 시험셋 묶음을 연다 (확정한 criteria 필요)")
    run.add_argument("--build-commit", help="서버가 /health 에 보고한 40자리 빌드 커밋 SHA")
    run.add_argument("--profile-key", help="/health/profiles 에서 확인한 프로필 키")
    run.add_argument("--profile-revision-id", type=int, help="/health/profiles 에서 확인한 서빙 프로필 버전 ID")
    run.add_argument("--health-checked-at")
    run.add_argument("--payload-check", choices=("strict", "record"), default="strict")
    run.add_argument("--input-price", type=float, help="USD/100만 입력 토큰")
    run.add_argument("--output-price", type=float, help="USD/100만 출력 토큰")
    run.add_argument("--cached-input-price", type=float, help="USD/100만 캐시 입력 토큰")
    run.add_argument("--concurrency", type=int, default=4)
    mode = run.add_mutually_exclusive_group(required=True)
    mode.add_argument("--estimate", "--dry-run", dest="estimate", action="store_true", help="호출 없이 호출 수와 비용 추정")
    mode.add_argument("--live", action="store_true", help="실제 API 를 부른다")

    score = sub.add_parser("score", help="결과 파일을 골든 라벨로 다시 채점")
    score.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    score.add_argument("--result", type=Path, required=True)
    score.add_argument("--holdout", action="store_true")

    comp = sub.add_parser("compare", help="같은 턴의 두 결과를 짝지어 비교")
    comp.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    comp.add_argument("--base", type=Path, required=True)
    comp.add_argument("--candidate", type=Path, required=True)
    comp.add_argument("--holdout", action="store_true")
    return parser


def _prices(args: argparse.Namespace) -> Prices:
    default = DEFAULT_PRICES[args.judge]
    return Prices(
        input=args.input_price if args.input_price is not None else default.input,
        output=args.output_price if args.output_price is not None else default.output,
        cached_input=args.cached_input_price if args.cached_input_price is not None else default.cached_input,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            config = RunConfig(
                suite=args.suite,
                judge=args.judge,
                golden_version=args.golden_version,
                registry=args.registry,
                output=args.output,
                ledger=args.ledger,
                criteria=args.criteria,
                holdout=args.holdout,
                build_commit=args.build_commit,
                profile_key=args.profile_key,
                profile_revision_id=args.profile_revision_id,
                health_checked_at=args.health_checked_at,
                payload_check=args.payload_check,
                prices=_prices(args),
                concurrency=args.concurrency,
                args=list(argv if argv is not None else sys.argv[1:]),
            )
            if args.estimate:
                print(json.dumps(run_estimate(config), ensure_ascii=False, indent=2))
                return 0
            result = asyncio.run(execute_run(config, None))
            print(json.dumps({"output": str(args.output), "runNo": result["runNo"], "cost": result["cost"]}, ensure_ascii=False))
            return 0
        if args.command == "score":
            _, scores, _ = rescore(args.registry, args.result, holdout=args.holdout)
            print(json.dumps(summarize(list(scores.values())), ensure_ascii=False, indent=2))
            return 0
        comparison = compare_results(args.registry, args.base, args.candidate, holdout=args.holdout)
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        return 0
    except GoldenError as exc:
        print(f"거부: {exc}", file=sys.stderr)
        return 2
