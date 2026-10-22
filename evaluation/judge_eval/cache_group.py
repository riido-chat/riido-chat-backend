"""Run model-only cache-group comparisons against the private frozen group.

This use case evaluates only the cache-path members that reach the judge
(P3/P4/P6). P1 exact hits intentionally bypass all judge models.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from evaluation.judge_eval import cli, golden as golden_mod
from evaluation.judge_eval.clients import (
    DEFAULT_PRICES,
    JudgeOutcome,
    Prices,
    estimate,
    turn_cost,
    variant_by_name,
)
from evaluation.judge_eval.golden import Golden, GoldenError, Label, Selection, TurnKey
from evaluation.judge_eval.inputs import PreparedTurn, prepare_turns
from evaluation.judge_eval.scoring import TurnScore
from app.question_grouping.prompt_v7_2 import JUDGE_INSTRUCTIONS


DEFAULT_GROUP = Path("golden/groups/cache-path/2.0.0.json")
MODEL_PATHS = {"P3", "P4", "P6"}
CLAUDE_BUDGET_USD = 10.0
CLAUDE_COMPARISON_MAX_ATTEMPTS = 1
CLAUDE_COMPARISON_MAX_OUTPUT_TOKENS = 512
CLAUDE_VARIANTS = ("claude-haiku", "claude-sonnet", "claude-opus", "claude-fable")


@dataclass
class CacheGroupInput:
    golden: Golden
    group_path: Path
    group: Dict[str, Any]
    turns: List[PreparedTurn]
    labels: Dict[TurnKey, Label]
    catalog: Dict[str, Dict[str, Any]]
    environment: Dict[str, Any]


def _canonical_sha256(value: Mapping[str, Any], *, omit: Optional[str] = None) -> str:
    content = {key: entry for key, entry in value.items() if key != omit}
    text = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_group(registry: Path, group_path: Path) -> None:
    for script_name in ("validate_evaluation_group.py", "validate_cache_group.py"):
        command = [
            sys.executable,
            str(registry / "scripts" / script_name),
            str(group_path),
            "--golden-manifest",
            str(registry / "golden" / "manifest.json"),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise GoldenError(f"캐시 그룹 검증 실패 ({script_name}): {result.stderr.strip()[-500:]}")


def _layers_for_members(golden: Golden, member_ids: Sequence[str]) -> Dict[str, str]:
    wanted = set(member_ids)
    found: Dict[str, str] = {}
    for layer, meta in golden.manifest["layers"].items():
        path = golden.root / meta["path"]
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item_id = json.loads(line).get("id")
            if item_id in wanted:
                if item_id in found:
                    raise GoldenError(f"골든셋 문항 ID가 여러 층에 중복되어 있습니다: {item_id}")
                found[item_id] = layer
    missing = wanted - set(found)
    if missing:
        raise GoldenError(f"캐시 그룹 문항을 골든셋에서 찾지 못했습니다: {sorted(missing)[:3]}")
    return found


def prepare_cache_group(registry: Path, group_file: Path = DEFAULT_GROUP) -> CacheGroupInput:
    registry = Path(registry).resolve()
    group_path = Path(group_file)
    if not group_path.is_absolute():
        group_path = registry / group_path
    group_path = group_path.resolve()
    try:
        group_path.relative_to(registry)
    except ValueError as exc:
        raise GoldenError("캐시 그룹 파일은 비공개 평가 등록부 안에 있어야 합니다") from exc

    try:
        group = json.loads(group_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldenError(f"캐시 그룹 파일을 읽지 못했습니다: {group_path}") from exc
    if not isinstance(group, dict) or group.get("groupId") != "cache-path":
        raise GoldenError("cache-path 평가 그룹 JSON이 아닙니다")
    if group.get("groupSha256") != _canonical_sha256(group, omit="groupSha256"):
        raise GoldenError("캐시 그룹 content hash 가 유효하지 않습니다")

    golden = golden_mod.open_golden(registry, group.get("sourceGoldenVersion"))
    _validate_group(registry, group_path)

    members = group.get("members") or []
    model_members = [member for member in members if (member.get("labels") or {}).get("path") in MODEL_PATHS]
    member_ids = [member["itemId"] for member in model_members]
    layers_by_id = _layers_for_members(golden, member_ids)
    refs = [(member["itemId"], layers_by_id[member["itemId"]], int(member["turnNo"])) for member in model_members]
    keys = [(item_id, turn_no) for item_id, _, turn_no in refs]

    source_selection = Selection(
        suite="cache-path@" + group["groupVersion"],
        layers=sorted(set(layers_by_id.values())),
        holdout=False,
        reportable=False,
        turns=refs,
    )
    source_labels = golden_mod.load_labels(golden, source_selection)
    labels: Dict[TurnKey, Label] = {}
    for member in model_members:
        labels_meta = member["labels"]
        key = (member["itemId"], int(member["turnNo"]))
        expected = labels_meta.get("expectedJudgeDecision")
        expected_keys = tuple(labels_meta.get("expectedSubproblemKeysAny") or ())
        source = source_labels[key]
        if source.decision != expected or set(source.keys) != set(expected_keys):
            raise GoldenError(f"캐시 그룹 라벨이 골든셋 라벨과 다릅니다: {key}")
        labels[key] = Label(expected, expected_keys, "cache-path-group", "confirmed")

    rows = golden_mod.load_frozen_rows(golden, keys)
    catalog = golden_mod.load_catalog(golden)
    turns = prepare_turns(refs, rows, catalog, golden_mod.load_supplement(golden))
    unverified = [turn.key for turn in turns if not turn.payload_verified]
    if unverified:
        raise GoldenError(
            f"운영 판별 payload 해시가 동결 입력과 다른 턴이 {len(unverified)}개 있습니다: {unverified[:3]}"
        )
    environment = golden_mod.load_environment(golden)
    expected_prompt_sha = (environment.get("judge") or {}).get("instructionsSha256")
    actual_prompt_sha = hashlib.sha256(JUDGE_INSTRUCTIONS.encode("utf-8")).hexdigest()
    if expected_prompt_sha != actual_prompt_sha:
        raise GoldenError("현재 운영 판별 지시문이 이 그룹의 동결 지시문과 다릅니다")
    return CacheGroupInput(golden, group_path, group, turns, labels, catalog, environment)


def _input_set_sha256(turns: Sequence[PreparedTurn]) -> str:
    joined = "\n".join(f"{turn.item_id}\t{turn.turn_no}\t{turn.input_sha256 or ''}" for turn in turns)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _score_rows(
    context: CacheGroupInput,
    turns: Sequence[PreparedTurn],
    outcomes: Sequence[Optional[JudgeOutcome]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    serving = {key: entry.get("servingState") for key, entry in context.catalog.items()}
    group_members = {(m["itemId"], int(m["turnNo"])): m for m in context.group["members"]}
    rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    scored: List[TurnScore] = []
    by_path: Dict[str, List[TurnScore]] = {}
    costs = {"cachedUsd": 0.0, "upperUsd": 0.0}
    usage_totals = {"inputTokens": 0, "outputTokens": 0, "cachedInputTokens": 0, "reasoningTokens": 0, "calls": 0}
    for turn, outcome in zip(turns, outcomes):
        member = group_members[turn.key]
        path = member["labels"]["path"]
        label = context.labels[turn.key]
        expected_keys = list(label.keys)
        if outcome is None:
            raise GoldenError(f"모델 판별 결과가 없습니다: {turn.key}")
        decision = outcome.decision or "UNCLASSIFIED"
        item = {
            "itemId": turn.item_id,
            "layer": turn.layer,
            "turnNo": turn.turn_no,
            "path": path,
            "inputSha256": turn.input_sha256,
            "frozenPayloadSha256": turn.frozen_payload_sha256,
            "expectedDecision": label.decision,
            "expectedSubproblemKeysAny": expected_keys,
            "actualDecision": outcome.decision,
            "actualSubproblemKey": outcome.subproblem_key,
            "actualModel": (outcome.raw or {}).get("responseModel") or (outcome.raw or {}).get("model"),
            "confidence": outcome.confidence,
            "invalidReason": outcome.invalid_reason,
            "error": outcome.error,
            "latencyMs": outcome.latency_ms,
            "usage": outcome.usage.to_json(),
            "attempts": outcome.attempts,
        }
        rows.append(item)
        raw_rows.append({
            "itemId": turn.item_id,
            "turnNo": turn.turn_no,
            "decision": outcome.decision,
            "subproblemKey": outcome.subproblem_key,
            "raw": outcome.raw,
        })
        score = cli.score_turn(turn.key, turn.layer, label, turn.presented_keys, decision, outcome.subproblem_key,
                               bool(outcome.error), serving)
        scored.append(score)
        by_path.setdefault(path, []).append(score)
        for name, value in outcome.usage.to_json().items():
            usage_totals[name] += value or 0
        usage_totals["calls"] += outcome.attempts
    summary = {
        "inputSetSha256": _input_set_sha256(turns),
        "turns": len(turns),
        "usage": usage_totals,
        "byPath": {
            path: {"turns": len(entries), "correct": sum(score.semantic_correct for score in entries),
                   "accuracy": sum(score.semantic_correct for score in entries) / len(entries)}
            for path, entries in sorted(by_path.items())
        },
    }
    return rows, raw_rows, summary


def estimate_cache_group(context: CacheGroupInput, judge: str, prices: Optional[Prices] = None) -> Dict[str, Any]:
    variant = variant_by_name(judge)
    if judge.startswith("claude-"):
        variant = replace(
            variant,
            max_attempts=CLAUDE_COMPARISON_MAX_ATTEMPTS,
            settings={**variant.settings, "maxOutputTokens": CLAUDE_COMPARISON_MAX_OUTPUT_TOKENS},
        )
    estimate_result = estimate(
        variant,
        context.turns,
        prices or DEFAULT_PRICES[judge],
        max_output_tokens=(CLAUDE_COMPARISON_MAX_OUTPUT_TOKENS if judge.startswith("claude-") else None),
    )
    path_counts = Counter(
        member["labels"]["path"] for member in context.group["members"] if member["labels"]["path"] in MODEL_PATHS
    )
    return {
        "groupId": context.group["groupId"],
        "groupVersion": context.group["groupVersion"],
        "groupSha256": context.group["groupSha256"],
        "goldenVersion": context.golden.version,
        "groupMembers": len(context.group["members"]),
        "judgeTurns": len(context.turns),
        "modelPathCounts": dict(sorted(path_counts.items())),
        "payloadHashesVerified": sum(turn.payload_verified for turn in context.turns),
        "p1BypassesJudge": sum(member["labels"]["path"] == "P1" for member in context.group["members"]),
        "model": variant.stage_model(),
        "estimate": estimate_result,
    }


def _claude_spent_or_reserved(registry: Path) -> float:
    ledger_path = registry / "evals" / "cache-path" / "model-comparison-ledger.jsonl"
    if not ledger_path.exists():
        return 0.0
    starts: Dict[str, Dict[str, Any]] = {}
    finishes: Dict[str, Dict[str, Any]] = {}
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if not str(entry.get("model", "")).startswith("claude-"):
            continue
        run_id = entry.get("runId")
        if not isinstance(run_id, str):
            continue
        if entry.get("kind") == "cache-model-comparison-started":
            starts[run_id] = entry
        elif entry.get("kind") == "cache-model-comparison-finished":
            finishes[run_id] = entry
    total = 0.0
    for run_id, started in starts.items():
        reservation = float(started.get("reservedUpperUsd") or 0.0)
        finished = finishes.get(run_id)
        if finished is None:
            total += reservation
            continue
        summary = finished.get("summary") or {}
        usage = summary.get("usage") or {}
        costs = summary.get("cost") or {}
        if (usage.get("inputTokens") or 0) + (usage.get("outputTokens") or 0) > 0:
            total += float(costs.get("upperUsd") or reservation)
        else:
            # A lost/failed response may still have been billed; retain the full reservation.
            total += reservation
    return total


async def execute(context: CacheGroupInput, judge: str, *, smoke: bool, output: Path, prices: Optional[Prices] = None) -> Dict[str, Any]:
    output = Path(output).resolve()
    try:
        output_relative = output.relative_to(context.golden.root)
    except ValueError as exc:
        raise GoldenError("실행 기록은 비공개 골든 저장소 안에 저장해야 합니다") from exc
    if output.exists() or output.with_suffix(output.suffix + ".raw.jsonl").exists():
        raise GoldenError(f"결과 파일이 이미 있습니다. 덮어쓰지 않습니다: {output}")
    variant = variant_by_name(judge)
    turns = context.turns[:1] if smoke else context.turns
    attempt_limit = 1 if smoke or judge.startswith("claude-") else None
    max_output_tokens = CLAUDE_COMPARISON_MAX_OUTPUT_TOKENS if judge.startswith("claude-") else None
    resolved_prices = DEFAULT_PRICES[judge] if judge.startswith("claude-") else (prices or DEFAULT_PRICES[judge])
    reserved_upper_usd = 0.0
    if judge.startswith("claude-"):
        budget_estimate = estimate_cache_group(context, judge, DEFAULT_PRICES[judge])["estimate"]
        if smoke:
            one_turn_variant = replace(
                variant,
                max_attempts=1,
                settings={**variant.settings, "maxOutputTokens": CLAUDE_COMPARISON_MAX_OUTPUT_TOKENS},
            )
            budget_estimate = estimate(
                one_turn_variant,
                turns,
                DEFAULT_PRICES[judge],
                max_output_tokens=CLAUDE_COMPARISON_MAX_OUTPUT_TOKENS,
            )
        reserved_upper_usd = float(budget_estimate.get("upperUsd") or 0.0)
        spent_or_reserved = _claude_spent_or_reserved(context.golden.root)
        # Reserve only this requested model/run. The full four-model plan can
        # exceed the user's shared cap; each later run is rechecked against the
        # append-only ledger before it starts.
        planned_upper_usd = reserved_upper_usd
        if spent_or_reserved + planned_upper_usd > CLAUDE_BUDGET_USD:
            raise GoldenError(
                f"Claude 예산 한도 초과를 막기 위해 호출하지 않았습니다: 이미 사용/예약 ${spent_or_reserved:.4f}, "
                f"남은 비교 계획 상한 ${planned_upper_usd:.4f}, 한도 ${CLAUDE_BUDGET_USD:.2f}"
            )
    adapter = cli._live_adapter(
        judge, max_attempts=attempt_limit, max_output_tokens=max_output_tokens
    )
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ledger_path = context.golden.root / "evals" / "cache-path" / "model-comparison-ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps({
            "kind": "cache-model-comparison-started",
            "mode": "smoke" if smoke else "full",
            "groupId": context.group["groupId"],
            "groupVersion": context.group["groupVersion"],
            "groupSha256": context.group["groupSha256"],
            "model": variant.model,
            "reservedUpperUsd": reserved_upper_usd if judge.startswith("claude-") else None,
            "runId": output.stem,
            "output": output_relative.as_posix(),
            "startedAt": started,
        }, ensure_ascii=False) + "\n")
    outcomes = await cli._judge_all(adapter, turns, concurrency=1 if smoke else 1)
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, raw_rows, summary = _score_rows(context, turns, outcomes)
    for row, outcome in zip(rows, outcomes):
        if outcome is None:
            continue
        cost = turn_cost(judge, outcome.usage, resolved_prices)
        row["costUsd"] = cost
        if resolved_prices.input is not None:
            summary.setdefault("cost", {"cachedUsd": 0.0, "upperUsd": 0.0})
            if cost["cachedUsd"] is not None:
                summary["cost"]["cachedUsd"] += cost["cachedUsd"]
            if cost["upperUsd"] is not None:
                summary["cost"]["upperUsd"] += cost["upperUsd"]
    result = {
        "schemaVersion": 1,
        "kind": "cache-model-comparison",
        "runId": output.stem,
        "mode": "smoke" if smoke else "full",
        "status": "completed" if all(outcome is not None and not outcome.error for outcome in outcomes) else "completed_with_errors",
        "execution": {
            "maxAttempts": attempt_limit or variant.max_attempts,
            "maxOutputTokens": max_output_tokens or variant.settings.get("maxOutputTokens"),
            "claudeBudgetLimitUsd": CLAUDE_BUDGET_USD if judge.startswith("claude-") else None,
            "reservedUpperUsd": reserved_upper_usd if judge.startswith("claude-") else None,
        },
        "group": {
            "id": context.group["groupId"],
            "version": context.group["groupVersion"],
            "sha256": context.group["groupSha256"],
            "goldenVersion": context.golden.version,
            "goldenCommit": context.golden.commit,
            "memberCount": len(context.group["members"]),
            "evaluatedJudgeTurns": len(turns),
            "p1BypassesJudge": sum(1 for m in context.group["members"] if m["labels"]["path"] == "P1"),
            "sourceGoldenCommit": context.golden.commit,
        },
        "runner": {"name": "evaluation.judge_eval.cache_group", "version": "1.0"},
        "app": cli._app_revision(),
        "model": variant.stage_model(),
        "versions": {
            "promptVersion": variant.prompt_version,
            "instructionsSha256": hashlib.sha256(JUDGE_INSTRUCTIONS.encode("utf-8")).hexdigest(),
            "indexVersionId": (context.environment.get("index") or {}).get("indexVersionId"),
            "catalogFingerprint": (context.environment.get("catalog") or {}).get("fingerprint"),
            "supplement": None if context.golden.supplement is None else context.golden.relative(context.golden.supplement),
            "supplementSha256": None if context.golden.supplement is None else golden_mod.sha256_file(context.golden.supplement),
        },
        "executedAt": started,
        "finishedAt": finished,
        "inputSetSha256": summary["inputSetSha256"],
        "summary": summary,
        "results": rows,
    }
    raw_path = output.with_suffix(output.suffix + ".raw.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_rows), encoding="utf-8")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with ledger_path.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps({
            "kind": "cache-model-comparison-finished",
            "mode": result["mode"],
            "status": result["status"],
            "groupId": context.group["groupId"],
            "groupVersion": context.group["groupVersion"],
            "groupSha256": context.group["groupSha256"],
            "model": variant.model,
            "reservedUpperUsd": reserved_upper_usd if judge.startswith("claude-") else None,
            "runId": output.stem,
            "output": output_relative.as_posix(),
            "finishedAt": finished,
            "summary": summary,
        }, ensure_ascii=False) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke", "run"))
    parser.add_argument("--registry", type=Path, default=cli.DEFAULT_REGISTRY)
    parser.add_argument("--group", type=Path, default=DEFAULT_GROUP)
    parser.add_argument("--judge", choices=sorted(DEFAULT_PRICES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-price", type=float)
    parser.add_argument("--output-price", type=float)
    parser.add_argument("--cached-input-price", type=float)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        context = prepare_cache_group(args.registry, args.group)
        if args.action == "prepare":
            variants = sorted(DEFAULT_PRICES)
            estimates = {name: estimate_cache_group(context, name) for name in variants}
            claude_upper = sum(
                float(estimates[name]["estimate"].get("upperUsd") or 0.0) for name in CLAUDE_VARIANTS
            )
            claude_spent = _claude_spent_or_reserved(context.golden.root)
            print(json.dumps({"group": {"id": context.group["groupId"], "version": context.group["groupVersion"],
                                           "sha256": context.group["groupSha256"]},
                              "turns": {"members": len(context.group["members"]),
                                        "judge": len(context.turns),
                                        "exactHitBypass": sum(m["labels"]["path"] == "P1" for m in context.group["members"]),
                                        "payloadHashesVerified": sum(t.payload_verified for t in context.turns)},
                              "sharedClaudeBudget": {"limitUsd": CLAUDE_BUDGET_USD,
                                                     "spentOrReservedUsd": round(claude_spent, 6),
                                                     "allClaudeVariantsUpperUsd": round(claude_upper, 6),
                                                     "remainingAfterFullPlanUsd": round(CLAUDE_BUDGET_USD - claude_spent - claude_upper, 6),
                                                     "maxAttempts": CLAUDE_COMPARISON_MAX_ATTEMPTS,
                                                     "maxOutputTokens": CLAUDE_COMPARISON_MAX_OUTPUT_TOKENS},
                              "estimates": estimates},
                             ensure_ascii=False, indent=2))
            return 0
        if not args.judge:
            raise GoldenError("smoke/run 에는 --judge 가 필요합니다")
        if args.output is None:
            raise GoldenError("smoke/run 에는 --output 이 필요합니다")
        default = DEFAULT_PRICES[args.judge]
        prices = Prices(
            input=args.input_price if args.input_price is not None else default.input,
            output=args.output_price if args.output_price is not None else default.output,
            cached_input=args.cached_input_price if args.cached_input_price is not None else default.cached_input,
        )
        if args.action == "smoke":
            if args.judge.startswith("claude-") and any(
                value is not None for value in (args.input_price, args.output_price, args.cached_input_price)
            ):
                raise GoldenError("Claude 예산 한도 보호를 위해 공식 단가는 재정의할 수 없습니다")
            result = asyncio.run(execute(context, args.judge, smoke=True, output=args.output, prices=prices))
        else:
            if args.judge.startswith("claude-") and any(
                value is not None for value in (args.input_price, args.output_price, args.cached_input_price)
            ):
                raise GoldenError("Claude 예산 한도 보호를 위해 공식 단가는 재정의할 수 없습니다")
            result = asyncio.run(execute(context, args.judge, smoke=False, output=args.output, prices=prices))
        print(json.dumps({"runId": result["runId"], "mode": result["mode"], "model": result["model"],
                          "group": result["group"], "summary": result["summary"]}, ensure_ascii=False, indent=2))
        return 0
    except (GoldenError, ValueError, OSError) as exc:
        print(f"거부: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
