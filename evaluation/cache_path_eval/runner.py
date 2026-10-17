"""캐시 경로 실행 관측값을 비공개 텍스트 없이 집계한다.

이 도구는 애플리케이션, DB, 임베딩 또는 모델을 실행하지 않는다. 캐시 경로별로
별도 수집한 JSONL 관측값과 동결된 텍스트 없는 선택 manifest를 받아 결과만 계산한다.
"""

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PATHS = ("P1", "P3", "P4", "P6")
OUTCOMES = ("SERVED", "SHADOW", "GROUP_DISABLED", "REJECTED", "SKIPPED", "FAILED")
NEXT_STAGE = ("PASS", "FAIL", "NOT_APPLICABLE")
OBSERVATION_FIELDS = {
    "itemId", "turnNo", "outcome", "nextStage", "latencyMs", "modelCalls", "embeddingCalls"
}
MANIFEST_TURN_FIELDS = {
    "itemId", "path", "turnNo", "kind", "expectedExactHit", "expectedServed",
    "expectedJudgeDecision", "expectedSubproblemKeysAny",
}


class CachePathEvalError(ValueError):
    """입력 manifest 또는 관측값이 유효하지 않다."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _non_negative_int(value: Any, field: str, key: Tuple[str, int]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CachePathEvalError(f"{key}: {field}는 0 이상의 정수여야 합니다")
    return value


def _load_observations(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CachePathEvalError(f"관측 JSONL {line_no}행을 읽지 못했습니다: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise CachePathEvalError(f"관측 JSONL {line_no}행은 객체여야 합니다")
                extra = set(row) - OBSERVATION_FIELDS
                missing = OBSERVATION_FIELDS - set(row)
                if extra or missing:
                    raise CachePathEvalError(
                        f"관측 JSONL {line_no}행의 필드가 맞지 않습니다 "
                        f"(누락={sorted(missing)}, 허용되지 않음={sorted(extra)})"
                    )
                rows.append(row)
    except OSError as exc:
        raise CachePathEvalError(f"관측 파일을 읽지 못했습니다: {exc}") from exc
    return rows, _sha256(path)


def _validate_manifest(manifest: Any) -> List[Dict[str, Any]]:
    if not isinstance(manifest, dict) or manifest.get("group") != "cache-path":
        raise CachePathEvalError("cache-path 선택 manifest가 아닙니다")
    if not isinstance(manifest.get("goldenVersion"), str) or not isinstance(manifest.get("inputs"), dict):
        raise CachePathEvalError("manifest에 골든셋 버전 또는 입력 해시가 없습니다")
    turns = manifest.get("turns")
    if not isinstance(turns, list) or not turns:
        raise CachePathEvalError("manifest에 선택된 턴이 없습니다")
    seen = set()
    for row in turns:
        if not isinstance(row, dict):
            raise CachePathEvalError("manifest 턴은 객체여야 합니다")
        extra = set(row) - MANIFEST_TURN_FIELDS
        if extra:
            raise CachePathEvalError(f"manifest 턴에 허용되지 않은 필드가 있습니다: {sorted(extra)}")
        key = (row.get("itemId"), row.get("turnNo"))
        if not isinstance(key[0], str) or not key[0] or not isinstance(key[1], int) or isinstance(key[1], bool) or key[1] < 1:
            raise CachePathEvalError("manifest 턴의 문항 ID 또는 턴 번호가 잘못되었습니다")
        if key in seen:
            raise CachePathEvalError(f"manifest에 중복 턴이 있습니다: {key}")
        seen.add(key)
        if row.get("path") not in PATHS or row.get("kind") not in ("exact_hit", "judge", "context_only"):
            raise CachePathEvalError(f"manifest 턴의 경로 또는 종류가 잘못되었습니다: {key}")
        if row["kind"] == "context_only" and "expectedServed" in row:
            raise CachePathEvalError(f"context-only 턴에는 캐시 기대 결과를 둘 수 없습니다: {key}")
        if row["kind"] != "context_only" and not isinstance(row.get("expectedServed"), bool):
            raise CachePathEvalError(f"캐시 기대 결과가 없습니다: {key}")
    return turns


def _summarize_group(expected_rows: Iterable[Dict[str, Any]], observations: Dict[Tuple[str, int], Dict[str, Any]]) -> Dict[str, Any]:
    expected_rows = list(expected_rows)
    counts = Counter()
    next_counts = Counter()
    latencies: List[int] = []
    model_calls = 0
    embedding_calls = 0
    ineligible_served = 0
    for expected in expected_rows:
        key = (expected["itemId"], expected["turnNo"])
        observed = observations[key]
        counts[observed["outcome"]] += 1
        if observed["nextStage"] != "NOT_APPLICABLE":
            next_counts[observed["nextStage"]] += 1
        latencies.append(observed["latencyMs"])
        model_calls += observed["modelCalls"]
        embedding_calls += observed["embeddingCalls"]
        if expected.get("expectedServed") is False and observed["outcome"] == "SERVED":
            ineligible_served += 1
    next_total = next_counts["PASS"] + next_counts["FAIL"]
    return {
        "turns": len(expected_rows),
        "outcomes": {outcome: counts[outcome] for outcome in OUTCOMES if counts[outcome]},
        "ineligibleServed": ineligible_served,
        "nextStage": {
            "passed": next_counts["PASS"],
            "failed": next_counts["FAIL"],
            "notApplicable": len(expected_rows) - next_total,
            "passRate": next_counts["PASS"] / next_total if next_total else None,
            "denominator": next_total,
        },
        "latencyMs": {
            "total": sum(latencies),
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "median": float(statistics.median(latencies)) if latencies else None,
        },
        "modelCalls": model_calls,
        "embeddingCalls": embedding_calls,
    }


def evaluate(manifest: Dict[str, Any], observations: Sequence[Dict[str, Any]], observation_sha256: str) -> Dict[str, Any]:
    turns = _validate_manifest(manifest)
    expected = {(row["itemId"], row["turnNo"]): row for row in turns}
    observed: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in observations:
        if not isinstance(row, dict):
            raise CachePathEvalError("관측 턴은 객체여야 합니다")
        extra = set(row) - OBSERVATION_FIELDS
        missing_fields = OBSERVATION_FIELDS - set(row)
        if extra or missing_fields:
            raise CachePathEvalError(
                "관측 필드가 맞지 않습니다 "
                f"(누락={sorted(missing_fields)}, 허용되지 않음={sorted(extra)})"
            )
        if (
            not isinstance(row["itemId"], str)
            or not row["itemId"]
            or not isinstance(row["turnNo"], int)
            or isinstance(row["turnNo"], bool)
            or row["turnNo"] < 1
        ):
            raise CachePathEvalError("관측 턴의 문항 ID 또는 턴 번호가 잘못되었습니다")
        key = (row.get("itemId"), row.get("turnNo"))
        if key not in expected:
            raise CachePathEvalError(f"manifest에 없는 관측 턴입니다: {key}")
        if key in observed:
            raise CachePathEvalError(f"중복 관측 턴입니다: {key}")
        if row.get("outcome") not in OUTCOMES:
            raise CachePathEvalError(f"{key}: 알 수 없는 캐시 결과입니다")
        if row.get("nextStage") not in NEXT_STAGE:
            raise CachePathEvalError(f"{key}: nextStage 값이 잘못되었습니다")
        for field in ("latencyMs", "modelCalls", "embeddingCalls"):
            _non_negative_int(row.get(field), field, key)
        observed[key] = dict(row)
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise CachePathEvalError(f"관측이 빠진 턴이 있습니다 ({len(missing)}건)")

    by_path: Dict[str, Dict[str, Any]] = {}
    for path in PATHS:
        path_rows = [row for row in turns if row["path"] == path]
        if not path_rows:
            continue
        by_path[path] = {
            kind: _summarize_group((row for row in path_rows if row["kind"] == kind), observed)
            for kind in ("exact_hit", "judge", "context_only")
            if any(row["kind"] == kind for row in path_rows)
        }
        by_path[path]["all"] = _summarize_group(path_rows, observed)
    overall = _summarize_group(turns, observed)
    return {
        "resultSchemaVersion": 1,
        "group": "cache-path",
        "runMode": "offline-observations",
        "goldenVersion": manifest["goldenVersion"],
        "sourceCorpusVersion": manifest.get("sourceCorpusVersion"),
        "inputs": {
            "draftsSha256": manifest["inputs"].get("draftsSha256"),
            "judgeLabelsSha256": manifest["inputs"].get("judgeLabelsSha256"),
            "observationsSha256": observation_sha256,
        },
        "summary": {"all": overall, "paths": by_path},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="합성 또는 별도 수집한 캐시 관측값을 오프라인 집계합니다")
    parser.add_argument("--manifest", required=True, type=Path, help="텍스트가 없는 cache-path 선택 manifest")
    parser.add_argument("--observations", required=True, type=Path, help="턴별 관측 JSONL; 질문/답변 텍스트를 넣지 마세요")
    parser.add_argument("--out", type=Path, help="결과 JSON 경로 (기존 파일은 덮어쓰지 않음)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        observations, observations_hash = _load_observations(args.observations)
        result = evaluate(manifest, observations, observations_hash)
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.out:
            if args.out.exists():
                raise CachePathEvalError(f"결과 파일이 이미 있습니다. 덮어쓰지 않습니다: {args.out}")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    except (OSError, json.JSONDecodeError, CachePathEvalError) as exc:
        print(f"거부: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
