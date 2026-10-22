"""판별 결과 채점 (riido-evals schemas/labels.md 6절).

턴 채점은 세 층이다.
  1. 의미상 정답: 결정이 라벨과 같고, CONNECT 면 키가 expectedSubproblemKeysAny 안에 있음
  2. 후보 검색 성공: 정답이 CONNECT 일 때 정답 키가 판별기에 보여 준 후보 안에 있었는지
  3. 판별기 정답: 후보가 주어졌을 때 맞혔는지. 정답 키가 후보에 없던 턴은 연결하지 않았으면 판별기 오류가
     아니다(후보 검색 실패로 따로 센다). 그런 턴에서 다른 키로 연결하면 판별기 오류다.

오류 지표는 분모를 섞지 않는다.
  - 잘못된 연결: 정답이 비연결(SEPARATE, UNCLASSIFIED)인 턴 중 연결한 턴
  - 잘못된 서빙: 실제로 정본을 서빙한 턴(CONNECT 이고 고른 세부 문제가 동결 카탈로그에서 SERVING) 중
    정답이 아닌 턴
  - 놓친 연결: 정답이 CONNECT 이고 정답 키가 후보에 있던 턴 중 연결하지 않은 턴 (판별기 층).
    후보 검색 실패를 포함한 값은 missedConnectSemantic 으로 따로 둔다
건수마다 단측 95% 상한을 붙인다. 0건이면 1 - 0.05^(1/n), 1건 이상이면 Clopper-Pearson.

호출 실패(ERROR)는 운영과 같이 UNCLASSIFIED 로 채점하고 건수를 따로 적는다. 판별하지 않은 턴(SKIPPED)은
모든 분모에서 뺀다. 결과는 층과 라벨 출처로 나눠 내고, 합친 값에도 출처별 수를 함께 둔다.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from evaluation.judge_eval.golden import JUDGE_DECISIONS, Label, TurnKey

ALPHA = 0.05
SERVING = "SERVING"
NON_CONNECT = ("SEPARATE", "UNCLASSIFIED")


# ---------------------------------------------------------------- 신뢰 상한


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k), X ~ Binomial(n, p)."""

    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q)
    return min(1.0, total)


def zero_error_upper(n: int, alpha: float = ALPHA) -> float:
    """0건일 때 단측 정확 상한 1 - alpha^(1/n)."""

    return 1.0 - alpha ** (1.0 / n)


def clopper_pearson_upper(k: int, n: int, alpha: float = ALPHA) -> float:
    """단측 Clopper-Pearson 상한: P(X <= k | n, p) = alpha 인 p. k = 0 이면 1 - alpha^(1/n) 과 같다."""

    if n <= 0:
        raise ValueError("n 은 1 이상이어야 합니다")
    if k >= n:
        return 1.0
    low, high = k / n, 1.0
    for _ in range(200):
        mid = (low + high) / 2
        if binom_cdf(k, n, mid) > alpha:
            low = mid
        else:
            high = mid
    return high


def rate_with_bound(count: int, n: int) -> Dict[str, Any]:
    if n == 0:
        return {"count": count, "n": 0, "rate": None, "upper95": None, "boundMethod": None}
    if count == 0:
        upper, method = zero_error_upper(n), "zero-error-exact"
    else:
        upper, method = clopper_pearson_upper(count, n), "clopper-pearson"
    return {"count": count, "n": n, "rate": count / n, "upper95": upper, "boundMethod": method}


def simple_rate(count: int, n: int) -> Dict[str, Any]:
    return {"count": count, "n": n, "rate": (count / n) if n else None}


def mcnemar_exact_p(b: int, c: int) -> Optional[float]:
    """짝 비교의 정확 McNemar 양측 p (b, c 는 불일치 쌍 수)."""

    if b + c == 0:
        return None
    return min(1.0, 2.0 * binom_cdf(min(b, c), b + c, 0.5))


# ---------------------------------------------------------------- 턴 채점


@dataclass(frozen=True)
class TurnScore:
    key: TurnKey
    layer: str
    label_source: str
    expected: str
    expected_keys: Tuple[str, ...]
    actual: Optional[str]  # None = SKIPPED
    actual_key: Optional[str]
    error: bool
    candidate_hit: Optional[bool]
    served: bool

    @property
    def scored(self) -> bool:
        return self.actual is not None

    @property
    def decision_correct(self) -> bool:
        return self.actual == self.expected

    @property
    def semantic_correct(self) -> bool:
        if not self.decision_correct:
            return False
        return self.expected != "CONNECT" or self.actual_key in self.expected_keys

    @property
    def judge_correct(self) -> bool:
        if self.expected == "CONNECT" and self.candidate_hit is False:
            return self.actual != "CONNECT"
        return self.semantic_correct

    @property
    def wrong_serve(self) -> bool:
        return self.served and not (self.expected == "CONNECT" and self.actual_key in self.expected_keys)


def score_turn(
    key: TurnKey,
    layer: str,
    label: Label,
    presented_keys: Sequence[str],
    actual: Optional[str],
    actual_key: Optional[str],
    error: bool,
    serving_state: Mapping[str, str],
) -> TurnScore:
    if actual is not None and actual not in JUDGE_DECISIONS:
        raise ValueError(f"{key}: 모르는 결정 {actual}")
    candidate_hit = None
    if label.decision == "CONNECT":
        candidate_hit = any(k in presented_keys for k in label.keys)
    served = actual == "CONNECT" and actual_key is not None and serving_state.get(actual_key) == SERVING
    return TurnScore(
        key=key,
        layer=layer,
        label_source=label.source,
        expected=label.decision,
        expected_keys=label.keys,
        actual=actual,
        actual_key=actual_key if actual == "CONNECT" else None,
        error=error,
        candidate_hit=candidate_hit,
        served=served,
    )


def metrics(scores: Iterable[TurnScore]) -> Dict[str, Any]:
    scores = list(scores)
    scored = [s for s in scores if s.scored]
    n = len(scored)
    non_connect = [s for s in scored if s.expected in NON_CONNECT]
    connect = [s for s in scored if s.expected == "CONNECT"]
    connect_hit = [s for s in connect if s.candidate_hit]
    served = [s for s in scored if s.served]
    confusion = {exp: {act: 0 for act in JUDGE_DECISIONS} for exp in JUDGE_DECISIONS}
    for s in scored:
        confusion[s.expected][s.actual] += 1
    sources: Dict[str, int] = {}
    for s in scores:
        sources[s.label_source] = sources.get(s.label_source, 0) + 1
    return {
        "turns": len(scores),
        "scored": n,
        "skipped": len(scores) - n,
        "errors": sum(1 for s in scored if s.error),
        "labelSourceCounts": dict(sorted(sources.items())),
        "decisionAccuracy": simple_rate(sum(s.decision_correct for s in scored), n),
        "semanticAccuracy": simple_rate(sum(s.semantic_correct for s in scored), n),
        "judgeAccuracy": simple_rate(sum(s.judge_correct for s in scored), n),
        "wrongConnect": rate_with_bound(sum(1 for s in non_connect if s.actual == "CONNECT"), len(non_connect)),
        "wrongServe": rate_with_bound(sum(1 for s in served if s.wrong_serve), len(served)),
        "wrongServePerTurn": rate_with_bound(sum(1 for s in served if s.wrong_serve), n),
        "missedConnect": rate_with_bound(sum(1 for s in connect_hit if s.actual != "CONNECT"), len(connect_hit)),
        "missedConnectSemantic": rate_with_bound(sum(1 for s in connect if s.actual != "CONNECT"), len(connect)),
        "wrongKeyConnect": rate_with_bound(
            sum(1 for s in connect_hit if s.actual == "CONNECT" and s.actual_key not in s.expected_keys), len(connect_hit)
        ),
        "candidateMiss": simple_rate(sum(1 for s in connect if not s.candidate_hit), len(connect)),
        "served": simple_rate(len(served), n),
        "unclassified": simple_rate(sum(1 for s in scored if s.actual == "UNCLASSIFIED"), n),
        "confusion": confusion,
    }


def _grouped(scores: Sequence[TurnScore]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for layer in sorted({s.layer for s in scores}):
        in_layer = [s for s in scores if s.layer == layer]
        out[layer] = {
            "all": metrics(in_layer),
            "byLabelSource": {
                source: metrics([s for s in in_layer if s.label_source == source])
                for source in sorted({s.label_source for s in in_layer})
            },
        }
    return out


def never_connect_baseline(scores: Sequence[TurnScore]) -> List[TurnScore]:
    """무조건 보류 기준선: 모든 턴을 UNCLASSIFIED 로 둔다 (판별 실패 시 운영 동작과 같음)."""

    return [
        TurnScore(
            key=s.key,
            layer=s.layer,
            label_source=s.label_source,
            expected=s.expected,
            expected_keys=s.expected_keys,
            actual=None if s.actual is None else "UNCLASSIFIED",
            actual_key=None,
            error=False,
            candidate_hit=s.candidate_hit,
            served=False,
        )
        for s in scores
    ]


def summarize(scores: Sequence[TurnScore]) -> Dict[str, Any]:
    return {
        "byLayer": _grouped(scores),
        "baselines": {"neverConnect": _grouped(never_connect_baseline(scores))},
        "definitions": {
            "wrongConnect": "정답 SEPARATE, UNCLASSIFIED 턴 중 CONNECT",
            "wrongServe": "CONNECT 이고 고른 세부 문제가 동결 카탈로그 SERVING 인 턴 중 정답 아님",
            "wrongServePerTurn": "잘못된 서빙 건수를 채점한 전체 턴으로 나눈 값 (모델마다 서빙 수가 달라도 같은 분모)",
            "missedConnect": "정답 CONNECT 이고 정답 키가 후보에 있던 턴 중 CONNECT 아님 (판별기 층)",
            "missedConnectSemantic": "정답 CONNECT 턴 중 CONNECT 아님 (후보 검색 실패 포함)",
            "wrongKeyConnect": "정답 CONNECT 이고 정답 키가 후보에 있던 턴 중 다른 키로 CONNECT",
            "candidateMiss": "정답 CONNECT 턴 중 정답 키가 후보에 없던 턴 (판별기 오류 아님)",
            "upper95": "단측 95% 상한. 0건이면 1-0.05^(1/n), 1건 이상이면 Clopper-Pearson",
            "errors": "호출 실패. UNCLASSIFIED 로 채점",
            "neverConnect": "모든 턴 UNCLASSIFIED 기준선",
        },
    }


# ---------------------------------------------------------------- 짝 비교


def compare(base: Mapping[TurnKey, TurnScore], candidate: Mapping[TurnKey, TurnScore]) -> Dict[str, Any]:
    """같은 턴끼리 짝지은 차이. 두 쪽 모두 채점한 턴만 쓴다."""

    keys = [key for key in base if key in candidate and base[key].scored and candidate[key].scored]
    out: Dict[str, Any] = {}
    for layer in sorted({base[key].layer for key in keys}):
        layer_keys = [key for key in keys if base[key].layer == layer]
        groups = {"all": layer_keys}
        for source in sorted({base[key].label_source for key in layer_keys}):
            groups[source] = [key for key in layer_keys if base[key].label_source == source]
        out[layer] = {name: _paired(group, base, candidate) for name, group in groups.items()}
    return out


def _paired(keys: Sequence[TurnKey], base: Mapping[TurnKey, TurnScore], cand: Mapping[TurnKey, TurnScore]) -> Dict[str, Any]:
    def flags(score: TurnScore) -> Dict[str, bool]:
        return {
            "judgeCorrect": score.judge_correct,
            "semanticCorrect": score.semantic_correct,
            "wrongServe": score.wrong_serve,
            "wrongConnect": score.expected in NON_CONNECT and score.actual == "CONNECT",
            "served": score.served,
        }

    result: Dict[str, Any] = {"pairs": len(keys)}
    for name in ("judgeCorrect", "semanticCorrect", "wrongServe", "wrongConnect", "served"):
        base_only = sum(1 for key in keys if flags(base[key])[name] and not flags(cand[key])[name])
        cand_only = sum(1 for key in keys if flags(cand[key])[name] and not flags(base[key])[name])
        both = sum(1 for key in keys if flags(base[key])[name] and flags(cand[key])[name])
        result[name] = {
            "base": both + base_only,
            "candidate": both + cand_only,
            "difference": (both + cand_only) - (both + base_only),
            "baseOnly": base_only,
            "candidateOnly": cand_only,
            "mcnemarExactP": mcnemar_exact_p(base_only, cand_only),
        }
    result["decisionChanged"] = sum(
        1 for key in keys if (base[key].actual, base[key].actual_key) != (cand[key].actual, cand[key].actual_key)
    )
    return result
