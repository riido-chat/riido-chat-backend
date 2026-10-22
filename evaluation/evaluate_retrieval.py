"""BM25 Retrieval 후보와 ground truth를 비교해 baseline metric을 계산한다."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple, Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES_PATH = (
    PROJECT_ROOT / "evaluation/retrieval_labeling_candidates.json"
)
DEFAULT_GROUND_TRUTH_PATH = PROJECT_ROOT / "evaluation/ground_truth.json"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "evaluation/bm25_evaluation_metrics.json"
)

METRIC_KEYS = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr_at_10",
    "ndcg_at_10",
)
SectionIdentity = str


def _section_identity(section: Dict[str, Any]) -> SectionIdentity:
    return section["section_id"]


def _relevant_identities(
    relevant_sections: Sequence[Dict[str, Any]],
) -> Set[SectionIdentity]:
    identities = {_section_identity(section) for section in relevant_sections}
    if not identities:
        raise ValueError("metric 계산에는 정답 Section이 하나 이상 필요합니다.")
    return identities


def calculate_recall_at_k(
    candidates: Sequence[Dict[str, Any]],
    relevant_sections: Sequence[Dict[str, Any]],
    k: int,
) -> float:
    """Top-K에서 찾은 고유 정답 Section 비율을 계산한다."""

    if k <= 0:
        raise ValueError("k는 1 이상이어야 합니다.")

    relevant = _relevant_identities(relevant_sections)
    retrieved = {
        _section_identity(candidate)
        for candidate in candidates
        if candidate["rank"] <= k
        and _section_identity(candidate) in relevant
    }
    return len(retrieved) / len(relevant)


def calculate_mrr_at_10(
    candidates: Sequence[Dict[str, Any]],
    relevant_sections: Sequence[Dict[str, Any]],
) -> float:
    """Top-10에서 처음 등장한 정답 Section 순위의 역수를 계산한다."""

    relevant = _relevant_identities(relevant_sections)
    for candidate in sorted(candidates, key=lambda item: item["rank"]):
        rank = candidate["rank"]
        if rank > 10:
            break
        if _section_identity(candidate) in relevant:
            return 1.0 / rank
    return 0.0


def calculate_ndcg_at_10(
    candidates: Sequence[Dict[str, Any]],
    relevant_sections: Sequence[Dict[str, Any]],
) -> float:
    """정답을 1로 보는 binary nDCG@10을 계산한다."""

    relevant = _relevant_identities(relevant_sections)
    seen_relevant: Set[SectionIdentity] = set()
    dcg = 0.0

    for candidate in sorted(candidates, key=lambda item: item["rank"]):
        rank = candidate["rank"]
        if rank > 10:
            break

        identity = _section_identity(candidate)
        if identity in relevant and identity not in seen_relevant:
            dcg += 1.0 / math.log2(rank + 1)
            seen_relevant.add(identity)

    ideal_count = min(len(relevant), 10)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )
    return dcg / idcg


def evaluate_question(
    candidate_item: Dict[str, Any],
    ground_truth_item: Dict[str, Any],
) -> Dict[str, Any]:
    """질문 하나의 Retrieval metric을 계산한다."""

    candidates = candidate_item["candidates"]
    relevant_sections = ground_truth_item["relevant_sections"]
    metrics = {
        "recall_at_1": calculate_recall_at_k(
            candidates, relevant_sections, 1
        ),
        "recall_at_3": calculate_recall_at_k(
            candidates, relevant_sections, 3
        ),
        "recall_at_5": calculate_recall_at_k(
            candidates, relevant_sections, 5
        ),
        "recall_at_10": calculate_recall_at_k(
            candidates, relevant_sections, 10
        ),
        "mrr_at_10": calculate_mrr_at_10(candidates, relevant_sections),
        "ndcg_at_10": calculate_ndcg_at_10(candidates, relevant_sections),
    }

    return {
        "question_id": candidate_item["question_id"],
        "question": candidate_item["question"],
        "relevant_section_count": len(
            _relevant_identities(relevant_sections)
        ),
        "metrics": metrics,
    }


def evaluate_retrieval(
    candidate_items: Sequence[Dict[str, Any]],
    ground_truth_items: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """전체 질문의 metric과 평균, Top-10 miss 목록을 계산한다."""

    candidates_by_id = {
        item["question_id"]: item for item in candidate_items
    }
    ground_truth_by_id = {
        item["question_id"]: item for item in ground_truth_items
    }

    if len(candidates_by_id) != len(candidate_items):
        raise ValueError("Retrieval 후보에 중복 question_id가 있습니다.")
    if len(ground_truth_by_id) != len(ground_truth_items):
        raise ValueError("Ground truth에 중복 question_id가 있습니다.")
    if set(candidates_by_id) != set(ground_truth_by_id):
        raise ValueError("Retrieval 후보와 ground truth의 question_id가 다릅니다.")
    if not candidate_items:
        raise ValueError("평가할 Retrieval 후보가 없습니다.")

    question_results = [
        evaluate_question(
            candidate_item,
            ground_truth_by_id[candidate_item["question_id"]],
        )
        for candidate_item in candidate_items
    ]
    average_metrics = {
        metric_key: math.fsum(
            result["metrics"][metric_key] for result in question_results
        )
        / len(question_results)
        for metric_key in METRIC_KEYS
    }
    missed_question_ids = [
        result["question_id"]
        for result in question_results
        if result["metrics"]["recall_at_10"] == 0.0
    ]

    return {
        "summary": {
            "question_count": len(question_results),
            "relevant_section_count": sum(
                result["relevant_section_count"]
                for result in question_results
            ),
            "average_metrics": average_metrics,
            "top_10_miss_question_ids": missed_question_ids,
        },
        "questions": question_results,
    }


def load_evaluation_data(
    candidates_path: Union[str, Path] = DEFAULT_CANDIDATES_PATH,
    ground_truth_path: Union[str, Path] = DEFAULT_GROUND_TRUTH_PATH,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Retrieval 후보와 ground truth JSON을 읽는다."""

    candidates = json.loads(
        Path(candidates_path).read_text(encoding="utf-8")
    )
    ground_truth = json.loads(
        Path(ground_truth_path).read_text(encoding="utf-8")
    )
    return candidates, ground_truth


def save_evaluation_metrics(
    evaluation_result: Dict[str, Any],
    output_path: Union[str, Path] = DEFAULT_OUTPUT_PATH,
) -> Path:
    """평가 결과를 UTF-8 JSON 파일로 저장한다."""

    path = Path(output_path)
    path.write_text(
        json.dumps(evaluation_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def print_evaluation_metrics(evaluation_result: Dict[str, Any]) -> None:
    """전체 평균, 질문별 metric, Top-10 miss 목록을 출력한다."""

    summary = evaluation_result["summary"]
    print("전체 평균 metric")
    for metric_key in METRIC_KEYS:
        print(
            f"{metric_key}: "
            f"{summary['average_metrics'][metric_key]:.6f}"
        )

    print("\n질문별 metric")
    for result in evaluation_result["questions"]:
        metrics = result["metrics"]
        metric_text = ", ".join(
            f"{metric_key}={metrics[metric_key]:.6f}"
            for metric_key in METRIC_KEYS
        )
        print(f"{result['question_id']}: {metric_text}")

    missed = summary["top_10_miss_question_ids"]
    print("\nTop-10 miss 질문")
    print(", ".join(missed) if missed else "없음")


def main() -> None:
    candidates, ground_truth = load_evaluation_data()
    evaluation_result = evaluate_retrieval(candidates, ground_truth)
    print_evaluation_metrics(evaluation_result)
    output_path = save_evaluation_metrics(evaluation_result)
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
