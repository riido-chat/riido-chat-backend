"""운영 DB를 변경하지 않고 임베딩 모델·차원별 Retrieval을 비교한다."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.bm25_retriever import BM25Retriever
from app.core.config import get_settings
from app.retrieval.corpus import build_retrieval_chunks
from app.retrieval.embedding import build_embedding_text
from app.retrieval.hybrid_retriever import CANDIDATE_K, fuse_rrf_results
from app.retrieval.models import RetrievalChunk, RetrievalResult
from evaluation.evaluate_retrieval import (
    DEFAULT_GROUND_TRUTH_PATH,
    evaluate_retrieval,
    load_evaluation_data,
)
from evaluation.run_bm25_evaluation import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_QUESTIONS_PATH,
    load_questions,
)


EXPERIMENT_VERSION = "embedding-model-comparison-v1"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "evaluation/baselines/embedding-model-comparison-v1.json"
)
PRICE_SOURCE_URL = (
    "https://developers.openai.com/api/docs/models/text-embedding-3-large"
)
PRICE_CHECKED_AT = "2026-09-08"
FLOAT_BYTES = 4
DEFAULT_BATCH_SIZE = 128


@dataclass(frozen=True)
class CandidateConfig:
    id: str
    model: str
    dimensions: int
    price_per_million_input_tokens_usd: float


CANDIDATE_CONFIGS: Tuple[CandidateConfig, ...] = (
    CandidateConfig("A", "text-embedding-3-large", 1536, 0.13),
    CandidateConfig("B", "text-embedding-3-small", 1536, 0.02),
    CandidateConfig("C", "text-embedding-3-large", 3072, 0.13),
    CandidateConfig("D", "text-embedding-3-large", 768, 0.13),
    CandidateConfig("E", "text-embedding-3-small", 768, 0.02),
)


@dataclass(frozen=True)
class EmbeddingBatchResult:
    embeddings: Tuple[Tuple[float, ...], ...]
    input_tokens: int
    latency_ms: int
    request_count: int


class ExperimentEmbedder:
    """실험 설정을 제품 상수와 분리해 OpenAI embedding을 생성한다."""

    def __init__(self, client: Optional[OpenAI] = None) -> None:
        if client is None:
            api_key = get_settings().openai_api_key
            if not api_key:
                raise ValueError("OPENAI_API_KEY 환경변수가 필요합니다.")
            client = OpenAI(api_key=api_key)
        self._client = client

    def embed_many(
        self,
        texts: Sequence[str],
        config: CandidateConfig,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> EmbeddingBatchResult:
        if not texts:
            raise ValueError("Embedding할 입력이 하나 이상이어야 합니다.")
        if batch_size <= 0:
            raise ValueError("batch_size는 1 이상이어야 합니다.")

        started = time.perf_counter()
        embeddings: List[Tuple[float, ...]] = []
        input_tokens = 0
        request_count = 0

        for batch in _batched(texts, batch_size):
            response = self._client.embeddings.create(
                model=config.model,
                input=list(batch),
                dimensions=config.dimensions,
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(batch))):
                raise RuntimeError("Embedding 응답 index가 입력 순서와 일치하지 않습니다.")

            for item in ordered:
                embedding = tuple(float(value) for value in item.embedding)
                if len(embedding) != config.dimensions:
                    raise ValueError(
                        f"{config.id} 응답은 {config.dimensions}차원이어야 합니다."
                    )
                embeddings.append(embedding)

            input_tokens += int(response.usage.prompt_tokens)
            request_count += 1

        if len(embeddings) != len(texts):
            raise RuntimeError("Embedding 응답 개수가 입력과 일치하지 않습니다.")

        return EmbeddingBatchResult(
            embeddings=tuple(embeddings),
            input_tokens=input_tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
            request_count=request_count,
        )


def _batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def attach_experiment_ids(
    chunks: Sequence[RetrievalChunk],
) -> Tuple[RetrievalChunk, ...]:
    """RRF가 요구하는 식별자를 manifest 순서로 결정적으로 부여한다."""

    return tuple(
        replace(
            chunk,
            chunk_id=index,
            document_version_id=1,
            index_version_id=1,
        )
        for index, chunk in enumerate(chunks, start=1)
    )


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("비교할 embedding 차원이 일치해야 합니다.")
    if not left:
        raise ValueError("빈 embedding은 비교할 수 없습니다.")

    dot = math.fsum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("영벡터는 cosine similarity를 계산할 수 없습니다.")
    return dot / (left_norm * right_norm)


def rank_vector_results(
    chunks: Sequence[RetrievalChunk],
    chunk_embeddings: Sequence[Sequence[float]],
    query_embedding: Sequence[float],
    top_k: int = CANDIDATE_K,
) -> List[RetrievalResult]:
    """메모리에서 pgvector cosine 검색과 같은 순위 입력을 만든다."""

    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 합니다.")
    if len(chunks) != len(chunk_embeddings):
        raise ValueError("Chunk와 embedding 개수가 일치해야 합니다.")

    scored = [
        (chunk, cosine_similarity(query_embedding, embedding))
        for chunk, embedding in zip(chunks, chunk_embeddings)
    ]
    scored.sort(
        key=lambda item: (
            -item[1],
            item[0].chunk_id if item[0].chunk_id is not None else 0,
        )
    )
    return [
        RetrievalResult(chunk=chunk, score=score, rank=rank)
        for rank, (chunk, score) in enumerate(scored[:top_k], start=1)
    ]


def _retrieval_candidate(result: RetrievalResult) -> Dict[str, Any]:
    return {
        "rank": result.rank,
        "section_id": result.chunk.section_id,
        "document_title": result.chunk.document_title,
        "section_path": " > ".join(result.chunk.section_path),
        "score": result.score,
    }


def _hybrid_candidate(result: Any) -> Dict[str, Any]:
    return {
        "rank": result.final_rank,
        "section_id": result.chunk.section_id,
        "document_title": result.chunk.document_title,
        "section_path": " > ".join(result.chunk.section_path),
        "score": result.rrf_score,
        "rrf_score": result.rrf_score,
        "final_rank": result.final_rank,
        "bm25_rank": result.bm25_rank,
        "vector_rank": result.vector_rank,
    }


def create_candidates(
    questions: Sequence[Dict[str, str]],
    chunks: Sequence[RetrievalChunk],
    chunk_embeddings: Sequence[Sequence[float]],
    query_embeddings: Sequence[Sequence[float]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """후보 하나의 Vector·Hybrid Top-10 결과를 생성한다."""

    if len(questions) != len(query_embeddings):
        raise ValueError("질문과 Query embedding 개수가 일치해야 합니다.")

    bm25 = BM25Retriever(chunks)
    vector_items: List[Dict[str, Any]] = []
    hybrid_items: List[Dict[str, Any]] = []

    for question, query_embedding in zip(questions, query_embeddings):
        vector_results = rank_vector_results(
            chunks,
            chunk_embeddings,
            query_embedding,
            top_k=CANDIDATE_K,
        )
        bm25_results = bm25.search(question["question"], top_k=CANDIDATE_K)
        hybrid_results = fuse_rrf_results(
            bm25_results,
            vector_results,
            top_k=10,
        )
        common = {
            "question_id": question["id"],
            "question": question["question"],
        }
        vector_items.append(
            {**common, "candidates": [_retrieval_candidate(r) for r in vector_results]}
        )
        hybrid_items.append(
            {**common, "candidates": [_hybrid_candidate(r) for r in hybrid_results]}
        )

    return vector_items, hybrid_items


def calculate_cost_usd(input_tokens: int, price_per_million: float) -> float:
    if input_tokens < 0:
        raise ValueError("input_tokens는 음수일 수 없습니다.")
    return input_tokens * price_per_million / 1_000_000


def calculate_raw_vector_storage_bytes(
    vector_count: int,
    dimensions: int,
) -> int:
    if vector_count < 0 or dimensions <= 0:
        raise ValueError("vector_count는 0 이상, dimensions는 1 이상이어야 합니다.")
    return vector_count * dimensions * FLOAT_BYTES


def _fingerprint(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _relevant_section_count(ground_truth: Sequence[Dict[str, Any]]) -> int:
    return sum(
        len({section["section_id"] for section in item["relevant_sections"]})
        for item in ground_truth
    )


def _git_metadata() -> Dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"head": head, "workingTreeClean": not bool(status.strip())}


def build_preflight() -> Dict[str, Any]:
    chunks = attach_experiment_ids(build_retrieval_chunks(DEFAULT_MANIFEST_PATH))
    questions = load_questions(DEFAULT_QUESTIONS_PATH)
    chunk_inputs = [build_embedding_text(chunk) for chunk in chunks]
    query_inputs = [question["question"] for question in questions]
    _, ground_truth = load_evaluation_data(
        PROJECT_ROOT / "evaluation/hybrid_retrieval_candidates.json",
        DEFAULT_GROUND_TRUTH_PATH,
    )
    if {item["id"] for item in questions} != {
        item["question_id"] for item in ground_truth
    }:
        raise ValueError("질문과 ground truth의 ID가 일치하지 않습니다.")

    return {
        "experimentVersion": EXPERIMENT_VERSION,
        "candidateConfigs": [asdict(config) for config in CANDIDATE_CONFIGS],
        "corpus": {
            "documentCount": len({chunk.document_id for chunk in chunks}),
            "chunkCount": len(chunks),
            "inputCharacterCount": sum(len(text) for text in chunk_inputs),
            "inputFingerprint": _fingerprint(chunk_inputs),
        },
        "questions": {
            "questionCount": len(questions),
            "relevantSectionCount": _relevant_section_count(ground_truth),
            "inputCharacterCount": sum(len(text) for text in query_inputs),
            "inputFingerprint": _fingerprint(query_inputs),
        },
        "pricing": {
            "currency": "USD",
            "unit": "per 1M input tokens",
            "checkedAt": PRICE_CHECKED_AT,
            "sourceUrl": PRICE_SOURCE_URL,
        },
        "storage": {
            config.id: {
                "rawDocumentVectorBytes": calculate_raw_vector_storage_bytes(
                    len(chunks), config.dimensions
                ),
                "excludesDatabaseAndIndexOverhead": True,
            }
            for config in CANDIDATE_CONFIGS
        },
    }


def compare_with_baseline(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_metrics = baseline["hybridMetrics"]["summary"]["average_metrics"]
    candidate_metrics = candidate["hybridMetrics"]["summary"]["average_metrics"]
    baseline_questions = {
        item["question_id"]: item for item in baseline["hybridMetrics"]["questions"]
    }
    candidate_questions = {
        item["question_id"]: item for item in candidate["hybridMetrics"]["questions"]
    }

    wins = ties = losses = 0
    changed: List[Dict[str, Any]] = []
    for question_id, baseline_item in baseline_questions.items():
        baseline_score = baseline_item["metrics"]["ndcg_at_10"]
        candidate_score = candidate_questions[question_id]["metrics"]["ndcg_at_10"]
        if candidate_score > baseline_score:
            wins += 1
        elif candidate_score < baseline_score:
            losses += 1
        else:
            ties += 1
        if candidate_score != baseline_score:
            changed.append(
                {
                    "questionId": question_id,
                    "baselineNdcgAt10": baseline_score,
                    "candidateNdcgAt10": candidate_score,
                    "delta": candidate_score - baseline_score,
                }
            )

    return {
        "averageMetricDeltas": {
            key: candidate_metrics[key] - baseline_metrics[key]
            for key in baseline_metrics
        },
        "questionNdcgAt10": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "changedQuestions": changed,
        },
    }


def execute_comparison(
    embedder: Optional[ExperimentEmbedder] = None,
) -> Dict[str, Any]:
    preflight = build_preflight()
    chunks = attach_experiment_ids(build_retrieval_chunks(DEFAULT_MANIFEST_PATH))
    questions = load_questions(DEFAULT_QUESTIONS_PATH)
    chunk_inputs = [build_embedding_text(chunk) for chunk in chunks]
    query_inputs = [question["question"] for question in questions]
    _, ground_truth = load_evaluation_data(
        PROJECT_ROOT / "evaluation/hybrid_retrieval_candidates.json",
        DEFAULT_GROUND_TRUTH_PATH,
    )
    runner = embedder if embedder is not None else ExperimentEmbedder()

    results: Dict[str, Dict[str, Any]] = {}
    for config in CANDIDATE_CONFIGS:
        chunk_batch = runner.embed_many(chunk_inputs, config)
        query_batch = runner.embed_many(query_inputs, config)
        vector_items, hybrid_items = create_candidates(
            questions,
            chunks,
            chunk_batch.embeddings,
            query_batch.embeddings,
        )
        input_tokens = chunk_batch.input_tokens + query_batch.input_tokens
        results[config.id] = {
            "config": asdict(config),
            "usage": {
                "inputTokens": input_tokens,
                "requestCount": chunk_batch.request_count + query_batch.request_count,
                "embeddingLatencyMs": chunk_batch.latency_ms + query_batch.latency_ms,
                "estimatedCostUsd": calculate_cost_usd(
                    input_tokens,
                    config.price_per_million_input_tokens_usd,
                ),
            },
            "vectorMetrics": evaluate_retrieval(vector_items, ground_truth),
            "hybridMetrics": evaluate_retrieval(hybrid_items, ground_truth),
            "vectorCandidates": vector_items,
            "hybridCandidates": hybrid_items,
        }

    baseline = results["A"]
    comparisons = {
        candidate_id: compare_with_baseline(baseline, result)
        for candidate_id, result in results.items()
        if candidate_id != "A"
    }
    return {
        "metadata": {
            **preflight,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "git": _git_metadata(),
            "databaseUsed": False,
            "activeIndexModified": False,
            "similarityImplementation": "in-memory cosine",
            "vectorTieBreak": "synthetic manifest-order chunk id",
        },
        "results": results,
        "comparisonsToA": comparisons,
    }


def save_result(
    result: Dict[str, Any],
    output_path: Path,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"기존 결과를 덮어쓰지 않습니다: {output_path}. "
            "새 경로를 지정하거나 --overwrite를 사용하세요."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="명시한 경우에만 OpenAI Embeddings API를 호출합니다.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="실험 결과 JSON 경로",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 출력 파일 덮어쓰기 허용",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.execute:
        print(json.dumps(build_preflight(), ensure_ascii=False, indent=2))
        print("\nAPI는 호출하지 않았습니다. 실제 실행에는 --execute가 필요합니다.")
        return

    result = execute_comparison()
    output_path = save_result(result, args.output, args.overwrite)
    print(f"비교 완료: {output_path}")


if __name__ == "__main__":
    main()
