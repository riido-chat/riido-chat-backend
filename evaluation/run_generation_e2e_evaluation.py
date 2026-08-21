"""30개 평가 질문의 실제 Retrieval과 Generation 결과를 저장한다."""

import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database.session import dispose_engine, get_session_factory
from app.rag.generation_service import GenerationService
from evaluation.evaluate_retrieval import DEFAULT_GROUND_TRUTH_PATH
from evaluation.run_bm25_evaluation import (
    DEFAULT_QUESTIONS_PATH,
    load_questions,
)
from generation.generator import OPENAI_GENERATION_MODEL, OpenAIGenerator
from generation.models import FinalGenerationResult
from retrieval.bm25_retriever import BM25Retriever
from retrieval.corpus import build_retrieval_chunks
from retrieval.embedding import OpenAIEmbedder
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import HybridRetrievalResult
from retrieval.pgvector_store import PgVectorStore
from retrieval.vector_retriever import VectorRetriever


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "evaluation/generation_e2e_results.json"


def load_ground_truth(
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH_PATH,
) -> List[Dict[str, Any]]:
    """질문별 정답 Section을 순서대로 읽는다."""

    return json.loads(ground_truth_path.read_text(encoding="utf-8"))


def to_retrieval_section(
    result: HybridRetrievalResult,
) -> Dict[str, Any]:
    """Hybrid 결과에서 평가에 필요한 Section 정보를 추출한다."""

    return {
        "rank": result.final_rank,
        "section_id": result.chunk.section_id,
        "document_title": result.chunk.document_title,
        "section_path": " > ".join(result.chunk.section_path),
    }


def to_citations(result: FinalGenerationResult) -> List[Dict[str, Any]]:
    """검증된 최종 Citation을 JSON 구조로 변환한다."""

    return [
        {
            "citationNumber": citation.citation_number,
            "documentTitle": citation.document_title,
            "sectionPath": " > ".join(citation.section_path),
            "sourceUrl": citation.source_url,
        }
        for citation in result.citations
    ]


def has_relevant_section(
    retrieval_results: Sequence[HybridRetrievalResult],
    ground_truth_item: Dict[str, Any],
) -> bool:
    """정답 Section 중 하나라도 Hybrid Top-5에 포함됐는지 확인한다."""

    relevant_section_ids = {
        section["section_id"]
        for section in ground_truth_item["relevant_sections"]
    }
    return any(
        result.chunk.section_id in relevant_section_ids
        for result in retrieval_results
    )


def to_evaluation_result(
    question: Dict[str, str],
    retrieval_results: Sequence[HybridRetrievalResult],
    ground_truth_item: Dict[str, Any],
    generation_result: FinalGenerationResult,
    generation_latency_seconds: float,
) -> Dict[str, Any]:
    """한 질문의 E2E 결과를 JSON 저장 구조로 변환한다."""

    return {
        "question_id": question["id"],
        "question": question["question"],
        "retrieval_top_5_sections": [
            to_retrieval_section(result) for result in retrieval_results
        ],
        "expected_section_in_top_5": has_relevant_section(
            retrieval_results,
            ground_truth_item,
        ),
        "final_status": generation_result.status.value,
        "answerMarkdown": generation_result.answer_markdown,
        "citations": to_citations(generation_result),
        "withheld_reason": (
            generation_result.withheld_reason.value
            if generation_result.withheld_reason is not None
            else None
        ),
        "error_code": generation_result.error_code,
        "generation_latency_seconds": round(
            generation_latency_seconds,
            3,
        ),
    }


def to_execution_error(
    question: Dict[str, str],
    error: Exception,
) -> Dict[str, Any]:
    """Retrieval 등 평가 실행 중 발생한 기술 오류를 기록한다."""

    return {
        "question_id": question["id"],
        "question": question["question"],
        "retrieval_top_5_sections": [],
        "expected_section_in_top_5": False,
        "final_status": "ERROR",
        "answerMarkdown": None,
        "citations": [],
        "withheld_reason": None,
        "error_code": "EVALUATION_EXECUTION_ERROR",
        "generation_latency_seconds": None,
        "evaluation_error_type": type(error).__name__,
    }


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """요청된 전체 실행 결과를 집계한다."""

    status_distribution = {
        "COMPLETED": 0,
        "WITHHELD": 0,
        "ERROR": 0,
    }
    for result in results:
        status_distribution[result["final_status"]] += 1

    latencies = [
        result["generation_latency_seconds"]
        for result in results
        if result["generation_latency_seconds"] is not None
    ]

    return {
        "total_count": len(results),
        "execution_success_count": (
            status_distribution["COMPLETED"]
            + status_distribution["WITHHELD"]
        ),
        "execution_failure_count": status_distribution["ERROR"],
        "retrieval_top_5_hit_count": sum(
            result["expected_section_in_top_5"] for result in results
        ),
        "status_distribution": status_distribution,
        "average_generation_latency_seconds": (
            round(sum(latencies) / len(latencies), 3)
            if latencies
            else None
        ),
    }


def save_results(
    results: Sequence[Dict[str, Any]],
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """질문별 결과와 전체 집계를 UTF-8 JSON으로 저장한다."""

    payload = {
        "model": OPENAI_GENERATION_MODEL,
        "summary": summarize(results),
        "results": list(results),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


async def run_evaluation() -> Path:
    """Hybrid Top-5부터 최종 Generation까지 30문항을 순차 평가한다."""

    questions = load_questions(DEFAULT_QUESTIONS_PATH)
    ground_truth = load_ground_truth()
    ground_truth_by_id = {
        item["question_id"]: item for item in ground_truth
    }
    if {question["id"] for question in questions} != set(ground_truth_by_id):
        raise ValueError("질문과 ground truth의 question_id가 일치하지 않습니다.")

    bm25_retriever = BM25Retriever(build_retrieval_chunks())
    evaluation_results = []

    try:
        async with get_session_factory()() as session:
            vector_retriever = VectorRetriever(
                OpenAIEmbedder(),
                PgVectorStore(session),
            )
            hybrid_retriever = HybridRetriever(
                bm25_retriever,
                vector_retriever,
            )
            generation_service = GenerationService(OpenAIGenerator())

            for index, question in enumerate(questions, start=1):
                print(
                    f"[{index}/{len(questions)}] "
                    f"{question['id']} 평가 중...",
                    flush=True,
                )
                try:
                    retrieval_results = await hybrid_retriever.search(
                        question["question"]
                    )
                    started_at = perf_counter()
                    generation_result = await generation_service.generate_answer(
                        question["question"],
                        retrieval_results,
                    )
                    generation_latency_seconds = perf_counter() - started_at
                    evaluation_result = to_evaluation_result(
                        question,
                        retrieval_results,
                        ground_truth_by_id[question["id"]],
                        generation_result,
                        generation_latency_seconds,
                    )
                except Exception as error:
                    evaluation_result = to_execution_error(question, error)

                evaluation_results.append(evaluation_result)
                save_results(evaluation_results)
                print(
                    f"[{index}/{len(questions)}] "
                    f"{question['id']} "
                    f"{evaluation_result['final_status']}",
                    flush=True,
                )
    finally:
        await dispose_engine()

    return save_results(evaluation_results)


def main() -> None:
    output_path = asyncio.run(run_evaluation())
    print(f"결과 저장: {output_path}")


if __name__ == "__main__":
    main()
