"""Vector Retrieval의 후보를 생성하고 기존 metric으로 평가한다."""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database.session import dispose_engine, get_session_factory
from evaluation.create_labeling_candidates import save_labeling_candidates
from evaluation.evaluate_retrieval import (
    DEFAULT_GROUND_TRUTH_PATH,
    evaluate_retrieval,
    load_evaluation_data,
    print_evaluation_metrics,
    save_evaluation_metrics,
)
from evaluation.run_bm25_evaluation import (
    DEFAULT_QUESTIONS_PATH,
    load_questions,
)
from app.retrieval.embedding import OpenAIEmbedder
from app.retrieval.models import RetrievalResult
from app.retrieval.search_reader import SearchReader
from app.retrieval.vector_retriever import VectorRetriever


DEFAULT_CANDIDATES_PATH = (
    PROJECT_ROOT / "evaluation/vector_retrieval_candidates.json"
)
DEFAULT_METRICS_PATH = (
    PROJECT_ROOT / "evaluation/vector_evaluation_metrics.json"
)


def to_candidate(result: RetrievalResult) -> Dict[str, Any]:
    """Vector 검색 결과를 기존 평가 candidate 구조로 변환한다."""

    return {
        "rank": result.rank,
        "section_id": result.chunk.section_id,
        "document_title": result.chunk.document_title,
        "section_path": " > ".join(result.chunk.section_path),
        "score": result.score,
    }


async def run_vector_evaluation(
    questions: Sequence[Dict[str, str]],
    retriever: VectorRetriever,
) -> List[Dict[str, Any]]:
    """모든 질문의 Vector Top-10을 기존 평가 candidate로 만든다."""

    candidate_items = []

    for question in questions:
        results = await retriever.search(question["question"], top_k=10)
        candidate_items.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "candidates": [to_candidate(result) for result in results],
            }
        )

    return candidate_items


async def create_vector_candidates(
    questions: Sequence[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """기존 session factory로 실제 Vector 검색 후보를 생성한다."""

    try:
        async with get_session_factory()() as session:
            retriever = VectorRetriever(
                OpenAIEmbedder(),
                SearchReader(session),
            )
            return await run_vector_evaluation(questions, retriever)
    finally:
        await dispose_engine()


async def main() -> None:
    questions = load_questions(DEFAULT_QUESTIONS_PATH)
    candidate_items = await create_vector_candidates(questions)
    candidates_path = save_labeling_candidates(
        candidate_items,
        DEFAULT_CANDIDATES_PATH,
    )

    candidates, ground_truth = load_evaluation_data(
        candidates_path,
        DEFAULT_GROUND_TRUTH_PATH,
    )
    evaluation_result = evaluate_retrieval(candidates, ground_truth)
    metrics_path = save_evaluation_metrics(
        evaluation_result,
        DEFAULT_METRICS_PATH,
    )

    print_evaluation_metrics(evaluation_result)
    print(f"\n후보 저장: {candidates_path}")
    print(f"평가 저장: {metrics_path}")


if __name__ == "__main__":
    asyncio.run(main())
