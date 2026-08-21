"""실제 Hybrid Retrieval부터 Generation까지 수동으로 확인한다."""

import argparse
import asyncio
import sys
from pathlib import Path
from time import perf_counter
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database.session import dispose_engine, get_session_factory
from app.rag.generation_service import GenerationService
from generation.generator import OpenAIGenerator
from generation.models import FinalGenerationResult
from retrieval.bm25_retriever import BM25Retriever
from retrieval.corpus import build_retrieval_chunks
from retrieval.embedding import OpenAIEmbedder
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import HybridRetrievalResult
from retrieval.pgvector_store import PgVectorStore
from retrieval.vector_retriever import VectorRetriever


def print_hybrid_results(results: Sequence[HybridRetrievalResult]) -> None:
    print("Hybrid Retrieval Top-5:")
    if not results:
        print("없음")
        return

    for result in results:
        print(f"- rank: {result.final_rank}")
        print(f"  document title: {result.chunk.document_title}")
        print(f"  section path: {' > '.join(result.chunk.section_path)}")


def print_generation_result(
    result: FinalGenerationResult,
    elapsed_seconds: float,
) -> None:
    print(f"최종 Generation 상태: {result.status.value}")
    print("최종 answerMarkdown:")
    print(result.answer_markdown if result.answer_markdown is not None else "None")
    print("citation 목록:")

    if not result.citations:
        print("없음")
    else:
        for citation in result.citations:
            print(f"- citationNumber: {citation.citation_number}")
            print(f"  documentTitle: {citation.document_title}")
            print(f"  sectionPath: {' > '.join(citation.section_path)}")
            print(f"  sourceUrl: {citation.source_url}")

    print(f"Generation 소요 시간: {elapsed_seconds:.3f}초")


async def run(question: str) -> None:
    bm25_retriever = BM25Retriever(build_retrieval_chunks())

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
            hybrid_results = await hybrid_retriever.search(question)

            print(f"입력 질문: {question}")
            print_hybrid_results(hybrid_results)

            generation_service = GenerationService(OpenAIGenerator())
            started_at = perf_counter()
            generation_result = await generation_service.generate_answer(
                question,
                hybrid_results,
            )
            elapsed_seconds = perf_counter() - started_at

            print_generation_result(generation_result, elapsed_seconds)
    finally:
        await dispose_engine()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid Retrieval과 Generation 결과를 수동 확인합니다."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="뤼이도 이용가이드에 질문할 내용",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.question is not None:
        asyncio.run(run(args.question))
        return

    while True:
        question = input("질문 > ").strip()
        if question.lower() in ("exit", "quit", "q"):
            return
        if not question:
            continue
        asyncio.run(run(question))


if __name__ == "__main__":
    main()
