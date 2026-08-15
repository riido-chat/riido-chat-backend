"""BM25 baseline의 질문별 Top-10 검색 결과를 출력한다."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.document.chunker import create_chunks
from pipeline.document.loader import load_normalized_documents
from pipeline.document.section_parser import parse_sections
from retrieval.bm25_retriever import BM25Retriever
from retrieval.models import RetrievalChunk


DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "evaluation/bm25_questions.json"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data/clean_manifest.json"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "evaluation/evaluation_retrieval_with_content.txt"
)


def load_questions(
    questions_path: Union[str, Path] = DEFAULT_QUESTIONS_PATH,
) -> List[Dict[str, str]]:
    """평가 질문 JSON을 순서대로 읽는다."""

    return json.loads(Path(questions_path).read_text(encoding="utf-8"))


def build_retrieval_chunks(
    manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
) -> List[RetrievalChunk]:
    """기존 문서 파이프라인의 결과를 RetrievalChunk로 변환한다."""

    retrieval_chunks = []

    for document in load_normalized_documents(manifest_path):
        sections = parse_sections(document)
        chunks = create_chunks(sections)
        retrieval_chunks.extend(
            RetrievalChunk.from_document_chunk(document, chunk)
            for chunk in chunks
        )

    return retrieval_chunks


def format_evaluation_results(
    questions: List[Dict[str, str]],
    retriever: BM25Retriever,
) -> str:
    """각 질문과 BM25 Top-10 검색 결과를 출력 문자열로 만든다."""

    lines = []

    for question in questions:
        lines.extend(
            [
                "=" * 80,
                f"질문 ID: {question['id']}",
                f"질문: {question['question']}",
            ]
        )

        for result in retriever.search(question["question"], top_k=10):
            lines.extend(
                [
                    "",
                    f"rank: {result.rank}",
                    f"score: {result.score:.6f}",
                    f"document_title: {result.chunk.document_title}",
                    f"section_path: {' > '.join(result.chunk.section_path)}",
                    f"source_url: {result.chunk.source_url}",
                    "content:",
                    result.chunk.content,
                ]
            )

    lines.append("=" * 80)
    return "\n".join(lines) + "\n"


def print_evaluation_results(
    questions: List[Dict[str, str]],
    retriever: BM25Retriever,
) -> None:
    """각 질문과 BM25 Top-10 검색 결과를 표준 출력으로 보여준다."""

    print(format_evaluation_results(questions, retriever), end="")


def save_evaluation_results(
    result_text: str,
    output_path: Union[str, Path] = DEFAULT_OUTPUT_PATH,
) -> Path:
    """BM25 평가 결과를 UTF-8 텍스트 파일로 저장한다."""

    path = Path(output_path)
    path.write_text(result_text, encoding="utf-8")
    return path


def main() -> None:
    questions = load_questions()
    retriever = BM25Retriever(build_retrieval_chunks())
    result_text = format_evaluation_results(questions, retriever)
    print(result_text, end="")
    output_path = save_evaluation_results(result_text)
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
