"""BM25 평가 결과에서 Section 라벨링 후보를 추출한다."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "evaluation/evaluation_retrieval_with_content.txt"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "evaluation/retrieval_labeling_candidates.json"
)

QUESTION_BLOCK_PATTERN = re.compile(
    r"^={80}\n"
    r"질문 ID: (?P<question_id>[^\n]+)\n"
    r"질문: (?P<question>[^\n]+)\n"
    r"(?P<results>.*?)"
    r"(?=^={80}$)",
    re.MULTILINE | re.DOTALL,
)
RESULT_PATTERN = re.compile(
    r"^rank: (?P<rank>\d+)\n"
    r"score: (?P<score>-?\d+(?:\.\d+)?)\n"
    r"section_id: (?P<section_id>[^\n]+)\n"
    r"document_title: (?P<document_title>[^\n]+)\n"
    r"section_path: (?P<section_path>[^\n]+)\n"
    r"source_url: [^\n]*\n"
    r"content:\n",
    re.MULTILINE,
)


def parse_labeling_candidates(text: str) -> List[Dict[str, Any]]:
    """평가 결과 문자열에서 질문과 검색 후보 metadata를 추출한다."""

    questions = []

    for question_match in QUESTION_BLOCK_PATTERN.finditer(text):
        candidates = [
            {
                "rank": int(result_match.group("rank")),
                "section_id": result_match.group("section_id"),
                "document_title": result_match.group("document_title"),
                "section_path": result_match.group("section_path"),
                "score": float(result_match.group("score")),
            }
            for result_match in RESULT_PATTERN.finditer(
                question_match.group("results")
            )
        ]

        if not candidates:
            question_id = question_match.group("question_id")
            raise ValueError(f"검색 후보가 없는 질문입니다: {question_id}")

        questions.append(
            {
                "question_id": question_match.group("question_id"),
                "question": question_match.group("question"),
                "candidates": candidates,
            }
        )

    if not questions:
        raise ValueError("평가 결과에서 질문을 찾을 수 없습니다.")

    return questions


def create_labeling_candidates(
    input_path: Union[str, Path] = DEFAULT_INPUT_PATH,
) -> List[Dict[str, Any]]:
    """평가 결과 파일을 읽어 라벨링 후보 목록을 만든다."""

    text = Path(input_path).read_text(encoding="utf-8")
    return parse_labeling_candidates(text)


def save_labeling_candidates(
    questions: List[Dict[str, Any]],
    output_path: Union[str, Path] = DEFAULT_OUTPUT_PATH,
) -> Path:
    """라벨링 후보 목록을 UTF-8 JSON 파일로 저장한다."""

    path = Path(output_path)
    path.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    questions = create_labeling_candidates()
    output_path = save_labeling_candidates(questions)
    candidate_count = sum(
        len(question["candidates"]) for question in questions
    )
    print(f"질문 {len(questions)}개, 후보 {candidate_count}개 저장")
    print(f"결과 저장: {output_path}")


if __name__ == "__main__":
    main()
