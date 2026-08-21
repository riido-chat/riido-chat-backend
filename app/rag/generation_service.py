"""Generation 결과 검증과 최종 답변 상태 결정을 담당한다."""

import re
from typing import Dict, Sequence, Tuple

from generation.generator import OpenAIGenerator, build_generation_context
from generation.models import (
    Citation,
    FinalAnswerStatus,
    FinalGenerationResult,
    FinalWithheldReason,
    GenerationContextSource,
    GenerationStatus,
    ValidatedAnswer,
)
from retrieval.models import HybridRetrievalResult


SOURCE_MARKER_PATTERN = re.compile(r"\[SOURCE_(\d+)\]")
UPSTREAM_ERROR_CODE = "UPSTREAM_ERROR"
CITATION_VALIDATION_ERROR_CODE = "CITATION_VALIDATION_ERROR"

WITHHELD_RESPONSES = {
    FinalWithheldReason.INSUFFICIENT_EVIDENCE: (
        "이용가이드에서 질문에 답할 충분한 근거를 찾지 못했습니다."
    ),
    FinalWithheldReason.AMBIGUOUS_QUESTION: (
        "질문의 의미가 명확하지 않아 답변하기 어렵습니다. "
        "질문을 조금 더 구체적으로 작성해주세요."
    ),
    FinalWithheldReason.OUT_OF_SCOPE: (
        "해당 질문은 이용가이드 범위를 벗어나 답변할 수 없습니다."
    ),
    FinalWithheldReason.UNVERIFIABLE_ANSWER: (
        "답변의 근거 출처를 확인할 수 없어 답변을 제공하지 않습니다."
    ),
}


class UnverifiableAnswerError(ValueError):
    """답변의 citation marker를 신뢰할 수 없을 때 발생한다."""


def validate_citations(
    answer_markdown: str,
    sources: Sequence[GenerationContextSource],
) -> ValidatedAnswer:
    """Citation marker를 검증하고 중복 출처를 병합해 번호를 부여한다."""

    markers = list(SOURCE_MARKER_PATTERN.finditer(answer_markdown))
    if not markers:
        raise UnverifiableAnswerError("citation marker가 없습니다.")

    source_by_id = {source.source_id: source for source in sources}
    used_source_ids = []
    for marker in markers:
        source_id = f"SOURCE_{marker.group(1)}"
        if source_id not in source_by_id:
            raise UnverifiableAnswerError(
                f"Generation Context에 없는 출처입니다: {source_id}"
            )
        if source_id not in used_source_ids:
            used_source_ids.append(source_id)

    if len(used_source_ids) > 3:
        raise UnverifiableAnswerError("서로 다른 SOURCE는 최대 3개까지 허용됩니다.")

    citation_number_by_identity: Dict[Tuple[str, Tuple[str, ...]], int] = {}
    citation_number_by_source_id: Dict[str, int] = {}
    citations = []

    for source_id in used_source_ids:
        source = source_by_id[source_id]
        identity = (source.chunk.source_url, source.chunk.section_path)
        citation_number = citation_number_by_identity.get(identity)

        if citation_number is None:
            citation_number = len(citations) + 1
            citation_number_by_identity[identity] = citation_number
            citations.append(
                Citation(
                    citation_number=citation_number,
                    document_title=source.chunk.document_title,
                    section_path=source.chunk.section_path,
                    source_url=source.chunk.source_url,
                )
            )

        citation_number_by_source_id[source_id] = citation_number

    def replace_marker(marker: re.Match[str]) -> str:
        source_id = f"SOURCE_{marker.group(1)}"
        return f"[{citation_number_by_source_id[source_id]}]"

    validated_markdown = SOURCE_MARKER_PATTERN.sub(replace_marker, answer_markdown)
    return ValidatedAnswer(
        answer_markdown=validated_markdown,
        citations=tuple(citations),
    )


def _withheld_result(reason: FinalWithheldReason) -> FinalGenerationResult:
    return FinalGenerationResult(
        status=FinalAnswerStatus.WITHHELD,
        answer_markdown=WITHHELD_RESPONSES[reason],
        citations=(),
        withheld_reason=reason,
    )


def _error_result(error_code: str) -> FinalGenerationResult:
    return FinalGenerationResult(
        status=FinalAnswerStatus.ERROR,
        answer_markdown=None,
        citations=(),
        error_code=error_code,
    )


class GenerationService:
    """Generator 호출과 Backend Citation Validation을 연결한다."""

    def __init__(self, generator: OpenAIGenerator) -> None:
        self._generator = generator

    async def generate_answer(
        self,
        question: str,
        retrieval_results: Sequence[HybridRetrievalResult],
    ) -> FinalGenerationResult:
        """Hybrid Top-5로 답변을 생성하고 최종 상태를 결정한다."""

        sources = build_generation_context(retrieval_results)

        try:
            generation_result = await self._generator.generate(question, sources)
        except Exception:
            return _error_result(UPSTREAM_ERROR_CODE)

        if generation_result.status == GenerationStatus.WITHHELD:
            reason = FinalWithheldReason(generation_result.withheld_reason.value)
            return _withheld_result(reason)

        try:
            validated_answer = validate_citations(
                generation_result.answer_markdown,
                sources,
            )
        except UnverifiableAnswerError:
            return _withheld_result(FinalWithheldReason.UNVERIFIABLE_ANSWER)
        except Exception:
            return _error_result(CITATION_VALIDATION_ERROR_CODE)

        return FinalGenerationResult(
            status=FinalAnswerStatus.COMPLETED,
            answer_markdown=validated_answer.answer_markdown,
            citations=validated_answer.citations,
        )
