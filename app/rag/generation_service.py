"""Generation 결과 검증과 최종 답변 상태 결정을 담당한다."""

import re
import time
from typing import Dict, Optional, Sequence, Tuple

from app.rag.model_trace import BeforeModelCallHook, ModelCallTrace
from generation.generator import (
    GENERATION_PROMPT_VERSION,
    OPENAI_GENERATION_MODEL,
    OPENAI_GENERATION_PROVIDER,
    OpenAIGenerator,
    build_generation_context,
)
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


SOURCE_MARKER_PATTERN = re.compile(r"\[SOURCE_([A-Za-z0-9_-]+)\]")
FORBIDDEN_ANSWER_CONTENT_PATTERNS = (
    re.compile(r"!?\[[^\]\r\n]*\]\([^\)\r\n]*\)"),
    re.compile(r"\[[^\]\r\n]+\]\[[^\]\r\n]*\]"),
    re.compile(r"(?m)^[ \t]{0,3}\[[^\]\r\n]+\]:[ \t]*\S+"),
    re.compile(r"(?i)\b(?:[a-z][a-z0-9+.-]*://|www\.)[^\s<]+"),
    re.compile(r"(?i)<[a-z][a-z0-9+.-]*:[^>\s]+>"),
    re.compile(r"<[^<>\s]+@[^<>\s]+>"),
    re.compile(
        r"<!--|<![A-Za-z]|</?[A-Za-z][A-Za-z0-9-]*"
        r"(?:\s[^<>]*?)?/?>"
    ),
)
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
    """답변이 외부 응답 계약에 맞게 검증될 수 없을 때 발생한다."""


def _validate_answer_content(answer_markdown: str) -> None:
    """출처 영역 밖에서 금지한 링크와 HTML이 답변 본문에 없는지 확인한다."""

    content_without_source_markers = SOURCE_MARKER_PATTERN.sub("", answer_markdown)
    if any(
        pattern.search(content_without_source_markers)
        for pattern in FORBIDDEN_ANSWER_CONTENT_PATTERNS
    ):
        raise UnverifiableAnswerError("답변 본문에 링크 또는 HTML이 포함됐습니다.")


def validate_citations(
    answer_markdown: str,
    sources: Sequence[GenerationContextSource],
) -> ValidatedAnswer:
    """본문 형식과 Citation marker를 검증하고 중복 출처를 병합한다."""

    _validate_answer_content(answer_markdown)

    markers = list(SOURCE_MARKER_PATTERN.finditer(answer_markdown))
    if not markers:
        raise UnverifiableAnswerError("citation marker가 없습니다.")

    source_by_id = {source.source_id: source for source in sources}
    used_source_ids = []
    for marker in markers:
        source_id = f"SOURCE_{marker.group(1)}"
        if source_id in source_by_id and source_id not in used_source_ids:
            used_source_ids.append(source_id)

    if not used_source_ids:
        raise UnverifiableAnswerError("유효한 citation marker가 없습니다.")

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
                    chunk_id=source.chunk.chunk_id,
                    document_version_id=source.chunk.document_version_id,
                )
            )

        citation_number_by_source_id[source_id] = citation_number

    if len(citations) > 3:
        raise UnverifiableAnswerError("최종 Citation은 최대 3개까지 허용됩니다.")

    def replace_marker(marker: re.Match[str]) -> str:
        source_id = f"SOURCE_{marker.group(1)}"
        citation_number = citation_number_by_source_id.get(source_id)
        return "" if citation_number is None else f"[{citation_number}]"

    validated_markdown = SOURCE_MARKER_PATTERN.sub(
        replace_marker,
        answer_markdown,
    ).strip()
    return ValidatedAnswer(
        answer_markdown=validated_markdown,
        citations=tuple(citations),
    )


def _withheld_result(
    reason: FinalWithheldReason,
    model_call: Optional[ModelCallTrace] = None,
) -> FinalGenerationResult:
    return FinalGenerationResult(
        status=FinalAnswerStatus.WITHHELD,
        answer_markdown=WITHHELD_RESPONSES[reason],
        citations=(),
        withheld_reason=reason,
        model_call=model_call,
    )


def _error_result(
    error_code: str,
    model_call: Optional[ModelCallTrace] = None,
) -> FinalGenerationResult:
    return FinalGenerationResult(
        status=FinalAnswerStatus.ERROR,
        answer_markdown=None,
        citations=(),
        error_code=error_code,
        model_call=model_call,
    )


def _failed_generation_trace(
    started: float,
    error: Exception,
) -> ModelCallTrace:
    """Generator가 trace를 만들기 전에 끝난 예상 밖 실패를 기록한다."""

    return ModelCallTrace(
        provider=OPENAI_GENERATION_PROVIDER,
        model_name=OPENAI_GENERATION_MODEL,
        succeeded=False,
        latency_ms=int((time.perf_counter() - started) * 1000),
        prompt_version=GENERATION_PROMPT_VERSION,
        error_message=str(error),
    )


class GenerationService:
    """Generator 호출과 Backend Citation Validation을 연결한다."""

    def __init__(self, generator: OpenAIGenerator) -> None:
        self._generator = generator

    async def generate_answer(
        self,
        question: str,
        retrieval_results: Sequence[HybridRetrievalResult],
        *,
        before_model_call: Optional[BeforeModelCallHook] = None,
    ) -> FinalGenerationResult:
        """Hybrid Top-5로 답변을 생성하고 최종 상태를 결정한다."""

        sources = build_generation_context(retrieval_results)
        if before_model_call is not None:
            await before_model_call(
                OPENAI_GENERATION_PROVIDER,
                OPENAI_GENERATION_MODEL,
                GENERATION_PROMPT_VERSION,
            )

        started = time.perf_counter()
        try:
            call = await self._generator.generate_with_trace(question, sources)
        except Exception as error:
            return _error_result(
                UPSTREAM_ERROR_CODE,
                _failed_generation_trace(started, error),
            )

        if call.error is not None:
            return _error_result(UPSTREAM_ERROR_CODE, call.trace)

        generation_result = call.result
        if generation_result.status == GenerationStatus.WITHHELD:
            reason = FinalWithheldReason(generation_result.withheld_reason.value)
            return _withheld_result(reason, call.trace)

        try:
            validated_answer = validate_citations(
                generation_result.answer_markdown,
                sources,
            )
        except UnverifiableAnswerError:
            return _withheld_result(
                FinalWithheldReason.UNVERIFIABLE_ANSWER,
                call.trace,
            )
        except Exception:
            return _error_result(CITATION_VALIDATION_ERROR_CODE, call.trace)

        return FinalGenerationResult(
            status=FinalAnswerStatus.COMPLETED,
            answer_markdown=validated_answer.answer_markdown,
            citations=validated_answer.citations,
            model_call=call.trace,
        )
