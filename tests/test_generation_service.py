import unittest
from unittest.mock import AsyncMock, patch

from app.rag.generation_service import (
    CITATION_VALIDATION_ERROR_CODE,
    UPSTREAM_ERROR_CODE,
    WITHHELD_RESPONSES,
    GenerationService,
)
from app.rag.model_trace import ModelCallTrace
from generation.generator import (
    GENERATION_PROMPT_VERSION,
    OPENAI_GENERATION_MODEL,
    OpenAIGenerator,
)
from generation.models import (
    FinalAnswerStatus,
    FinalWithheldReason,
    GenerationCall,
    GenerationResult,
    GenerationStatus,
    GenerationWithheldReason,
)
from retrieval.models import HybridRetrievalResult, RetrievalChunk


def _trace(succeeded: bool = True) -> ModelCallTrace:
    return ModelCallTrace(
        provider="openai",
        model_name=OPENAI_GENERATION_MODEL,
        succeeded=succeeded,
        latency_ms=120,
        retry_count=1 if not succeeded else 0,
        prompt_version=GENERATION_PROMPT_VERSION,
    )


def _failed_trace() -> ModelCallTrace:
    return _trace(succeeded=False)


def _call(result: GenerationResult) -> GenerationCall:
    return GenerationCall(trace=_trace(), result=result)


class GenerationServiceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.generator = AsyncMock(spec=OpenAIGenerator)
        self.service = GenerationService(self.generator)

    async def test_completes_with_first_marker_order_and_repeated_source(self) -> None:
        results = [self._result(1), self._result(2)]
        self.generator.generate_with_trace.return_value = _call(
            self._answerable(
                "두 번째 근거입니다. [SOURCE_2] 첫 번째 근거입니다. "
                "[SOURCE_1] 다시 두 번째입니다. [SOURCE_2]"
            )
        )

        result = await self.service.generate_answer("질문", results)

        self.assertEqual(FinalAnswerStatus.COMPLETED, result.status)
        self.assertEqual(
            "두 번째 근거입니다. [1] 첫 번째 근거입니다. [2] "
            "다시 두 번째입니다. [1]",
            result.answer_markdown,
        )
        self.assertEqual([1, 2], [item.citation_number for item in result.citations])
        self.assertEqual(
            [results[1].chunk.source_url, results[0].chunk.source_url],
            [item.source_url for item in result.citations],
        )
        self.assertIsNone(result.withheld_reason)
        self.assertIsNone(result.error_code)

    async def test_merges_different_source_ids_with_same_source_identity(self) -> None:
        first = self._result(
            1,
            source_url="https://same",
            document_title="같은 문서",
            section_name="같은 섹션",
        )
        duplicate = self._result(
            2,
            source_url="https://same",
            document_title="같은 문서",
            section_name="같은 섹션",
        )
        third = self._result(3)
        self.generator.generate_with_trace.return_value = _call(
            self._answerable(
                "첫 근거 [SOURCE_1] 중복 근거 [SOURCE_2] 다른 근거 [SOURCE_3]"
            )
        )

        result = await self.service.generate_answer(
            "질문",
            [first, duplicate, third],
        )

        self.assertEqual(FinalAnswerStatus.COMPLETED, result.status)
        self.assertEqual(
            "첫 근거 [1] 중복 근거 [1] 다른 근거 [2]",
            result.answer_markdown,
        )
        self.assertEqual(2, len(result.citations))
        self.assertEqual([1, 2], [item.citation_number for item in result.citations])

    async def test_keeps_same_url_with_different_section_paths_separate(self) -> None:
        first = self._result(1, source_url="https://same", section_name="섹션 A")
        second = self._result(2, source_url="https://same", section_name="섹션 B")
        self.generator.generate_with_trace.return_value = _call(
            self._answerable(
                "A 근거 [SOURCE_1] B 근거 [SOURCE_2]"
            )
        )

        result = await self.service.generate_answer("질문", [first, second])

        self.assertEqual(2, len(result.citations))
        self.assertEqual("A 근거 [1] B 근거 [2]", result.answer_markdown)

    async def test_withholds_unverifiable_answers_without_citations(self) -> None:
        cases = (
            "marker가 없는 답변",
            "존재하지 않는 근거 [SOURCE_9]",
            "근거 [SOURCE_1] [SOURCE_2] [SOURCE_3] [SOURCE_4]",
        )

        for answer_markdown in cases:
            with self.subTest(answer_markdown=answer_markdown):
                self.generator.generate_with_trace.return_value = _call(
                    self._answerable(
                        answer_markdown
                    )
                )
                result = await self.service.generate_answer(
                    "질문",
                    [self._result(index) for index in range(1, 5)],
                )

                self.assertEqual(FinalAnswerStatus.WITHHELD, result.status)
                self.assertEqual(
                    FinalWithheldReason.UNVERIFIABLE_ANSWER,
                    result.withheld_reason,
                )
                self.assertEqual(
                    WITHHELD_RESPONSES[
                        FinalWithheldReason.UNVERIFIABLE_ANSWER
                    ],
                    result.answer_markdown,
                )
                self.assertEqual((), result.citations)
                self.assertIsNone(result.error_code)

    async def test_uses_fixed_response_for_generator_withheld_reason(self) -> None:
        for reason in GenerationWithheldReason:
            with self.subTest(reason=reason):
                self.generator.generate_with_trace.return_value = _call(
                    GenerationResult(
                        status=GenerationStatus.WITHHELD,
                        answer_markdown=None,
                        withheld_reason=reason,
                    )
                )

                result = await self.service.generate_answer("질문", [])

                final_reason = FinalWithheldReason(reason.value)
                self.assertEqual(FinalAnswerStatus.WITHHELD, result.status)
                self.assertEqual(final_reason, result.withheld_reason)
                self.assertEqual(WITHHELD_RESPONSES[final_reason], result.answer_markdown)
                self.assertEqual((), result.citations)

    def test_keeps_exact_backend_withheld_responses(self) -> None:
        self.assertEqual(
            {
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
            },
            WITHHELD_RESPONSES,
        )

    async def test_returns_error_when_generation_call_fails(self) -> None:
        self.generator.generate_with_trace.return_value = GenerationCall(
            trace=_failed_trace(),
            error=RuntimeError("API failure"),
        )

        result = await self.service.generate_answer("질문", [])

        self.assertEqual(FinalAnswerStatus.ERROR, result.status)
        self.assertEqual(UPSTREAM_ERROR_CODE, result.error_code)
        self.assertIsNone(result.answer_markdown)
        self.assertEqual((), result.citations)
        self.assertIsNone(result.withheld_reason)

    async def test_returns_error_when_citation_validation_processing_fails(
        self,
    ) -> None:
        self.generator.generate_with_trace.return_value = _call(
            self._answerable("답변 [SOURCE_1]")
        )

        with patch(
            "app.rag.generation_service.validate_citations",
            side_effect=RuntimeError("validation failure"),
        ):
            result = await self.service.generate_answer("질문", [self._result(1)])

        self.assertEqual(FinalAnswerStatus.ERROR, result.status)
        self.assertEqual(CITATION_VALIDATION_ERROR_CODE, result.error_code)
        self.assertIsNone(result.answer_markdown)
        self.assertEqual((), result.citations)

    async def test_citations_carry_chunk_identifiers_for_logging(self) -> None:
        results = [self._result(1), self._result(2)]
        self.generator.generate_with_trace.return_value = _call(
            self._answerable("첫 근거 [SOURCE_2]")
        )

        result = await self.service.generate_answer("질문", results)

        citation = result.citations[0]
        self.assertEqual(results[1].chunk.chunk_id, citation.chunk_id)
        self.assertEqual(
            results[1].chunk.document_version_id,
            citation.document_version_id,
        )

    async def test_passes_model_call_trace_through_every_outcome(self) -> None:
        results = [self._result(1)]

        self.generator.generate_with_trace.return_value = _call(
            self._answerable("근거 [SOURCE_1]")
        )
        completed = await self.service.generate_answer("질문", results)
        self.assertIsNotNone(completed.model_call)
        self.assertTrue(completed.model_call.succeeded)

        self.generator.generate_with_trace.return_value = _call(
            GenerationResult(
                status=GenerationStatus.WITHHELD,
                answer_markdown=None,
                withheld_reason=GenerationWithheldReason.OUT_OF_SCOPE,
            )
        )
        withheld = await self.service.generate_answer("질문", results)
        self.assertIsNotNone(withheld.model_call)

        self.generator.generate_with_trace.return_value = GenerationCall(
            trace=_failed_trace(),
            error=RuntimeError("API failure"),
        )
        failed = await self.service.generate_answer("질문", results)
        self.assertIsNotNone(failed.model_call)
        self.assertFalse(failed.model_call.succeeded)
        self.assertEqual(1, failed.model_call.retry_count)

    @staticmethod
    def _answerable(answer_markdown: str) -> GenerationResult:
        return GenerationResult(
            status=GenerationStatus.ANSWERABLE,
            answer_markdown=answer_markdown,
            withheld_reason=None,
        )

    @staticmethod
    def _result(
        index: int,
        source_url: str = "",
        document_title: str = "",
        section_name: str = "",
    ) -> HybridRetrievalResult:
        title = document_title or f"문서 {index}"
        chunk = RetrievalChunk(
            document_id=f"document-{index}",
            section_id=f"section-{index}",
            document_title=title,
            section_path=(title, section_name or f"섹션 {index}"),
            source_url=source_url or f"https://docs.riido.io/{index}",
            category="guide",
            content=f"본문 {index}",
            chunk_id=index,
            document_version_id=100 + index,
            index_version_id=1,
        )
        return HybridRetrievalResult(
            chunk=chunk,
            rrf_score=0.1,
            final_rank=index,
            bm25_rank=index,
            vector_rank=index,
        )


if __name__ == "__main__":
    unittest.main()
