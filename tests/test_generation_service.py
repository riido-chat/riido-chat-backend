import unittest
from dataclasses import replace
from typing import Optional
from unittest.mock import AsyncMock, patch

import httpx
from openai import APITimeoutError

from app.answering.service import (
    CITATION_VALIDATION_ERROR_CODE,
    INTERNAL_ERROR_CODE,
    UPSTREAM_ERROR_CODE,
    WITHHELD_RESPONSES,
    GenerationService,
)
from app.core.model_trace import ModelCallTrace
from app.document.document_key import (
    DEFAULT_DOCUMENT_GROUP_KEY,
    build_console_canonical_uri,
    build_upload_document_key,
)
from app.answering.generator import (
    GENERATION_PROMPT_VERSION,
    OPENAI_GENERATION_MODEL,
    OPENAI_GENERATION_PROVIDER,
    OpenAIGenerator,
    build_generation_context,
)
from app.answering.models import (
    CitationSourceKind,
    FinalAnswerStatus,
    FinalWithheldReason,
    GenerationAnswerType,
    GenerationAnswerScope,
    GenerationCall,
    GenerationEvidenceRequirement,
    GenerationResult,
    GenerationSourcePlan,
    GenerationStageTrace,
    GenerationStatus,
    GenerationWithheldReason,
)
from app.retrieval.models import HybridRetrievalResult, RetrievalChunk


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


def _call(
    result: GenerationResult,
    stage_trace: Optional[GenerationStageTrace] = None,
) -> GenerationCall:
    return GenerationCall(
        trace=_trace(),
        result=result,
        stage_trace=stage_trace,
    )


class GenerationServiceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.generator = AsyncMock(spec=OpenAIGenerator)
        self.generator.model_name = OPENAI_GENERATION_MODEL
        self.service = GenerationService(self.generator)

    async def test_runs_checkpoint_before_generation(self) -> None:
        events = []

        async def checkpoint(*_args) -> None:
            events.append("checkpoint")

        async def generate(*_args):
            events.append("generation")
            return _call(
                GenerationResult(
                    status=GenerationStatus.WITHHELD,
                    answer_markdown=None,
                    withheld_reason=GenerationWithheldReason.OUT_OF_SCOPE,
                )
            )

        before_model_call = AsyncMock(side_effect=checkpoint)
        self.generator.generate_with_trace.side_effect = generate

        await self.service.generate_answer(
            "질문",
            [],
            before_model_call=before_model_call,
        )

        self.assertEqual(["checkpoint", "generation"], events)
        before_model_call.assert_awaited_once_with(
            OPENAI_GENERATION_PROVIDER,
            OPENAI_GENERATION_MODEL,
            GENERATION_PROMPT_VERSION,
        )

    async def test_checkpoints_configured_generation_model(self) -> None:
        self.generator.model_name = "generation-test"
        self.generator.generate_with_trace.return_value = _call(
            GenerationResult(
                status=GenerationStatus.WITHHELD,
                answer_markdown=None,
                withheld_reason=GenerationWithheldReason.OUT_OF_SCOPE,
            )
        )
        before_model_call = AsyncMock()

        result = await self.service.generate_answer(
            "질문",
            [],
            before_model_call=before_model_call,
        )

        before_model_call.assert_awaited_once_with(
            OPENAI_GENERATION_PROVIDER,
            "generation-test",
            GENERATION_PROMPT_VERSION,
        )
        self.assertEqual(FinalAnswerStatus.WITHHELD, result.status)

    async def test_does_not_generate_when_checkpoint_fails(self) -> None:
        before_model_call = AsyncMock(side_effect=RuntimeError("checkpoint failed"))

        with self.assertRaisesRegex(RuntimeError, "checkpoint failed"):
            await self.service.generate_answer(
                "질문",
                [],
                before_model_call=before_model_call,
            )

        self.generator.generate_with_trace.assert_not_awaited()

    async def test_maps_unexpected_generator_exception_to_internal_error(
        self,
    ) -> None:
        before_model_call = AsyncMock()
        self.generator.generate_with_trace.side_effect = RuntimeError(
            "unexpected provider failure"
        )

        with patch(
            "app.answering.service.time.perf_counter",
            side_effect=[10.0, 12.5],
        ):
            result = await self.service.generate_answer(
                "질문",
                [],
                before_model_call=before_model_call,
            )

        self.assertEqual(FinalAnswerStatus.ERROR, result.status)
        self.assertEqual(INTERNAL_ERROR_CODE, result.error_code)
        self.assertIsNotNone(result.model_call)
        self.assertFalse(result.model_call.succeeded)
        self.assertEqual(2500, result.model_call.latency_ms)
        self.assertEqual(0, result.model_call.retry_count)
        self.assertEqual(OPENAI_GENERATION_MODEL, result.model_call.model_name)
        self.assertEqual(
            "unexpected provider failure",
            result.model_call.error_message,
        )
        before_model_call.assert_awaited_once()

    async def test_completes_with_first_marker_order_and_repeated_source(self) -> None:
        results = [self._result(1), self._result(2)]
        generated = self._answerable(
            "두 번째 근거입니다. [SOURCE_2] 첫 번째 근거입니다. "
            "[SOURCE_1] 다시 두 번째입니다. [SOURCE_2]"
        )
        stage_trace = GenerationStageTrace(
            pre_validation_result=generated,
        )
        self.generator.generate_with_trace.return_value = _call(
            generated,
            stage_trace,
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
        self.assertIs(stage_trace, result.stage_trace)

    async def test_marks_citation_source_kind_by_canonical_uri_scheme(self) -> None:
        console = self._result(
            1,
            source_url=build_console_canonical_uri(
                DEFAULT_DOCUMENT_GROUP_KEY,
                build_upload_document_key("업로드 문서"),
            ),
        )
        gitbook = self._result(2)
        self.generator.generate_with_trace.return_value = _call(
            self._answerable("콘솔 근거 [SOURCE_1] 깃북 근거 [SOURCE_2]")
        )

        result = await self.service.generate_answer("질문", [console, gitbook])

        self.assertEqual(
            [CitationSourceKind.CONSOLE, CitationSourceKind.GITBOOK],
            [item.source_kind for item in result.citations],
        )
        # 내부 인용에는 원문 위치자를 그대로 남긴다. 감추는 일은 응답 변환이 맡는다.
        self.assertEqual(
            console.chunk.source_url,
            result.citations[0].source_url,
        )

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

    async def test_merges_duplicates_among_four_selected_sources(self) -> None:
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
        self.generator.generate_with_trace.return_value = _call(
            self._answerable(
                "중복 근거[SOURCE_1][SOURCE_2], "
                "두 번째 근거[SOURCE_3], 세 번째 근거[SOURCE_4]"
            )
        )

        result = await self.service.generate_answer(
            "질문",
            [first, duplicate, self._result(3), self._result(4)],
        )

        self.assertEqual(FinalAnswerStatus.COMPLETED, result.status)
        self.assertEqual(
            "중복 근거[1][1], 두 번째 근거[2], 세 번째 근거[3]",
            result.answer_markdown,
        )
        self.assertEqual(3, len(result.citations))

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

    async def test_withholds_mixed_valid_and_invalid_source_markers(
        self,
    ) -> None:
        self.generator.generate_with_trace.return_value = _call(
            self._answerable(
                "잘못된 근거[SOURCE_9], 유효한 근거[SOURCE_1], "
                "잘못된 형식[SOURCE_unknown]"
            )
        )

        result = await self.service.generate_answer("질문", [self._result(1)])

        self.assertEqual(FinalAnswerStatus.WITHHELD, result.status)
        self.assertEqual(
            FinalWithheldReason.UNVERIFIABLE_ANSWER,
            result.withheld_reason,
        )
        self.assertEqual((), result.citations)
        self.generator.regenerate_answer_with_trace.assert_not_awaited()

    async def test_validates_against_selected_sources_and_regenerates_answer_once(
        self,
    ) -> None:
        results = [self._result(1), self._result(2)]
        generated = self._answerable("미선택 근거 [SOURCE_2]")
        stage_trace = self._stage_trace(results[:1], generated)
        repaired = self._answerable("선택 근거 [SOURCE_1]")
        initial_trace = replace(
            _trace(),
            input_tokens=100,
            output_tokens=20,
        )
        self.generator.generate_with_trace.return_value = GenerationCall(
            trace=initial_trace,
            result=generated,
            stage_trace=stage_trace,
        )

        async def regenerate(_question, failed_trace, _reason):
            repair_trace = replace(
                _trace(),
                input_tokens=30,
                output_tokens=10,
            )
            return GenerationCall(
                trace=repair_trace,
                result=repaired,
                stage_trace=replace(
                    failed_trace,
                    validation_regeneration_result=repaired,
                    validation_regeneration_count=1,
                    validation_regeneration_model_call=repair_trace,
                ),
            )

        self.generator.regenerate_answer_with_trace.side_effect = regenerate

        result = await self.service.generate_answer("질문", results)

        self.assertEqual(FinalAnswerStatus.COMPLETED, result.status)
        self.assertEqual("선택 근거 [1]", result.answer_markdown)
        self.assertEqual(1, result.stage_trace.validation_regeneration_count)
        self.assertEqual(generated, result.stage_trace.pre_validation_result)
        self.assertEqual(
            repaired,
            result.stage_trace.validation_regeneration_result,
        )
        self.assertEqual(1, len(result.stage_trace.validation_errors))
        self.assertIn("SOURCE_2", result.stage_trace.validation_errors[0])
        self.assertEqual(240, result.model_call.latency_ms)
        self.assertEqual(130, result.model_call.input_tokens)
        self.assertEqual(30, result.model_call.output_tokens)
        self.assertEqual(
            30,
            result.stage_trace.validation_regeneration_model_call.input_tokens,
        )
        self.generator.regenerate_answer_with_trace.assert_awaited_once()

    async def test_regenerates_when_answer_withholds_after_answerable_plan(
        self,
    ) -> None:
        results = [self._result(1)]
        generated = GenerationResult(
            status=GenerationStatus.WITHHELD,
            answer_markdown=None,
            withheld_reason=GenerationWithheldReason.INSUFFICIENT_EVIDENCE,
        )
        stage_trace = self._stage_trace(results, generated)
        stage_trace = replace(
            stage_trace,
            source_plan=stage_trace.source_plan.model_copy(
                update={"answer_type": GenerationAnswerType.GENERAL},
            ),
        )
        repaired = self._answerable("계획에 따른 답변 [SOURCE_1]")
        self.generator.generate_with_trace.return_value = _call(
            generated,
            stage_trace,
        )
        self.generator.regenerate_answer_with_trace.return_value = GenerationCall(
            trace=_trace(),
            result=repaired,
            stage_trace=replace(
                stage_trace,
                validation_error=(
                    "ANSWERABLE Source Plan과 Answer의 WITHHELD 상태가 일치하지 않습니다."
                ),
                validation_errors=(
                    "ANSWERABLE Source Plan과 Answer의 WITHHELD 상태가 일치하지 않습니다.",
                ),
                validation_regeneration_result=repaired,
                validation_regeneration_count=1,
            ),
        )

        result = await self.service.generate_answer("질문", results)

        self.assertEqual(FinalAnswerStatus.COMPLETED, result.status)
        self.assertEqual("계획에 따른 답변 [1]", result.answer_markdown)
        self.assertEqual(1, result.stage_trace.validation_regeneration_count)
        self.assertIn("상태가 일치하지 않습니다", result.stage_trace.validation_error)
        self.generator.regenerate_answer_with_trace.assert_awaited_once()

    async def test_withholds_when_answerable_plan_repair_still_withholds(
        self,
    ) -> None:
        results = [self._result(1)]
        generated = GenerationResult(
            status=GenerationStatus.WITHHELD,
            answer_markdown=None,
            withheld_reason=GenerationWithheldReason.INSUFFICIENT_EVIDENCE,
        )
        stage_trace = self._stage_trace(results, generated)
        stage_trace = replace(
            stage_trace,
            source_plan=stage_trace.source_plan.model_copy(
                update={"answer_type": GenerationAnswerType.GENERAL},
            ),
        )
        self.generator.generate_with_trace.return_value = _call(
            generated,
            stage_trace,
        )
        self.generator.regenerate_answer_with_trace.return_value = GenerationCall(
            trace=_trace(),
            result=generated,
            stage_trace=replace(
                stage_trace,
                validation_regeneration_result=generated,
                validation_regeneration_count=1,
            ),
        )

        result = await self.service.generate_answer("질문", results)

        self.assertEqual(FinalAnswerStatus.WITHHELD, result.status)
        self.assertEqual(
            FinalWithheldReason.UNVERIFIABLE_ANSWER,
            result.withheld_reason,
        )
        self.generator.regenerate_answer_with_trace.assert_awaited_once()

    async def test_does_not_force_procedure_answer_over_answer_withholding(
        self,
    ) -> None:
        results = [self._result(1)]
        generated = GenerationResult(
            status=GenerationStatus.WITHHELD,
            answer_markdown=None,
            withheld_reason=GenerationWithheldReason.INSUFFICIENT_EVIDENCE,
        )
        stage_trace = self._stage_trace(results, generated)
        self.generator.generate_with_trace.return_value = _call(
            generated,
            stage_trace,
        )

        result = await self.service.generate_answer("구독 해지 방법", results)

        self.assertEqual(FinalAnswerStatus.WITHHELD, result.status)
        self.assertEqual(
            FinalWithheldReason.INSUFFICIENT_EVIDENCE,
            result.withheld_reason,
        )
        self.generator.regenerate_answer_with_trace.assert_not_awaited()

    async def test_withholds_when_regenerated_answer_is_still_unverifiable(
        self,
    ) -> None:
        results = [self._result(1), self._result(2)]
        generated = self._answerable("미선택 근거 [SOURCE_2]")
        stage_trace = self._stage_trace(results[:1], generated)
        self.generator.generate_with_trace.return_value = _call(
            generated,
            stage_trace,
        )

        async def regenerate(_question, failed_trace, _reason):
            return GenerationCall(
                trace=_trace(),
                result=generated,
                stage_trace=replace(
                    failed_trace,
                    validation_regeneration_result=generated,
                    validation_regeneration_count=1,
                ),
            )

        self.generator.regenerate_answer_with_trace.side_effect = regenerate

        result = await self.service.generate_answer("질문", results)

        self.assertEqual(FinalAnswerStatus.WITHHELD, result.status)
        self.assertEqual(
            FinalWithheldReason.UNVERIFIABLE_ANSWER,
            result.withheld_reason,
        )
        self.assertEqual(2, len(result.stage_trace.validation_errors))
        self.generator.regenerate_answer_with_trace.assert_awaited_once()

    async def test_returns_error_when_validation_regeneration_api_fails(
        self,
    ) -> None:
        results = [self._result(1), self._result(2)]
        generated = self._answerable("미선택 근거 [SOURCE_2]")
        stage_trace = self._stage_trace(results[:1], generated)
        self.generator.generate_with_trace.return_value = _call(
            generated,
            stage_trace,
        )
        error = RuntimeError("repair API failure")
        self.generator.regenerate_answer_with_trace.return_value = GenerationCall(
            trace=_failed_trace(),
            error=error,
            stage_trace=replace(
                stage_trace,
                validation_regeneration_count=1,
            ),
        )

        result = await self.service.generate_answer("질문", results)

        self.assertEqual(FinalAnswerStatus.ERROR, result.status)
        self.assertEqual(INTERNAL_ERROR_CODE, result.error_code)
        self.assertFalse(result.model_call.succeeded)
        self.generator.regenerate_answer_with_trace.assert_awaited_once()

    async def test_withholds_unverifiable_answers_without_citations(self) -> None:
        cases = (
            "marker가 없는 답변",
            "존재하지 않는 근거 [SOURCE_9]",
            "SOURCE_1에 따르면 접근할 수 있습니다. [SOURCE_1]",
            "내부 식별자는 `SOURCE_1`입니다. [SOURCE_1]",
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

    async def test_completes_with_four_and_five_valid_citations(self) -> None:
        for count in (4, 5):
            with self.subTest(count=count):
                results = [self._result(index) for index in range(1, count + 1)]
                generated = self._answerable(
                    " ".join(f"근거 [SOURCE_{index}]" for index in range(1, count + 1))
                )
                self.generator.generate_with_trace.return_value = _call(
                    generated, self._stage_trace(results, generated)
                )
                result = await self.service.generate_answer("질문", results)
                self.assertEqual(FinalAnswerStatus.COMPLETED, result.status)
                self.assertEqual(count, len(result.citations))
                self.assertEqual(
                    list(range(1, count + 1)),
                    [citation.citation_number for citation in result.citations],
                )
                self.generator.regenerate_answer_with_trace.assert_not_awaited()

    async def test_withholds_answers_containing_links_or_html(self) -> None:
        cases = (
            "상세 내용은 [가이드](/guide)를 확인하세요. [SOURCE_1]",
            "상세 주소는 https://docs.riido.io/guide 입니다. [SOURCE_1]",
            "문의 주소는 <support@example.com>입니다. [SOURCE_1]",
            "<strong>중요한 내용</strong>입니다. [SOURCE_1]",
        )

        for answer_markdown in cases:
            with self.subTest(answer_markdown=answer_markdown):
                self.generator.generate_with_trace.return_value = _call(
                    self._answerable(answer_markdown)
                )

                result = await self.service.generate_answer(
                    "질문",
                    [self._result(1)],
                )

                self.assertEqual(FinalAnswerStatus.WITHHELD, result.status)
                self.assertEqual(
                    FinalWithheldReason.UNVERIFIABLE_ANSWER,
                    result.withheld_reason,
                )
                self.assertEqual((), result.citations)

    async def test_allows_links_and_html_inside_code_regions(self) -> None:
        cases = (
            (
                "설정 예시입니다. [SOURCE_1]\n\n"
                "```\nbase_url = https://docs.riido.io/guide\n```",
                "설정 예시입니다. [1]\n\n"
                "```\nbase_url = https://docs.riido.io/guide\n```",
            ),
            (
                "엔드포인트는 `https://api.riido.io/v1/chat` 입니다. [SOURCE_1]",
                "엔드포인트는 `https://api.riido.io/v1/chat` 입니다. [1]",
            ),
            (
                "예시는 다음과 같습니다. [SOURCE_1]\n\n"
                "```html\n<strong>강조</strong> [가이드](/guide)\n```",
                "예시는 다음과 같습니다. [1]\n\n"
                "```html\n<strong>강조</strong> [가이드](/guide)\n```",
            ),
            (
                "설정 예시입니다. [SOURCE_1]\n\n"
                "~~~\nwww.riido.io/guide\n~~~",
                "설정 예시입니다. [1]\n\n~~~\nwww.riido.io/guide\n~~~",
            ),
        )

        for answer_markdown, expected_markdown in cases:
            with self.subTest(answer_markdown=answer_markdown):
                self.generator.generate_with_trace.return_value = _call(
                    self._answerable(answer_markdown)
                )

                result = await self.service.generate_answer(
                    "질문",
                    [self._result(1)],
                )

                self.assertEqual(FinalAnswerStatus.COMPLETED, result.status)
                self.assertEqual(expected_markdown, result.answer_markdown)
                self.assertIsNone(result.withheld_reason)

    async def test_withholds_link_syntax_outside_code_regions(self) -> None:
        cases = (
            "자세한 내용은 [가이드][guide]를 확인하세요. [SOURCE_1]",
            "[guide]: https://docs.riido.io/guide\n안내 내용입니다. [SOURCE_1]",
            "주소는 <https://docs.riido.io/guide> 입니다. [SOURCE_1]",
            "코드 밖 주소는 https://docs.riido.io 입니다. `옵션` [SOURCE_1]",
            "```\ncode\n```\n주소는 https://docs.riido.io 입니다. [SOURCE_1]",
        )

        for answer_markdown in cases:
            with self.subTest(answer_markdown=answer_markdown):
                self.generator.generate_with_trace.return_value = _call(
                    self._answerable(answer_markdown)
                )

                result = await self.service.generate_answer(
                    "질문",
                    [self._result(1)],
                )

                self.assertEqual(FinalAnswerStatus.WITHHELD, result.status)
                self.assertEqual(
                    FinalWithheldReason.UNVERIFIABLE_ANSWER,
                    result.withheld_reason,
                )
                self.assertEqual((), result.citations)

    async def test_handles_malformed_code_markup_without_error(self) -> None:
        # 닫히지 않은 펜스와 짝이 맞지 않는 백틱에서도 예외 없이 상태를 결정한다.
        cases = (
            (
                "설정 예시입니다. [SOURCE_1]\n\n```\nurl = https://docs.riido.io",
                FinalAnswerStatus.COMPLETED,
            ),
            (
                "값은 `option 입니다. [SOURCE_1]",
                FinalAnswerStatus.COMPLETED,
            ),
            (
                "값은 `option 이고 주소는 https://docs.riido.io 입니다. [SOURCE_1]",
                FinalAnswerStatus.WITHHELD,
            ),
        )

        for answer_markdown, expected_status in cases:
            with self.subTest(answer_markdown=answer_markdown):
                self.generator.generate_with_trace.return_value = _call(
                    self._answerable(answer_markdown)
                )

                result = await self.service.generate_answer(
                    "질문",
                    [self._result(1)],
                )

                self.assertEqual(expected_status, result.status)

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

        self.generator.regenerate_answer_with_trace.assert_not_awaited()

    def test_keeps_exact_backend_withheld_responses(self) -> None:
        self.assertEqual(
            {
                FinalWithheldReason.INSUFFICIENT_EVIDENCE: (
                    "이용가이드에서 질문에 답할 충분한 근거를 찾지 못했습니다."
                ),
                FinalWithheldReason.AMBIGUOUS_QUESTION: (
                    "질문의 범위가 너무 넓거나 의미가 명확하지 않아 "
                    "답변하기 어렵습니다. "
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
        error = APITimeoutError(
            httpx.Request("POST", "https://api.openai.com")
        )
        self.generator.generate_with_trace.return_value = GenerationCall(
            trace=_failed_trace(),
            error=error,
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
        generated = self._answerable("답변 [SOURCE_1]")
        stage_trace = GenerationStageTrace(pre_validation_result=generated)
        self.generator.generate_with_trace.return_value = _call(
            generated,
            stage_trace,
        )

        with patch(
            "app.answering.service.validate_citations",
            side_effect=RuntimeError("validation failure"),
        ):
            result = await self.service.generate_answer("질문", [self._result(1)])

        self.assertEqual(FinalAnswerStatus.ERROR, result.status)
        self.assertEqual(CITATION_VALIDATION_ERROR_CODE, result.error_code)
        self.assertIsNone(result.answer_markdown)
        self.assertEqual((), result.citations)
        self.assertEqual(
            "validation failure",
            result.stage_trace.validation_error,
        )
        self.assertEqual(
            generated,
            result.stage_trace.pre_validation_result,
        )

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

    @staticmethod
    def _stage_trace(
        results: list[HybridRetrievalResult],
        generated: GenerationResult,
    ) -> GenerationStageTrace:
        sources = tuple(build_generation_context(results))
        return GenerationStageTrace(
            source_plan=GenerationSourcePlan(
                status=GenerationStatus.ANSWERABLE,
                answer_type=GenerationAnswerType.PROCEDURE,
                answer_scope=(
                    GenerationAnswerScope.SUMMARY
                    if len(sources) == 1
                    else GenerationAnswerScope.MULTI_DETAIL
                ),
                evidence_requirements=[
                    GenerationEvidenceRequirement(
                        information_unit="답변 정보",
                        source_ids=[source.source_id for source in sources],
                    )
                ],
                withheld_reason=None,
            ),
            selected_sources=sources,
            pre_validation_result=generated,
            planning_attempt_count=1,
            answer_attempt_count=1,
        )


if __name__ == "__main__":
    unittest.main()
