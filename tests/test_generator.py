import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, Mock, call, patch

import httpx
from openai import APITimeoutError
from pydantic import ValidationError

from generation.generator import (
    ANSWER_PROMPT_V6,
    GENERATION_PROMPT_VERSION,
    MAX_CONTEXT_SOURCES,
    MAX_PLANNED_CITATIONS,
    OPENAI_GENERATION_MODEL,
    SOURCE_PLANNING_PROMPT_V6,
    OpenAIGenerator,
    build_answer_input,
    build_generation_context,
    build_generation_input,
    count_distinct_citations,
    select_required_sources,
)
from generation.models import (
    GenerationAnswerScope,
    GenerationEvidenceRequirement,
    GenerationResult,
    GenerationSourcePlan,
    GenerationStatus,
    GenerationWithheldReason,
)
from retrieval.models import HybridRetrievalResult, RetrievalChunk


class GenerationResultTest(unittest.TestCase):
    def test_answerable_requires_answer_and_null_reason(self) -> None:
        result = GenerationResult(
            status=GenerationStatus.ANSWERABLE,
            answer_markdown="핵심 답변입니다. [SOURCE_1]",
            withheld_reason=None,
        )

        self.assertEqual(GenerationStatus.ANSWERABLE, result.status)

        invalid_cases = (
            {
                "status": GenerationStatus.ANSWERABLE,
                "answer_markdown": None,
                "withheld_reason": None,
            },
            {
                "status": GenerationStatus.ANSWERABLE,
                "answer_markdown": "  ",
                "withheld_reason": None,
            },
            {
                "status": GenerationStatus.ANSWERABLE,
                "answer_markdown": "답변",
                "withheld_reason": GenerationWithheldReason.OUT_OF_SCOPE,
            },
        )
        for values in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaises(ValidationError):
                    GenerationResult(**values)

    def test_withheld_requires_null_answer_and_reason(self) -> None:
        for reason in GenerationWithheldReason:
            with self.subTest(reason=reason):
                result = GenerationResult(
                    status=GenerationStatus.WITHHELD,
                    answer_markdown=None,
                    withheld_reason=reason,
                )
                self.assertEqual(reason, result.withheld_reason)

        invalid_cases = (
            {
                "status": GenerationStatus.WITHHELD,
                "answer_markdown": "답변",
                "withheld_reason": GenerationWithheldReason.OUT_OF_SCOPE,
            },
            {
                "status": GenerationStatus.WITHHELD,
                "answer_markdown": None,
                "withheld_reason": None,
            },
        )
        for values in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaises(ValidationError):
                    GenerationResult(**values)

    def test_structured_output_requires_all_fields_and_forbids_extra_fields(self) -> None:
        schema = GenerationResult.model_json_schema()

        self.assertEqual(
            {"status", "answer_markdown", "withheld_reason"},
            set(schema["required"]),
        )
        self.assertFalse(schema["additionalProperties"])

        with self.assertRaises(ValidationError):
            GenerationResult(
                status=GenerationStatus.WITHHELD,
                answer_markdown=None,
            )
        with self.assertRaises(ValidationError):
            GenerationResult(
                status=GenerationStatus.WITHHELD,
                answer_markdown=None,
                withheld_reason=GenerationWithheldReason.OUT_OF_SCOPE,
                unexpected="field",
            )


class GenerationSourcePlanTest(unittest.TestCase):
    def test_answerable_requires_information_unit_evidence(self) -> None:
        plan = self._answerable_plan("SOURCE_1", "SOURCE_2")

        self.assertEqual(GenerationStatus.ANSWERABLE, plan.status)
        self.assertEqual(
            ["SOURCE_1", "SOURCE_2"],
            plan.evidence_requirements[0].source_ids,
        )

        invalid_cases = (
            {
                "status": GenerationStatus.ANSWERABLE,
                "answer_scope": GenerationAnswerScope.SUMMARY,
                "evidence_requirements": [],
                "withheld_reason": None,
            },
            {
                "status": GenerationStatus.ANSWERABLE,
                "answer_scope": GenerationAnswerScope.SUMMARY,
                "evidence_requirements": [
                    GenerationEvidenceRequirement(
                        information_unit="설정 방법",
                        source_ids=["SOURCE_1"],
                    )
                ],
                "withheld_reason": GenerationWithheldReason.OUT_OF_SCOPE,
            },
        )
        for values in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaises(ValidationError):
                    GenerationSourcePlan(**values)

    def test_withheld_requires_empty_evidence_and_reason(self) -> None:
        plan = GenerationSourcePlan(
            status=GenerationStatus.WITHHELD,
            answer_scope=GenerationAnswerScope.MULTI_DETAIL,
            evidence_requirements=[],
            withheld_reason=GenerationWithheldReason.INSUFFICIENT_EVIDENCE,
        )

        self.assertEqual(
            GenerationWithheldReason.INSUFFICIENT_EVIDENCE,
            plan.withheld_reason,
        )

        with self.assertRaises(ValidationError):
            GenerationSourcePlan(
                status=GenerationStatus.WITHHELD,
                answer_scope=GenerationAnswerScope.SUMMARY,
                evidence_requirements=[],
                withheld_reason=None,
            )

    def test_evidence_rejects_duplicate_source_ids(self) -> None:
        with self.assertRaises(ValidationError):
            GenerationEvidenceRequirement(
                information_unit="설정 방법",
                source_ids=["SOURCE_1", "SOURCE_1"],
            )

    def test_structured_output_requires_all_fields_and_forbids_extra(self) -> None:
        schema = GenerationSourcePlan.model_json_schema()

        self.assertEqual(
            {
                "status",
                "answer_scope",
                "evidence_requirements",
                "withheld_reason",
            },
            set(schema["required"]),
        )
        self.assertFalse(schema["additionalProperties"])

    @staticmethod
    def _answerable_plan(*source_ids: str) -> GenerationSourcePlan:
        return GenerationSourcePlan(
            status=GenerationStatus.ANSWERABLE,
            answer_scope=GenerationAnswerScope.SUMMARY,
            evidence_requirements=[
                GenerationEvidenceRequirement(
                    information_unit="설정 방법",
                    source_ids=list(source_ids),
                )
            ],
            withheld_reason=None,
        )


class GenerationContextTest(unittest.TestCase):
    def test_assigns_source_ids_in_hybrid_order(self) -> None:
        results = [self._result(index) for index in range(1, 6)]

        sources = build_generation_context(results)

        self.assertEqual(
            [f"SOURCE_{index}" for index in range(1, 6)],
            [source.source_id for source in sources],
        )
        self.assertEqual(
            [result.chunk for result in results],
            [source.chunk for source in sources],
        )

    def test_rejects_more_than_five_sources(self) -> None:
        results = [self._result(index) for index in range(MAX_CONTEXT_SOURCES + 1)]

        with self.assertRaisesRegex(ValueError, "최대 5개"):
            build_generation_context(results)

    def test_generation_input_exposes_only_allowed_context_fields(self) -> None:
        source = build_generation_context([self._result(1)])[0]

        generation_input = build_generation_input("사용자 질문", [source])

        self.assertIn("SOURCE_1", generation_input)
        self.assertIn(source.chunk.document_title, generation_input)
        self.assertIn(" > ".join(source.chunk.section_path), generation_input)
        self.assertIn(source.chunk.content, generation_input)
        self.assertIn("사용자 질문", generation_input)
        self.assertNotIn("Chunk ID:", generation_input)
        self.assertNotIn(source.chunk.source_url, generation_input)
        self.assertNotIn("rrf", generation_input.lower())

    def test_generation_input_normalizes_escaped_opening_bracket(self) -> None:
        result = self._result(1)
        result = replace(
            result,
            chunk=replace(
                result.chunk,
                content=r"1. \[설정 > 워크스페이스 > 알림]에서 관리",
            ),
        )
        source = build_generation_context([result])[0]

        generation_input = build_generation_input("질문", [source])

        self.assertIn("[설정 > 워크스페이스 > 알림]", generation_input)
        self.assertNotIn(r"\[설정", generation_input)

    def test_source_planning_prompt_preserves_scope_and_all_evidence(self) -> None:
        self.assertIn("질문이 직접 요구한 정보 단위만", SOURCE_PLANNING_PROMPT_V6)
        self.assertIn("필요한 SOURCE 수에 상한은 없습니다", SOURCE_PLANNING_PROMPT_V6)
        self.assertIn("절대로 일부를 빼지 마세요", SOURCE_PLANNING_PROMPT_V6)
        self.assertIn("정보 단위 하나라도", SOURCE_PLANNING_PROMPT_V6)
        self.assertIn("INSUFFICIENT_EVIDENCE", SOURCE_PLANNING_PROMPT_V6)
        self.assertIn("자동화", SOURCE_PLANNING_PROMPT_V6)

    def test_answer_prompt_avoids_partial_or_duplicate_answers(self) -> None:
        self.assertIn("모든 정보 단위에 답하세요", ANSWER_PROMPT_V6)
        self.assertIn("정보 단위를 생략하거나", ANSWER_PROMPT_V6)
        self.assertIn("반드시 가장 직접적인 SOURCE 하나만 선택", ANSWER_PROMPT_V6)
        self.assertIn("중복 SOURCE는 사용하거나 인용하지 마세요", ANSWER_PROMPT_V6)
        self.assertIn("SOURCE별로 답변 문단을 만들거나", ANSWER_PROMPT_V6)
        self.assertIn("필요한 최소한의 SOURCE만 인용", ANSWER_PROMPT_V6)

    def test_prompt_forbids_multiline_content_inside_table_cells(self) -> None:
        self.assertIn("표를 사용하지 말고", ANSWER_PROMPT_V6)
        self.assertIn("항목별 소제목이나 목록으로 설명하세요", ANSWER_PROMPT_V6)
        self.assertIn("모든 셀은 반드시 한 줄로 끝내고", ANSWER_PROMPT_V6)
        self.assertIn("코드 블록이나", ANSWER_PROMPT_V6)
        self.assertIn("줄바꿈을 절대 넣지 마세요", ANSWER_PROMPT_V6)

    def test_prompt_defines_terms_before_riido_role_and_value(self) -> None:
        self.assertIn("용어의 의미를 직접 물으면", ANSWER_PROMPT_V6)
        self.assertIn("첫 문장에 그 용어 자체의 쉬운 의미", ANSWER_PROMPT_V6)
        self.assertIn("그 용어가 어떤 종류인지", ANSWER_PROMPT_V6)
        self.assertIn("핵심 사용 주체나 목적", ANSWER_PROMPT_V6)
        self.assertIn("첫 문장만으로 이해할 수 있게 끝내고", ANSWER_PROMPT_V6)
        self.assertIn("같은 문장에", ANSWER_PROMPT_V6)
        self.assertIn("이어 붙이지 마세요", ANSWER_PROMPT_V6)
        self.assertIn("둘째 문장으로 미루지 마세요", ANSWER_PROMPT_V6)
        self.assertIn("용어 의미를 먼저 설명한 뒤", ANSWER_PROMPT_V6)
        self.assertIn("뤼이도에서의 역할과 사용 가치", ANSWER_PROMPT_V6)

    def test_prompt_does_not_invent_unsupported_term_definitions(self) -> None:
        self.assertIn("일반 지식으로 정의를 보완하지 말고", ANSWER_PROMPT_V6)
        self.assertIn("WITHHELD 여부를 판단하세요", ANSWER_PROMPT_V6)

    def test_prompt_forbids_links_urls_and_html(self) -> None:
        self.assertEqual("v6", GENERATION_PROMPT_VERSION)
        self.assertIn(
            "Markdown 링크 문법과 HTML을 사용하지 마세요",
            ANSWER_PROMPT_V6,
        )
        self.assertIn(
            "코드 블록이나 백틱 인라인 코드 안에 넣고",
            ANSWER_PROMPT_V6,
        )
        self.assertIn("그 밖의 URL은 본문에 쓰지 마세요", ANSWER_PROMPT_V6)
        self.assertIn("별도 citations 영역", ANSWER_PROMPT_V6)

    def test_selects_required_sources_in_first_evidence_order(self) -> None:
        sources = build_generation_context(
            [self._result(index) for index in range(1, 4)]
        )
        plan = GenerationSourcePlan(
            status=GenerationStatus.ANSWERABLE,
            answer_scope=GenerationAnswerScope.MULTI_DETAIL,
            evidence_requirements=[
                GenerationEvidenceRequirement(
                    information_unit="두 번째",
                    source_ids=["SOURCE_2", "SOURCE_1"],
                ),
                GenerationEvidenceRequirement(
                    information_unit="세 번째",
                    source_ids=["SOURCE_1", "SOURCE_3"],
                ),
            ],
            withheld_reason=None,
        )

        selected = select_required_sources(plan, sources)

        self.assertEqual(
            ["SOURCE_2", "SOURCE_1", "SOURCE_3"],
            [source.source_id for source in selected],
        )

    def test_rejects_source_plan_id_missing_from_context(self) -> None:
        plan = GenerationSourcePlanTest._answerable_plan("SOURCE_9")

        with self.assertRaisesRegex(RuntimeError, "SOURCE_9"):
            select_required_sources(plan, [])

    def test_counts_citations_after_same_section_merge(self) -> None:
        results = [self._result(index) for index in range(1, 5)]
        results[1] = replace(
            results[1],
            chunk=replace(
                results[1].chunk,
                source_url=results[0].chunk.source_url,
                section_path=results[0].chunk.section_path,
            ),
        )
        sources = build_generation_context(results)

        self.assertEqual(MAX_PLANNED_CITATIONS, count_distinct_citations(sources))

    def test_answer_input_contains_required_coverage(self) -> None:
        sources = build_generation_context([self._result(1)])
        plan = GenerationSourcePlanTest._answerable_plan("SOURCE_1")

        answer_input = build_answer_input("질문", sources, plan)

        self.assertIn("## Required Answer Coverage", answer_input)
        self.assertIn("설정 방법: SOURCE_1", answer_input)

    @staticmethod
    def _result(index: int) -> HybridRetrievalResult:
        chunk = RetrievalChunk(
            document_id=f"document-{index}",
            section_id=f"section-{index}",
            document_title=f"문서 {index}",
            section_path=(f"문서 {index}", f"섹션 {index}"),
            source_url=f"https://docs.riido.io/{index}",
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


class OpenAIGeneratorTest(unittest.IsolatedAsyncioTestCase):
    async def test_selects_sources_before_requesting_answer_with_prompt_v6(
        self,
    ) -> None:
        plan = self._answerable_plan("SOURCE_1")
        expected = self._answerable_result()
        client = self._client_with_responses(plan, expected)
        generator = OpenAIGenerator(client=client)
        sources = build_generation_context([GenerationContextTest._result(1)])

        result = await generator.generate("질문", sources)

        self.assertEqual(expected, result)
        self.assertEqual(
            [
                call(
                    model=OPENAI_GENERATION_MODEL,
                    instructions=SOURCE_PLANNING_PROMPT_V6,
                    input=build_generation_input("질문", sources),
                    text_format=GenerationSourcePlan,
                ),
                call(
                    model=OPENAI_GENERATION_MODEL,
                    instructions=ANSWER_PROMPT_V6,
                    input=build_answer_input("질문", sources, plan),
                    text_format=GenerationResult,
                ),
            ],
            client.responses.parse.await_args_list,
        )

    async def test_answer_receives_only_sources_required_by_plan(self) -> None:
        sources = build_generation_context(
            [GenerationContextTest._result(index) for index in range(1, 4)]
        )
        plan = self._answerable_plan("SOURCE_2")
        client = self._client_with_responses(plan, self._answerable_result())
        generator = OpenAIGenerator(client=client)

        await generator.generate("질문", sources)

        answer_input = client.responses.parse.await_args_list[1].kwargs["input"]
        self.assertIn("### SOURCE_2", answer_input)
        self.assertNotIn("### SOURCE_1", answer_input)
        self.assertNotIn("### SOURCE_3", answer_input)

    async def test_source_plan_withheld_skips_answer_generation(self) -> None:
        plan = GenerationSourcePlan(
            status=GenerationStatus.WITHHELD,
            answer_scope=GenerationAnswerScope.MULTI_DETAIL,
            evidence_requirements=[],
            withheld_reason=GenerationWithheldReason.INSUFFICIENT_EVIDENCE,
        )
        client = self._client_with_responses(plan)
        generator = OpenAIGenerator(client=client)

        result = await generator.generate("질문", [])

        self.assertEqual(GenerationStatus.WITHHELD, result.status)
        self.assertEqual(
            GenerationWithheldReason.INSUFFICIENT_EVIDENCE,
            result.withheld_reason,
        )
        client.responses.parse.assert_awaited_once()

    async def test_withholds_before_answer_when_plan_needs_over_three_citations(
        self,
    ) -> None:
        sources = build_generation_context(
            [GenerationContextTest._result(index) for index in range(1, 5)]
        )
        plan = self._answerable_plan(
            "SOURCE_1",
            "SOURCE_2",
            "SOURCE_3",
            "SOURCE_4",
        )
        client = self._client_with_responses(plan)
        generator = OpenAIGenerator(client=client)

        result = await generator.generate("복합 질문", sources)

        self.assertEqual(GenerationStatus.WITHHELD, result.status)
        self.assertEqual(
            GenerationWithheldReason.AMBIGUOUS_QUESTION,
            result.withheld_reason,
        )
        client.responses.parse.assert_awaited_once()

    async def test_invalid_planned_source_returns_error_without_answer_call(
        self,
    ) -> None:
        client = self._client_with_responses(self._answerable_plan("SOURCE_9"))
        generator = OpenAIGenerator(client=client)

        call_result = await generator.generate_with_trace("질문", [])

        self.assertIsInstance(call_result.error, RuntimeError)
        self.assertIn("SOURCE_9", str(call_result.error))
        self.assertFalse(call_result.trace.succeeded)
        client.responses.parse.assert_awaited_once()

    async def test_retries_one_transient_source_planning_error(self) -> None:
        expected = self._answerable_result()
        client = Mock()
        client.responses.parse = AsyncMock(
            side_effect=[
                APITimeoutError(httpx.Request("POST", "https://api.openai.com")),
                self._response(self._answerable_plan("SOURCE_1")),
                self._response(expected),
            ]
        )
        generator = OpenAIGenerator(client=client)
        sources = build_generation_context([GenerationContextTest._result(1)])

        result = await generator.generate("질문", sources)

        self.assertEqual(expected, result)
        self.assertEqual(3, client.responses.parse.await_count)

    async def test_stops_after_one_retry_when_source_planning_error_continues(
        self,
    ) -> None:
        error = APITimeoutError(
            httpx.Request("POST", "https://api.openai.com")
        )
        client = Mock()
        client.responses.parse = AsyncMock(side_effect=[error, error])
        generator = OpenAIGenerator(client=client)

        with self.assertRaises(APITimeoutError):
            await generator.generate("질문", [])

        self.assertEqual(2, client.responses.parse.await_count)

    async def test_does_not_retry_non_transient_source_planning_error(self) -> None:
        client = Mock()
        client.responses.parse = AsyncMock(side_effect=ValueError("invalid"))
        generator = OpenAIGenerator(client=client)

        with self.assertRaisesRegex(ValueError, "invalid"):
            await generator.generate("질문", [])

        client.responses.parse.assert_awaited_once()

    async def test_rejects_missing_source_plan_output(self) -> None:
        client = self._client_with_responses(None)
        generator = OpenAIGenerator(client=client)

        with self.assertRaisesRegex(RuntimeError, "Structured Output"):
            await generator.generate("질문", [])

    def test_requires_api_key_and_configures_client_retry_and_timeout(self) -> None:
        with patch(
            "generation.generator.get_settings",
            return_value=SimpleNamespace(openai_api_key=None),
        ):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                OpenAIGenerator()

        with patch(
            "generation.generator.get_settings",
            return_value=SimpleNamespace(openai_api_key="test-key"),
        ), patch("generation.generator.AsyncOpenAI") as client_class:
            OpenAIGenerator()

        client_class.assert_called_once_with(
            api_key="test-key",
            max_retries=0,
            timeout=30.0,
        )

    async def test_trace_reports_tokens_without_retry_on_first_success(self) -> None:
        client = Mock()
        client.responses.parse = AsyncMock(
            side_effect=[
                self._response(
                    self._answerable_plan("SOURCE_1"),
                    input_tokens=400,
                    output_tokens=50,
                ),
                self._response(
                    self._answerable_result(),
                    input_tokens=1200,
                    output_tokens=300,
                ),
            ]
        )
        generator = OpenAIGenerator(client=client)
        sources = build_generation_context([GenerationContextTest._result(1)])

        generation_call = await generator.generate_with_trace("질문", sources)

        self.assertIsNone(generation_call.error)
        self.assertTrue(generation_call.trace.succeeded)
        self.assertEqual(0, generation_call.trace.retry_count)
        self.assertEqual(1600, generation_call.trace.input_tokens)
        self.assertEqual(350, generation_call.trace.output_tokens)
        self.assertEqual(
            GENERATION_PROMPT_VERSION,
            generation_call.trace.prompt_version,
        )
        self.assertEqual(
            OPENAI_GENERATION_MODEL,
            generation_call.trace.model_name,
        )

    async def test_trace_counts_answer_retry_as_a_single_logical_call(self) -> None:
        client = Mock()
        client.responses.parse = AsyncMock(
            side_effect=[
                self._response(self._answerable_plan("SOURCE_1")),
                APITimeoutError(httpx.Request("POST", "https://api.openai.com")),
                self._response(self._answerable_result()),
            ]
        )
        generator = OpenAIGenerator(client=client)
        sources = build_generation_context([GenerationContextTest._result(1)])

        generation_call = await generator.generate_with_trace("질문", sources)

        self.assertIsNone(generation_call.error)
        self.assertEqual(1, generation_call.trace.retry_count)

    async def test_trace_keeps_last_source_planning_error_when_attempts_fail(
        self,
    ) -> None:
        error = APITimeoutError(httpx.Request("POST", "https://api.openai.com"))
        client = Mock()
        client.responses.parse = AsyncMock(side_effect=[error, error])
        generator = OpenAIGenerator(client=client)

        generation_call = await generator.generate_with_trace("질문", [])

        self.assertIs(error, generation_call.error)
        self.assertIsNone(generation_call.result)
        self.assertFalse(generation_call.trace.succeeded)
        self.assertEqual(1, generation_call.trace.retry_count)
        self.assertIsNotNone(generation_call.trace.error_message)

    @staticmethod
    def _client_with_responses(*results: object) -> Mock:
        client = Mock()
        client.responses.parse = AsyncMock(
            side_effect=[OpenAIGeneratorTest._response(result) for result in results]
        )
        return client

    @staticmethod
    def _response(
        result: object,
        *,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> SimpleNamespace:
        usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return SimpleNamespace(output_parsed=result, usage=usage)

    @staticmethod
    def _answerable_plan(*source_ids: str) -> GenerationSourcePlan:
        return GenerationSourcePlanTest._answerable_plan(*source_ids)

    @staticmethod
    def _answerable_result() -> GenerationResult:
        return GenerationResult(
            status=GenerationStatus.ANSWERABLE,
            answer_markdown="답변입니다. [SOURCE_1]",
            withheld_reason=None,
        )


if __name__ == "__main__":
    unittest.main()
