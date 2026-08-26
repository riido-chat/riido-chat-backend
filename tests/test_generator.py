import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from openai import APITimeoutError
from pydantic import ValidationError

from generation.generator import (
    GENERATION_PROMPT_VERSION,
    MAX_CONTEXT_SOURCES,
    OPENAI_GENERATION_MODEL,
    PROMPT_V2,
    OpenAIGenerator,
    build_generation_context,
    build_generation_input,
)
from generation.models import (
    GenerationResult,
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

    def test_prompt_avoids_duplicate_answers_and_uses_minimum_sources(self) -> None:
        self.assertIn("반드시 가장 직접적인 SOURCE 하나만 선택", PROMPT_V2)
        self.assertIn("중복 SOURCE는 사용하거나 인용하지 마세요", PROMPT_V2)
        self.assertIn("SOURCE별로 답변 문단을 만들거나", PROMPT_V2)
        self.assertIn("필요한 최소한의 SOURCE만 인용", PROMPT_V2)

    def test_prompt_forbids_links_urls_and_html(self) -> None:
        self.assertEqual("v2", GENERATION_PROMPT_VERSION)
        self.assertIn("Markdown 링크, URL, HTML을 포함하지 마세요", PROMPT_V2)
        self.assertIn("별도 citations 영역", PROMPT_V2)

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
    async def test_requests_structured_output_with_prompt_v2(self) -> None:
        expected = self._answerable_result()
        client = self._client_with_response(expected)
        generator = OpenAIGenerator(client=client)
        sources = build_generation_context([GenerationContextTest._result(1)])

        result = await generator.generate("질문", sources)

        self.assertEqual(expected, result)
        client.responses.parse.assert_awaited_once_with(
            model=OPENAI_GENERATION_MODEL,
            instructions=PROMPT_V2,
            input=build_generation_input("질문", sources),
            text_format=GenerationResult,
        )

    async def test_retries_one_transient_error(self) -> None:
        expected = self._answerable_result()
        client = Mock()
        client.responses.parse = AsyncMock(
            side_effect=[
                APITimeoutError(httpx.Request("POST", "https://api.openai.com")),
                SimpleNamespace(output_parsed=expected),
            ]
        )
        generator = OpenAIGenerator(client=client)

        result = await generator.generate("질문", [])

        self.assertEqual(expected, result)
        self.assertEqual(2, client.responses.parse.await_count)

    async def test_stops_after_one_retry_when_transient_error_continues(self) -> None:
        error = APITimeoutError(
            httpx.Request("POST", "https://api.openai.com")
        )
        client = Mock()
        client.responses.parse = AsyncMock(side_effect=[error, error])
        generator = OpenAIGenerator(client=client)

        with self.assertRaises(APITimeoutError):
            await generator.generate("질문", [])

        self.assertEqual(2, client.responses.parse.await_count)

    async def test_does_not_retry_non_transient_error(self) -> None:
        client = Mock()
        client.responses.parse = AsyncMock(side_effect=ValueError("invalid"))
        generator = OpenAIGenerator(client=client)

        with self.assertRaisesRegex(ValueError, "invalid"):
            await generator.generate("질문", [])

        client.responses.parse.assert_awaited_once()

    async def test_rejects_missing_parsed_output(self) -> None:
        client = self._client_with_response(None)
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
            return_value=SimpleNamespace(
                output_parsed=self._answerable_result(),
                usage=SimpleNamespace(input_tokens=1200, output_tokens=300),
            )
        )
        generator = OpenAIGenerator(client=client)

        call = await generator.generate_with_trace("질문", [])

        self.assertIsNone(call.error)
        self.assertTrue(call.trace.succeeded)
        self.assertEqual(0, call.trace.retry_count)
        self.assertEqual(1200, call.trace.input_tokens)
        self.assertEqual(300, call.trace.output_tokens)
        self.assertEqual(GENERATION_PROMPT_VERSION, call.trace.prompt_version)
        self.assertEqual(OPENAI_GENERATION_MODEL, call.trace.model_name)

    async def test_trace_counts_one_retry_as_a_single_logical_call(self) -> None:
        client = Mock()
        client.responses.parse = AsyncMock(
            side_effect=[
                APITimeoutError(httpx.Request("POST", "https://api.openai.com")),
                SimpleNamespace(output_parsed=self._answerable_result()),
            ]
        )
        generator = OpenAIGenerator(client=client)

        call = await generator.generate_with_trace("질문", [])

        self.assertIsNone(call.error)
        self.assertEqual(1, call.trace.retry_count)

    async def test_trace_keeps_last_error_when_every_attempt_fails(self) -> None:
        error = APITimeoutError(httpx.Request("POST", "https://api.openai.com"))
        client = Mock()
        client.responses.parse = AsyncMock(side_effect=[error, error])
        generator = OpenAIGenerator(client=client)

        call = await generator.generate_with_trace("질문", [])

        self.assertIs(error, call.error)
        self.assertIsNone(call.result)
        self.assertFalse(call.trace.succeeded)
        self.assertEqual(1, call.trace.retry_count)
        self.assertIsNotNone(call.trace.error_message)

    @staticmethod
    def _client_with_response(result: object) -> Mock:
        client = Mock()
        client.responses.parse = AsyncMock(
            return_value=SimpleNamespace(output_parsed=result)
        )
        return client

    @staticmethod
    def _answerable_result() -> GenerationResult:
        return GenerationResult(
            status=GenerationStatus.ANSWERABLE,
            answer_markdown="답변입니다. [SOURCE_1]",
            withheld_reason=None,
        )


if __name__ == "__main__":
    unittest.main()
