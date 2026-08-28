import unittest
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import Mock, patch

import httpx
from openai import InternalServerError, OpenAI

from retrieval.embedding import (
    OPENAI_EMBEDDING_DIMENSIONS,
    OPENAI_EMBEDDING_MODEL,
    OpenAIEmbedder,
    build_embedding_text,
)
from retrieval.models import RetrievalChunk


class EmbeddingTextTest(unittest.TestCase):
    def test_builds_text_from_semantic_chunk_fields(self) -> None:
        chunk = self._retrieval_chunk(
            section_path=("문서 제목", "상위 섹션", "하위 섹션"),
        )

        text = build_embedding_text(chunk)

        self.assertEqual(
            "문서 제목\n상위 섹션\n하위 섹션\n전체 본문",
            text,
        )
        self.assertEqual(1, text.count(chunk.document_title))

    def test_excludes_empty_values_and_metadata(self) -> None:
        chunk = self._retrieval_chunk(
            section_path=("문서 제목", "", "설정"),
        )

        text = build_embedding_text(chunk)

        self.assertEqual("문서 제목\n설정\n전체 본문", text)
        self.assertIsNone(chunk.chunk_id)
        self.assertNotIn(chunk.document_id, text)
        self.assertNotIn(chunk.section_id, text)
        self.assertNotIn(chunk.source_url, text)
        self.assertNotIn(chunk.category, text)

    @staticmethod
    def _retrieval_chunk(section_path: Tuple[str, ...]) -> RetrievalChunk:
        return RetrievalChunk(
            document_id="document-metadata",
            section_id="section-metadata",
            document_title="문서 제목",
            section_path=section_path,
            source_url="https://metadata.example.com",
            category="category-metadata",
            content="전체 본문",
        )


class OpenAIEmbedderTest(unittest.TestCase):
    def test_requests_multiple_embeddings_once_and_restores_input_order(self) -> None:
        first_embedding = [0.1] * OPENAI_EMBEDDING_DIMENSIONS
        second_embedding = [0.2] * OPENAI_EMBEDDING_DIMENSIONS
        client, raw_response = self._client_with_response(
            SimpleNamespace(
                data=[
                    SimpleNamespace(index=1, embedding=second_embedding),
                    SimpleNamespace(index=0, embedding=first_embedding),
                ]
            )
        )
        embedder = OpenAIEmbedder(client=client)

        embeddings = embedder.embed_many(["첫 번째 텍스트", "두 번째 텍스트"])

        client.with_options.assert_not_called()
        client.embeddings.with_raw_response.create.assert_called_once_with(
            model=OPENAI_EMBEDDING_MODEL,
            input=["첫 번째 텍스트", "두 번째 텍스트"],
            dimensions=OPENAI_EMBEDDING_DIMENSIONS,
            encoding_format="float",
        )
        raw_response.parse.assert_called_once_with()
        self.assertEqual([first_embedding, second_embedding], embeddings)

    def test_rejects_mismatched_response_count(self) -> None:
        client, _ = self._client_with_response(
            SimpleNamespace(
                data=[
                    SimpleNamespace(
                        index=0,
                        embedding=[0.1] * OPENAI_EMBEDDING_DIMENSIONS,
                    )
                ]
            )
        )
        embedder = OpenAIEmbedder(client=client)

        with self.assertRaisesRegex(RuntimeError, "응답 개수"):
            embedder.embed_many(["첫 번째 텍스트", "두 번째 텍스트"])

    def test_rejects_invalid_response_indexes(self) -> None:
        client, _ = self._client_with_response(
            SimpleNamespace(
                data=[
                    SimpleNamespace(
                        index=0,
                        embedding=[0.1] * OPENAI_EMBEDDING_DIMENSIONS,
                    ),
                    SimpleNamespace(
                        index=0,
                        embedding=[0.2] * OPENAI_EMBEDDING_DIMENSIONS,
                    ),
                ]
            )
        )
        embedder = OpenAIEmbedder(client=client)

        with self.assertRaisesRegex(RuntimeError, "응답 index"):
            embedder.embed_many(["첫 번째 텍스트", "두 번째 텍스트"])

    def test_rejects_invalid_embedding_dimension(self) -> None:
        client, _ = self._client_with_response(
            SimpleNamespace(
                data=[
                    SimpleNamespace(
                        index=0,
                        embedding=[0.1] * (OPENAI_EMBEDDING_DIMENSIONS - 1),
                    )
                ]
            )
        )
        embedder = OpenAIEmbedder(client=client)

        with self.assertRaisesRegex(ValueError, "1536차원"):
            embedder.embed_many(["검색할 텍스트"])

    def test_rejects_empty_texts_without_request(self) -> None:
        client = Mock()
        embedder = OpenAIEmbedder(client=client)

        with self.assertRaisesRegex(ValueError, "하나 이상"):
            embedder.embed_many([])

        client.embeddings.with_raw_response.create.assert_not_called()

    def test_embed_keeps_single_embedding_contract(self) -> None:
        expected_embedding = [0.1] * OPENAI_EMBEDDING_DIMENSIONS
        client, _ = self._client_with_response(
            SimpleNamespace(
                data=[SimpleNamespace(index=0, embedding=expected_embedding)]
            )
        )
        embedder = OpenAIEmbedder(client=client)

        embedding = embedder.embed("검색할 텍스트")

        client.embeddings.with_raw_response.create.assert_called_once_with(
            model=OPENAI_EMBEDDING_MODEL,
            input=["검색할 텍스트"],
            dimensions=OPENAI_EMBEDDING_DIMENSIONS,
            encoding_format="float",
        )
        self.assertEqual(expected_embedding, embedding)
        self.assertEqual(1536, len(embedding))

    def test_reports_zero_retries_on_first_success(self) -> None:
        client, _ = self._client_with_response(
            SimpleNamespace(
                data=[
                    SimpleNamespace(
                        index=0,
                        embedding=[0.1] * OPENAI_EMBEDDING_DIMENSIONS,
                    )
                ]
            )
        )

        response = OpenAIEmbedder(client=client).embed_many_with_usage(["질문"])

        self.assertEqual(0, response.retry_count)

    def test_reports_actual_sdk_retry_count_after_retries(self) -> None:
        attempts = 0

        def handle_request(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                return httpx.Response(
                    500,
                    headers={"retry-after": "0.001"},
                    json={"error": {"message": "temporary error"}},
                    request=request,
                )
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "index": 0,
                            "embedding": [0.1] * OPENAI_EMBEDDING_DIMENSIONS,
                        }
                    ],
                    "model": OPENAI_EMBEDDING_MODEL,
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
                request=request,
            )

        client = OpenAI(
            api_key="test-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handle_request)),
        )
        self.addCleanup(client.close)

        response = OpenAIEmbedder(client=client).embed_many_with_usage(["질문"])

        self.assertEqual(3, attempts)
        self.assertEqual(2, response.retry_count)

    def test_can_disable_sdk_retry_for_query_embedding(self) -> None:
        attempts = 0

        def handle_request(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(
                500,
                json={"error": {"message": "temporary error"}},
                request=request,
            )

        client = OpenAI(
            api_key="test-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handle_request)),
        )
        self.addCleanup(client.close)

        with self.assertRaises(InternalServerError):
            OpenAIEmbedder(client=client).embed_many_with_usage(
                ["질문"],
                sdk_max_retries=0,
            )

        self.assertEqual(1, attempts)

    def test_applies_explicit_timeout_and_sdk_retry_override(self) -> None:
        client, _ = self._client_with_response(
            SimpleNamespace(
                data=[
                    SimpleNamespace(
                        index=0,
                        embedding=[0.1] * OPENAI_EMBEDDING_DIMENSIONS,
                    )
                ]
            )
        )

        OpenAIEmbedder(client=client).embed_many_with_usage(
            ["질문"],
            sdk_max_retries=0,
            timeout=30.0,
        )

        client.with_options.assert_called_once_with(
            max_retries=0,
            timeout=30.0,
        )

    def test_configures_default_client_without_timeout_override(self) -> None:
        settings = SimpleNamespace(openai_api_key="test-key")

        with patch("retrieval.embedding.get_settings", return_value=settings):
            with patch("retrieval.embedding.OpenAI") as client_class:
                OpenAIEmbedder()

        client_class.assert_called_once_with(api_key="test-key")

    def test_requires_api_key_when_client_is_not_injected(self) -> None:
        settings = SimpleNamespace(openai_api_key=None)

        with patch("retrieval.embedding.get_settings", return_value=settings):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                OpenAIEmbedder()

    @staticmethod
    def _client_with_response(
        response: SimpleNamespace,
        retries_taken: int = 0,
    ) -> Tuple[Mock, Mock]:
        raw_response = Mock()
        raw_response.parse.return_value = response
        raw_response.retries_taken = retries_taken
        client = Mock()
        client.with_options.return_value = client
        client.embeddings.with_raw_response.create.return_value = raw_response
        return client, raw_response


if __name__ == "__main__":
    unittest.main()
