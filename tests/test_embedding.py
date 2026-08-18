import unittest
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import Mock, patch

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
        self.assertNotIn(chunk.chunk_id, text)
        self.assertNotIn(chunk.document_id, text)
        self.assertNotIn(chunk.section_id, text)
        self.assertNotIn(chunk.source_url, text)
        self.assertNotIn(chunk.category, text)

    @staticmethod
    def _retrieval_chunk(section_path: Tuple[str, ...]) -> RetrievalChunk:
        return RetrievalChunk(
            chunk_id="chunk-metadata",
            document_id="document-metadata",
            section_id="section-metadata",
            document_title="문서 제목",
            section_path=section_path,
            source_url="https://metadata.example.com",
            category="category-metadata",
            content="전체 본문",
        )


class OpenAIEmbedderTest(unittest.TestCase):
    def test_requests_embedding_with_confirmed_model_and_dimensions(self) -> None:
        expected_embedding = [0.1] * OPENAI_EMBEDDING_DIMENSIONS
        client = Mock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=expected_embedding)]
        )
        embedder = OpenAIEmbedder(client=client)

        embedding = embedder.embed("검색할 텍스트")

        client.embeddings.create.assert_called_once_with(
            model=OPENAI_EMBEDDING_MODEL,
            input="검색할 텍스트",
            dimensions=OPENAI_EMBEDDING_DIMENSIONS,
            encoding_format="float",
        )
        self.assertEqual(expected_embedding, embedding)
        self.assertEqual(1536, len(embedding))

    def test_requires_api_key_when_client_is_not_injected(self) -> None:
        settings = SimpleNamespace(openai_api_key=None)

        with patch("retrieval.embedding.get_settings", return_value=settings):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                OpenAIEmbedder()


if __name__ == "__main__":
    unittest.main()
