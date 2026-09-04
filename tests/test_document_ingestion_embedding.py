import unittest
from unittest.mock import AsyncMock, Mock

from app.core.hashing import sha256_hex
from app.document.ingestion import prepare_chunk_embeddings
from app.retrieval.embedding import (
    OPENAI_EMBEDDING_DIMENSIONS,
    EmbeddingResponse,
    build_embedding_text,
)
from app.retrieval.models import RetrievalChunk


def _chunk(index: int) -> RetrievalChunk:
    return RetrievalChunk(
        document_id=f"doc-{index}",
        section_id=f"doc-{index}#s",
        document_title=f"문서 {index}",
        section_path=(f"문서 {index}", f"섹션 {index}"),
        source_url=f"https://docs.riido.io/doc-{index}.md",
        category="test",
        content=f"본문 {index}",
    )


def _embedding(value: float) -> list:
    return [value] * OPENAI_EMBEDDING_DIMENSIONS


class PrepareChunkEmbeddingsTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.chunks = [_chunk(0), _chunk(1)]
        self.hashes = [
            sha256_hex(build_embedding_text(chunk)) for chunk in self.chunks
        ]
        self.store = Mock()
        self.embedder = Mock()

    async def test_generates_every_embedding_when_nothing_is_reusable(self) -> None:
        self.store.load_reusable_embeddings = AsyncMock(return_value={})
        self.embedder.embed_many_with_usage.return_value = EmbeddingResponse(
            embeddings=[_embedding(0.1), _embedding(0.2)],
            input_tokens=40,
            retry_count=1,
        )

        embeddings, call = await prepare_chunk_embeddings(
            self.store,
            11,
            self.chunks,
            self.embedder,
        )

        self.assertEqual([_embedding(0.1), _embedding(0.2)], embeddings)
        self.assertEqual(2, call.generated_count)
        self.assertEqual(0, call.reused_count)
        self.assertEqual(40, call.input_tokens)
        self.assertEqual(1, call.retry_count)
        self.embedder.embed_many_with_usage.assert_called_once_with(
            [build_embedding_text(chunk) for chunk in self.chunks]
        )

    async def test_reuses_stored_embedding_with_same_input_hash(self) -> None:
        self.store.load_reusable_embeddings = AsyncMock(
            return_value={self.hashes[0]: _embedding(0.9)}
        )
        self.embedder.embed_many_with_usage.return_value = EmbeddingResponse(
            embeddings=[_embedding(0.2)],
            input_tokens=20,
            retry_count=0,
        )

        embeddings, call = await prepare_chunk_embeddings(
            self.store,
            11,
            self.chunks,
            self.embedder,
        )

        # 재사용한 첫 Chunk는 저장된 vector 그대로이고 두 번째만 새로 만든다
        self.assertEqual([_embedding(0.9), _embedding(0.2)], embeddings)
        self.assertEqual(1, call.generated_count)
        self.assertEqual(1, call.reused_count)
        self.embedder.embed_many_with_usage.assert_called_once_with(
            [build_embedding_text(self.chunks[1])]
        )

    async def test_skips_model_call_when_every_chunk_is_reusable(self) -> None:
        self.store.load_reusable_embeddings = AsyncMock(
            return_value={
                self.hashes[0]: _embedding(0.9),
                self.hashes[1]: _embedding(0.8),
            }
        )

        embeddings, call = await prepare_chunk_embeddings(
            self.store,
            11,
            self.chunks,
            self.embedder,
        )

        self.assertEqual([_embedding(0.9), _embedding(0.8)], embeddings)
        self.assertIsNone(call)
        self.embedder.embed_many_with_usage.assert_not_called()

    async def test_rejects_empty_chunks(self) -> None:
        with self.assertRaisesRegex(ValueError, "하나 이상"):
            await prepare_chunk_embeddings(self.store, 11, [], self.embedder)


if __name__ == "__main__":
    unittest.main()
