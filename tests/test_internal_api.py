import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.chat.schema import ChatErrorCode, ChatResponseStatus
from app.main import create_app
from app.retrieval.corpus_state import (
    CorpusNotLoadedError,
    CorpusSnapshot,
    CorpusState,
)
from app.retrieval.pgvector_store import ActiveIndexNotFoundError


@asynccontextmanager
async def test_lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


LOADED_SNAPSHOT = CorpusSnapshot(
    loaded=True,
    chunk_count=142,
    document_count=39,
    loaded_at=datetime(2026, 8, 22, 3, 0, tzinfo=timezone.utc),
    source="data/clean_manifest.json",
)
EMPTY_SNAPSHOT = CorpusSnapshot(
    loaded=False,
    chunk_count=0,
    document_count=0,
    loaded_at=None,
    source="data/clean_manifest.json",
)


class InternalCorpusApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus_state = Mock(spec=CorpusState)
        self.active_chunks = [Mock()]
        self.store = Mock()
        self.store.load_active_chunks = AsyncMock(return_value=self.active_chunks)
        self.store_patcher = patch(
            "app.api.internal.PgVectorStore",
            return_value=self.store,
        )
        self.store_patcher.start()

        with patch("app.main.lifespan", test_lifespan):
            self.app = create_app()

        self.app.state.corpus_state = self.corpus_state
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.store_patcher.stop()
        self.client.close()

    def test_reports_unloaded_state(self) -> None:
        self.corpus_state.snapshot.return_value = EMPTY_SNAPSHOT

        response = self.client.get("/internal/corpus")

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "loaded": False,
                "chunkCount": 0,
                "documentCount": 0,
                "loadedAt": None,
                "source": "data/clean_manifest.json",
            },
            response.json(),
        )

    def test_reports_loaded_state(self) -> None:
        self.corpus_state.snapshot.return_value = LOADED_SNAPSHOT

        body = self.client.get("/internal/corpus").json()

        self.assertTrue(body["loaded"])
        self.assertEqual(142, body["chunkCount"])
        self.assertEqual(39, body["documentCount"])
        self.assertIsNotNone(body["loadedAt"])

    def test_reload_returns_new_state(self) -> None:
        self.corpus_state.replace.return_value = LOADED_SNAPSHOT

        response = self.client.post("/internal/corpus/reload")

        self.store.load_active_chunks.assert_awaited_once_with()
        self.corpus_state.replace.assert_called_once_with(self.active_chunks)
        self.assertEqual(200, response.status_code)
        self.assertEqual(142, response.json()["chunkCount"])

    def test_reload_returns_503_when_corpus_is_missing(self) -> None:
        self.store.load_active_chunks.side_effect = ActiveIndexNotFoundError(
            "ACTIVE index version이 없습니다."
        )

        response = self.client.post("/internal/corpus/reload")

        self.assertEqual(503, response.status_code)
        self.assertIn("ACTIVE index", response.json()["detail"])

    def test_chat_returns_503_when_corpus_is_not_loaded(self) -> None:
        self.corpus_state.get_retriever.side_effect = CorpusNotLoadedError(
            "corpus가 적재되지 않았습니다."
        )

        response = self.client.post("/api/chat", json={"question": "질문"})

        self.assertEqual(503, response.status_code)
        body = response.json()
        self.assertEqual(ChatResponseStatus.ERROR.value, body["status"])
        self.assertEqual(
            ChatErrorCode.SERVICE_UNAVAILABLE.value,
            body["error"]["code"],
        )
        self.assertIsNone(body["answer"])
        self.assertEqual([], body["citations"])


if __name__ == "__main__":
    unittest.main()
