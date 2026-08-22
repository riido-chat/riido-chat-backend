import unittest
from collections.abc import AsyncIterator
from contextlib import ExitStack, contextmanager
from typing import List
from unittest.mock import AsyncMock, Mock, patch

from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db_session
from app.main import create_app
from app.rag.chat_service import ChatService
from app.rag.dependencies import get_chat_service
from app.rag.generation_service import GenerationService
from generation.generator import OpenAIGenerator
from retrieval.bm25_retriever import BM25Retriever
from retrieval.embedding import OpenAIEmbedder


class ChatDependencyLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = [Mock()]
        self.bm25_retriever = Mock(spec=BM25Retriever)
        self.embedder = Mock(spec=OpenAIEmbedder)
        self.generator = Mock(spec=OpenAIGenerator)
        self.generation_service = Mock(spec=GenerationService)

    def test_lifespan_initializes_shared_dependencies_once(self) -> None:
        with self._patched_lifespan_dependencies() as dependencies:
            app = create_app()

            with TestClient(app) as client:
                self.assertIs(self.corpus, app.state.retrieval_corpus)
                self.assertIs(self.bm25_retriever, app.state.bm25_retriever)
                self.assertIs(self.embedder, app.state.embedder)
                self.assertIs(
                    self.generation_service,
                    app.state.generation_service,
                )

                self.assertEqual(200, client.get("/health").status_code)
                self.assertEqual(200, client.get("/health").status_code)

            dependencies["build_corpus"].assert_called_once_with()
            dependencies["bm25"].assert_called_once_with(self.corpus)
            dependencies["embedder"].assert_called_once_with()
            dependencies["generator"].assert_called_once_with()
            dependencies["generation_service"].assert_called_once_with(
                self.generator
            )
            dependencies["dispose_engine"].assert_awaited_once_with()

    def test_builds_request_scoped_chat_dependency_graph(self) -> None:
        sessions = [
            AsyncMock(spec=AsyncSession),
            AsyncMock(spec=AsyncSession),
        ]
        session_iterator = iter(sessions)
        released_sessions = []
        observed_services: List[ChatService] = []

        async def override_db_session() -> AsyncIterator[AsyncSession]:
            session = next(session_iterator)
            try:
                yield session
            finally:
                released_sessions.append(session)

        with self._patched_lifespan_dependencies():
            app = create_app()
            app.dependency_overrides[get_db_session] = override_db_session

            @app.get("/_test/chat-dependencies")
            async def inspect_chat_dependencies(
                service: ChatService = Depends(get_chat_service),
            ) -> dict[str, str]:
                observed_services.append(service)
                return {"status": "ok"}

            with TestClient(app) as client:
                self.assertEqual(
                    {"status": "ok"},
                    client.get("/_test/chat-dependencies").json(),
                )
                self.assertEqual(
                    {"status": "ok"},
                    client.get("/_test/chat-dependencies").json(),
                )

        first, second = observed_services
        self.assertIsInstance(first, ChatService)
        self.assertIsInstance(second, ChatService)
        self.assertIsNot(first, second)
        self.assertIsNot(first._retriever, second._retriever)
        self.assertIsNot(
            first._retriever._vector_retriever,
            second._retriever._vector_retriever,
        )
        self.assertIsNot(
            first._retriever._vector_retriever._store,
            second._retriever._vector_retriever._store,
        )
        self.assertIs(
            self.bm25_retriever,
            first._retriever._bm25_retriever,
        )
        self.assertIs(
            self.bm25_retriever,
            second._retriever._bm25_retriever,
        )
        self.assertIs(
            self.embedder,
            first._retriever._vector_retriever._embedder,
        )
        self.assertIs(
            self.embedder,
            second._retriever._vector_retriever._embedder,
        )
        self.assertIs(
            sessions[0],
            first._retriever._vector_retriever._store._session,
        )
        self.assertIs(
            sessions[1],
            second._retriever._vector_retriever._store._session,
        )
        self.assertIs(self.generation_service, first._generation_service)
        self.assertIs(self.generation_service, second._generation_service)
        self.assertEqual(sessions, released_sessions)

    @contextmanager
    def _patched_lifespan_dependencies(self):
        patches = {
            "build_corpus": patch(
                "app.main.build_retrieval_chunks",
                return_value=self.corpus,
            ),
            "bm25": patch(
                "app.main.BM25Retriever",
                return_value=self.bm25_retriever,
            ),
            "embedder": patch(
                "app.main.OpenAIEmbedder",
                return_value=self.embedder,
            ),
            "generator": patch(
                "app.main.OpenAIGenerator",
                return_value=self.generator,
            ),
            "generation_service": patch(
                "app.main.GenerationService",
                return_value=self.generation_service,
            ),
            "dispose_engine": patch(
                "app.main.dispose_engine",
                new_callable=AsyncMock,
            ),
        }

        with ExitStack() as stack:
            yield {
                name: stack.enter_context(dependency_patch)
                for name, dependency_patch in patches.items()
            }


if __name__ == "__main__":
    unittest.main()
