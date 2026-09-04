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
from app.chat.service import ChatService
from app.chat.dependencies import get_chat_service
from app.answering.service import GenerationService
from app.chat.query_rewrite import QueryRewriteService
from app.answering.generator import OpenAIGenerator
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.embedding import OpenAIEmbedder
from app.retrieval.models import RetrievalChunk


class ChatDependencyLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = [
            RetrievalChunk(
                document_id="document-1",
                section_id="section-1",
                document_title="문서 1",
                section_path=("문서 1", "섹션 1"),
                source_url="https://docs.riido.io/1.md",
                category="guide",
                content="본문 1",
                chunk_id=1,
                document_version_id=2,
                index_version_id=3,
            )
        ]
        self.bm25_retriever = Mock(spec=BM25Retriever)
        self.embedder = Mock(spec=OpenAIEmbedder)
        self.generator = Mock(spec=OpenAIGenerator)
        self.generation_service = Mock(spec=GenerationService)
        self.query_rewrite_service = Mock(spec=QueryRewriteService)

    def test_lifespan_initializes_shared_dependencies_once(self) -> None:
        with self._patched_lifespan_dependencies() as dependencies:
            app = create_app()

            with TestClient(app) as client:
                self.assertIs(
                    self.bm25_retriever,
                    app.state.corpus_state.get_retriever(),
                )
                self.assertIs(self.embedder, app.state.embedder)
                self.assertIs(
                    self.generation_service,
                    app.state.generation_service,
                )
                self.assertIs(
                    self.query_rewrite_service,
                    app.state.query_rewrite_service,
                )

                self.assertEqual(200, client.get("/health").status_code)
                self.assertEqual(200, client.get("/health").status_code)

            dependencies["store"].load_active_chunks.assert_awaited_once_with()
            dependencies["store_class"].assert_called_once_with(
                dependencies["lifespan_session"]
            )
            dependencies["bm25"].assert_called_once_with(self.corpus)
            dependencies["embedder"].assert_called_once_with()
            dependencies["generator"].assert_called_once_with()
            dependencies["generation_service"].assert_called_once_with(
                self.generator
            )
            dependencies["query_rewrite_service"].assert_called_once_with()
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
        self.assertIs(self.query_rewrite_service, first._query_rewrite_service)
        self.assertIs(self.query_rewrite_service, second._query_rewrite_service)
        self.assertEqual(sessions, released_sessions)

        # 검색과 로그가 같은 session을 공유해야 2단계 커밋이 하나의 경계가 된다
        self.assertIs(sessions[0], first._session)
        self.assertIs(sessions[0], first._log_store._session)
        self.assertIs(sessions[1], second._session)
        self.assertIs(sessions[1], second._log_store._session)
        self.assertIsNot(first._log_store, second._log_store)
        self.assertEqual(3, first._index_version_id)
        self.assertEqual(3, second._index_version_id)

    @contextmanager
    def _patched_lifespan_dependencies(self):
        lifespan_session = AsyncMock(spec=AsyncSession)
        session_context = AsyncMock()
        session_context.__aenter__.return_value = lifespan_session
        session_factory = Mock(return_value=session_context)
        store = Mock()
        store.load_active_chunks = AsyncMock(return_value=self.corpus)
        patches = {
            "session_factory": patch(
                "app.main.get_session_factory",
                return_value=session_factory,
            ),
            "store_class": patch(
                "app.main.SearchReader",
                return_value=store,
            ),
            "bm25": patch(
                "app.retrieval.corpus_state.BM25Retriever",
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
            "query_rewrite_service": patch(
                "app.main.QueryRewriteService",
                return_value=self.query_rewrite_service,
            ),
            "dispose_engine": patch(
                "app.main.dispose_engine",
                new_callable=AsyncMock,
            ),
        }

        with ExitStack() as stack:
            dependencies = {
                name: stack.enter_context(dependency_patch)
                for name, dependency_patch in patches.items()
            }
            dependencies["lifespan_session"] = lifespan_session
            dependencies["store"] = store
            yield dependencies


if __name__ == "__main__":
    unittest.main()
