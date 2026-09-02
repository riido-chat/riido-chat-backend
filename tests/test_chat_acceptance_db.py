"""Chat HTTP 응답과 PostgreSQL 실행 로그를 함께 검증하는 수용 테스트."""

import asyncio
import unittest
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock

import httpx
from httpx import ASGITransport, AsyncClient
from openai import APITimeoutError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.config import get_settings
from app.database.models import (
    AnswerStatus,
    ChunkingConfig,
    ContentNode,
    ContextStrategy,
    DocumentChunk,
    DocumentSource,
    DocumentVersion,
    DocumentVersionStatus,
    EmbeddingConfig,
    ExecutionStatus,
    Feedback,
    IndexVersion,
    IndexVersionStatus,
    ModelCallPurpose,
)
from app.database.session import get_db_session
from app.main import create_app
from app.rag.chat_service import ChatService
from app.rag.dependencies import get_chat_service, get_rag_log_store
from app.rag.generation_service import GenerationService
from app.rag.log_store import RagLogStore
from app.rag.model_trace import ModelCallTrace
from app.rag.query_rewrite import (
    MODEL_OUTPUT_INVALID_ERROR_CODE,
    QUERY_REWRITE_PROMPT_VERSION,
    QueryResolution,
    QueryRewriteCall,
    QueryRewriteDecision,
    QueryRewriteService,
    QueryRewriteTurnStatus,
)
from generation.models import (
    Citation,
    FinalAnswerStatus,
    FinalGenerationResult,
    FinalWithheldReason,
)
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import (
    HybridRetrievalResult,
    HybridSearchCall,
    RetrievalChunk,
    RetrievalResult,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _check_database_available(url: str) -> bool:
    engine = create_async_engine(url)
    try:
        async with engine.connect():
            return True
    except Exception:
        return False
    finally:
        await engine.dispose()


class ChatApiDbAcceptanceTest(unittest.IsolatedAsyncioTestCase):
    """실제 API 계층과 DB 로그 사이의 수용 계약을 고정한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_check_database_available(cls.database_url)):
            raise unittest.SkipTest(
                "로컬 DB에 연결할 수 없어 Chat 수용 테스트를 건너뜁니다."
            )

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(self.database_url)
        self.connection = await self.engine.connect()
        self.transaction = await self.connection.begin()
        self.session = AsyncSession(
            bind=self.connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        self.store = RagLogStore(self.session)
        await self._seed_content_chain()

        self.service: ChatService
        self.app = create_app()

        async def override_chat_service() -> ChatService:
            return self.service

        async def override_db_session() -> AsyncIterator[AsyncSession]:
            yield self.session

        self.app.dependency_overrides[get_chat_service] = override_chat_service
        self.app.dependency_overrides[get_db_session] = override_db_session
        self.app.dependency_overrides[get_rag_log_store] = lambda: self.store
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://testserver",
            headers={"Accept": "application/json"},
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        self.app.dependency_overrides.clear()
        await self.session.close()
        await self.transaction.rollback()
        await self.connection.close()
        await self.engine.dispose()

    async def _seed_content_chain(self) -> None:
        suffix = uuid.uuid4().hex[:8]
        chunking = ChunkingConfig(
            version=f"acceptance-chunking-{suffix}",
            strategy="SECTION",
            max_tokens=512,
            created_at=_now(),
        )
        embedding = EmbeddingConfig(
            version=f"acceptance-embedding-{suffix}",
            provider="openai",
            model_name="text-embedding-test",
            dimensions=1536,
            input_template_version="v1",
            created_at=_now(),
        )
        source = DocumentSource(
            source_type="LLMS_TXT",
            canonical_uri=f"https://docs.riido.io/acceptance/{suffix}.md",
            title="스프린트",
            created_at=_now(),
            updated_at=_now(),
        )
        self.session.add_all([chunking, embedding, source])
        await self.session.flush()

        version = DocumentVersion(
            document_source_id=source.id,
            version_no=1,
            raw_content_uri=f"raw/acceptance/{suffix}.md",
            mime_type="text/markdown",
            raw_content_hash="raw-" + suffix,
            normalized_content_hash="norm-" + suffix,
            parser_name="markdown",
            parser_version="1",
            status=DocumentVersionStatus.READY,
            collected_at=_now(),
            created_at=_now(),
        )
        self.session.add(version)
        await self.session.flush()

        node = ContentNode(
            document_version_id=version.id,
            node_type="SECTION",
            node_path="스프린트>설정",
            node_order=1,
            title="설정",
            normalized_content="스프린트 주기와 시작 요일을 설정할 수 있습니다.",
            content_hash="content-" + suffix,
            node_identity_hash=f"{suffix}:identity",
            node_identity_kind="path",
            created_at=_now(),
        )
        self.session.add(node)
        await self.session.flush()

        chunk = DocumentChunk(
            id=node.id,
            chunking_config_id=chunking.id,
            chunk_index=0,
            token_count=42,
            created_at=_now(),
        )
        index_version = IndexVersion(
            version=f"acceptance-index-{suffix}",
            status=IndexVersionStatus.ACTIVE,
            chunking_config_id=chunking.id,
            embedding_config_id=embedding.id,
            created_at=_now(),
        )
        self.session.add_all([chunk, index_version])
        await self.session.flush()

        self.chunk_id = chunk.id
        self.document_version_id = version.id
        self.index_version_id = index_version.id

    def _search_call(
        self,
        *,
        error: Optional[Exception] = None,
        embedding_succeeded: bool = True,
    ) -> HybridSearchCall:
        embedding_trace = ModelCallTrace(
            provider="openai",
            model_name="text-embedding-test",
            succeeded=embedding_succeeded,
            latency_ms=15,
            input_tokens=3 if embedding_succeeded else None,
            retry_count=1 if error is not None else 0,
            error_message=None if error is None else str(error),
        )
        if error is not None:
            return HybridSearchCall(embedding_call=embedding_trace, error=error)

        chunk = RetrievalChunk(
            document_id="sprint-document",
            section_id="sprint-settings",
            document_title="스프린트",
            section_path=("스프린트", "설정"),
            source_url="https://docs.riido.io/projects/sprints.md",
            category="projects",
            content="스프린트 주기와 시작 요일을 설정할 수 있습니다.",
            chunk_id=self.chunk_id,
            document_version_id=self.document_version_id,
            index_version_id=self.index_version_id,
        )
        bm25 = RetrievalResult(chunk=chunk, score=7.5, rank=1)
        vector = RetrievalResult(chunk=chunk, score=0.9, rank=1)
        fused = HybridRetrievalResult(
            chunk=chunk,
            rrf_score=0.032,
            final_rank=1,
            bm25_rank=1,
            vector_rank=1,
        )
        return HybridSearchCall(
            bm25_results=(bm25,),
            vector_results=(vector,),
            fused_results=(fused,),
            bm25_latency_ms=10,
            vector_latency_ms=20,
            embedding_call=embedding_trace,
        )

    def _completed_result(self, answer: str) -> FinalGenerationResult:
        return FinalGenerationResult(
            status=FinalAnswerStatus.COMPLETED,
            answer_markdown=answer,
            citations=(
                Citation(
                    citation_number=1,
                    document_title="스프린트",
                    section_path=("스프린트", "설정"),
                    source_url="https://docs.riido.io/projects/sprints.md",
                    chunk_id=self.chunk_id,
                    document_version_id=self.document_version_id,
                ),
            ),
            model_call=ModelCallTrace(
                provider="openai",
                model_name="gpt-test",
                succeeded=True,
                latency_ms=30,
                input_tokens=20,
                output_tokens=10,
                prompt_version="v3",
            ),
        )

    def _build_service(
        self,
        *,
        search_side_effect=None,
        generation_side_effect=None,
        rewrite_side_effect=None,
    ) -> tuple[AsyncMock, AsyncMock, AsyncMock]:
        retriever = AsyncMock(spec=HybridRetriever)
        generation_service = AsyncMock(spec=GenerationService)
        query_rewrite_service = AsyncMock(spec=QueryRewriteService)

        async def default_search(_query, *, before_model_call):
            await before_model_call("openai", "text-embedding-test", None)
            return self._search_call()

        async def default_generation(query, _results, *, before_model_call):
            await before_model_call("openai", "gpt-test", "v3")
            return self._completed_result(f"{query}에 대한 답변입니다. [1]")

        retriever.search_with_trace.side_effect = search_side_effect or default_search
        generation_service.generate_answer.side_effect = (
            generation_side_effect or default_generation
        )
        if rewrite_side_effect is not None:
            query_rewrite_service.rewrite.side_effect = rewrite_side_effect

        self.service = ChatService(
            retriever=retriever,
            generation_service=generation_service,
            query_rewrite_service=query_rewrite_service,
            log_store=self.store,
            session=self.session,
            index_version_id=self.index_version_id,
        )
        return retriever, generation_service, query_rewrite_service

    async def test_sprint_follow_up_keeps_api_and_db_context_consistent(self) -> None:
        retrieval_queries = []
        generation_queries = []
        rewrite_candidates = []

        async def search(query, *, before_model_call):
            retrieval_queries.append(query)
            await before_model_call("openai", "text-embedding-test", None)
            return self._search_call()

        async def generate(query, _results, *, before_model_call):
            generation_queries.append(query)
            await before_model_call("openai", "gpt-test", "v3")
            return self._completed_result(f"{query}에 대한 답변입니다. [1]")

        async def rewrite(question, candidates, *, before_model_call):
            rewrite_candidates.extend(candidates)
            await before_model_call(
                "openai",
                "gpt-test",
                QUERY_REWRITE_PROMPT_VERSION,
            )
            return QueryRewriteCall(
                trace=ModelCallTrace(
                    provider="openai",
                    model_name="gpt-test",
                    succeeded=True,
                    latency_ms=25,
                    input_tokens=30,
                    output_tokens=8,
                    prompt_version=QUERY_REWRITE_PROMPT_VERSION,
                ),
                resolution=QueryResolution(
                    decision=QueryRewriteDecision.FOLLOW_UP_RESOLVED,
                    resolved_query="스프린트 설정 방법",
                    selected_turns=(candidates[0],),
                ),
            )

        self._build_service(
            search_side_effect=search,
            generation_side_effect=generate,
            rewrite_side_effect=rewrite,
        )

        first = await self.client.post(
            "/api/chat",
            json={"question": "스프린트가 뭐야?"},
        )
        second = await self.client.post(
            "/api/chat",
            json={
                "question": "그건 어떻게 설정해?",
                "conversationId": first.json()["conversationId"],
            },
        )

        self.assertEqual(200, first.status_code)
        self.assertEqual("COMPLETED", first.json()["status"])
        self.assertEqual(200, second.status_code)
        self.assertEqual("COMPLETED", second.json()["status"])
        self.assertEqual(
            ["스프린트가 뭐야?", "스프린트 설정 방법"],
            retrieval_queries,
        )
        self.assertEqual(retrieval_queries, generation_queries)
        self.assertEqual(1, len(rewrite_candidates))
        self.assertEqual(1, rewrite_candidates[0].turn_no)
        self.assertEqual("스프린트가 뭐야?", rewrite_candidates[0].user_query)

        detail = await self.store.get_rag_run_detail(
            uuid.UUID(second.json()["ragRunId"])
        )
        self.assertEqual(AnswerStatus.COMPLETED, detail.run.status)
        self.assertEqual("스프린트 설정 방법", detail.run.resolved_query)
        self.assertEqual(ContextStrategy.FOLLOW_UP_WINDOW, detail.run.context_strategy)
        self.assertEqual(1, detail.run.context_turn_count)
        self.assertEqual(
            [1],
            [
                turn["turnNo"]
                for turn in detail.run.context_snapshot["selectedTurns"]
            ],
        )
        self.assertEqual(
            [
                ModelCallPurpose.QUERY_REWRITE,
                ModelCallPurpose.QUERY_EMBEDDING,
                ModelCallPurpose.ANSWER_GENERATION,
            ],
            [call.purpose for call in detail.model_calls],
        )
        self.assertEqual(2, len(detail.retrieval_results))
        self.assertEqual(1, len(detail.citations))
        self.assertEqual(
            second.json()["citations"][0]["sourceUrl"],
            detail.citations[0].source_uri_snapshot,
        )

        rag_run_id = second.json()["ragRunId"]
        registered = await self.client.put(
            f"/api/chat/{rag_run_id}/feedback",
            json={"rating": "GOOD"},
        )
        changed = await self.client.put(
            f"/api/chat/{rag_run_id}/feedback",
            json={"rating": "BAD"},
        )
        cleared = await self.client.delete(f"/api/chat/{rag_run_id}/feedback")

        self.assertEqual("GOOD", registered.json()["rating"])
        self.assertEqual("BAD", changed.json()["rating"])
        self.assertIsNone(cleared.json()["rating"])
        feedback = await self.session.scalar(
            select(Feedback).where(Feedback.rag_run_id == uuid.UUID(rag_run_id))
        )
        self.assertIsNone(feedback)

    async def test_previous_withheld_turn_is_available_to_query_rewrite(self) -> None:
        generation_count = 0
        seen_candidates = []

        async def generate(query, _results, *, before_model_call):
            nonlocal generation_count
            generation_count += 1
            await before_model_call("openai", "gpt-test", "v3")
            if generation_count == 1:
                return FinalGenerationResult(
                    status=FinalAnswerStatus.WITHHELD,
                    answer_markdown=(
                        "이용가이드에서 질문에 답할 충분한 근거를 찾지 못했습니다."
                    ),
                    citations=(),
                    withheld_reason=FinalWithheldReason.INSUFFICIENT_EVIDENCE,
                    model_call=ModelCallTrace(
                        provider="openai",
                        model_name="gpt-test",
                        succeeded=True,
                        latency_ms=20,
                        prompt_version="v3",
                    ),
                )
            return self._completed_result(f"{query}에 대한 답변입니다. [1]")

        async def rewrite(_question, candidates, *, before_model_call):
            seen_candidates.extend(candidates)
            await before_model_call(
                "openai",
                "gpt-test",
                QUERY_REWRITE_PROMPT_VERSION,
            )
            return QueryRewriteCall(
                trace=ModelCallTrace(
                    provider="openai",
                    model_name="gpt-test",
                    succeeded=True,
                    latency_ms=20,
                    prompt_version=QUERY_REWRITE_PROMPT_VERSION,
                ),
                resolution=QueryResolution(
                    decision=QueryRewriteDecision.NEW_TOPIC,
                    resolved_query="스프린트 설정 방법은?",
                    selected_turns=(),
                ),
            )

        self._build_service(
            generation_side_effect=generate,
            rewrite_side_effect=rewrite,
        )
        first = await self.client.post(
            "/api/chat",
            json={"question": "문서에 없는 스프린트 정책을 알려줘"},
        )
        second = await self.client.post(
            "/api/chat",
            json={
                "question": "스프린트 설정 방법은?",
                "conversationId": first.json()["conversationId"],
            },
        )

        self.assertEqual("WITHHELD", first.json()["status"])
        self.assertEqual("COMPLETED", second.json()["status"])
        self.assertEqual(1, len(seen_candidates))
        self.assertEqual(QueryRewriteTurnStatus.WITHHELD, seen_candidates[0].status)
        self.assertIsNone(seen_candidates[0].answer_content)
        self.assertEqual(
            FinalWithheldReason.INSUFFICIENT_EVIDENCE,
            seen_candidates[0].withheld_reason_code,
        )

    async def test_ambiguous_follow_up_stops_before_retrieval_and_generation(
        self,
    ) -> None:
        async def rewrite(_question, _candidates, *, before_model_call):
            await before_model_call(
                "openai",
                "gpt-test",
                QUERY_REWRITE_PROMPT_VERSION,
            )
            return QueryRewriteCall(
                trace=ModelCallTrace(
                    provider="openai",
                    model_name="gpt-test",
                    succeeded=True,
                    latency_ms=20,
                    prompt_version=QUERY_REWRITE_PROMPT_VERSION,
                ),
                resolution=QueryResolution(
                    decision=QueryRewriteDecision.FOLLOW_UP_UNRESOLVED,
                    resolved_query=None,
                    selected_turns=(),
                ),
            )

        retriever, generation, _ = self._build_service(rewrite_side_effect=rewrite)
        first = await self.client.post(
            "/api/chat",
            json={"question": "스프린트와 프로젝트 차이를 알려줘"},
        )
        retriever.reset_mock()
        generation.reset_mock()
        second = await self.client.post(
            "/api/chat",
            json={
                "question": "그건 어떻게 설정해?",
                "conversationId": first.json()["conversationId"],
            },
        )

        self.assertEqual(200, second.status_code)
        self.assertEqual("WITHHELD", second.json()["status"])
        self.assertEqual(
            "AMBIGUOUS_QUESTION",
            second.json()["withheld"]["reasonCode"],
        )
        retriever.search_with_trace.assert_not_awaited()
        generation.generate_answer.assert_not_awaited()
        detail = await self.store.get_rag_run_detail(
            uuid.UUID(second.json()["ragRunId"])
        )
        self.assertEqual(AnswerStatus.WITHHELD, detail.run.status)
        self.assertEqual("AMBIGUOUS_QUESTION", detail.run.withheld_reason_code)
        self.assertEqual(1, len(detail.model_calls))
        self.assertEqual(ModelCallPurpose.QUERY_REWRITE, detail.model_calls[0].purpose)
        self.assertEqual([], detail.retrieval_results)

    async def test_fault_injection_keeps_http_and_db_error_policy_consistent(
        self,
    ) -> None:
        timeout = APITimeoutError(
            request=httpx.Request("POST", "https://api.openai.com")
        )

        async def failed_search(_query, *, before_model_call):
            await before_model_call("openai", "text-embedding-test", None)
            return self._search_call(error=timeout, embedding_succeeded=False)

        self._build_service(search_side_effect=failed_search)
        search_response = await self.client.post(
            "/api/chat",
            json={"question": "검색 실패를 확인해줘"},
        )
        await self._assert_error_run(
            search_response,
            error_code="UPSTREAM_ERROR",
            retryable=True,
            purposes=[ModelCallPurpose.QUERY_EMBEDDING],
            statuses=[ExecutionStatus.FAILED],
            retrieval_count=0,
        )

        async def failed_generation(_query, _results, *, before_model_call):
            await before_model_call("openai", "gpt-test", "v3")
            return FinalGenerationResult(
                status=FinalAnswerStatus.ERROR,
                answer_markdown=None,
                citations=(),
                error_code="UPSTREAM_ERROR",
                model_call=ModelCallTrace(
                    provider="openai",
                    model_name="gpt-test",
                    succeeded=False,
                    latency_ms=40,
                    retry_count=1,
                    prompt_version="v3",
                    error_message="generation timeout",
                ),
            )

        self._build_service(generation_side_effect=failed_generation)
        generation_response = await self.client.post(
            "/api/chat",
            json={"question": "생성 실패를 확인해줘"},
        )
        await self._assert_error_run(
            generation_response,
            error_code="UPSTREAM_ERROR",
            retryable=True,
            purposes=[
                ModelCallPurpose.QUERY_EMBEDDING,
                ModelCallPurpose.ANSWER_GENERATION,
            ],
            statuses=[ExecutionStatus.SUCCESS, ExecutionStatus.FAILED],
            retrieval_count=2,
        )

        async def citation_failure(_query, _results, *, before_model_call):
            await before_model_call("openai", "gpt-test", "v3")
            return FinalGenerationResult(
                status=FinalAnswerStatus.ERROR,
                answer_markdown=None,
                citations=(),
                error_code="CITATION_VALIDATION_ERROR",
                model_call=ModelCallTrace(
                    provider="openai",
                    model_name="gpt-test",
                    succeeded=True,
                    latency_ms=35,
                    prompt_version="v3",
                ),
            )

        self._build_service(generation_side_effect=citation_failure)
        citation_response = await self.client.post(
            "/api/chat",
            json={"question": "인용 실패를 확인해줘"},
        )
        await self._assert_error_run(
            citation_response,
            error_code="CITATION_VALIDATION_ERROR",
            retryable=False,
            purposes=[
                ModelCallPurpose.QUERY_EMBEDDING,
                ModelCallPurpose.ANSWER_GENERATION,
            ],
            statuses=[ExecutionStatus.SUCCESS, ExecutionStatus.SUCCESS],
            retrieval_count=2,
        )

    async def test_query_rewrite_contract_failure_is_retryable_and_persisted(
        self,
    ) -> None:
        async def failed_rewrite(_question, _candidates, *, before_model_call):
            await before_model_call(
                "openai",
                "gpt-test",
                QUERY_REWRITE_PROMPT_VERSION,
            )
            return QueryRewriteCall(
                trace=ModelCallTrace(
                    provider="openai",
                    model_name="gpt-test",
                    succeeded=False,
                    latency_ms=45,
                    retry_count=1,
                    prompt_version=QUERY_REWRITE_PROMPT_VERSION,
                    error_message="structured output invalid",
                ),
                error=ValueError("structured output invalid"),
                error_code=MODEL_OUTPUT_INVALID_ERROR_CODE,
            )

        retriever, generation, _ = self._build_service(
            rewrite_side_effect=failed_rewrite
        )
        first = await self.client.post(
            "/api/chat",
            json={"question": "스프린트가 뭐야?"},
        )
        retriever.reset_mock()
        generation.reset_mock()
        second = await self.client.post(
            "/api/chat",
            json={
                "question": "그건 어떻게 설정해?",
                "conversationId": first.json()["conversationId"],
            },
        )

        await self._assert_error_run(
            second,
            error_code="MODEL_OUTPUT_INVALID",
            retryable=True,
            purposes=[ModelCallPurpose.QUERY_REWRITE],
            statuses=[ExecutionStatus.FAILED],
            retrieval_count=0,
        )
        retriever.search_with_trace.assert_not_awaited()
        generation.generate_answer.assert_not_awaited()

    async def _assert_error_run(
        self,
        response,
        *,
        error_code: str,
        retryable: bool,
        purposes: list[ModelCallPurpose],
        statuses: list[ExecutionStatus],
        retrieval_count: int,
    ) -> None:
        self.assertEqual(500, response.status_code)
        body = response.json()
        self.assertEqual("ERROR", body["status"])
        self.assertEqual(error_code, body["error"]["code"])
        self.assertEqual(retryable, body["error"]["retryable"])
        detail = await self.store.get_rag_run_detail(uuid.UUID(body["ragRunId"]))
        self.assertEqual(AnswerStatus.ERROR, detail.run.status)
        self.assertEqual(error_code, detail.run.error_code)
        self.assertEqual(purposes, [call.purpose for call in detail.model_calls])
        self.assertEqual(statuses, [call.status for call in detail.model_calls])
        self.assertEqual(retrieval_count, len(detail.retrieval_results))
        self.assertEqual([], detail.citations)


if __name__ == "__main__":
    unittest.main()
