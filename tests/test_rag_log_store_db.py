"""RagLogStore 로컬 DB 통합 테스트.

- 로컬 PostgreSQL(docker compose)과 적용된 마이그레이션(head)이 필요하다.
  DB에 연결할 수 없으면 전체를 skip 한다.
- 모든 작업을 하나의 외부 트랜잭션 안에서 수행하고 마지막에 rollback 하므로
  로컬 DB에 데이터를 남기지 않는다.
- 적재 브리지 구현 전이므로 문서·청크·색인 체인은 시드 데이터로 직접 생성한다.

실행: python -m unittest tests.test_rag_log_store_db -v
"""

import asyncio
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import httpx
from openai import APITimeoutError
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.api.chat_schema import ChatCompletedResponse, ChatErrorResponse
from app.core.config import get_settings
from app.database.models import (
    AnswerStatus,
    ChunkingConfig,
    ContentNode,
    Conversation,
    ConversationStatus,
    ContextStrategy,
    DocumentChunk,
    DocumentSource,
    DocumentVersion,
    DocumentVersionStatus,
    EmbeddingConfig,
    ExecutionStatus,
    Feedback,
    FeedbackRating,
    IndexVersion,
    IndexVersionStatus,
    ModelCall,
    ModelCallPurpose,
    RagRun,
    RetrieverType,
)
from app.rag.chat_service import ChatService
from app.rag.generation_service import GenerationService
from app.rag.log_store import (
    CitationLog,
    CONVERSATION_EXPIRE_AFTER,
    ConversationBusyError,
    ConversationUnavailableError,
    FeedbackNotAllowedError,
    RagLogStore,
    RagRunNotFoundError,
    RetrievalCandidateLog,
)
from app.rag.model_trace import ModelCallTrace
from app.rag.query_rewrite import (
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


class RagLogStoreDbTest(unittest.IsolatedAsyncioTestCase):
    """대화 → 턴 → 후보·호출 → 답변·인용 → 조회의 전체 생명주기를 검증한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_check_database_available(cls.database_url)):
            raise unittest.SkipTest(
                "로컬 DB에 연결할 수 없어 통합 테스트를 건너뜁니다. "
                "docker compose -f docker-compose.db.yml up -d && "
                "alembic upgrade head 후 재실행하세요."
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

    async def asyncTearDown(self) -> None:
        await self.session.close()
        await self.transaction.rollback()
        await self.connection.close()
        await self.engine.dispose()

    async def _seed_content_chain(self) -> None:
        """문서 원본 → 버전 → 노드 → 청크 → 색인 버전 시드를 생성한다."""

        suffix = uuid.uuid4().hex[:8]

        chunking = ChunkingConfig(
            version=f"test-chunking-{suffix}",
            strategy="SECTION",
            max_tokens=512,
            created_at=_now(),
        )
        embedding = EmbeddingConfig(
            version=f"test-embedding-{suffix}",
            provider="openai",
            model_name="text-embedding-test",
            dimensions=1536,
            input_template_version="v1",
            created_at=_now(),
        )
        source = DocumentSource(
            source_type="LLMS_TXT",
            canonical_uri=f"https://docs.riido.io/test/{suffix}.md",
            title="테스트 문서",
            created_at=_now(),
            updated_at=_now(),
        )
        self.session.add_all([chunking, embedding, source])
        await self.session.flush()

        version = DocumentVersion(
            document_source_id=source.id,
            version_no=1,
            raw_content_uri=f"raw/test/{suffix}.md",
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
            node_path="테스트 문서>섹션",
            node_order=1,
            title="섹션",
            normalized_content="테스트 본문",
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
            version=f"test-index-{suffix}",
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

    # ------------------------------------------------------------------

    async def test_model_call_checkpoint_is_visible_between_transactions(
        self,
    ) -> None:
        """PROCESSING commit과 같은 행의 최종 갱신을 별도 session에서 확인한다."""

        suffix = uuid.uuid4().hex[:8]
        conversation_id = None
        index_version_id = None
        chunking_config_id = None
        embedding_config_id = None

        try:
            async with AsyncSession(
                bind=self.engine,
                expire_on_commit=False,
            ) as setup_session:
                chunking = ChunkingConfig(
                    version=f"checkpoint-chunking-{suffix}",
                    strategy="SECTION",
                    max_tokens=512,
                    created_at=_now(),
                )
                embedding = EmbeddingConfig(
                    version=f"checkpoint-embedding-{suffix}",
                    provider="openai",
                    model_name="text-embedding-test",
                    dimensions=1536,
                    input_template_version="v1",
                    created_at=_now(),
                )
                setup_session.add_all([chunking, embedding])
                await setup_session.flush()

                index_version = IndexVersion(
                    version=f"checkpoint-index-{suffix}",
                    status=IndexVersionStatus.INACTIVE,
                    chunking_config_id=chunking.id,
                    embedding_config_id=embedding.id,
                    created_at=_now(),
                )
                setup_session.add(index_version)
                await setup_session.flush()

                setup_store = RagLogStore(setup_session)
                conversation = await setup_store.create_conversation()
                run = await setup_store.start_rag_run(
                    conversation.id,
                    user_query="checkpoint 가시성 확인",
                    index_version_id=index_version.id,
                )
                conversation_id = conversation.id
                rag_run_id = run.id
                index_version_id = index_version.id
                chunking_config_id = chunking.id
                embedding_config_id = embedding.id
                await setup_session.commit()

            async with AsyncSession(
                bind=self.engine,
                expire_on_commit=False,
            ) as start_session:
                started = await RagLogStore(start_session).start_model_call(
                    rag_run_id=rag_run_id,
                    purpose=ModelCallPurpose.GENERATION.value,
                    provider="openai",
                    model_name="gpt-test",
                    prompt_version="v1",
                )
                model_call_id = started.id
                await start_session.commit()

            async with AsyncSession(bind=self.engine) as read_session:
                processing = await read_session.get(ModelCall, model_call_id)
                self.assertIsNotNone(processing)
                self.assertEqual(ExecutionStatus.PROCESSING, processing.status)
                self.assertIsNone(processing.latency_ms)

            # 최종 갱신 transaction이 실패하면 checkpoint 행은 보존돼야 한다.
            async with AsyncSession(bind=self.engine) as rollback_session:
                await RagLogStore(rollback_session).finish_model_call(
                    model_call_id,
                    status=ExecutionStatus.SUCCESS,
                    latency_ms=1200,
                )
                await rollback_session.rollback()

            async with AsyncSession(bind=self.engine) as read_session:
                processing = await read_session.get(ModelCall, model_call_id)
                self.assertEqual(ExecutionStatus.PROCESSING, processing.status)

            async with AsyncSession(bind=self.engine) as finish_session:
                finished = await RagLogStore(finish_session).finish_model_call(
                    model_call_id,
                    status=ExecutionStatus.SUCCESS,
                    input_tokens=1000,
                    output_tokens=300,
                    latency_ms=1200,
                    retry_count=1,
                )
                self.assertEqual(model_call_id, finished.id)
                await finish_session.commit()

            async with AsyncSession(bind=self.engine) as read_session:
                finished = await read_session.get(ModelCall, model_call_id)
                row_count = await read_session.scalar(
                    select(func.count(ModelCall.id)).where(
                        ModelCall.rag_run_id == rag_run_id,
                        ModelCall.purpose == ModelCallPurpose.GENERATION,
                    )
                )
                self.assertEqual(1, row_count)
                self.assertEqual(model_call_id, finished.id)
                self.assertEqual(ExecutionStatus.SUCCESS, finished.status)
                self.assertEqual(1, finished.retry_count)
        finally:
            async with AsyncSession(bind=self.engine) as cleanup_session:
                if conversation_id is not None:
                    await cleanup_session.execute(
                        delete(Conversation).where(Conversation.id == conversation_id)
                    )
                if index_version_id is not None:
                    await cleanup_session.execute(
                        delete(IndexVersion).where(IndexVersion.id == index_version_id)
                    )
                if chunking_config_id is not None:
                    await cleanup_session.execute(
                        delete(ChunkingConfig).where(
                            ChunkingConfig.id == chunking_config_id
                        )
                    )
                if embedding_config_id is not None:
                    await cleanup_session.execute(
                        delete(EmbeddingConfig).where(
                            EmbeddingConfig.id == embedding_config_id
                        )
                    )
                await cleanup_session.commit()

    # ------------------------------------------------------------------

    async def test_full_turn_lifecycle_completed(self) -> None:
        conversation = await self.store.create_conversation()
        self.assertEqual(ConversationStatus.ACTIVE, conversation.status)

        run = await self.store.start_rag_run(
            conversation.id,
            user_query="멤버 초대는 어떻게 하나요?",
            index_version_id=self.index_version_id,
        )
        self.assertEqual(1, run.turn_no)
        self.assertEqual(AnswerStatus.PROCESSING, run.status)
        self.assertIsNotNone(run.trace_id)

        await self.store.record_retrieval_results(
            run.id,
            [
                RetrievalCandidateLog(
                    chunk_id=self.chunk_id,
                    retriever_type=RetrieverType.BM25.value,
                    raw_score=11.5,
                    retriever_rank=1,
                    fused_rank=1,
                    fused_score=0.0328,
                    selected_as_evidence=True,
                ),
                RetrievalCandidateLog(
                    chunk_id=self.chunk_id,
                    retriever_type=RetrieverType.VECTOR.value,
                    raw_score=0.87,
                    retriever_rank=1,
                    fused_rank=2,
                    fused_score=0.0164,
                ),
            ],
        )
        model_call = await self.store.start_model_call(
            rag_run_id=run.id,
            purpose=ModelCallPurpose.GENERATION.value,
            provider="openai",
            model_name="gpt-test",
            prompt_version="v1",
        )
        model_call_id = model_call.id
        self.assertEqual(ExecutionStatus.PROCESSING, model_call.status)
        self.assertIsNone(model_call.input_tokens)
        self.assertIsNone(model_call.latency_ms)

        finished_model_call = await self.store.finish_model_call(
            model_call_id,
            status=ExecutionStatus.SUCCESS,
            input_tokens=1000,
            output_tokens=300,
            latency_ms=1200,
        )
        self.assertEqual(model_call_id, finished_model_call.id)
        self.assertEqual(ExecutionStatus.SUCCESS, finished_model_call.status)
        with self.assertRaisesRegex(ValueError, "PROCESSING"):
            await self.store.finish_model_call(
                model_call_id,
                status=ExecutionStatus.FAILED,
                error_message="중복 마감",
            )
        await self.store.complete_rag_run(
            run.id,
            answer_content="## 멤버 초대 방법\n\n1. 설정으로 이동합니다. [1]",
            citations=[
                CitationLog(
                    chunk_id=self.chunk_id,
                    document_version_id=self.document_version_id,
                    citation_order=1,
                    document_title_snapshot="테스트 문서",
                    node_path_snapshot="테스트 문서>섹션",
                    source_uri_snapshot="https://docs.riido.io/test.md",
                )
            ],
            total_latency_ms=1500,
        )

        detail = await self.store.get_rag_run_detail(run.id)
        self.assertIsNotNone(detail)
        self.assertEqual(AnswerStatus.COMPLETED, detail.run.status)
        self.assertTrue(detail.run.citation_validated)
        self.assertEqual(2, len(detail.retrieval_results))
        self.assertTrue(detail.retrieval_results[0].selected_as_evidence)
        self.assertEqual(
            [0.0328, 0.0164],
            [float(row.fused_score) for row in detail.retrieval_results],
        )
        self.assertEqual(1, len(detail.model_calls))
        self.assertEqual(model_call_id, detail.model_calls[0].id)
        self.assertEqual(
            ModelCallPurpose.GENERATION,
            detail.model_calls[0].purpose,
        )
        self.assertEqual(ExecutionStatus.SUCCESS, detail.model_calls[0].status)
        self.assertEqual(1, len(detail.citations))
        self.assertEqual(1, detail.citations[0].citation_order)
        self.assertIsNone(detail.feedback)

    async def test_generation_exception_preserves_retrieval_and_failed_call(
        self,
    ) -> None:
        chunk = RetrievalChunk(
            document_id="test-document",
            section_id="test-section",
            document_title="테스트 문서",
            section_path=("테스트 문서", "섹션"),
            source_url="https://docs.riido.io/test.md",
            category="guide",
            content="테스트 본문",
            chunk_id=self.chunk_id,
            document_version_id=self.document_version_id,
            index_version_id=self.index_version_id,
        )
        bm25_result = RetrievalResult(chunk=chunk, score=7.5, rank=1)
        vector_result = RetrievalResult(chunk=chunk, score=0.9, rank=1)
        fused_result = HybridRetrievalResult(
            chunk=chunk,
            rrf_score=0.032,
            final_rank=1,
            bm25_rank=1,
            vector_rank=1,
        )
        search = HybridSearchCall(
            bm25_results=(bm25_result,),
            vector_results=(vector_result,),
            fused_results=(fused_result,),
            bm25_latency_ms=10,
            vector_latency_ms=20,
            embedding_call=ModelCallTrace(
                provider="openai",
                model_name="text-embedding-test",
                succeeded=True,
                latency_ms=15,
                input_tokens=3,
            ),
        )

        retriever = AsyncMock(spec=HybridRetriever)

        async def search_after_checkpoint(_question, *, before_model_call):
            await before_model_call("openai", "text-embedding-test", None)
            return search

        retriever.search_with_trace.side_effect = search_after_checkpoint

        generation_service = AsyncMock(spec=GenerationService)

        async def raise_after_checkpoint(
            _question,
            _results,
            *,
            before_model_call,
        ):
            await before_model_call("openai", "gpt-test", "v1")
            raise RuntimeError("generation unavailable")

        generation_service.generate_answer.side_effect = raise_after_checkpoint
        service = ChatService(
            retriever=retriever,
            generation_service=generation_service,
            query_rewrite_service=AsyncMock(spec=QueryRewriteService),
            log_store=self.store,
            session=self.session,
            index_version_id=self.index_version_id,
        )

        response = await service.answer_question("실패 로그를 확인해줘")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertIsNotNone(response.conversation_id)
        self.assertIsNotNone(response.rag_run_id)
        detail = await self.store.get_rag_run_detail(response.rag_run_id)
        self.assertIsNotNone(detail)
        self.assertEqual(AnswerStatus.ERROR, detail.run.status)
        self.assertEqual("INTERNAL_ERROR", detail.run.error_code)
        self.assertEqual(2, len(detail.retrieval_results))
        self.assertTrue(
            all(row.selected_as_evidence for row in detail.retrieval_results)
        )
        self.assertEqual(2, len(detail.model_calls))
        generation_calls = [
            call
            for call in detail.model_calls
            if call.purpose == ModelCallPurpose.ANSWER_GENERATION
        ]
        self.assertEqual(1, len(generation_calls))
        self.assertEqual(ExecutionStatus.FAILED, generation_calls[0].status)
        self.assertEqual(
            "generation unavailable",
            generation_calls[0].error_message,
        )
        self.assertEqual(0, generation_calls[0].retry_count)
        self.assertEqual([], detail.citations)

    async def test_multi_turn_uses_resolved_query_and_persists_snapshot(self) -> None:
        chunk = RetrievalChunk(
            document_id="test-document",
            section_id="test-section",
            document_title="테스트 문서",
            section_path=("테스트 문서", "섹션"),
            source_url="https://docs.riido.io/test.md",
            category="guide",
            content="멤버 초대 방법",
            chunk_id=self.chunk_id,
            document_version_id=self.document_version_id,
            index_version_id=self.index_version_id,
        )
        bm25_result = RetrievalResult(chunk=chunk, score=7.5, rank=1)
        vector_result = RetrievalResult(chunk=chunk, score=0.9, rank=1)
        fused_result = HybridRetrievalResult(
            chunk=chunk,
            rrf_score=0.032,
            final_rank=1,
            bm25_rank=1,
            vector_rank=1,
        )
        search = HybridSearchCall(
            bm25_results=(bm25_result,),
            vector_results=(vector_result,),
            fused_results=(fused_result,),
            bm25_latency_ms=10,
            vector_latency_ms=20,
            embedding_call=ModelCallTrace(
                provider="openai",
                model_name="text-embedding-test",
                succeeded=True,
                latency_ms=15,
                input_tokens=3,
            ),
        )
        retrieval_queries = []
        generation_queries = []
        seen_candidates = []
        retriever = AsyncMock(spec=HybridRetriever)

        async def search_after_checkpoint(query, *, before_model_call):
            retrieval_queries.append(query)
            await before_model_call("openai", "text-embedding-test", None)
            return search

        retriever.search_with_trace.side_effect = search_after_checkpoint
        generation_service = AsyncMock(spec=GenerationService)

        async def generate_after_checkpoint(
            query,
            _results,
            *,
            before_model_call,
        ):
            generation_queries.append(query)
            await before_model_call("openai", "gpt-test", "v3")
            return FinalGenerationResult(
                status=FinalAnswerStatus.COMPLETED,
                answer_markdown="멤버를 초대할 수 있습니다. [1]",
                citations=(
                    Citation(
                        citation_number=1,
                        document_title="테스트 문서",
                        section_path=("테스트 문서", "섹션"),
                        source_url="https://docs.riido.io/test.md",
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

        generation_service.generate_answer.side_effect = generate_after_checkpoint
        query_rewrite_service = AsyncMock(spec=QueryRewriteService)

        async def rewrite_after_checkpoint(
            _question,
            candidates,
            *,
            before_model_call,
        ):
            seen_candidates.extend(candidates)
            await before_model_call(
                "openai",
                "gpt-5.4-mini",
                QUERY_REWRITE_PROMPT_VERSION,
            )
            return QueryRewriteCall(
                trace=ModelCallTrace(
                    provider="openai",
                    model_name="gpt-5.4-mini",
                    succeeded=True,
                    latency_ms=25,
                    input_tokens=30,
                    output_tokens=8,
                    prompt_version=QUERY_REWRITE_PROMPT_VERSION,
                ),
                resolution=QueryResolution(
                    decision=QueryRewriteDecision.FOLLOW_UP_RESOLVED,
                    resolved_query="리두 멤버 초대 권한 설정 방법",
                    selected_turns=(candidates[0],),
                ),
            )

        query_rewrite_service.rewrite.side_effect = rewrite_after_checkpoint
        service = ChatService(
            retriever=retriever,
            generation_service=generation_service,
            query_rewrite_service=query_rewrite_service,
            log_store=self.store,
            session=self.session,
            index_version_id=self.index_version_id,
        )

        first = await service.answer_question("멤버를 어떻게 초대해?")
        second = await service.answer_question(
            "권한은 어떻게 설정해?",
            first.conversation_id,
        )

        self.assertIsInstance(first, ChatCompletedResponse)
        self.assertIsInstance(second, ChatCompletedResponse)
        self.assertEqual(
            ["멤버를 어떻게 초대해?", "리두 멤버 초대 권한 설정 방법"],
            retrieval_queries,
        )
        self.assertEqual(retrieval_queries, generation_queries)
        self.assertEqual([1], [candidate.turn_no for candidate in seen_candidates])
        self.assertEqual(
            "멤버를 어떻게 초대해?",
            seen_candidates[0].user_query,
        )
        self.assertEqual(
            "멤버를 초대할 수 있습니다. [1]",
            seen_candidates[0].answer_content,
        )

        detail = await self.store.get_rag_run_detail(second.rag_run_id)
        self.assertEqual(AnswerStatus.COMPLETED, detail.run.status)
        self.assertEqual(
            "리두 멤버 초대 권한 설정 방법",
            detail.run.resolved_query,
        )
        self.assertEqual(
            ContextStrategy.FOLLOW_UP_WINDOW,
            detail.run.context_strategy,
        )
        self.assertEqual(1, detail.run.context_turn_count)
        self.assertEqual("v1", detail.run.context_snapshot["schemaVersion"])
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

    async def test_embedding_final_failure_persists_retry_count(self) -> None:
        error = APITimeoutError(
            httpx.Request("POST", "https://api.openai.com")
        )
        failed_search = HybridSearchCall(
            embedding_call=ModelCallTrace(
                provider="openai",
                model_name="text-embedding-test",
                succeeded=False,
                latency_ms=3500,
                retry_count=2,
                error_message="embedding unavailable",
            ),
            error=error,
        )
        retriever = AsyncMock(spec=HybridRetriever)

        async def search_after_checkpoint(_question, *, before_model_call):
            await before_model_call("openai", "text-embedding-test", None)
            return failed_search

        retriever.search_with_trace.side_effect = search_after_checkpoint
        generation_service = AsyncMock(spec=GenerationService)
        service = ChatService(
            retriever=retriever,
            generation_service=generation_service,
            query_rewrite_service=AsyncMock(spec=QueryRewriteService),
            log_store=self.store,
            session=self.session,
            index_version_id=self.index_version_id,
        )

        response = await service.answer_question("Embedding 실패를 확인해줘")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertIsNotNone(response.conversation_id)
        self.assertIsNotNone(response.rag_run_id)
        generation_service.generate_answer.assert_not_awaited()
        detail = await self.store.get_rag_run_detail(response.rag_run_id)
        self.assertIsNotNone(detail)
        self.assertEqual(AnswerStatus.ERROR, detail.run.status)
        self.assertEqual("UPSTREAM_ERROR", detail.run.error_code)
        self.assertEqual("UPSTREAM_ERROR", response.error.code.value)
        self.assertTrue(response.error.retryable)
        self.assertEqual([], detail.retrieval_results)
        self.assertEqual(1, len(detail.model_calls))
        embedding_call = detail.model_calls[0]
        self.assertEqual(ModelCallPurpose.QUERY_EMBEDDING, embedding_call.purpose)
        self.assertEqual(ExecutionStatus.FAILED, embedding_call.status)
        self.assertEqual(3500, embedding_call.latency_ms)
        self.assertEqual(2, embedding_call.retry_count)
        self.assertEqual("embedding unavailable", embedding_call.error_message)

    async def test_turn_numbers_increase_within_conversation(self) -> None:
        conversation = await self.store.create_conversation()
        first = await self.store.start_rag_run(
            conversation.id,
            user_query="첫 질문",
            index_version_id=self.index_version_id,
        )
        await self.store.withhold_rag_run(
            first.id, reason_code="INSUFFICIENT_EVIDENCE"
        )
        second = await self.store.start_rag_run(
            conversation.id,
            user_query="두 번째 질문",
            index_version_id=self.index_version_id,
        )

        self.assertEqual((1, 2), (first.turn_no, second.turn_no))
        self.assertEqual(ContextStrategy.NEW_TOPIC, first.context_strategy)
        self.assertEqual("첫 질문", first.resolved_query)
        self.assertEqual(ContextStrategy.UNRESOLVED, second.context_strategy)
        self.assertIsNone(second.resolved_query)
        runs = await self.store.list_conversation_runs(conversation.id)
        self.assertEqual([1, 2], [r.turn_no for r in runs])
        self.assertEqual(AnswerStatus.WITHHELD, runs[0].status)
        self.assertEqual("INSUFFICIENT_EVIDENCE", runs[0].withheld_reason_code)
        # 인용 검증에 도달하지 못한 보류는 판정 자체가 없다
        self.assertIsNone(runs[0].citation_validated)

    async def test_query_rewrite_candidates_use_latest_five_valid_turns(self) -> None:
        conversation = await self.store.create_conversation()
        previous_runs = []
        for turn_no in range(1, 7):
            run = await self.store.start_rag_run(
                conversation.id,
                user_query=f"이전 질문 {turn_no}",
                index_version_id=self.index_version_id,
            )
            previous_runs.append(run)
            if turn_no % 2 == 0:
                await self.store.complete_rag_run(
                    run.id,
                    answer_content=f"이전 답변 {turn_no}",
                    citations=[
                        CitationLog(
                            chunk_id=self.chunk_id,
                            document_version_id=self.document_version_id,
                            citation_order=1,
                        )
                    ],
                )
            else:
                await self.store.withhold_rag_run(
                    run.id,
                    reason_code="OUT_OF_SCOPE",
                )

        failed = await self.store.start_rag_run(
            conversation.id,
            user_query="실패한 질문",
            index_version_id=self.index_version_id,
        )
        await self.store.fail_rag_run(failed.id, error_code="INTERNAL_ERROR")
        cancelled = await self.store.start_rag_run(
            conversation.id,
            user_query="취소된 질문",
            index_version_id=self.index_version_id,
        )
        await self.store.cancel_rag_run(cancelled.id)
        current = await self.store.start_rag_run(
            conversation.id,
            user_query="현재 후속 질문",
            index_version_id=self.index_version_id,
        )

        candidates = await self.store.get_query_rewrite_candidates(current.id)

        self.assertEqual([2, 3, 4, 5, 6], [turn.turn_no for turn in candidates])
        self.assertEqual(
            [
                QueryRewriteTurnStatus.COMPLETED,
                QueryRewriteTurnStatus.WITHHELD,
                QueryRewriteTurnStatus.COMPLETED,
                QueryRewriteTurnStatus.WITHHELD,
                QueryRewriteTurnStatus.COMPLETED,
            ],
            [turn.status for turn in candidates],
        )
        self.assertEqual("이전 답변 2", candidates[0].answer_content)
        self.assertIsNone(candidates[0].withheld_reason_code)
        self.assertIsNone(candidates[1].answer_content)
        self.assertEqual(
            "OUT_OF_SCOPE",
            candidates[1].withheld_reason_code.value,
        )
        self.assertNotIn(1, [turn.turn_no for turn in candidates])
        self.assertNotIn(failed.id, [turn.rag_run_id for turn in candidates])
        self.assertNotIn(cancelled.id, [turn.rag_run_id for turn in candidates])

    async def test_records_v1_query_resolution_snapshot_consistently(self) -> None:
        conversation = await self.store.create_conversation()
        previous = await self.store.start_rag_run(
            conversation.id,
            user_query="이전 질문",
            index_version_id=self.index_version_id,
        )
        await self.store.withhold_rag_run(
            previous.id,
            reason_code="INSUFFICIENT_EVIDENCE",
        )
        current = await self.store.start_rag_run(
            conversation.id,
            user_query="그건 어떻게 해?",
            index_version_id=self.index_version_id,
        )
        snapshot = {
            "schemaVersion": "v1",
            "selectedTurns": [
                {
                    "ragRunId": str(previous.id),
                    "turnNo": previous.turn_no,
                    "status": "WITHHELD",
                    "userQuery": previous.user_query,
                    "answerContent": None,
                    "withheldReasonCode": "INSUFFICIENT_EVIDENCE",
                }
            ],
        }

        resolved = await self.store.record_query_resolution(
            current.id,
            resolved_query="이전 질문의 구체적인 처리 방법",
            context_strategy=ContextStrategy.FOLLOW_UP_WINDOW,
            context_turn_count=1,
            context_snapshot=snapshot,
        )

        self.assertEqual(
            "이전 질문의 구체적인 처리 방법",
            resolved.resolved_query,
        )
        self.assertEqual(ContextStrategy.FOLLOW_UP_WINDOW, resolved.context_strategy)
        self.assertEqual(1, resolved.context_turn_count)
        self.assertEqual(snapshot, resolved.context_snapshot)

        with self.assertRaisesRegex(ValueError, "selectedTurns 길이"):
            await self.store.record_query_resolution(
                current.id,
                resolved_query="잘못된 기록",
                context_strategy=ContextStrategy.FOLLOW_UP_WINDOW,
                context_turn_count=2,
                context_snapshot=snapshot,
            )

    async def test_fresh_run_is_busy_then_stale_run_and_calls_are_recovered(
        self,
    ) -> None:
        conversation = await self.store.create_conversation()
        stale_run = await self.store.start_rag_run(
            conversation.id,
            user_query="오래 걸린 질문",
            index_version_id=self.index_version_id,
        )
        processing_call = await self.store.start_model_call(
            rag_run_id=stale_run.id,
            purpose=ModelCallPurpose.EMBEDDING.value,
            provider="openai",
            model_name="text-embedding-test",
        )
        successful_call = await self.store.start_model_call(
            rag_run_id=stale_run.id,
            purpose=ModelCallPurpose.GENERATION.value,
            provider="openai",
            model_name="gpt-test",
        )
        await self.store.finish_model_call(
            successful_call.id,
            status=ExecutionStatus.SUCCESS,
        )
        last_active_at = conversation.last_active_at

        with self.assertRaises(ConversationBusyError):
            await self.store.start_rag_run(
                conversation.id,
                user_query="동시 질문",
                index_version_id=self.index_version_id,
            )

        self.assertEqual(last_active_at, conversation.last_active_at)
        self.assertEqual(
            1,
            await self.session.scalar(
                select(func.count(RagRun.id)).where(
                    RagRun.conversation_id == conversation.id
                )
            ),
        )
        self.assertEqual(AnswerStatus.PROCESSING, stale_run.status)
        self.assertEqual(ExecutionStatus.PROCESSING, processing_call.status)

        stale_run.created_at = _now() - timedelta(minutes=11)
        await self.session.flush()
        next_run = await self.store.start_rag_run(
            conversation.id,
            user_query="복구 뒤 질문",
            index_version_id=self.index_version_id,
        )

        self.assertEqual(2, next_run.turn_no)
        self.assertEqual(ContextStrategy.UNRESOLVED, next_run.context_strategy)
        self.assertEqual(AnswerStatus.ERROR, stale_run.status)
        self.assertEqual("INTERNAL_ERROR", stale_run.error_code)
        self.assertIsNotNone(stale_run.completed_at)
        self.assertEqual(ExecutionStatus.FAILED, processing_call.status)
        self.assertIn("stale recovery", processing_call.error_message)
        self.assertEqual(ExecutionStatus.SUCCESS, successful_call.status)

        # 복구 뒤 늦게 돌아온 기존 worker는 ERROR run이나 ModelCall을 되살릴 수 없다.
        with self.assertRaisesRegex(ValueError, "PROCESSING"):
            await self.store.finish_model_call(
                processing_call.id,
                status=ExecutionStatus.SUCCESS,
            )
        with self.assertRaisesRegex(ValueError, "PROCESSING"):
            await self.store.withhold_rag_run(
                stale_run.id,
                reason_code="INSUFFICIENT_EVIDENCE",
            )

    async def test_failure_recovery_closes_processing_model_calls(self) -> None:
        conversation = await self.store.create_conversation()
        run = await self.store.start_rag_run(
            conversation.id,
            user_query="마감 실패 질문",
            index_version_id=self.index_version_id,
        )
        processing_call = await self.store.start_model_call(
            rag_run_id=run.id,
            purpose=ModelCallPurpose.GENERATION.value,
            provider="openai",
            model_name="gpt-test",
        )

        failed_calls = await self.store.fail_processing_model_calls(run.id)
        failed_run = await self.store.fail_rag_run(
            run.id,
            error_code="INTERNAL_ERROR",
        )

        self.assertEqual([processing_call.id], [call.id for call in failed_calls])
        self.assertEqual(ExecutionStatus.FAILED, processing_call.status)
        self.assertIn("RagRun 실패 복구", processing_call.error_message)
        self.assertEqual(AnswerStatus.ERROR, failed_run.status)

    async def test_records_citation_validation_only_when_it_ran(self) -> None:
        conversation = await self.store.create_conversation()
        run = await self.store.start_rag_run(
            conversation.id,
            user_query="인용 검증 실패",
            index_version_id=self.index_version_id,
        )

        withheld = await self.store.withhold_rag_run(
            run.id, reason_code="UNVERIFIABLE_ANSWER"
        )

        self.assertFalse(withheld.citation_validated)

    async def test_feedback_registers_changes_and_clears(self) -> None:
        run = await self._withheld_run("피드백 대상")

        registered = await self.store.set_feedback(
            run.id, rating=FeedbackRating.GOOD
        )
        self.assertEqual(FeedbackRating.GOOD, registered.rating)
        self.assertEqual(registered.created_at, registered.updated_at)

        # 같은 값 재전송은 무시하므로 updated_at이 그대로다
        stamped_at = registered.updated_at
        resent = await self.store.set_feedback(
            run.id, rating=FeedbackRating.GOOD
        )
        self.assertEqual(stamped_at, resent.updated_at)

        changed = await self.store.set_feedback(
            run.id, rating=FeedbackRating.BAD
        )
        self.assertEqual(FeedbackRating.BAD, changed.rating)
        self.assertGreater(changed.updated_at, changed.created_at)

        detail = await self.store.get_rag_run_detail(run.id)
        self.assertEqual(FeedbackRating.BAD, detail.feedback.rating)

        self.assertTrue(await self.store.clear_feedback(run.id))
        self.assertFalse(await self.store.clear_feedback(run.id))
        self.assertEqual(
            0,
            await self.session.scalar(
                select(func.count())
                .select_from(Feedback)
                .where(Feedback.rag_run_id == run.id)
            ),
        )

    async def test_feedback_keeps_one_row_per_answer(self) -> None:
        first = await self._withheld_run("첫 답변")
        second = await self._withheld_run("두 번째 답변")

        await self.store.set_feedback(first.id, rating=FeedbackRating.GOOD)
        await self.store.set_feedback(first.id, rating=FeedbackRating.BAD)
        await self.store.set_feedback(second.id, rating=FeedbackRating.GOOD)

        self.assertEqual(
            2,
            await self.session.scalar(
                select(func.count()).select_from(Feedback)
            ),
        )

    async def test_feedback_rejects_turns_without_a_rateable_answer(self) -> None:
        conversation = await self.store.create_conversation()
        run = await self.store.start_rag_run(
            conversation.id,
            user_query="처리 중인 턴",
            index_version_id=self.index_version_id,
        )

        # PROCESSING 상태에서는 평가할 답변이 아직 없다
        with self.assertRaises(FeedbackNotAllowedError):
            await self.store.set_feedback(run.id, rating=FeedbackRating.GOOD)

        await self.store.fail_rag_run(run.id, error_code="UPSTREAM_ERROR")
        with self.assertRaises(FeedbackNotAllowedError):
            await self.store.set_feedback(run.id, rating=FeedbackRating.GOOD)
        with self.assertRaises(FeedbackNotAllowedError):
            await self.store.clear_feedback(run.id)

        with self.assertRaises(RagRunNotFoundError):
            await self.store.set_feedback(
                uuid.uuid4(), rating=FeedbackRating.GOOD
            )

    async def _withheld_run(self, user_query: str):
        """평가 가능한 상태로 마감한 턴을 만든다."""

        conversation = await self.store.create_conversation()
        run = await self.store.start_rag_run(
            conversation.id,
            user_query=user_query,
            index_version_id=self.index_version_id,
        )
        await self.store.withhold_rag_run(
            run.id, reason_code="INSUFFICIENT_EVIDENCE"
        )
        return run

    async def test_status_transitions_and_guards(self) -> None:
        conversation = await self.store.create_conversation()
        run = await self.store.start_rag_run(
            conversation.id,
            user_query="오류 케이스",
            index_version_id=self.index_version_id,
        )
        failed = await self.store.fail_rag_run(run.id, error_code="UPSTREAM_ERROR")
        self.assertEqual(AnswerStatus.ERROR, failed.status)
        self.assertEqual("UPSTREAM_ERROR", failed.error_code)

        # 종료된 턴은 다시 전이할 수 없다
        with self.assertRaisesRegex(ValueError, "PROCESSING"):
            await self.store.cancel_rag_run(run.id)

        # 잘못된 보류 사유는 거부한다
        run2 = await self.store.start_rag_run(
            conversation.id,
            user_query="보류 검증",
            index_version_id=self.index_version_id,
        )
        with self.assertRaisesRegex(ValueError, "보류 사유"):
            await self.store.withhold_rag_run(run2.id, reason_code="UNKNOWN")

        # COMPLETED는 인용 1~3개를 강제한다
        with self.assertRaisesRegex(ValueError, "1~3개"):
            await self.store.complete_rag_run(
                run2.id, answer_content="본문", citations=[]
            )

    async def test_closed_conversation_rejects_new_turns(self) -> None:
        conversation = await self.store.create_conversation()
        await self.store.close_conversation(conversation.id)

        with self.assertRaisesRegex(ConversationUnavailableError, "후속 질문"):
            await self.store.start_rag_run(
                conversation.id,
                user_query="닫힌 대화 질문",
                index_version_id=self.index_version_id,
            )

        expired = await self.store.create_conversation()
        await self.store.expire_conversation(expired.id)
        refreshed = await self.store.get_conversation(expired.id)
        self.assertEqual(ConversationStatus.EXPIRED, refreshed.status)
        self.assertIsNotNone(refreshed.closed_at)

    async def test_inactive_conversation_expires_lazily_on_next_turn(self) -> None:
        conversation = await self.store.create_conversation()
        conversation.last_active_at = _now() - CONVERSATION_EXPIRE_AFTER - timedelta(
            seconds=1
        )
        await self.session.flush()

        with self.assertRaisesRegex(ConversationUnavailableError, "후속 질문"):
            await self.store.start_rag_run(
                conversation.id,
                user_query="24시간 지난 대화 질문",
                index_version_id=self.index_version_id,
            )

        refreshed = await self.store.get_conversation(conversation.id)
        self.assertEqual(ConversationStatus.EXPIRED, refreshed.status)
        self.assertIsNotNone(refreshed.closed_at)

    async def test_recently_active_conversation_is_not_expired(self) -> None:
        conversation = await self.store.create_conversation()
        conversation.last_active_at = _now() - CONVERSATION_EXPIRE_AFTER + timedelta(
            seconds=1
        )
        await self.session.flush()

        run = await self.store.start_rag_run(
            conversation.id,
            user_query="23시간대 후속 질문",
            index_version_id=self.index_version_id,
        )

        self.assertEqual(1, run.turn_no)
        refreshed = await self.store.get_conversation(conversation.id)
        self.assertEqual(ConversationStatus.ACTIVE, refreshed.status)


class RagLogStoreConcurrencyDbTest(unittest.IsolatedAsyncioTestCase):
    """독립 PostgreSQL connection으로 Conversation row lock의 범위를 검증한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_check_database_available(cls.database_url)):
            raise unittest.SkipTest(
                "로컬 DB에 연결할 수 없어 동시성 테스트를 건너뜁니다."
            )

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(self.database_url)
        suffix = uuid.uuid4().hex[:8]

        async with AsyncSession(
            bind=self.engine,
            expire_on_commit=False,
        ) as session:
            chunking = ChunkingConfig(
                version=f"concurrency-chunking-{suffix}",
                strategy="SECTION",
                max_tokens=512,
                created_at=_now(),
            )
            embedding = EmbeddingConfig(
                version=f"concurrency-embedding-{suffix}",
                provider="openai",
                model_name="text-embedding-test",
                dimensions=1536,
                input_template_version="v1",
                created_at=_now(),
            )
            session.add_all([chunking, embedding])
            await session.flush()

            index_version = IndexVersion(
                version=f"concurrency-index-{suffix}",
                status=IndexVersionStatus.INACTIVE,
                chunking_config_id=chunking.id,
                embedding_config_id=embedding.id,
                created_at=_now(),
            )
            session.add(index_version)
            await session.flush()

            store = RagLogStore(session)
            first_conversation = await store.create_conversation()
            second_conversation = await store.create_conversation()
            self.index_version_id = index_version.id
            self.chunking_config_id = chunking.id
            self.embedding_config_id = embedding.id
            self.conversation_ids = (
                first_conversation.id,
                second_conversation.id,
            )
            await session.commit()

    async def asyncTearDown(self) -> None:
        async with AsyncSession(bind=self.engine) as session:
            await session.execute(
                delete(Conversation).where(
                    Conversation.id.in_(self.conversation_ids)
                )
            )
            await session.execute(
                delete(IndexVersion).where(IndexVersion.id == self.index_version_id)
            )
            await session.execute(
                delete(ChunkingConfig).where(
                    ChunkingConfig.id == self.chunking_config_id
                )
            )
            await session.execute(
                delete(EmbeddingConfig).where(
                    EmbeddingConfig.id == self.embedding_config_id
                )
            )
            await session.commit()
        await self.engine.dispose()

    async def test_same_conversation_concurrent_start_creates_one_run(self) -> None:
        conversation_id = self.conversation_ids[0]
        first_session = AsyncSession(bind=self.engine, expire_on_commit=False)
        second_session = AsyncSession(bind=self.engine, expire_on_commit=False)
        second_task = None

        try:
            first_run = await RagLogStore(first_session).start_rag_run(
                conversation_id,
                user_query="첫 동시 요청",
                index_version_id=self.index_version_id,
            )
            second_backend_pid = await second_session.scalar(
                text("SELECT pg_backend_pid()")
            )

            second_task = asyncio.create_task(
                RagLogStore(second_session).start_rag_run(
                    conversation_id,
                    user_query="두 번째 동시 요청",
                    index_version_id=self.index_version_id,
                )
            )
            await self._wait_until_backend_waits_for_lock(second_backend_pid)
            self.assertFalse(second_task.done())

            await first_session.commit()
            with self.assertRaises(ConversationBusyError):
                await asyncio.wait_for(second_task, timeout=3)
            await second_session.rollback()

            async with AsyncSession(bind=self.engine) as read_session:
                runs = list(
                    (
                        await read_session.scalars(
                            select(RagRun).where(
                                RagRun.conversation_id == conversation_id
                            )
                        )
                    ).all()
                )
            self.assertEqual(1, len(runs))
            self.assertEqual(first_run.id, runs[0].id)
            self.assertEqual(1, runs[0].turn_no)
        finally:
            if second_task is not None and not second_task.done():
                second_task.cancel()
                await asyncio.gather(second_task, return_exceptions=True)
            await first_session.rollback()
            await second_session.rollback()
            await first_session.close()
            await second_session.close()

    async def test_stale_recovery_rejects_cached_late_worker_state(self) -> None:
        conversation_id = self.conversation_ids[0]
        worker_session = AsyncSession(bind=self.engine, expire_on_commit=False)

        try:
            worker_store = RagLogStore(worker_session)
            stale_run = await worker_store.start_rag_run(
                conversation_id,
                user_query="stale 작업",
                index_version_id=self.index_version_id,
            )
            stale_call = await worker_store.start_model_call(
                rag_run_id=stale_run.id,
                purpose=ModelCallPurpose.GENERATION.value,
                provider="openai",
                model_name="gpt-test",
            )
            stale_run_id = stale_run.id
            stale_call_id = stale_call.id
            stale_run.created_at = _now() - timedelta(minutes=11)
            await worker_session.commit()

            async with AsyncSession(
                bind=self.engine,
                expire_on_commit=False,
            ) as recovery_session:
                next_run = await RagLogStore(recovery_session).start_rag_run(
                    conversation_id,
                    user_query="복구 뒤 요청",
                    index_version_id=self.index_version_id,
                )
                await recovery_session.commit()

            # expire_on_commit=False라 worker에는 복구 전 PROCESSING 값이 남아 있다.
            self.assertEqual(AnswerStatus.PROCESSING, stale_run.status)
            self.assertEqual(ExecutionStatus.PROCESSING, stale_call.status)
            with self.assertRaisesRegex(ValueError, "PROCESSING"):
                await worker_store.finish_model_call(
                    stale_call_id,
                    status=ExecutionStatus.SUCCESS,
                )
            await worker_session.rollback()

            async with AsyncSession(bind=self.engine) as read_session:
                persisted_run = await read_session.get(RagRun, stale_run_id)
                persisted_call = await read_session.get(ModelCall, stale_call_id)
            self.assertEqual(AnswerStatus.ERROR, persisted_run.status)
            self.assertEqual(ExecutionStatus.FAILED, persisted_call.status)
            self.assertEqual(2, next_run.turn_no)
        finally:
            await worker_session.rollback()
            await worker_session.close()

    async def test_different_conversations_do_not_share_a_lock(self) -> None:
        first_session = AsyncSession(bind=self.engine, expire_on_commit=False)
        second_session = AsyncSession(bind=self.engine, expire_on_commit=False)

        try:
            first_run = await RagLogStore(first_session).start_rag_run(
                self.conversation_ids[0],
                user_query="첫 대화 질문",
                index_version_id=self.index_version_id,
            )
            second_run = await asyncio.wait_for(
                RagLogStore(second_session).start_rag_run(
                    self.conversation_ids[1],
                    user_query="둘째 대화 질문",
                    index_version_id=self.index_version_id,
                ),
                timeout=3,
            )
            await second_session.commit()
            await first_session.commit()

            self.assertEqual(1, first_run.turn_no)
            self.assertEqual(1, second_run.turn_no)
            async with AsyncSession(bind=self.engine) as read_session:
                count = await read_session.scalar(
                    select(func.count(RagRun.id)).where(
                        RagRun.conversation_id.in_(self.conversation_ids)
                    )
                )
            self.assertEqual(2, count)
        finally:
            await first_session.rollback()
            await second_session.rollback()
            await first_session.close()
            await second_session.close()

    async def _wait_until_backend_waits_for_lock(self, backend_pid: int) -> None:
        deadline = asyncio.get_running_loop().time() + 3
        async with AsyncSession(bind=self.engine) as monitor_session:
            while True:
                wait_event_type = await monitor_session.scalar(
                    text(
                        "SELECT wait_event_type FROM pg_stat_activity "
                        "WHERE pid = :backend_pid"
                    ),
                    {"backend_pid": backend_pid},
                )
                if wait_event_type == "Lock":
                    return
                if asyncio.get_running_loop().time() >= deadline:
                    self.fail("두 번째 요청이 Conversation row lock 대기에 진입하지 않았습니다.")
                await asyncio.sleep(0.01)


if __name__ == "__main__":
    unittest.main()
