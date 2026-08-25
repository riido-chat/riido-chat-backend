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
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.config import get_settings
from app.database.models import (
    AnswerStatus,
    ChunkingConfig,
    ContentNode,
    ConversationStatus,
    DocumentChunk,
    DocumentSource,
    DocumentVersion,
    DocumentVersionStatus,
    EmbeddingConfig,
    ExecutionStatus,
    IndexVersion,
    IndexVersionStatus,
    ModelCallPurpose,
    RetrieverType,
)
from app.rag.log_store import (
    CitationLog,
    RagLogStore,
    RetrievalCandidateLog,
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
                "docker compose up -d && alembic upgrade head 후 재실행하세요."
            )

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(self.database_url)
        self.connection = await self.engine.connect()
        self.transaction = await self.connection.begin()
        self.session = AsyncSession(bind=self.connection, expire_on_commit=False)
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

    async def test_full_turn_lifecycle_completed(self) -> None:
        conversation = await self.store.create_conversation()
        self.assertEqual(ConversationStatus.ACTIVE, conversation.status)

        run = await self.store.start_rag_run(
            conversation.id,
            user_query="멤버 초대는 어떻게 하나요?",
            index_version_id=self.index_version_id,
            context_strategy="WINDOW",
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
        await self.store.record_model_call(
            rag_run_id=run.id,
            purpose=ModelCallPurpose.GENERATION.value,
            provider="openai",
            model_name="gpt-test",
            status=ExecutionStatus.SUCCESS,
            input_tokens=1000,
            output_tokens=300,
            latency_ms=1200,
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
        self.assertEqual(
            ModelCallPurpose.GENERATION,
            detail.model_calls[0].purpose,
        )
        self.assertEqual(1, len(detail.citations))
        self.assertEqual(1, detail.citations[0].citation_order)
        self.assertIsNone(detail.feedback)

    async def test_turn_numbers_increase_within_conversation(self) -> None:
        conversation = await self.store.create_conversation()
        first = await self.store.start_rag_run(
            conversation.id,
            user_query="첫 질문",
            index_version_id=self.index_version_id,
            context_strategy="WINDOW",
        )
        await self.store.withhold_rag_run(
            first.id, reason_code="INSUFFICIENT_EVIDENCE"
        )
        second = await self.store.start_rag_run(
            conversation.id,
            user_query="두 번째 질문",
            index_version_id=self.index_version_id,
            context_strategy="WINDOW",
        )

        self.assertEqual((1, 2), (first.turn_no, second.turn_no))
        runs = await self.store.list_conversation_runs(conversation.id)
        self.assertEqual([1, 2], [r.turn_no for r in runs])
        self.assertEqual(AnswerStatus.WITHHELD, runs[0].status)
        self.assertEqual("INSUFFICIENT_EVIDENCE", runs[0].withheld_reason_code)

    async def test_status_transitions_and_guards(self) -> None:
        conversation = await self.store.create_conversation()
        run = await self.store.start_rag_run(
            conversation.id,
            user_query="오류 케이스",
            index_version_id=self.index_version_id,
            context_strategy="FULL",
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
            context_strategy="FULL",
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

        with self.assertRaisesRegex(ValueError, "후속 질문"):
            await self.store.start_rag_run(
                conversation.id,
                user_query="닫힌 대화 질문",
                index_version_id=self.index_version_id,
                context_strategy="WINDOW",
            )

        expired = await self.store.create_conversation()
        await self.store.expire_conversation(expired.id)
        refreshed = await self.store.get_conversation(expired.id)
        self.assertEqual(ConversationStatus.EXPIRED, refreshed.status)
        self.assertIsNotNone(refreshed.closed_at)


if __name__ == "__main__":
    unittest.main()
