"""서비스 B 로컬 DB 검색 경로가 서비스 A 문서 근거를 섞지 않는지 확인한다.

각 실행은 합성 문서만 담은 일회용 PostgreSQL DB를 사용한다. 질의 임베딩과
답변 생성은 로컬 stub이며 외부 모델 API를 호출하지 않는다.
"""

import asyncio
import os
import subprocess
import sys
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.answering.generator import GENERATION_PROMPT_VERSION
from app.answering.models import Citation, CitationSourceKind, FinalAnswerStatus, FinalGenerationResult
from app.answering.service import GenerationService
from app.chat.query_rewrite import (
    OPENAI_QUERY_REWRITE_MODEL,
    QUERY_REWRITE_PROMPT_VERSION,
    QueryResolution,
    QueryRewriteCall,
    QueryRewriteDecision,
    QueryRewriteService,
)
from app.core.config import get_settings
from app.core.model_trace import ModelCallTrace
from app.database.models import (
    ChatProfile,
    ChatProfileRevision,
    ChatProfileRevisionStatus,
    ChunkEmbedding,
    ChunkingConfig,
    ContentNode,
    DocumentChunk,
    DocumentGroup,
    DocumentSource,
    DocumentVersion,
    DocumentVersionStatus,
    EmbeddingConfig,
    IndexDocument,
    IndexVersion,
    IndexVersionStatus,
)
from app.database.session import get_db_session
from app.main import create_app
from app.profiles.services import HELP_CHATBOT, HELP_CHATBOT_TEST
from app.retrieval.corpus_state import CorpusRegistry, CorpusState
from app.retrieval.embedding import EmbeddingResponse, OPENAI_EMBEDDING_DIMENSIONS
from app.retrieval.search_reader import SearchReader
from tests.profile_components import component_models


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATION_MODEL = "gpt-test-generation"
SYNTHETIC_EXAMPLES = {
    "public": ("민들레 재고 안내", "민들레 창고의 재고 확인 절차는 매주 월요일 점검표를 기록하는 것입니다."),
    "service_b": ("청록 달팽이 동기화 안내", "청록 달팽이 자료 동기화는 합성 규칙에 따라 저녁마다 실행합니다."),
}
SHARED_SYNTHETIC_DOCUMENT = (
    "공통 설정 안내",
    "합성 환경의 공통 설정은 관리 화면에서 옵션을 저장하는 방식입니다.",
)


async def _available(url: str) -> bool:
    engine = create_async_engine(url)
    try:
        async with engine.connect():
            return True
    except Exception:
        return False
    finally:
        await engine.dispose()


def _alembic(url: str, *args: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO_ROOT,
        env={**os.environ, "DATABASE_URL": url},
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise AssertionError(f"alembic {' '.join(args)} 실패:\n{result.stderr}")


class _LocalEmbedder:
    """외부 호출 없이 동일한 query vector를 반환한다."""

    def embed_many_with_usage(self, texts, *, sdk_max_retries=None, timeout=None):
        if sdk_max_retries != 0:
            raise AssertionError("query embedding retry 설정이 예상과 다릅니다.")
        return EmbeddingResponse(
            embeddings=[[0.1] * OPENAI_EMBEDDING_DIMENSIONS for _ in texts],
            input_tokens=len(texts),
        )


class ServiceBDocumentsDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_available(cls.database_url)):
            raise unittest.SkipTest("로컬 PostgreSQL에 연결할 수 없습니다.")

    async def asyncSetUp(self) -> None:
        self.database_name = f"riido_service_b_{uuid.uuid4().hex[:12]}"
        self.scratch_url = make_url(self.database_url).set(
            database=self.database_name
        ).render_as_string(hide_password=False)
        await self._maintenance(f'CREATE DATABASE "{self.database_name}"')
        self.engine = None
        self.addAsyncCleanup(self._cleanup_scratch_database)
        self.engine = create_async_engine(self.scratch_url)
        _alembic(self.scratch_url, "upgrade", "head")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.registry = CorpusRegistry()
        async with self.sessions() as session:
            migrated = await session.scalar(
                select(ChatProfileRevision).where(
                    ChatProfileRevision.status == ChatProfileRevisionStatus.PUBLISHED
                )
            )
            migrated.status = ChatProfileRevisionStatus.RETIRED
            await session.flush()
            public_group = await session.get(DocumentGroup, migrated.document_group_id)
            self.public = await self._make_profile(
                session, public_group, version=2
            )
            test_group = DocumentGroup(
                group_key=HELP_CHATBOT_TEST.profile_key,
                name="서비스 B 합성 문서 그룹",
                consumer_key=HELP_CHATBOT.profile_key,
            )
            session.add(test_group)
            await session.flush()
            test_profile = ChatProfile(
                profile_key=HELP_CHATBOT_TEST.profile_key,
                name="서비스 B 합성 서비스",
                document_group_id=test_group.id,
            )
            session.add(test_profile)
            await session.flush()
            self.service_b = await self._make_profile(
                session, test_group, version=1, profile=test_profile
            )
            self.public.update(await self._seed_document(
                session, public_group.id, *SYNTHETIC_EXAMPLES["public"]
            ))
            self.service_b.update(await self._seed_document(
                session, test_group.id, *SYNTHETIC_EXAMPLES["service_b"]
            ))
            await session.commit()

        async with self.sessions() as session:
            for service in (self.public, self.service_b):
                chunks = await SearchReader(
                    session, document_group_id=service["group_id"]
                ).load_active_chunks()
                self.registry.replace(service["group_id"], chunks)

    async def _cleanup_scratch_database(self) -> None:
        if self.engine is not None:
            await self.engine.dispose()
        await self._maintenance(
            f'DROP DATABASE IF EXISTS "{self.database_name}" WITH (FORCE)'
        )

    async def _maintenance(self, statement: str) -> None:
        url = make_url(self.database_url).set(database="postgres")
        engine = create_async_engine(
            url.render_as_string(hide_password=False), isolation_level="AUTOCOMMIT"
        )
        try:
            async with engine.connect() as connection:
                await connection.execute(text(statement))
        finally:
            await engine.dispose()

    async def _make_profile(self, session, group, *, version, profile=None):
        if profile is None:
            profile = await session.scalar(
                select(ChatProfile).where(ChatProfile.document_group_id == group.id)
            )
        revision = ChatProfileRevision(
            profile_id=profile.id,
            version=version,
            document_group_id=group.id,
            status=ChatProfileRevisionStatus.PUBLISHED,
            generation_model_name=GENERATION_MODEL,
            generation_prompt_version=GENERATION_PROMPT_VERSION,
            query_rewrite_model_name=OPENAI_QUERY_REWRITE_MODEL,
            query_rewrite_prompt_version=QUERY_REWRITE_PROMPT_VERSION,
        )
        session.add(revision)
        await session.flush()
        components = component_models(
            revision.id,
            generation_model=GENERATION_MODEL,
            rewrite_model=OPENAI_QUERY_REWRITE_MODEL,
        )
        for component in components:
            if component.stage.value == "rewrite":
                component.prompt_version = QUERY_REWRITE_PROMPT_VERSION
                component.prompt_key = f"rewrite@{QUERY_REWRITE_PROMPT_VERSION}"
        session.add_all(components)
        await session.flush()
        return {"profile_id": profile.id, "revision_id": revision.id, "group_id": group.id}

    async def _seed_document(self, session: AsyncSession, group_id: int, title: str, body: str) -> dict:
        suffix = uuid.uuid4().hex[:10]
        chunking = ChunkingConfig(
            version=f"service-b-chunk-{suffix}", strategy="SECTION", max_tokens=512
        )
        embedding = EmbeddingConfig(
            version=f"service-b-embedding-{suffix}", provider="openai",
            model_name="text-embedding-test", dimensions=OPENAI_EMBEDDING_DIMENSIONS,
            input_template_version="v1",
        )
        source = DocumentSource(
            document_group_id=group_id, document_key=f"synthetic/{suffix}",
            source_type="LLMS_TXT", canonical_uri=f"https://example.invalid/{suffix}.md",
            title=title, created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        )
        session.add_all([chunking, embedding, source])
        await session.flush()
        version = DocumentVersion(
            document_source_id=source.id, version_no=1,
            raw_content_uri=f"synthetic/{suffix}.md", mime_type="text/markdown",
            raw_content_hash=f"raw-{suffix}", normalized_content_hash=f"normalized-{suffix}",
            parser_name="test", parser_version="1", status=DocumentVersionStatus.READY,
            collected_at=datetime.now(timezone.utc),
        )
        session.add(version)
        await session.flush()
        node = ContentNode(
            document_version_id=version.id, node_type="SECTION",
            node_path=f"{title}>절차", node_order=1, title="절차",
            normalized_content=body, content_hash=f"content-{suffix}",
            node_identity_hash=f"{suffix}:identity", node_identity_kind="path",
            metadata_={"document_id": suffix, "section_id": "procedure", "section_path": [title, "절차"]},
        )
        session.add(node)
        await session.flush()
        index = IndexVersion(
            document_group_id=group_id, version=f"service-b-index-{suffix}",
            status=IndexVersionStatus.ACTIVE, chunking_config_id=chunking.id,
            embedding_config_id=embedding.id,
        )
        session.add_all([
            DocumentChunk(id=node.id, chunking_config_id=chunking.id, chunk_index=0, token_count=20),
            index,
        ])
        await session.flush()
        session.add_all([
            IndexDocument(index_version_id=index.id, document_version_id=version.id),
            ChunkEmbedding(
                chunk_id=node.id, embedding_config_id=embedding.id,
                embedding=[0.1] * OPENAI_EMBEDDING_DIMENSIONS,
                embedding_input_hash=f"embedding-{suffix}",
            ),
        ])
        await session.flush()
        return {
            "index_version_id": index.id,
            "source_id": source.id,
            "content_node_id": node.id,
        }

    async def _use_shared_synthetic_document(self) -> None:
        title, body = SHARED_SYNTHETIC_DOCUMENT
        async with self.sessions() as session:
            for service in (self.public, self.service_b):
                source = await session.get(DocumentSource, service["source_id"])
                node = await session.get(ContentNode, service["content_node_id"])
                source.title = title
                node.normalized_content = body
                metadata = dict(node.metadata_ or {})
                metadata["section_path"] = [title, "절차"]
                node.metadata_ = metadata
            await session.commit()

        async with self.sessions() as session:
            for service in (self.public, self.service_b):
                chunks = await SearchReader(
                    session, document_group_id=service["group_id"]
                ).load_active_chunks()
                self.registry.replace(service["group_id"], chunks)

    def _chat_app(self):
        retrieved_titles_by_document = {}
        retrieved_candidates_by_chunk = {}
        allowed_titles = {
            *(title for title, _body in SYNTHETIC_EXAMPLES.values()),
            SHARED_SYNTHETIC_DOCUMENT[0],
        }
        generation = AsyncMock(spec=GenerationService)
        generation.model_name = GENERATION_MODEL
        generation.prompt_version = GENERATION_PROMPT_VERSION

        async def generate(query, results, *, before_model_call, stages):
            candidate_titles = {result.chunk.document_title for result in results}
            self.assertEqual(1, len(candidate_titles))
            self.assertTrue(candidate_titles.issubset(allowed_titles))
            chunk = results[0].chunk
            retrieved_titles_by_document[chunk.document_title] = candidate_titles
            retrieved_candidates_by_chunk[chunk.chunk_id] = {
                "chunk_ids": {result.chunk.chunk_id for result in results},
                "index_version_ids": {
                    result.chunk.index_version_id for result in results
                },
            }
            await before_model_call("openai", stages.model_name, stages.prompt_version)
            return FinalGenerationResult(
                status=FinalAnswerStatus.COMPLETED,
                answer_markdown=f"{chunk.content} [1]",
                citations=(Citation(
                    citation_number=1, document_title=chunk.document_title,
                    section_path=chunk.section_path, source_url=chunk.source_url,
                    source_kind=CitationSourceKind.GITBOOK, chunk_id=chunk.chunk_id,
                    document_version_id=chunk.document_version_id,
                ),),
                model_call=ModelCallTrace(
                    provider="openai", model_name=stages.model_name,
                    succeeded=True, latency_ms=1, input_tokens=5, output_tokens=5,
                    prompt_version=stages.prompt_version,
                ),
            )

        generation.generate_answer.side_effect = generate
        rewrite = AsyncMock(spec=QueryRewriteService)
        rewrite.model_name = OPENAI_QUERY_REWRITE_MODEL
        rewrite.prompt_version = QUERY_REWRITE_PROMPT_VERSION

        async def keep_query(question, _candidates, *, before_model_call, stage):
            await before_model_call("openai", stage.model_name, stage.prompt_version)
            return QueryRewriteCall(
                trace=ModelCallTrace(
                    provider="openai", model_name=stage.model_name,
                    succeeded=True, latency_ms=1, prompt_version=stage.prompt_version,
                ),
                resolution=QueryResolution(
                    decision=QueryRewriteDecision.NEW_TOPIC,
                    resolved_query=question, selected_turns=(),
                ),
            )

        rewrite.rewrite.side_effect = keep_query
        app = create_app()
        app.state.corpus_state = Mock(spec=CorpusState)
        app.state.corpus_registry = self.registry
        app.state.embedder = _LocalEmbedder()
        app.state.generation_service = generation
        app.state.query_rewrite_service = rewrite

        async def request_session():
            async with self.sessions() as session:
                yield session

        app.dependency_overrides[get_db_session] = request_session
        return app, retrieved_titles_by_document, retrieved_candidates_by_chunk

    async def test_service_b_search_uses_only_its_synthetic_document_group(self) -> None:
        app, retrieved_titles_by_document, _retrieved_candidates = self._chat_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            public, service_b = await asyncio.gather(
                client.post("/api/chat", json={"question": "청록 달팽이 동기화"}),
                client.post("/api/internal/test-chat", json={"question": "청록 달팽이 동기화"}),
            )

        self.assertEqual(200, public.status_code, public.text)
        self.assertEqual(200, service_b.status_code, service_b.text)
        public_body, service_b_body = public.json(), service_b.json()
        self.assertEqual(SYNTHETIC_EXAMPLES["public"][0], public_body["citations"][0]["documentTitle"])
        self.assertEqual(SYNTHETIC_EXAMPLES["service_b"][0], service_b_body["citations"][0]["documentTitle"])
        for body, expected_title in (
            (public_body, SYNTHETIC_EXAMPLES["public"][0]),
            (service_b_body, SYNTHETIC_EXAMPLES["service_b"][0]),
        ):
            response_title = body["citations"][0]["documentTitle"]
            self.assertEqual(expected_title, response_title)
            self.assertEqual({expected_title}, retrieved_titles_by_document[response_title])
        self.assertNotIn("청록 달팽이", public_body["answer"]["answerMarkdown"])
        self.assertIn("청록 달팽이", service_b_body["answer"]["answerMarkdown"])

        for body, service in ((public_body, self.public), (service_b_body, self.service_b)):
            run = (await self._rows(
                "SELECT profile_revision_id, index_version_id FROM rag_runs WHERE id = :id",
                {"id": uuid.UUID(body["ragRunId"])},
            ))[0]
            self.assertEqual((service["revision_id"], service["index_version_id"]), tuple(run))

        # 서비스 B의 새 PUBLISHED revision이 선택되어도 문서 그룹은 B에 고정된다.
        async with self.sessions() as session:
            current = await session.get(ChatProfileRevision, self.service_b["revision_id"])
            current.status = ChatProfileRevisionStatus.RETIRED
            await session.flush()
            profile = await session.get(ChatProfile, self.service_b["profile_id"])
            switched = await self._make_profile(
                session,
                await session.get(DocumentGroup, self.service_b["group_id"]),
                version=2, profile=profile,
            )
            await session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            switched_b, still_public = await asyncio.gather(
                client.post("/api/internal/test-chat", json={
                    "question": "청록 달팽이 동기화",
                    "conversationId": service_b_body["conversationId"],
                }),
                client.post("/api/chat", json={"question": "청록 달팽이 동기화"}),
            )
        self.assertEqual(200, switched_b.status_code, switched_b.text)
        self.assertEqual(200, still_public.status_code, still_public.text)
        self.assertEqual(
            SYNTHETIC_EXAMPLES["service_b"][0],
            switched_b.json()["citations"][0]["documentTitle"],
        )
        self.assertEqual(
            SYNTHETIC_EXAMPLES["public"][0],
            still_public.json()["citations"][0]["documentTitle"],
        )
        self.assertEqual(
            switched["revision_id"],
            (await self._rows(
                "SELECT profile_revision_id FROM rag_runs WHERE id = :id",
                {"id": uuid.UUID(switched_b.json()["ragRunId"])},
            ))[0][0],
        )

    async def test_identical_document_and_answer_format_stay_service_scoped(self) -> None:
        await self._use_shared_synthetic_document()
        app, _retrieved_titles, retrieved_candidates_by_chunk = self._chat_app()
        expected_title, expected_markdown = SHARED_SYNTHETIC_DOCUMENT
        self.assertNotEqual(self.public["group_id"], self.service_b["group_id"])
        self.assertNotEqual(self.public["index_version_id"], self.service_b["index_version_id"])
        self.assertNotEqual(self.public["profile_id"], self.service_b["profile_id"])
        self.assertNotEqual(self.public["revision_id"], self.service_b["revision_id"])

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            public, service_b = await asyncio.gather(
                client.post("/api/chat", json={"question": "공통 설정 저장 방식"}),
                client.post("/api/internal/test-chat", json={"question": "공통 설정 저장 방식"}),
            )
            self.assertEqual(200, public.status_code, public.text)
            self.assertEqual(200, service_b.status_code, service_b.text)
            public_body, service_b_body = public.json(), service_b.json()
            for body in (public_body, service_b_body):
                self.assertEqual(expected_title, body["citations"][0]["documentTitle"])
                self.assertEqual(f"{expected_markdown} [1]", body["answer"]["answerMarkdown"])
            self.assertEqual(public_body["answer"], service_b_body["answer"])
            self.assertEqual(set(public_body), set(service_b_body))
            self.assertNotEqual(public_body["conversationId"], service_b_body["conversationId"])

            expected_services = (
                (public_body, self.public, "PUBLIC"),
                (service_b_body, self.service_b, "INTERNAL_TEST"),
            )
            for body, service, channel in expected_services:
                expected_chunk_id = service["content_node_id"]
                self.assertEqual(
                    {expected_chunk_id},
                    retrieved_candidates_by_chunk[expected_chunk_id]["chunk_ids"],
                )
                self.assertEqual(
                    {service["index_version_id"]},
                    retrieved_candidates_by_chunk[expected_chunk_id]["index_version_ids"],
                )
                row = (await self._rows(
                    "SELECT c.id, c.profile_id, c.chat_profile_revision_id, c.channel, "
                    "r.profile_revision_id, r.index_version_id "
                    "FROM rag_runs r JOIN conversations c ON c.id = r.conversation_id "
                    "WHERE r.id = :id",
                    {"id": uuid.UUID(body["ragRunId"])},
                ))[0]
                self.assertEqual(uuid.UUID(body["conversationId"]), row[0])
                self.assertEqual(
                    (service["profile_id"], service["revision_id"], channel,
                     service["revision_id"], service["index_version_id"]),
                    tuple(row[1:]),
                )

            async with self.sessions() as session:
                current = await session.get(
                    ChatProfileRevision, self.service_b["revision_id"]
                )
                current.status = ChatProfileRevisionStatus.RETIRED
                await session.flush()
                profile = await session.get(ChatProfile, self.service_b["profile_id"])
                new_revision = await self._make_profile(
                    session,
                    await session.get(DocumentGroup, self.service_b["group_id"]),
                    version=2,
                    profile=profile,
                )
                await session.commit()

            switched_b, still_public = await asyncio.gather(
                client.post("/api/internal/test-chat", json={
                    "question": "공통 설정 저장 방식",
                    "conversationId": service_b_body["conversationId"],
                }),
                client.post("/api/chat", json={
                    "question": "공통 설정 저장 방식",
                    "conversationId": public_body["conversationId"],
                }),
            )

        self.assertEqual(200, switched_b.status_code, switched_b.text)
        self.assertEqual(200, still_public.status_code, still_public.text)
        for response in (switched_b, still_public):
            self.assertEqual(expected_title, response.json()["citations"][0]["documentTitle"])
            self.assertEqual(
                f"{expected_markdown} [1]",
                response.json()["answer"]["answerMarkdown"],
            )
        switched_runs = await self._rows(
            "SELECT r.profile_revision_id, c.profile_id, c.chat_profile_revision_id, "
            "c.channel, r.index_version_id FROM rag_runs r "
            "JOIN conversations c ON c.id = r.conversation_id WHERE r.id = :id",
            {"id": uuid.UUID(switched_b.json()["ragRunId"])},
        )
        self.assertEqual(
            (new_revision["revision_id"], self.service_b["profile_id"],
             self.service_b["revision_id"], "INTERNAL_TEST", self.service_b["index_version_id"]),
            tuple(switched_runs[0]),
        )
        public_runs = await self._rows(
            "SELECT r.profile_revision_id, c.profile_id, c.chat_profile_revision_id, "
            "c.channel, r.index_version_id FROM rag_runs r "
            "JOIN conversations c ON c.id = r.conversation_id WHERE r.id = :id",
            {"id": uuid.UUID(still_public.json()["ragRunId"])},
        )
        self.assertEqual(
            (self.public["revision_id"], self.public["profile_id"],
             self.public["revision_id"], "PUBLIC", self.public["index_version_id"]),
            tuple(public_runs[0]),
        )

    async def _rows(self, statement: str, parameters=None):
        async with self.engine.connect() as connection:
            return (await connection.execute(text(statement), parameters or {})).fetchall()


if __name__ == "__main__":
    unittest.main()
