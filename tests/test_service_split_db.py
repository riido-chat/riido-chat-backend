"""서비스 나누기(20261010_20) migration 과 두 서비스의 주소별 서빙 통합 테스트.

- migration: DEV 와 같은 모양(문서 그룹 둘, HELP_CHATBOT 아래 PUBLISHED 와 TESTING 프로필 버전,
  두 channel 의 대화)과 빈 DB 에서 upgrade 와 downgrade 를 확인한다.
- 서빙: /api/chat 과 /api/internal/test-chat 이 동시에 각자 서비스의 PUBLISHED 프로필 버전과 문서
  그룹으로 답하고, 대화는 자기 서비스의 주소로만 이어지는지 확인한다. 검색과 생성은 가짜다.

로컬 PostgreSQL 에 연결할 수 없으면 skip 한다. 테스트마다 일회용 DB 를 만들고 지운다.
"""

import asyncio
import os
import subprocess
import sys
import unittest
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.answering.generator import GENERATION_PROMPT_VERSION
from app.answering.models import (
    Citation,
    CitationSourceKind,
    FinalAnswerStatus,
    FinalGenerationResult,
)
from app.answering.service import GenerationService
from app.chat.profile import (
    ConversationProfileMismatchError,
    resolve_chat_profile_revision,
)
from app.chat.query_rewrite import (
    OPENAI_QUERY_REWRITE_MODEL,
    QUERY_REWRITE_PROMPT_VERSION,
    QueryResolution,
    QueryRewriteCall,
    QueryRewriteDecision,
    QueryRewriteService,
)
from app.chat.service import conversation_not_found_response
from app.core.config import get_settings
from app.core.model_trace import ModelCallTrace
from app.database.models import (
    ChatProfile,
    ChatProfileRevision,
    ChatProfileRevisionStatus,
    ChunkingConfig,
    ContentNode,
    DocumentChunk,
    DocumentGroup,
    DocumentSource,
    DocumentVersion,
    DocumentVersionStatus,
    EmbeddingConfig,
    IndexVersion,
    IndexVersionStatus,
)
from app.database.session import get_db_session
from app.main import create_app
from app.ops.bootstrap_testing_corpus import _test_service_revision
from app.profiles.readiness import check_published_profile_versions
from app.profiles.services import HELP_CHATBOT, HELP_CHATBOT_TEST
from app.profiles.stage_settings import component_problems, read_profile_components
from app.retrieval.corpus_state import CorpusRegistry, CorpusState
from app.retrieval.models import (
    HybridRetrievalResult,
    HybridSearchCall,
    RetrievalChunk,
    RetrievalResult,
)
from tests.profile_components import component_models


REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_REVISION = "20261001_19"
SPLIT_REVISION = "20261010_20"
PUBLIC_CONVERSATIONS = 3
TEST_CONVERSATIONS = 2
# 서빙 테스트의 가짜 생성 모델. 프롬프트 버전은 등록부의 것을 쓴다.
GENERATION_MODEL = "gpt-test-generation"


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


def _now() -> datetime:
    return datetime.now(timezone.utc)


class _ScratchDatabaseTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_available(cls.database_url)):
            raise unittest.SkipTest("로컬 PostgreSQL에 연결할 수 없습니다.")

    async def asyncSetUp(self) -> None:
        self.database_name = f"riido_service_split_{uuid.uuid4().hex[:12]}"
        self.scratch_url = make_url(self.database_url).set(
            database=self.database_name
        ).render_as_string(hide_password=False)
        await self._maintenance(f'CREATE DATABASE "{self.database_name}"')
        self.engine = create_async_engine(self.scratch_url)

    async def asyncTearDown(self) -> None:
        await self.engine.dispose()
        await self._maintenance(
            f'DROP DATABASE IF EXISTS "{self.database_name}" WITH (FORCE)'
        )

    async def _maintenance(self, statement: str) -> None:
        url = make_url(self.database_url).set(database="postgres")
        engine = create_async_engine(
            url.render_as_string(hide_password=False),
            isolation_level="AUTOCOMMIT",
        )
        try:
            async with engine.connect() as connection:
                await connection.execute(text(statement))
        finally:
            await engine.dispose()

    async def _execute(self, statement: str, parameters=None) -> None:
        async with self.engine.begin() as connection:
            await connection.execute(text(statement), parameters or {})

    async def _scalar_write(self, statement: str, parameters=None):
        async with self.engine.begin() as connection:
            return (await connection.execute(text(statement), parameters or {})).scalar_one()

    async def _scalar(self, statement: str, parameters=None):
        async with self.engine.connect() as connection:
            return (await connection.execute(text(statement), parameters or {})).scalar_one()

    async def _rows(self, statement: str, parameters=None):
        async with self.engine.connect() as connection:
            return (await connection.execute(text(statement), parameters or {})).fetchall()

    async def _column(self, table: str, column: str) -> bool:
        return bool(await self._scalar(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = :table AND column_name = :column",
            {"table": table, "column": column},
        ))


class ServiceSplitMigrationDbTest(_ScratchDatabaseTest):
    async def test_dev_shape_upgrade_and_downgrade(self) -> None:
        _alembic(self.scratch_url, "upgrade", PARENT_REVISION)
        seed = await self._seed_dev_shape()
        before = await self._revision_rows()
        runs_before = await self._rows(
            "SELECT id, profile_revision_id FROM rag_runs ORDER BY id"
        )

        _alembic(self.scratch_url, "upgrade", SPLIT_REVISION)

        profiles = await self._rows(
            "SELECT id, profile_key, name, document_group_id FROM chat_profiles ORDER BY id"
        )
        self.assertEqual(
            [
                (seed["public_profile_id"], "HELP_CHATBOT", "도움말 챗봇", seed["public_group_id"]),
                (profiles[1][0], "HELP_CHATBOT_TEST", "테스트 서비스", seed["test_group_id"]),
            ],
            profiles,
        )
        test_profile_id = profiles[1][0]
        # 프로필 버전 id 와 version 은 그대로, 서비스 B 의 TESTING 은 그 서비스의 PUBLISHED 가 된다.
        self.assertEqual(
            [
                (seed["public_revision_id"], seed["public_profile_id"], 1, "PUBLISHED", seed["public_group_id"]),
                (seed["test_revision_id"], test_profile_id, 2, "PUBLISHED", seed["test_group_id"]),
            ],
            await self._revision_rows(),
        )
        self.assertEqual(
            2,
            await self._scalar(
                "SELECT count(*) FROM profile_revision_components WHERE revision_id = :id",
                {"id": seed["test_revision_id"]},
            ),
        )
        self.assertEqual(
            {
                ("PUBLIC", seed["public_revision_id"], seed["public_profile_id"]): PUBLIC_CONVERSATIONS,
                ("INTERNAL_TEST", seed["test_revision_id"], test_profile_id): TEST_CONVERSATIONS,
            },
            {
                (row[0], row[1], row[2]): row[3]
                for row in await self._rows(
                    "SELECT channel, chat_profile_revision_id, profile_id, count(*) "
                    "FROM conversations GROUP BY 1, 2, 3"
                )
            },
        )
        self.assertEqual(
            runs_before,
            await self._rows("SELECT id, profile_revision_id FROM rag_runs ORDER BY id"),
        )
        self.assertEqual(
            "HELP_CHATBOT",
            await self._scalar(
                "SELECT consumer_key FROM document_groups WHERE id = :id",
                {"id": seed["test_group_id"]},
            ),
        )

        # 옮긴 프로필 버전으로 시작한 대화는 test-chat 주소에서만 그대로 이어진다.
        sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with sessions() as session:
            resolved = await resolve_chat_profile_revision(
                session,
                service=HELP_CHATBOT_TEST,
                conversation_id=seed["test_conversation_id"],
            )
            self.assertEqual(seed["test_revision_id"], resolved.id)
            with self.assertRaises(ConversationProfileMismatchError):
                await resolve_chat_profile_revision(
                    session,
                    service=HELP_CHATBOT,
                    conversation_id=seed["test_conversation_id"],
                )
            with self.assertRaises(ConversationProfileMismatchError):
                await resolve_chat_profile_revision(
                    session,
                    service=HELP_CHATBOT_TEST,
                    conversation_id=seed["public_conversation_id"],
                )
            self.assertEqual(
                seed["public_revision_id"],
                (
                    await resolve_chat_profile_revision(
                        session,
                        service=HELP_CHATBOT,
                        conversation_id=seed["public_conversation_id"],
                    )
                ).id,
            )

        _alembic(self.scratch_url, "downgrade", PARENT_REVISION)

        self.assertEqual(before, await self._revision_rows())
        self.assertEqual(
            [("HELP_CHATBOT",)],
            await self._rows("SELECT profile_key FROM chat_profiles"),
        )
        self.assertFalse(await self._column("chat_profiles", "document_group_id"))
        self.assertFalse(await self._column("conversations", "profile_id"))
        self.assertTrue(await self._column("conversations", "channel"))
        self.assertEqual(
            PUBLIC_CONVERSATIONS + TEST_CONVERSATIONS,
            await self._scalar("SELECT count(*) FROM conversations"),
        )
        self.assertEqual(
            runs_before,
            await self._rows("SELECT id, profile_revision_id FROM rag_runs ORDER BY id"),
        )

        # 다시 올려도 같은 모양이 된다.
        _alembic(self.scratch_url, "upgrade", SPLIT_REVISION)
        self.assertEqual(
            ["HELP_CHATBOT", "HELP_CHATBOT_TEST"],
            [row[0] for row in await self._rows(
                "SELECT profile_key FROM chat_profiles ORDER BY id"
            )],
        )

    async def test_empty_database_gets_no_test_service(self) -> None:
        _alembic(self.scratch_url, "upgrade", PARENT_REVISION)
        # 문서 그룹만 있고 그 그룹을 쓰는 프로필 버전이 없어도 서비스를 만들지 않는다.
        await self._execute(
            "INSERT INTO document_groups (group_key, name, consumer_key) "
            "VALUES ('HELP_CHATBOT_TEST', '테스트', 'HELP_CHATBOT')"
        )

        _alembic(self.scratch_url, "upgrade", SPLIT_REVISION)

        self.assertEqual(
            [("HELP_CHATBOT", "HELP_CHATBOT")],
            await self._rows(
                "SELECT p.profile_key, g.group_key FROM chat_profiles p "
                "JOIN document_groups g ON g.id = p.document_group_id"
            ),
        )
        _alembic(self.scratch_url, "downgrade", PARENT_REVISION)
        self.assertFalse(await self._column("chat_profiles", "document_group_id"))
        self.assertEqual(1, await self._scalar("SELECT count(*) FROM chat_profiles"))

    async def test_database_rejects_revision_on_another_services_group(self) -> None:
        _alembic(self.scratch_url, "upgrade", PARENT_REVISION)
        seed = await self._seed_dev_shape()
        _alembic(self.scratch_url, "upgrade", SPLIT_REVISION)
        test_profile_id = await self._scalar(
            "SELECT id FROM chat_profiles WHERE profile_key = 'HELP_CHATBOT_TEST'"
        )

        for profile_id, group_id in (
            (seed["public_profile_id"], seed["test_group_id"]),
            (test_profile_id, seed["public_group_id"]),
        ):
            with self.subTest(profile_id=profile_id, group_id=group_id):
                with self.assertRaises(IntegrityError):
                    await self._insert_revision(profile_id, 9, group_id, "DRAFT")
                with self.assertRaises(IntegrityError):
                    await self._execute(
                        "UPDATE chat_profile_revisions SET document_group_id = :group_id "
                        "WHERE profile_id = :profile_id",
                        {"profile_id": profile_id, "group_id": group_id},
                    )
        # 같은 서비스의 문서 그룹이면 받는다.
        await self._insert_revision(seed["public_profile_id"], 9, seed["public_group_id"], "DRAFT")
        # 문서 그룹 하나에 서비스 하나다.
        with self.assertRaises(IntegrityError):
            await self._execute(
                "INSERT INTO chat_profiles (profile_key, name, document_group_id) "
                "VALUES ('OTHER', '다른 서비스', :group_id)",
                {"group_id": seed["public_group_id"]},
            )
        with self.assertRaises(IntegrityError):
            await self._execute(
                "INSERT INTO conversations (id, chat_profile_revision_id, status) "
                "VALUES (:id, :revision_id, 'ACTIVE')",
                {"id": uuid.uuid4(), "revision_id": seed["public_revision_id"]},
            )

    async def test_downgrade_renumbers_version_taken_after_upgrade(self) -> None:
        _alembic(self.scratch_url, "upgrade", PARENT_REVISION)
        seed = await self._seed_dev_shape()
        _alembic(self.scratch_url, "upgrade", SPLIT_REVISION)
        draft_id = await self._insert_revision(
            seed["public_profile_id"], 2, seed["public_group_id"], "DRAFT"
        )

        _alembic(self.scratch_url, "downgrade", PARENT_REVISION)

        self.assertEqual(
            [
                (seed["public_revision_id"], seed["public_profile_id"], 1, "PUBLISHED", seed["public_group_id"]),
                (seed["test_revision_id"], seed["public_profile_id"], 3, "TESTING", seed["test_group_id"]),
                (draft_id, seed["public_profile_id"], 2, "DRAFT", seed["public_group_id"]),
            ],
            await self._revision_rows(),
        )

    async def _seed_dev_shape(self) -> dict:
        seed = {
            "public_profile_id": await self._scalar(
                "SELECT id FROM chat_profiles WHERE profile_key = 'HELP_CHATBOT'"
            ),
            "public_group_id": await self._scalar(
                "SELECT id FROM document_groups WHERE group_key = 'HELP_CHATBOT'"
            ),
        }
        seed["public_revision_id"] = await self._scalar(
            "SELECT id FROM chat_profile_revisions WHERE status = 'PUBLISHED'"
        )
        seed["test_group_id"] = await self._scalar_write(
            "INSERT INTO document_groups (group_key, name, consumer_key) "
            "VALUES ('HELP_CHATBOT_TEST', '테스트', 'HELP_CHATBOT') RETURNING id"
        )
        seed["test_revision_id"] = await self._insert_revision(
            seed["public_profile_id"], 2, seed["test_group_id"], "TESTING"
        )
        await self._execute(
            "INSERT INTO profile_revision_components "
            "(revision_id, stage, model_name, prompt_key, prompt_version, recorded_by) "
            "SELECT :id, 'rewrite', query_rewrite_model_name, "
            "'rewrite@' || query_rewrite_prompt_version, query_rewrite_prompt_version, "
            "'BACKFILL' FROM chat_profile_revisions WHERE id = :id "
            "UNION ALL SELECT :id, 'generation', generation_model_name, "
            "'generation@' || generation_prompt_version, generation_prompt_version, "
            "'BACKFILL' FROM chat_profile_revisions WHERE id = :id",
            {"id": seed["test_revision_id"]},
        )
        chunking_id = await self._scalar_write(
            "INSERT INTO chunking_configs (version, strategy, max_tokens) "
            "VALUES ('split-test', 'section', 500) RETURNING id"
        )
        embedding_id = await self._scalar_write(
            "INSERT INTO embedding_configs "
            "(version, provider, model_name, dimensions, input_template_version) "
            "VALUES ('split-test', 'openai', 'text-embedding-3-large', 1536, 'v1') "
            "RETURNING id"
        )
        index_ids = {}
        for key in ("public", "test"):
            index_ids[key] = await self._scalar_write(
                "INSERT INTO index_versions "
                "(document_group_id, version, status, chunking_config_id, embedding_config_id) "
                "VALUES (:group_id, :version, 'ACTIVE', :chunking_id, :embedding_id) "
                "RETURNING id",
                {
                    "group_id": seed[f"{key}_group_id"],
                    "version": f"split-test-{key}",
                    "chunking_id": chunking_id,
                    "embedding_id": embedding_id,
                },
            )
        for key, channel, count in (
            ("public", "PUBLIC", PUBLIC_CONVERSATIONS),
            ("test", "INTERNAL_TEST", TEST_CONVERSATIONS),
        ):
            for number in range(count):
                conversation_id = uuid.uuid4()
                seed.setdefault(f"{key}_conversation_id", conversation_id)
                await self._execute(
                    "INSERT INTO conversations "
                    "(id, chat_profile_revision_id, channel, status) "
                    "VALUES (:id, :revision_id, :channel, 'ACTIVE')",
                    {
                        "id": conversation_id,
                        "revision_id": seed[f"{key}_revision_id"],
                        "channel": channel,
                    },
                )
                # 턴 기록의 프로필 버전은 일부만 채워져 있다(빈 칸은 이력이라 그대로 둔다).
                await self._execute(
                    "INSERT INTO rag_runs "
                    "(id, trace_id, conversation_id, turn_no, index_version_id, user_query, "
                    "context_strategy, status, profile_revision_id) "
                    "VALUES (:id, :trace_id, :conversation_id, 1, :index_id, '질문', "
                    "'NEW_TOPIC', 'COMPLETED', :revision_id)",
                    {
                        "id": uuid.uuid4(),
                        "trace_id": uuid.uuid4(),
                        "conversation_id": conversation_id,
                        "index_id": index_ids[key],
                        "revision_id": seed[f"{key}_revision_id"] if number == 0 else None,
                    },
                )
        return seed

    async def _insert_revision(
        self, profile_id: int, version: int, group_id: int, status: str
    ) -> int:
        return await self._scalar_write(
            "INSERT INTO chat_profile_revisions "
            "(profile_id, version, document_group_id, status, "
            "generation_model_name, generation_prompt_version, "
            "query_rewrite_model_name, query_rewrite_prompt_version, "
            "semantic_cache_enabled, exact_cache_enabled) "
            "SELECT :profile_id, :version, :group_id, :status, "
            "generation_model_name, generation_prompt_version, "
            "query_rewrite_model_name, query_rewrite_prompt_version, false, true "
            "FROM chat_profile_revisions ORDER BY id LIMIT 1 RETURNING id",
            {
                "profile_id": profile_id,
                "version": version,
                "group_id": group_id,
                "status": status,
            },
        )

    async def _revision_rows(self):
        return await self._rows(
            "SELECT id, profile_id, version, status, document_group_id "
            "FROM chat_profile_revisions ORDER BY id"
        )


@asynccontextmanager
async def _no_lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


class _GroupRetriever:
    """문서 그룹 하나의 청크만 돌려주는 가짜 검색기."""

    def __init__(self, chunk: RetrievalChunk) -> None:
        self.chunk = chunk

    async def search_with_trace(self, _query, *, before_model_call) -> HybridSearchCall:
        await before_model_call("openai", "text-embedding-test", None)
        # 다른 요청과 번갈아 실행되도록 한 번 양보한다.
        await asyncio.sleep(0)
        return HybridSearchCall(
            bm25_results=(RetrievalResult(chunk=self.chunk, score=7.5, rank=1),),
            vector_results=(RetrievalResult(chunk=self.chunk, score=0.9, rank=1),),
            fused_results=(
                HybridRetrievalResult(
                    chunk=self.chunk,
                    rrf_score=0.032,
                    final_rank=1,
                    bm25_rank=1,
                    vector_rank=1,
                ),
            ),
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


class TwoServicesServingDbTest(_ScratchDatabaseTest):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        _alembic(self.scratch_url, "upgrade", "head")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.generation_stages = []
        self.rewrite_stages = []
        self.model_config = (
            GENERATION_MODEL,
            GENERATION_PROMPT_VERSION,
            OPENAI_QUERY_REWRITE_MODEL,
            QUERY_REWRITE_PROMPT_VERSION,
        )
        async with self.sessions() as session:
            migrated = await session.scalar(
                select(ChatProfileRevision).where(
                    ChatProfileRevision.status == ChatProfileRevisionStatus.PUBLISHED
                )
            )
            # 빈 DB migration 이 만든 프로필 버전(v38, v8)은 지금 코드로 서빙할 수 없다. 상태 칸만
            # 내리고 지금 코드 구성으로 새 프로필 버전을 게시한다.
            migrated.status = ChatProfileRevisionStatus.RETIRED
            await session.flush()
            public_group = await session.get(DocumentGroup, migrated.document_group_id)
            public_revision = await self._publish(
                session, migrated.profile_id, public_group.id, version=2
            )
            test_group = DocumentGroup(
                group_key="HELP_CHATBOT_TEST", name="테스트", consumer_key="HELP_CHATBOT"
            )
            session.add(test_group)
            await session.flush()
            test_profile = ChatProfile(
                profile_key=HELP_CHATBOT_TEST.profile_key,
                name="테스트 서비스",
                document_group_id=test_group.id,
            )
            session.add(test_profile)
            await session.flush()
            test_revision = await self._publish(
                session, test_profile.id, test_group.id, version=1
            )
            self.public = {
                "profile_id": public_revision.profile_id,
                "revision_id": public_revision.id,
                "group_id": public_group.id,
            }
            self.test = {
                "profile_id": test_profile.id,
                "revision_id": test_revision.id,
                "group_id": test_group.id,
            }
            for service, title in ((self.public, "공개 문서"), (self.test, "테스트 문서")):
                service.update(await self._seed_content(session, service["group_id"], title))
            await session.commit()

    async def test_two_services_answer_from_own_groups_concurrently(self) -> None:
        async with self._client() as client:
            public_response, test_response = await asyncio.gather(
                client.post("/api/chat", json={"question": "질문"}),
                client.post("/api/internal/test-chat", json={"question": "질문"}),
            )

        for response, service, title in (
            (public_response, self.public, "공개 문서"),
            (test_response, self.test, "테스트 문서"),
        ):
            with self.subTest(title=title):
                self.assertEqual(200, response.status_code, response.text)
                body = response.json()
                self.assertEqual("COMPLETED", body["status"])
                self.assertEqual(title, body["citations"][0]["documentTitle"])
                row = (
                    await self._rows(
                        "SELECT c.profile_id, c.chat_profile_revision_id, "
                        "r.profile_revision_id, r.index_version_id "
                        "FROM rag_runs r JOIN conversations c ON c.id = r.conversation_id "
                        "WHERE r.id = :id",
                        {"id": uuid.UUID(body["ragRunId"])},
                    )
                )[0]
                self.assertEqual(
                    (
                        service["profile_id"],
                        service["revision_id"],
                        service["revision_id"],
                        service["index_version_id"],
                    ),
                    tuple(row),
                )
        self.assertEqual(
            {("PUBLIC", self.public["profile_id"]), ("INTERNAL_TEST", self.test["profile_id"])},
            {
                tuple(row)
                for row in await self._rows("SELECT channel, profile_id FROM conversations")
            },
        )

    async def test_conversation_continues_only_on_its_own_service_address(self) -> None:
        async with self._client() as client:
            public_body = (await client.post("/api/chat", json={"question": "질문"})).json()
            test_body = (
                await client.post("/api/internal/test-chat", json={"question": "질문"})
            ).json()

            for path, foreign in (
                ("/api/internal/test-chat", public_body["conversationId"]),
                ("/api/chat", test_body["conversationId"]),
            ):
                with self.subTest(path=path):
                    response = await client.post(
                        path,
                        json={"question": "후속 질문", "conversationId": foreign},
                    )
                    # 기존과 같은 이어갈 수 없는 대화 응답이다.
                    self.assertEqual(404, response.status_code, response.text)
                    self.assertEqual(
                        conversation_not_found_response().model_dump(
                            mode="json", by_alias=True
                        ),
                        response.json(),
                    )
        self.assertEqual(2, await self._scalar("SELECT count(*) FROM rag_runs"))

    async def test_bootstrap_keeps_or_adds_published_revision_without_editing(self) -> None:
        async with self.sessions() as session:
            test_group = await session.get(DocumentGroup, self.test["group_id"])
            # 지금 코드로 서빙할 수 있는 PUBLISHED 프로필 버전은 그대로 쓴다(구성 행 검사).
            profile, kept = await _test_service_revision(session, test_group)
            self.assertEqual((self.test["profile_id"], self.test["revision_id"]), (profile.id, kept.id))

        # 새 프로필 버전과 구성 행은 같은 트랜잭션에서 만든다. 되돌리면 둘 다 남지 않는다.
        async with self.sessions() as session:
            test_group = await session.get(DocumentGroup, self.test["group_id"])
            with patch(
                "app.ops.bootstrap_testing_corpus._matches_code", return_value=False
            ):
                _, discarded = await _test_service_revision(session, test_group)
            discarded_id = discarded.id
            await session.rollback()
        self.assertEqual(
            (0, 0),
            (
                await self._scalar(
                    "SELECT count(*) FROM chat_profile_revisions WHERE id = :id",
                    {"id": discarded_id},
                ),
                await self._scalar(
                    "SELECT count(*) FROM profile_revision_components WHERE revision_id = :id",
                    {"id": discarded_id},
                ),
            ),
        )

        async with self.sessions() as session:
            test_group = await session.get(DocumentGroup, self.test["group_id"])
            with patch(
                "app.ops.bootstrap_testing_corpus._matches_code", return_value=False
            ):
                profile, created = await _test_service_revision(session, test_group)
            await session.commit()

        async with self.sessions() as session:
            rows = await read_profile_components(session, created.id)
            self.assertEqual(7, len(rows))
            self.assertEqual({"PUBLISH"}, {row.recorded_by.value for row in rows})
            self.assertEqual((), component_problems(created, rows))

        self.assertNotEqual(self.test["revision_id"], created.id)
        self.assertEqual(
            [
                (self.test["revision_id"], 1, "RETIRED") + self.model_config,
                (created.id, 2, "PUBLISHED") + (
                    created.generation_model_name,
                    created.generation_prompt_version,
                    created.query_rewrite_model_name,
                    created.query_rewrite_prompt_version,
                ),
            ],
            [
                tuple(row)
                for row in await self._rows(
                    "SELECT id, version, status, generation_model_name, "
                    "generation_prompt_version, query_rewrite_model_name, "
                    "query_rewrite_prompt_version FROM chat_profile_revisions "
                    "WHERE profile_id = :profile_id ORDER BY id",
                    {"profile_id": self.test["profile_id"]},
                )
            ],
        )

    @staticmethod
    async def _publish(
        session: AsyncSession,
        profile_id: int,
        group_id: int,
        *,
        version: int,
        generation_model: str = GENERATION_MODEL,
        rewrite_model: str = OPENAI_QUERY_REWRITE_MODEL,
        rewrite_prompt_version: str = QUERY_REWRITE_PROMPT_VERSION,
    ) -> ChatProfileRevision:
        """지금 코드 구성의 PUBLISHED 프로필 버전과 구성 행을 한 트랜잭션에서 만든다."""

        revision = ChatProfileRevision(
            profile_id=profile_id,
            version=version,
            document_group_id=group_id,
            status=ChatProfileRevisionStatus.PUBLISHED,
            generation_model_name=generation_model,
            generation_prompt_version=GENERATION_PROMPT_VERSION,
            query_rewrite_model_name=rewrite_model,
            query_rewrite_prompt_version=rewrite_prompt_version,
        )
        session.add(revision)
        await session.flush()
        rows = component_models(
            revision.id,
            generation_model=generation_model,
            rewrite_model=rewrite_model,
        )
        for row in rows:
            if row.stage.value == "rewrite":
                row.prompt_version = rewrite_prompt_version
                row.prompt_key = f"rewrite@{rewrite_prompt_version}"
        session.add_all(rows)
        await session.flush()
        return revision

    async def test_next_turn_uses_newly_published_revision(self) -> None:
        async with self._client() as client:
            first = (await client.post("/api/chat", json={"question": "질문"})).json()
            self.assertEqual("COMPLETED", first["status"], first)

            # 대화 도중 새 프로필 버전을 게시한다. 이전 PUBLISHED 는 상태 칸만 바뀐다.
            async with self.sessions() as session:
                current = await session.get(ChatProfileRevision, self.public["revision_id"])
                current.status = ChatProfileRevisionStatus.RETIRED
                await session.flush()
                switched = await self._publish(
                    session,
                    self.public["profile_id"],
                    self.public["group_id"],
                    version=3,
                    generation_model="gpt-test-switched",
                    rewrite_model="gpt-test-rewrite-switched",
                )
                switched_id = switched.id
                await session.commit()

            second = await client.post(
                "/api/chat",
                json={"question": "후속 질문", "conversationId": first["conversationId"]},
            )

        self.assertEqual(200, second.status_code, second.text)
        self.assertEqual("COMPLETED", second.json()["status"])
        # 턴마다 고른 프로필 버전이 턴 행에 남고, 대화 행은 시작 프로필 버전을 그대로 둔다.
        self.assertEqual(
            [(1, self.public["revision_id"]), (2, switched_id)],
            [
                tuple(row)
                for row in await self._rows(
                    "SELECT turn_no, profile_revision_id FROM rag_runs "
                    "WHERE conversation_id = :id ORDER BY turn_no",
                    {"id": uuid.UUID(first["conversationId"])},
                )
            ],
        )
        self.assertEqual(
            self.public["revision_id"],
            await self._scalar(
                "SELECT chat_profile_revision_id FROM conversations WHERE id = :id",
                {"id": uuid.UUID(first["conversationId"])},
            ),
        )
        self.assertEqual(
            [GENERATION_MODEL, "gpt-test-switched"],
            [stages.model_name for stages in self.generation_stages],
        )
        self.assertEqual(
            ["gpt-test-rewrite-switched"],
            [stage.model_name for stage in self.rewrite_stages],
        )
        # model_calls 에는 실제로 부른 모델과 프롬프트 버전이 남는다.
        self.assertEqual(
            [
                (1, "ANSWER_GENERATION", GENERATION_MODEL, GENERATION_PROMPT_VERSION),
                (2, "ANSWER_GENERATION", "gpt-test-switched", GENERATION_PROMPT_VERSION),
                (2, "QUERY_REWRITE", "gpt-test-rewrite-switched", QUERY_REWRITE_PROMPT_VERSION),
            ],
            sorted(
                (
                    tuple(row)
                    for row in await self._rows(
                        "SELECT r.turn_no, m.purpose, m.model_name, m.prompt_version "
                        "FROM model_calls m JOIN rag_runs r ON r.id = m.rag_run_id "
                        "WHERE r.conversation_id = :id AND m.purpose <> 'QUERY_EMBEDDING'",
                        {"id": uuid.UUID(first["conversationId"])},
                    )
                ),
                key=lambda row: (row[1], row[0]),
            ),
        )

    async def test_readiness_reports_each_services_published_revision(self) -> None:
        async with self.sessions() as session:
            results = await check_published_profile_versions(session)
        self.assertEqual(
            [
                ("HELP_CHATBOT", self.public["revision_id"], ()),
                ("HELP_CHATBOT_TEST", self.test["revision_id"], ()),
            ],
            [(r.profile_key, r.profile_version_id, r.problems) for r in results],
        )

        # 구성 행이 등록부에 없는 버전을 가리키는 프로필 버전을 게시하면 그 서비스만 실패한다.
        async with self.sessions() as session:
            current = await session.get(ChatProfileRevision, self.test["revision_id"])
            current.status = ChatProfileRevisionStatus.RETIRED
            await session.flush()
            # 구성은 생성부터 불변이다. 잘못된 등록부 참조도 최초 INSERT로 확정한다.
            broken = await self._publish(
                session, self.test["profile_id"], self.test["group_id"], version=2,
                rewrite_prompt_version="v9-missing",
            )
            await session.commit()
            broken_id = broken.id
        async with self.sessions() as session:
            results = await check_published_profile_versions(session)
        self.assertTrue(results[0].servable)
        self.assertEqual((HELP_CHATBOT_TEST.profile_key, broken_id), (results[1].profile_key, results[1].profile_version_id))
        self.assertEqual(
            ("rewrite: 실행 중인 코드에 없는 프롬프트 버전 rewrite@v9-missing",),
            results[1].problems,
        )

        # 그 서비스의 채팅은 기존 503 응답으로 답하지 않고, 다른 서비스는 계속 답한다.
        async with self._client() as client:
            public_response, test_response = await asyncio.gather(
                client.post("/api/chat", json={"question": "질문"}),
                client.post("/api/internal/test-chat", json={"question": "질문"}),
            )
        self.assertEqual(200, public_response.status_code, public_response.text)
        self.assertEqual(503, test_response.status_code, test_response.text)
        self.assertEqual("SERVICE_UNAVAILABLE", test_response.json()["error"]["code"])
        self.assertEqual(
            0,
            await self._scalar(
                "SELECT count(*) FROM rag_runs WHERE profile_revision_id = :id",
                {"id": broken_id},
            ),
        )

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[AsyncClient]:
        retrievers = {
            service["group_id"]: (_GroupRetriever(service["chunk"]), service["index_version_id"])
            for service in (self.public, self.test)
        }
        generation = AsyncMock(spec=GenerationService)
        generation.model_name, generation.prompt_version = self.model_config[:2]
        generation.generate_answer.side_effect = self._generate
        rewrite = AsyncMock(spec=QueryRewriteService)
        rewrite.model_name, rewrite.prompt_version = self.model_config[2:]
        rewrite.rewrite.side_effect = self._rewrite
        registry = Mock(spec=CorpusRegistry)
        registry.is_legacy_compat = False

        with patch("app.main.lifespan", _no_lifespan):
            app = create_app()
        app.state.corpus_state = Mock(spec=CorpusState)
        app.state.corpus_registry = registry
        app.state.embedder = Mock()
        app.state.generation_service = generation
        app.state.query_rewrite_service = rewrite

        async def request_session() -> AsyncIterator[AsyncSession]:
            async with self.sessions() as session:
                yield session

        app.dependency_overrides[get_db_session] = request_session
        with patch(
            "app.chat.dependencies._build_group_retriever",
            side_effect=lambda _registry, _session, _embedder, group_id: retrievers[group_id],
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
                headers={"Accept": "application/json"},
            ) as client:
                yield client

    async def _rewrite(self, question, _candidates, *, before_model_call, stage) -> QueryRewriteCall:
        self.rewrite_stages.append(stage)
        await before_model_call("openai", stage.model_name, stage.prompt_version)
        return QueryRewriteCall(
            trace=ModelCallTrace(
                provider="openai",
                model_name=stage.model_name,
                succeeded=True,
                latency_ms=5,
                prompt_version=stage.prompt_version,
            ),
            resolution=QueryResolution(
                decision=QueryRewriteDecision.NEW_TOPIC,
                resolved_query=question,
                selected_turns=(),
            ),
        )

    async def _generate(
        self, query, results, *, before_model_call, stages
    ) -> FinalGenerationResult:
        # 턴의 프로필 버전 구성 행이 고른 생성 모델과 묶음 버전으로 부른다.
        self.generation_stages.append(stages)
        await before_model_call("openai", stages.model_name, stages.prompt_version)
        chunk = results[0].chunk
        return FinalGenerationResult(
            status=FinalAnswerStatus.COMPLETED,
            answer_markdown=f"{chunk.document_title}에 따른 답변입니다. [1]",
            citations=(
                Citation(
                    citation_number=1,
                    document_title=chunk.document_title,
                    section_path=chunk.section_path,
                    source_url=chunk.source_url,
                    source_kind=CitationSourceKind.GITBOOK,
                    chunk_id=chunk.chunk_id,
                    document_version_id=chunk.document_version_id,
                ),
            ),
            model_call=ModelCallTrace(
                provider="openai",
                model_name=stages.model_name,
                succeeded=True,
                latency_ms=30,
                input_tokens=20,
                output_tokens=10,
                prompt_version=stages.prompt_version,
            ),
        )

    @staticmethod
    async def _seed_content(session: AsyncSession, group_id: int, title: str) -> dict:
        suffix = uuid.uuid4().hex[:8]
        chunking = ChunkingConfig(
            version=f"split-chunking-{suffix}", strategy="SECTION", max_tokens=512
        )
        embedding = EmbeddingConfig(
            version=f"split-embedding-{suffix}",
            provider="openai",
            model_name="text-embedding-test",
            dimensions=1536,
            input_template_version="v1",
        )
        source = DocumentSource(
            document_group_id=group_id,
            document_key=f"split/{suffix}",
            source_type="LLMS_TXT",
            canonical_uri=f"https://docs.riido.io/split/{suffix}.md",
            title=title,
            created_at=_now(),
            updated_at=_now(),
        )
        session.add_all([chunking, embedding, source])
        await session.flush()
        version = DocumentVersion(
            document_source_id=source.id,
            version_no=1,
            raw_content_uri=f"raw/split/{suffix}.md",
            mime_type="text/markdown",
            raw_content_hash="raw-" + suffix,
            normalized_content_hash="norm-" + suffix,
            parser_name="markdown",
            parser_version="1",
            status=DocumentVersionStatus.READY,
            collected_at=_now(),
        )
        session.add(version)
        await session.flush()
        node = ContentNode(
            document_version_id=version.id,
            node_type="SECTION",
            node_path=f"{title}>설정",
            node_order=1,
            title="설정",
            normalized_content=f"{title}의 설정 방법입니다.",
            content_hash="content-" + suffix,
            node_identity_hash=f"{suffix}:identity",
            node_identity_kind="path",
        )
        session.add(node)
        await session.flush()
        index_version = IndexVersion(
            document_group_id=group_id,
            version=f"split-index-{suffix}",
            status=IndexVersionStatus.ACTIVE,
            chunking_config_id=chunking.id,
            embedding_config_id=embedding.id,
        )
        session.add_all(
            [
                DocumentChunk(
                    id=node.id,
                    chunking_config_id=chunking.id,
                    chunk_index=0,
                    token_count=10,
                ),
                index_version,
            ]
        )
        await session.flush()
        return {
            "index_version_id": index_version.id,
            "chunk": RetrievalChunk(
                document_id=f"split-{suffix}",
                section_id="settings",
                document_title=title,
                section_path=(title, "설정"),
                source_url=f"https://docs.riido.io/split/{suffix}.md",
                category="split",
                content=f"{title}의 설정 방법입니다.",
                chunk_id=node.id,
                document_version_id=version.id,
                index_version_id=index_version.id,
            ),
        }


if __name__ == "__main__":
    unittest.main()
