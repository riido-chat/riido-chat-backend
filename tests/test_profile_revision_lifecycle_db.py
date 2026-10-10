"""임시 PostgreSQL에서 수명주기, 직렬화와 DB 불변 규칙을 검증한다.

PROFILE_LIFECYCLE_TEST_DATABASE_URL 을 명시했을 때만 실행한다. 고유한 임시 DB를 생성하고
삭제하며 합성 서비스와 대화만 사용한다. TESTING은 모델 호출을 수행하지 않는다.
"""

import asyncio
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import unittest
import uuid
from unittest.mock import patch

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import prompts
from app.chat.profile import resolve_chat_profile_revision
from app.database.models import (
    ChatProfile, ChatProfileRevision, ChatProfileRevisionStatus as Status, Conversation,
    ConversationChannel, DocumentGroup, ProfileRevisionComponent,
)
from app.profiles.lifecycle import ProfileRevisionError, ProfileRevisionService
from app.profiles.services import ChatServiceRoute
from app.profiles.stage_settings import code_components
from app.prompts.fingerprints import load_fingerprint_table, sha256_text


class ProfileRevisionLifecycleDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = os.environ.get("PROFILE_LIFECYCLE_TEST_DATABASE_URL")
        if not cls.base_url:
            raise unittest.SkipTest("전용 임시 PostgreSQL URL을 명시해야 합니다.")
        # 실행자가 실수로 공유 DB를 지정해도 기존 DB 테이블을 수정하지 않는다.
        cls.database_name = "riido_lifecycle_" + uuid.uuid4().hex[:12]
        cls.url = make_url(cls.base_url).set(database=cls.database_name).render_as_string(hide_password=False)
        asyncio.run(cls._maintenance(f'CREATE DATABASE "{cls.database_name}"'))
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "20261010_21"],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "DATABASE_URL": cls.url}, capture_output=True, text=True,
        )
        if result.returncode:
            asyncio.run(cls._maintenance(f'DROP DATABASE "{cls.database_name}" WITH (FORCE)'))
            raise AssertionError("임시 DB migration 실패: " + result.stderr)

    @classmethod
    def tearDownClass(cls):
        if cls.base_url:
            asyncio.run(cls._maintenance(f'DROP DATABASE "{cls.database_name}" WITH (FORCE)'))

    @classmethod
    async def _maintenance(cls, statement):
        engine = create_async_engine(make_url(cls.base_url).set(database="postgres"), isolation_level="AUTOCOMMIT")
        try:
            async with engine.connect() as connection:
                await connection.execute(text(statement))
        finally:
            await engine.dispose()

    async def asyncSetUp(self):
        self.engine = create_async_engine(self.url)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.keys = ["SYNTHETIC_" + uuid.uuid4().hex[:12] for _ in range(2)]
        async with self.sessions() as session, session.begin():
            for key in self.keys:
                group = DocumentGroup(group_key=key, name="Synthetic", consumer_key=key)
                session.add(group)
                await session.flush()
                session.add(ChatProfile(profile_key=key, name="Synthetic", document_group_id=group.id))
        self.a, self.b = self.keys

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def call(self, method, key, *args, **kwargs):
        async with self.sessions() as session:
            return await getattr(ProfileRevisionService(session), method)(key, *args, **kwargs)

    async def published(self, key):
        detail = await self.call("create", key)
        await self.call("testing", key, detail.revision.id)
        return await self.call("publish", key, detail.revision.id)

    async def state(self, revision_id):
        async with self.sessions() as session:
            return (await session.get(ChatProfileRevision, revision_id)).status

    async def test_publish_and_rollback_affect_next_turn_and_keep_services_separate(self):
        a1, b1 = await self.published(self.a), await self.published(self.b)
        async with self.sessions() as session, session.begin():
            conversation = Conversation(profile_id=a1.revision.profile_id, chat_profile_revision_id=a1.revision.id)
            session.add(conversation)
            await session.flush()
            conversation_id = conversation.id
        a2 = await self.call("create", self.a)
        await self.call("testing", self.a, a2.revision.id)
        # TESTING은 현재 서빙 버전을 바꾸지 않는다.
        self.assertEqual(Status.PUBLISHED, await self.state(a1.revision.id))
        await self.call("publish", self.a, a2.revision.id)
        self.assertEqual(Status.RETIRED, await self.state(a1.revision.id))
        self.assertEqual(Status.PUBLISHED, await self.state(b1.revision.id))
        service = ChatServiceRoute(self.a, ConversationChannel.PUBLIC)
        async with self.sessions() as session:
            resolved = await resolve_chat_profile_revision(session, service=service, conversation_id=conversation_id)
            self.assertEqual(a2.revision.id, resolved.id)
        await self.call("rollback", self.a, a1.revision.id)
        async with self.sessions() as session:
            resolved = await resolve_chat_profile_revision(session, service=service, conversation_id=conversation_id)
            self.assertEqual(a1.revision.id, resolved.id)
            self.assertEqual(a1.revision.id, (await session.get(Conversation, conversation_id)).chat_profile_revision_id)
        self.assertEqual(Status.RETIRED, await self.state(a2.revision.id))

    async def test_cross_service_revision_ids_are_not_found(self):
        foreign = await self.call("create", self.b)
        for method in ("detail", "testing", "publish", "rollback", "delete"):
            with self.subTest(method=method), self.assertRaises(ProfileRevisionError) as error:
                await self.call(method, self.a, foreign.revision.id)
            self.assertEqual(404, error.exception.status_code)
        self.assertEqual(Status.DRAFT, await self.state(foreign.revision.id))

    async def test_registered_older_prompt_and_selected_models_are_allowed(self):
        specs = list(code_components())
        old_version = "v9-synthetic-registered"
        body = "synthetic previous registered rewrite instructions"
        table = dict(load_fingerprint_table())
        table[f"rewrite.output_schema@{old_version}"] = table[f"rewrite.output_schema@{specs[0].prompt_version}"]
        original_get = prompts.get
        def registered(component, version):
            return body if (component, version) == ("rewrite", old_version) else original_get(component, version)
        specs[0] = replace(specs[0], model_name="selected-rewrite-model", prompt_version=old_version,
            prompt_key=f"rewrite@{old_version}", prompt_sha256=sha256_text(body))
        specs = [replace(spec, model_name="selected-generation-model") if spec.stage.value not in ("rewrite", "judge") else spec for spec in specs]
        with patch("app.profiles.stage_settings.prompts.get", side_effect=registered), patch("app.profiles.stage_settings._fingerprint_table", return_value=table):
            detail = await self.call("create", self.a, components=specs)
            await self.call("testing", self.a, detail.revision.id)
            await self.call("publish", self.a, detail.revision.id)
        self.assertEqual(old_version, detail.revision.query_rewrite_prompt_version)
        self.assertEqual("selected-generation-model", detail.revision.generation_model_name)

    async def test_invalid_creation_and_validation_failure_are_atomic(self):
        active = await self.published(self.a)
        specs = list(code_components())
        specs[0] = replace(specs[0], prompt_sha256="0" * 64)
        with self.assertRaises(ProfileRevisionError) as error:
            await self.call("create", self.a, components=specs)
        self.assertEqual(400, error.exception.status_code)
        self.assertEqual(1, len(await self.call("list", self.a)))
        candidate = await self.call("create", self.a)
        await self.call("testing", self.a, candidate.revision.id)
        with patch("app.profiles.lifecycle.component_problems", return_value=("synthetic incompatible deployment",)):
            with self.assertRaises(ProfileRevisionError):
                await self.call("publish", self.a, candidate.revision.id)
        self.assertEqual(Status.PUBLISHED, await self.state(active.revision.id))
        self.assertEqual(Status.TESTING, await self.state(candidate.revision.id))

    async def test_failure_after_retiring_previous_rolls_back_both_statuses(self):
        active = await self.published(self.a)
        candidate = await self.call("create", self.a)
        await self.call("testing", self.a, candidate.revision.id)
        async with self.sessions() as session:
            original_flush = session.flush
            count = 0
            async def failing_flush(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 2:
                    raise IntegrityError("synthetic constraint", {}, RuntimeError("synthetic"))
                return await original_flush(*args, **kwargs)
            with patch.object(session, "flush", side_effect=failing_flush):
                with self.assertRaises(ProfileRevisionError) as error:
                    await ProfileRevisionService(session).publish(self.a, candidate.revision.id)
            self.assertEqual(409, error.exception.status_code)
        self.assertEqual(Status.PUBLISHED, await self.state(active.revision.id))
        self.assertEqual(Status.TESTING, await self.state(candidate.revision.id))

    async def test_concurrent_creations_get_distinct_versions(self):
        details = await asyncio.gather(*(self.call("create", self.a) for _ in range(8)))
        self.assertEqual(list(range(1, 9)), sorted(detail.revision.version for detail in details))
        self.assertEqual(8, len({detail.revision.id for detail in details}))

    async def test_concurrent_transitions_keep_single_active_version(self):
        old = await self.published(self.a)
        candidates = await asyncio.gather(self.call("create", self.a), self.call("create", self.a))
        await asyncio.gather(*(self.call("testing", self.a, item.revision.id) for item in candidates))
        states = [await self.state(item.revision.id) for item in candidates]
        self.assertEqual({Status.TESTING, Status.RETIRED}, set(states))
        first = candidates[states.index(Status.TESTING)]
        results = await asyncio.gather(*(self.call("publish", self.a, first.revision.id) for _ in range(2)), return_exceptions=True)
        self.assertEqual(1, sum(isinstance(item, ProfileRevisionError) for item in results))
        # RETIRED 버전 두 개의 동시 복원도 서비스 잠금으로 직렬화한다.
        retired = candidates[states.index(Status.RETIRED)]
        await asyncio.gather(self.call("rollback", self.a, old.revision.id), self.call("rollback", self.a, retired.revision.id))
        revisions = await self.call("list", self.a)
        self.assertEqual(1, sum(row.status == Status.PUBLISHED for row in revisions))

    async def test_only_unreferenced_drafts_can_be_deleted(self):
        draft = await self.call("create", self.a)
        await self.call("delete", self.a, draft.revision.id)
        with self.assertRaises(ProfileRevisionError):
            await self.call("detail", self.a, draft.revision.id)
        draft = await self.call("create", self.a)
        async with self.sessions() as session, session.begin():
            session.add(Conversation(profile_id=draft.revision.profile_id, chat_profile_revision_id=draft.revision.id))
        with self.assertRaises(ProfileRevisionError) as error:
            await self.call("delete", self.a, draft.revision.id)
        self.assertEqual(409, error.exception.status_code)
        self.assertEqual(7, len((await self.call("detail", self.a, draft.revision.id)).components))
        await self.call("testing", self.a, draft.revision.id)
        with self.assertRaises(ProfileRevisionError):
            await self.call("delete", self.a, draft.revision.id)

    async def test_db_triggers_protect_configuration_and_transitions(self):
        draft = await self.call("create", self.a)
        statements = (
            "UPDATE chat_profile_revisions SET generation_model_name = 'changed' WHERE id = :id",
            "UPDATE chat_profile_revisions SET status = 'RETIRED' WHERE id = :id",
            "UPDATE profile_revision_components SET model_name = 'changed' WHERE revision_id = :id",
            "INSERT INTO profile_revision_components (revision_id, stage, model_name, prompt_key, prompt_version, prompt_sha256, recorded_by) VALUES (:id, 'rewrite', 'm', 'rewrite@v10', 'v10', NULL, 'PUBLISH')",
        )
        for statement in statements:
            with self.subTest(statement=statement), self.assertRaises(IntegrityError):
                async with self.engine.begin() as connection:
                    await connection.execute(text(statement), {"id": draft.revision.id})
        self.assertEqual(Status.DRAFT, await self.state(draft.revision.id))

    async def test_actual_http_create_response_after_commit(self):
        from httpx import ASGITransport, AsyncClient
        from app.core.config import Settings
        from app.database.session import get_db_session
        from app.main import create_app
        with patch("app.main.get_settings", return_value=Settings(_env_file=None, profile_revision_admin_enabled=True)):
            app = create_app()
        async def database():
            async with self.sessions() as session:
                yield session
        app.dependency_overrides[get_db_session] = database
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://synthetic") as client:
            base = f"/api/admin/profiles/{self.a}/revisions"
            response = await client.post(base, json={})
            self.assertEqual(201, response.status_code)
            created = response.json()
            self.assertEqual("DRAFT", created["status"])
            self.assertEqual(7, len(created["components"]))
            self.assertTrue(created["createdAt"])
            response = await client.post(base + f"/{created['id']}/testing")
            self.assertEqual("TESTING", response.json()["status"])
            response = await client.post(base + f"/{created['id']}/publish")
            self.assertEqual("PUBLISHED", response.json()["status"])
            response = await client.get(base + f"/{created['id']}")
            self.assertEqual(200, response.status_code)
            self.assertEqual(created["components"], response.json()["components"])

    async def test_rag_run_reference_blocks_draft_deletion(self):
        draft = await self.call("create", self.a)
        owner = await self.published(self.b)
        # 참조가 자기 서비스의 대화와 달라도 rag_runs FK를 검사한다.
        async with self.engine.begin() as connection:
            conversation = uuid.uuid4()
            await connection.execute(text(
                "INSERT INTO conversations (id, profile_id, chat_profile_revision_id) "
                "VALUES (:id, :profile, :revision)"
            ), {"id": conversation, "profile": owner.revision.profile_id, "revision": owner.revision.id})
            chunking = (await connection.execute(text(
                "INSERT INTO chunking_configs (version, strategy, max_tokens) "
                "VALUES (:version, 'synthetic', 500) RETURNING id"
            ), {"version": self.a})).scalar_one()
            embedding = (await connection.execute(text(
                "INSERT INTO embedding_configs (version, provider, model_name, dimensions, input_template_version) "
                "VALUES (:version, 'synthetic', 'synthetic', 1536, 'v1') RETURNING id"
            ), {"version": self.a})).scalar_one()
            index = (await connection.execute(text(
                "INSERT INTO index_versions (document_group_id, version, status, chunking_config_id, embedding_config_id) "
                "VALUES (:group, :version, 'ACTIVE', :chunking, :embedding) RETURNING id"
            ), {"group": owner.revision.document_group_id, "version": self.a,
                "chunking": chunking, "embedding": embedding})).scalar_one()
            await connection.execute(text(
                "INSERT INTO rag_runs (id, trace_id, turn_no, context_strategy, conversation_id, index_version_id, profile_revision_id, "
                "user_query, status) VALUES (:id, :trace, 1, 'NEW_TOPIC', :conversation, :index, :revision, 'synthetic question', 'PROCESSING')"
            ), {"id": uuid.uuid4(), "trace": uuid.uuid4(), "conversation": conversation, "index": index, "revision": draft.revision.id})
        with self.assertRaises(ProfileRevisionError) as error:
            await self.call("delete", self.a, draft.revision.id)
        self.assertEqual(409, error.exception.status_code)
        self.assertEqual(7, len((await self.call("detail", self.a, draft.revision.id)).components))

    async def test_cache_defaults_inherit_current_published_only(self):
        first = await self.call("create", self.a, semantic_cache_enabled=True, exact_cache_enabled=True)
        second = await self.call("create", self.a)
        self.assertFalse(second.revision.semantic_cache_enabled)
        self.assertFalse(second.revision.exact_cache_enabled)
        await self.call("testing", self.a, first.revision.id)
        await self.call("publish", self.a, first.revision.id)
        third = await self.call("create", self.a)
        self.assertTrue(third.revision.semantic_cache_enabled)
        self.assertTrue(third.revision.exact_cache_enabled)
        override = await self.call("create", self.a, semantic_cache_enabled=False)
        self.assertFalse(override.revision.semantic_cache_enabled)
        self.assertTrue(override.revision.exact_cache_enabled)
