"""질문 그룹핑 migration(20260915_13, 20260915_14)의 upgrade, 제약, downgrade 통합 테스트."""

import asyncio
import os
import subprocess
import sys
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings


REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_REVISION = "20260912_12"
GROUPING_REVISION = "20260915_13"
CLEANUP_REVISION = "20260915_14"

NEW_TABLES = (
    "question_problem_groups",
    "question_subproblems",
    "question_subproblem_revisions",
    "classification_runs",
    "question_classifications",
    "question_embeddings",
    "canonical_answers",
    "canonical_answer_citations",
    "question_cache_attempts",
)
REMOVED_TABLES = (
    "question_group_revisions",
    "question_reviews",
    "question_review_intents",
    "legacy_document_chunks",
    "legacy_chunk_embeddings",
)
PREVIOUS_GROUP_SOURCE_FK = (
    "fk_document_sources_group_source_id_document_group_sources"
)
GROUP_SOURCE_DOCUMENT_GROUP_FK = (
    "fk_document_sources_group_source_id_document_group_id"
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


class QuestionGroupingMigrationDbTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_available(cls.database_url)):
            raise unittest.SkipTest("로컬 PostgreSQL에 연결할 수 없습니다.")

    async def asyncSetUp(self) -> None:
        self.database_name = f"riido_grouping_{uuid.uuid4().hex[:12]}"
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

    async def test_upgrade_enforces_constraints_and_downgrade_restores(self) -> None:
        _alembic(self.scratch_url, "upgrade", PARENT_REVISION)
        seed = await self._seed_parent_schema()

        _alembic(self.scratch_url, "upgrade", "head")
        self.assertEqual(
            CLEANUP_REVISION,
            await self._scalar("SELECT version_num FROM alembic_version"),
        )
        for table in NEW_TABLES:
            self.assertTrue(await self._table(table), table)
        for table in REMOVED_TABLES:
            self.assertFalse(await self._table(table), table)
        self.assertFalse(await self._table("subproblem_centroids"))
        self.assertTrue(await self._table("question_access_audits"))
        self.assertFalse(await self._column("rag_runs", "sanitized_query"))
        self.assertTrue(await self._column("rag_runs", "query_hash"))
        # 기존 모델 호출 행은 새 허용 조합을 그대로 만족한다.
        self.assertEqual(2, await self._scalar("SELECT count(*) FROM model_calls"))

        await self._assert_document_source_group_match(seed)
        await self._assert_model_call_owner_combination(seed)
        await self._assert_open_online_run_is_unique(seed)
        grouping = await self._assert_problem_group_and_subproblem(seed)
        await self._assert_classification_rows(seed, grouping)
        answer_id = await self._assert_canonical_answers(seed, grouping)
        await self._assert_cache_attempts(seed, grouping, answer_id)

        # 턴을 지우면 연결 행과 캐시 시도가 함께 지워진다.
        async with self.engine.begin() as connection:
            await connection.execute(
                text("DELETE FROM rag_runs WHERE id = :id"),
                {"id": seed["rag_run_id"]},
            )
        for table in ("question_classifications", "question_cache_attempts"):
            self.assertEqual(
                0,
                await self._scalar(
                    f"SELECT count(*) FROM {table} WHERE rag_run_id = :id",
                    {"id": seed["rag_run_id"]},
                ),
            )
        self.assertEqual(
            2, await self._scalar("SELECT count(*) FROM question_cache_attempts")
        )

        # 판별 호출 행은 이전 purpose CHECK 를 어기므로 지운 뒤 되돌린다.
        async with self.engine.begin() as connection:
            await connection.execute(
                text(
                    "DELETE FROM model_calls "
                    "WHERE purpose = 'QUESTION_CLASSIFICATION'"
                )
            )
        _alembic(self.scratch_url, "downgrade", PARENT_REVISION)
        for table in NEW_TABLES:
            self.assertFalse(await self._table(table), table)
        for table in REMOVED_TABLES:
            self.assertTrue(await self._table(table), table)
        self.assertTrue(await self._column("rag_runs", "sanitized_query"))
        self.assertFalse(await self._column("model_calls", "classification_run_id"))
        self.assertFalse(await self._column("model_calls", "cached_input_tokens"))
        self.assertFalse(await self._column("model_calls", "reasoning_tokens"))
        self.assertEqual(
            {PREVIOUS_GROUP_SOURCE_FK},
            await self._foreign_keys("document_sources", "group_source_id"),
        )
        _alembic(self.scratch_url, "upgrade", "head")

    async def test_upgrade_stops_when_document_group_differs_from_group_source(self) -> None:
        _alembic(self.scratch_url, "upgrade", PARENT_REVISION)
        seed = await self._seed_parent_schema()
        async with self.engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO document_sources "
                    "(document_group_id, group_source_id, document_key, "
                    "source_type, canonical_uri) "
                    "VALUES (:group_id, :group_source_id, 'other/doc.md', "
                    "'GITBOOK', 'https://docs.riido.io/other/doc')"
                ),
                {
                    "group_id": seed["other_group_id"],
                    "group_source_id": seed["group_source_id"],
                },
            )

        with self.assertRaisesRegex(AssertionError, "원천의 문서 그룹과 다른"):
            _alembic(self.scratch_url, "upgrade", GROUPING_REVISION)
        self.assertEqual(
            PARENT_REVISION,
            await self._scalar("SELECT version_num FROM alembic_version"),
        )

    async def test_upgrade_stops_when_model_call_owner_is_ambiguous(self) -> None:
        _alembic(self.scratch_url, "upgrade", PARENT_REVISION)
        seed = await self._seed_parent_schema()
        async with self.engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO model_calls "
                    "(purpose, provider, model_name, status) "
                    "VALUES ('GENERATION', 'openai', 'gpt-test', 'SUCCESS')"
                )
            )
            await connection.execute(
                text(
                    "INSERT INTO model_calls "
                    "(rag_run_id, index_run_id, purpose, provider, model_name, status) "
                    "VALUES (:rag_run_id, :index_run_id, 'QUERY_EMBEDDING', "
                    "'openai', 'emb', 'SUCCESS')"
                ),
                seed,
            )

        with self.assertRaisesRegex(AssertionError, "실행 칸을 하나만 채우지 않은"):
            _alembic(self.scratch_url, "upgrade", GROUPING_REVISION)

    async def test_cleanup_stops_when_sanitized_query_has_value(self) -> None:
        _alembic(self.scratch_url, "upgrade", PARENT_REVISION)
        seed = await self._seed_parent_schema()
        async with self.engine.begin() as connection:
            await connection.execute(
                text("UPDATE rag_runs SET sanitized_query = 'x' WHERE id = :id"),
                {"id": seed["rag_run_id"]},
            )

        _alembic(self.scratch_url, "upgrade", GROUPING_REVISION)
        with self.assertRaisesRegex(AssertionError, "rag_runs.sanitized_query=1"):
            _alembic(self.scratch_url, "upgrade", CLEANUP_REVISION)
        self.assertTrue(await self._column("rag_runs", "sanitized_query"))

    # ------------------------------------------------------------------
    # 제약 검사
    # ------------------------------------------------------------------

    async def _assert_document_source_group_match(self, seed) -> None:
        await self._rejects(
            "INSERT INTO document_sources "
            "(document_group_id, group_source_id, document_key, source_type, "
            "canonical_uri) "
            "VALUES (:other_group_id, :group_source_id, 'mismatch.md', "
            "'GITBOOK', 'https://docs.riido.io/mismatch')",
            seed,
        )
        # 원천이 없는 업로드 문서는 복합 외래키 검사를 받지 않는다.
        await self._execute(
            "INSERT INTO document_sources "
            "(document_group_id, document_key, source_type, canonical_uri) "
            "VALUES (:other_group_id, 'upload.md', 'UPLOAD', "
            "'console://upload.md')",
            seed,
        )
        self.assertEqual(
            {GROUP_SOURCE_DOCUMENT_GROUP_FK},
            await self._foreign_keys("document_sources", "group_source_id"),
        )

    async def _assert_model_call_owner_combination(self, seed) -> None:
        run_id = await self._scalar_write(
            "INSERT INTO classification_runs "
            "(document_group_id, index_version_id, kind, model, prompt_version, "
            "started_at) "
            "VALUES (:group_id, :index_version_id, 'ONLINE', 'gpt-5.6-luna', "
            "'question-grouping-v7-2', now()) RETURNING id",
            seed,
        )
        seed["classification_run_id"] = run_id
        insert = (
            "INSERT INTO model_calls "
            "(rag_run_id, index_run_id, classification_run_id, purpose, provider, "
            "model_name, status) "
            "VALUES (:rag_run_id, :index_run_id, :classification_run_id, "
            ":purpose, 'openai', 'model', 'SUCCESS')"
        )

        def values(purpose, rag=False, index=False, classification=False):
            return {
                "purpose": purpose,
                "rag_run_id": seed["rag_run_id"] if rag else None,
                "index_run_id": seed["index_run_id"] if index else None,
                "classification_run_id": run_id if classification else None,
            }

        # 즉시 판별의 임베딩과 판정, 백필 판정
        await self._execute(insert, values("QUERY_EMBEDDING", rag=True, classification=True))
        await self._execute(insert, values("QUESTION_CLASSIFICATION", rag=True, classification=True))
        await self._execute(insert, values("QUESTION_CLASSIFICATION", classification=True))
        # 기존 턴과 색인 호출
        await self._execute(insert, values("ANSWER_GENERATION", rag=True))
        await self._execute(insert, values("CHUNK_EMBEDDING", index=True))

        await self._rejects(insert, values("QUESTION_CLASSIFICATION", rag=True))
        await self._rejects(insert, values("QUERY_EMBEDDING", classification=True))
        await self._rejects(insert, values("ANSWER_GENERATION", rag=True, classification=True))
        await self._rejects(insert, values("CHUNK_EMBEDDING", rag=True, index=True))
        await self._rejects(insert, values("QUESTION_CLASSIFICATION", index=True, classification=True))
        await self._rejects(insert, values("GENERATION"))

        # 캐시 입력과 추론 출력은 총량에 포함된 부분값이다.
        usage_insert = (
            "INSERT INTO model_calls "
            "(rag_run_id, purpose, provider, model_name, status, input_tokens, "
            "output_tokens, cached_input_tokens, reasoning_tokens) "
            "VALUES (:rag_run_id, 'ANSWER_GENERATION', 'openai', 'model', "
            "'SUCCESS', :input, :output, :cached, :reasoning)"
        )

        def usage(input=None, output=None, cached=None, reasoning=None):
            return {
                "rag_run_id": seed["rag_run_id"],
                "input": input,
                "output": output,
                "cached": cached,
                "reasoning": reasoning,
            }

        await self._execute(usage_insert, usage(100, 20, 100, 20))
        await self._execute(usage_insert, usage(100, 20))
        await self._execute(usage_insert, usage(cached=10, reasoning=2))
        await self._rejects(usage_insert, usage(100, 20, cached=101))
        await self._rejects(usage_insert, usage(100, 20, reasoning=21))
        await self._rejects(usage_insert, usage(cached=-1))
        await self._rejects(usage_insert, usage(reasoning=-1))

    async def _assert_open_online_run_is_unique(self, seed) -> None:
        insert = (
            "INSERT INTO classification_runs "
            "(document_group_id, index_version_id, kind, model, prompt_version, "
            "started_at, finished_at) "
            "VALUES (:group_id, :index_version_id, :kind, 'gpt-5.6-luna', "
            ":prompt_version, now(), :finished_at)"
        )

        def run(kind="ONLINE", prompt_version="question-grouping-v7-2", finished=False):
            return {
                "group_id": seed["group_id"],
                "index_version_id": seed["index_version_id"],
                "kind": kind,
                "prompt_version": prompt_version,
                "finished_at": (
                    datetime(2026, 9, 15, tzinfo=timezone.utc) if finished else None
                ),
            }

        # 같은 설정의 열린 ONLINE 실행은 이미 하나 있다.
        await self._rejects(insert, run())
        await self._execute(insert, run(finished=True))
        await self._execute(insert, run(kind="BACKFILL"))
        await self._execute(insert, run(prompt_version="question-grouping-v7-3"))
        # 기존 실행을 닫으면 같은 설정으로 새 실행을 열 수 있다.
        await self._execute(
            "UPDATE classification_runs SET finished_at = now() WHERE id = :id",
            {"id": seed["classification_run_id"]},
        )
        await self._execute(insert, run())
        await self._rejects(insert, run())

    async def _assert_problem_group_and_subproblem(self, seed):
        insert_group = (
            "INSERT INTO question_problem_groups "
            "(id, kind, document_source_id, document_group_id) "
            "VALUES (:id, :kind, :document_source_id, :document_group_id)"
        )

        def group(kind, source=None, document_group=None):
            return {
                "id": str(uuid.uuid4()),
                "kind": kind,
                "document_source_id": source,
                "document_group_id": document_group,
            }

        document_group = group("DOCUMENT", source=seed["document_source_id"])
        no_document_group = group("NO_DOCUMENT", document_group=seed["group_id"])
        await self._execute(insert_group, document_group)
        await self._execute(insert_group, no_document_group)

        await self._rejects(insert_group, group("DOCUMENT", document_group=seed["other_group_id"]))
        await self._rejects(
            insert_group,
            group("DOCUMENT", source=seed["upload_source_id"], document_group=seed["other_group_id"]),
        )
        await self._rejects(insert_group, group("NO_DOCUMENT", source=seed["upload_source_id"]))
        await self._rejects(insert_group, group("NO_DOCUMENT"))
        # 한 문서에 문제 그룹 하나, 문서 그룹당 가이드 밖 그룹 하나
        await self._rejects(insert_group, group("DOCUMENT", source=seed["document_source_id"]))
        await self._rejects(insert_group, group("NO_DOCUMENT", document_group=seed["group_id"]))

        insert_subproblem = (
            "INSERT INTO question_subproblems "
            "(id, problem_group_id, key, name, inclusion_criteria, current_version, "
            "status, created_by) "
            "VALUES (:id, :group_id, :key, '알림 끄기', '알림을 끄는 방법', 1, "
            "'APPROVED', 'backend')"
        )
        subproblem_id = str(uuid.uuid4())
        await self._execute(
            insert_subproblem,
            {
                "id": subproblem_id,
                "group_id": document_group["id"],
                "key": "notifications.turn-off",
            },
        )
        # key 는 문제 그룹 안에서만 유일하다.
        await self._rejects(
            insert_subproblem,
            {
                "id": str(uuid.uuid4()),
                "group_id": document_group["id"],
                "key": "notifications.turn-off",
            },
        )
        await self._execute(
            insert_subproblem,
            {
                "id": str(uuid.uuid4()),
                "group_id": no_document_group["id"],
                "key": "notifications.turn-off",
            },
        )
        await self._rejects(
            insert_subproblem,
            {
                "id": str(uuid.uuid4()),
                "group_id": document_group["id"],
                "key": None,
            },
        )
        self.assertEqual(
            "UNUSED",
            await self._scalar(
                "SELECT serving_state FROM question_subproblems WHERE id = :id",
                {"id": subproblem_id},
            ),
        )
        await self._rejects(
            "UPDATE question_subproblems SET serving_state = 'PAUSED' WHERE id = :id",
            {"id": subproblem_id},
        )
        insert_revision = (
            "INSERT INTO question_subproblem_revisions "
            "(subproblem_id, version, name_snapshot, inclusion_snapshot, approved_by) "
            "VALUES (:id, 1, '알림 끄기', '알림을 끄는 방법', 'backend')"
        )
        await self._execute(insert_revision, {"id": subproblem_id})
        await self._rejects(insert_revision, {"id": subproblem_id})

        # 포함 기준 벡터, 임베딩 설정, 입력 문장 구성 판은 함께 채우거나 함께 비운다.
        insert_embedded_revision = (
            "INSERT INTO question_subproblem_revisions "
            "(subproblem_id, version, name_snapshot, inclusion_snapshot, approved_by, "
            "inclusion_embedding, embedding_config_id, embedding_text_version) "
            "VALUES (:id, :version, '알림 끄기', '알림을 끄는 방법', 'backend', "
            "CAST(:embedding AS vector), :embedding_config_id, :text_version)"
        )
        embedding = "[" + ",".join(["0.01"] * 1536) + "]"
        full = {
            "id": subproblem_id,
            "version": 2,
            "embedding": embedding,
            "embedding_config_id": seed["embedding_config_id"],
            "text_version": "name-inclusion-v1",
        }
        await self._execute(insert_embedded_revision, full)
        for missing in ("embedding", "embedding_config_id", "text_version"):
            await self._rejects(
                insert_embedded_revision,
                {**full, "version": 3, missing: None},
            )
        await self._rejects(
            insert_embedded_revision,
            {**full, "version": 3, "embedding": None, "embedding_config_id": None},
        )
        # 개정 이력이 쓰는 임베딩 설정은 지울 수 없다.
        await self._rejects(
            "DELETE FROM embedding_configs WHERE id = :id",
            {"id": seed["embedding_config_id"]},
        )

        return {
            "document_group_id": document_group["id"],
            "no_document_group_id": no_document_group["id"],
            "subproblem_id": subproblem_id,
        }

    async def _assert_classification_rows(self, seed, grouping) -> None:
        insert = (
            "INSERT INTO question_classifications "
            "(rag_run_id, subproblem_id, problem_group_id, run_id, decision, "
            "subproblem_version, attribution_source, judgment_input, effective_to) "
            "VALUES (:rag_run_id, :subproblem_id, :problem_group_id, :run_id, "
            ":decision, :version, :source, '{}'::jsonb, :effective_to) "
            "RETURNING id"
        )

        def row(decision, source, subproblem=False, group=None, effective_to=None, rag_run_id=None):
            return {
                "rag_run_id": rag_run_id or seed["rag_run_id"],
                "subproblem_id": grouping["subproblem_id"] if subproblem else None,
                "version": 1 if subproblem else None,
                "problem_group_id": group or grouping["document_group_id"],
                "run_id": seed["classification_run_id"],
                "decision": decision,
                "source": source,
                "effective_to": effective_to,
            }

        await self._rejects(insert, row("CONNECT", "SUBPROBLEM"))
        await self._rejects(insert, row("CONNECT", "CITATION", subproblem=True))
        await self._rejects(insert, row("SEPARATE", "SUBPROBLEM"))
        await self._rejects(insert, row("SEPARATE", "DOCUMENT", subproblem=True))
        await self._rejects(
            insert,
            {**row("UNCLASSIFIED", "NONE"), "version": 1},
        )

        first_id = await self._scalar_write(
            insert,
            row("UNCLASSIFIED", "NONE", group=grouping["no_document_group_id"]),
        )
        # 질문마다 현재 행은 하나다.
        await self._rejects(insert, row("CONNECT", "SUBPROBLEM", subproblem=True))
        await self._execute(
            "UPDATE question_classifications SET effective_to = effective_from "
            "WHERE id = :id",
            {"id": first_id},
        )
        current_id = await self._scalar_write(
            insert, row("CONNECT", "SUBPROBLEM", subproblem=True)
        )
        seed["classification_id"] = current_id
        seed["closed_classification_id"] = first_id
        await self._rejects(
            "UPDATE question_classifications "
            "SET effective_to = effective_from - interval '1 second' WHERE id = :id",
            {"id": current_id},
        )

    async def _assert_canonical_answers(self, seed, grouping) -> str:
        insert = (
            "INSERT INTO canonical_answers "
            "(id, subproblem_id, origin, content_markdown, subproblem_version, approval) "
            "VALUES (:id, :subproblem_id, 'AUTHORED', '알림은 설정에서 끕니다 [1]', "
            "1, :approval)"
        )
        revoked_id = str(uuid.uuid4())
        approved_id = str(uuid.uuid4())
        subproblem_id = grouping["subproblem_id"]
        await self._execute(insert, {"id": revoked_id, "subproblem_id": subproblem_id, "approval": "REVOKED"})
        await self._execute(insert, {"id": approved_id, "subproblem_id": subproblem_id, "approval": "APPROVED"})
        await self._rejects(
            insert,
            {"id": str(uuid.uuid4()), "subproblem_id": subproblem_id, "approval": "APPROVED"},
        )

        citation = (
            "INSERT INTO canonical_answer_citations "
            "(canonical_answer_id, citation_order, chunk_id, document_version_id, "
            "document_title_snapshot) "
            "VALUES (:answer_id, 1, :chunk_id, :document_version_id, '알림')"
        )
        values = {
            "answer_id": approved_id,
            "chunk_id": seed["chunk_id"],
            "document_version_id": seed["document_version_id"],
        }
        await self._execute(citation, values)
        await self._rejects(citation, values)
        # 정본이 쓰는 청크와 문서 판은 지울 수 없다.
        await self._rejects(
            "DELETE FROM document_chunks WHERE id = :chunk_id", values
        )
        await self._rejects(
            "DELETE FROM document_versions WHERE id = :document_version_id", values
        )

        # 출처 턴이 지워지면 정본은 남고 출처만 비운다.
        source_run_id = await self._insert_rag_run(seed, turn_no=9)
        await self._execute(
            "UPDATE canonical_answers SET source_rag_run_id = :run_id WHERE id = :id",
            {"run_id": source_run_id, "id": revoked_id},
        )
        await self._execute("DELETE FROM rag_runs WHERE id = :id", {"id": source_run_id})
        self.assertIsNone(
            await self._scalar(
                "SELECT source_rag_run_id FROM canonical_answers WHERE id = :id",
                {"id": revoked_id},
            )
        )
        return approved_id

    async def _assert_cache_attempts(self, seed, grouping, answer_id) -> None:
        insert = (
            "INSERT INTO question_cache_attempts "
            "(rag_run_id, classification_id, outcome, canonical_answer_id, "
            "rejection_reasons) "
            "VALUES (:rag_run_id, :classification_id, :outcome, :answer_id, "
            "CAST(:reasons AS varchar[]))"
        )

        def attempt(outcome, answer=False, reasons=None, rag_run_id=None, classification_id=None):
            return {
                "rag_run_id": rag_run_id or seed["rag_run_id"],
                "classification_id": classification_id,
                "outcome": outcome,
                "answer_id": answer_id if answer else None,
                "reasons": reasons,
            }

        for outcome in ("SERVED", "SHADOW", "GROUP_DISABLED"):
            await self._rejects(insert, attempt(outcome))
        for outcome in ("REJECTED", "SKIPPED", "FAILED"):
            await self._rejects(insert, attempt(outcome, answer=True, reasons=["X"]))
        await self._rejects(insert, attempt("REJECTED"))
        await self._rejects(insert, attempt("REJECTED", reasons=[]))
        await self._rejects(insert, attempt("SKIPPED", reasons=["SUBPROBLEM_NOT_SERVING"]))
        await self._rejects(insert, attempt("SERVED", answer=True, reasons=["X"]))

        await self._execute(
            insert,
            attempt("SERVED", answer=True, classification_id=seed["classification_id"]),
        )
        # 턴 하나에 시도는 많아야 하나
        await self._rejects(insert, attempt("SKIPPED"))

        other_run_id = await self._insert_rag_run(seed, turn_no=2)
        third_run_id = await self._insert_rag_run(seed, turn_no=3)
        # 연결 행 하나에 시도는 많아야 하나
        await self._rejects(
            insert,
            attempt(
                "FAILED",
                rag_run_id=other_run_id,
                classification_id=seed["classification_id"],
            ),
        )
        await self._execute(
            insert,
            attempt("REJECTED", reasons=["GROUP_REVISION_MISMATCH"], rag_run_id=other_run_id),
        )
        await self._execute(insert, attempt("SKIPPED", rag_run_id=third_run_id))
        # 캐시 시도가 가리키는 정본은 지울 수 없다.
        await self._rejects(
            "DELETE FROM canonical_answers WHERE id = :id", {"id": answer_id}
        )

    # ------------------------------------------------------------------
    # 시드와 조회 도우미
    # ------------------------------------------------------------------

    async def _seed_parent_schema(self) -> dict:
        seed = {}
        async with self.engine.begin() as connection:
            async def scalar(statement, parameters=None):
                return (
                    await connection.execute(text(statement), parameters or {})
                ).scalar_one()

            seed["group_id"] = await scalar(
                "SELECT id FROM document_groups WHERE group_key = 'HELP_CHATBOT'"
            )
            seed["other_group_id"] = await scalar(
                "INSERT INTO document_groups (group_key, name, consumer_key) "
                "VALUES ('OTHER_GROUP', '다른 그룹', 'OTHER') RETURNING id"
            )
            seed["group_source_id"] = await scalar(
                "INSERT INTO document_group_sources "
                "(document_group_id, provider, root_url) "
                "VALUES (:group_id, 'GITBOOK', 'https://docs.riido.io') "
                "ON CONFLICT (document_group_id, root_url) DO UPDATE "
                "SET enabled = true RETURNING id",
                seed,
            )
            seed["document_source_id"] = await scalar(
                "INSERT INTO document_sources "
                "(document_group_id, group_source_id, document_key, source_type, "
                "canonical_uri) "
                "VALUES (:group_id, :group_source_id, 'notifications.md', "
                "'GITBOOK', 'https://docs.riido.io/notifications') RETURNING id",
                seed,
            )
            seed["upload_source_id"] = await scalar(
                "INSERT INTO document_sources "
                "(document_group_id, document_key, source_type, canonical_uri) "
                "VALUES (:group_id, 'faq.md', 'UPLOAD', 'console://faq.md') "
                "RETURNING id",
                seed,
            )
            seed["document_version_id"] = await scalar(
                "INSERT INTO document_versions "
                "(document_source_id, version_no, raw_content, mime_type, "
                "raw_content_hash, normalized_content_hash, parser_name, "
                "parser_version, status, collected_at) "
                "VALUES (:document_source_id, 1, '# 알림', 'text/markdown', "
                "'raw', 'normalized', 'markdown', 'v1', 'READY', now()) "
                "RETURNING id",
                seed,
            )
            seed["chunking_config_id"] = await scalar(
                "INSERT INTO chunking_configs (version, strategy, max_tokens) "
                "VALUES ('grouping-test', 'section', 500) RETURNING id"
            )
            seed["embedding_config_id"] = await scalar(
                "INSERT INTO embedding_configs "
                "(version, provider, model_name, dimensions, input_template_version) "
                "VALUES ('grouping-test', 'openai', 'text-embedding-3-small', "
                "1536, 'v1') RETURNING id"
            )
            node_id = await scalar(
                "INSERT INTO content_nodes "
                "(document_version_id, node_type, node_order, normalized_content, "
                "content_hash) "
                "VALUES (:document_version_id, 'SECTION', 0, '알림', 'hash') "
                "RETURNING id",
                seed,
            )
            await connection.execute(
                text(
                    "INSERT INTO document_chunks (id, chunking_config_id, chunk_index) "
                    "VALUES (:id, :chunking_config_id, 0)"
                ),
                {"id": node_id, **seed},
            )
            seed["chunk_id"] = node_id
            seed["index_version_id"] = await scalar(
                "INSERT INTO index_versions "
                "(document_group_id, version, status, chunking_config_id, "
                "embedding_config_id) "
                "VALUES (:group_id, 'grouping-test', 'READY', "
                ":chunking_config_id, :embedding_config_id) RETURNING id",
                seed,
            )
            seed["index_run_id"] = await scalar(
                "INSERT INTO index_runs "
                "(index_version_id, trigger_type, operation_type, stage, status, "
                "started_at) "
                "VALUES (:index_version_id, 'MANUAL', 'BUILD_AND_APPLY', "
                "'BUILDING', 'SUCCESS', now()) RETURNING id",
                seed,
            )
            seed["profile_revision_id"] = await scalar(
                "SELECT r.id FROM chat_profile_revisions r "
                "JOIN chat_profiles p ON p.id = r.profile_id "
                "WHERE p.profile_key = 'HELP_CHATBOT' AND r.status = 'PUBLISHED'"
            )
            seed["conversation_id"] = str(uuid.uuid4())
            await connection.execute(
                text(
                    "INSERT INTO conversations "
                    "(id, chat_profile_revision_id, status, created_at, last_active_at) "
                    "VALUES (:conversation_id, :profile_revision_id, 'ACTIVE', "
                    "now(), now())"
                ),
                seed,
            )
            seed["rag_run_id"] = str(uuid.uuid4())
            await connection.execute(
                text(
                    "INSERT INTO rag_runs "
                    "(id, trace_id, conversation_id, turn_no, index_version_id, "
                    "user_query, context_strategy, status) "
                    "VALUES (:rag_run_id, :trace_id, :conversation_id, 1, "
                    ":index_version_id, '알림 끄는 법', 'NEW_TOPIC', 'COMPLETED')"
                ),
                {"trace_id": str(uuid.uuid4()), **seed},
            )
            await connection.execute(
                text(
                    "INSERT INTO model_calls "
                    "(rag_run_id, purpose, provider, model_name, status) "
                    "VALUES (:rag_run_id, 'ANSWER_GENERATION', 'openai', "
                    "'gpt-test', 'SUCCESS')"
                ),
                seed,
            )
            await connection.execute(
                text(
                    "INSERT INTO model_calls "
                    "(index_run_id, purpose, provider, model_name, status) "
                    "VALUES (:index_run_id, 'CHUNK_EMBEDDING', 'openai', "
                    "'text-embedding-3-small', 'SUCCESS')"
                ),
                seed,
            )
        return seed

    async def _insert_rag_run(self, seed, *, turn_no: int) -> str:
        rag_run_id = str(uuid.uuid4())
        await self._execute(
            "INSERT INTO rag_runs "
            "(id, trace_id, conversation_id, turn_no, index_version_id, "
            "user_query, context_strategy, status) "
            "VALUES (:id, :trace_id, :conversation_id, :turn_no, "
            ":index_version_id, '질문', 'NEW_TOPIC', 'COMPLETED')",
            {
                "id": rag_run_id,
                "trace_id": str(uuid.uuid4()),
                "turn_no": turn_no,
                "conversation_id": seed["conversation_id"],
                "index_version_id": seed["index_version_id"],
            },
        )
        return rag_run_id

    async def _execute(self, statement: str, parameters=None) -> None:
        async with self.engine.begin() as connection:
            await connection.execute(text(statement), parameters or {})

    async def _scalar_write(self, statement: str, parameters=None):
        async with self.engine.begin() as connection:
            return (
                await connection.execute(text(statement), parameters or {})
            ).scalar_one()

    async def _rejects(self, statement: str, parameters=None) -> None:
        with self.assertRaises(IntegrityError, msg=statement):
            await self._execute(statement, parameters)

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

    async def _scalar(self, statement: str, parameters=None):
        async with self.engine.connect() as connection:
            return (
                await connection.execute(text(statement), parameters or {})
            ).scalar_one()

    async def _table(self, name: str) -> bool:
        return await self._scalar("SELECT to_regclass(:name)", {"name": name}) is not None

    async def _column(self, table: str, column: str) -> bool:
        return bool(await self._scalar(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = :table AND column_name = :column",
            {"table": table, "column": column},
        ))

    async def _foreign_keys(self, table: str, column: str) -> set:
        async with self.engine.connect() as connection:
            result = await connection.execute(
                text(
                    "SELECT DISTINCT tc.constraint_name "
                    "FROM information_schema.table_constraints tc "
                    "JOIN information_schema.key_column_usage kcu "
                    "ON kcu.constraint_name = tc.constraint_name "
                    "AND kcu.table_schema = tc.table_schema "
                    "WHERE tc.table_name = :table "
                    "AND tc.constraint_type = 'FOREIGN KEY' "
                    "AND kcu.column_name = :column"
                ),
                {"table": table, "column": column},
            )
            return set(result.scalars().all())


if __name__ == "__main__":
    unittest.main()
