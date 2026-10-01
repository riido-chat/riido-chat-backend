"""프로필 판 구성 migration(20261001_17)의 고정값과 백필 규칙 단위 테스트.

DB 없이 돈다. 백필 SQL 은 SQLite 메모리 DB 에서 실제로 실행해 규칙을 확인하고,
PostgreSQL 에서의 upgrade, 제약, downgrade 는 test_profile_revision_provenance_migration_db 가 본다.
"""

import importlib.util
import json
import unittest
from pathlib import Path

import sqlalchemy as sa

from app.core.prompt_fingerprints import load_fingerprint_table
from app.database.models import (
    PROFILE_REVISION_COMPONENT_PROMPT_KEY_CHECK,
    PROFILE_REVISION_COMPONENT_PROMPT_KEYS,
    ProfileRevisionComponentRecordedBy,
    ProfileRevisionComponentStage,
)


MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "alembic"
    / "versions"
    / "20261001_17_add_profile_revision_components.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("profile_components_migration", MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


migration = _load_migration()

# (id, status, generation 모델, generation 버전, rewrite 모델, rewrite 버전)
REVISIONS = (
    (1, "PUBLISHED", "gpt-5.6-terra", "v40", "gpt-5.4-mini", "v10"),
    (2, "TESTING", "gpt-5.6-terra", "v40", "gpt-5.4-mini", "v8"),
    (3, "DRAFT", "gpt-5.6-terra", "v40", "gpt-5.4-mini", "v10"),
    (4, "RETIRED", "gpt-5.6-terra", "v38", "gpt-5.4-mini", "v8"),
    (5, "PUBLISHED", "gpt-other", "v24", "gpt-other-mini", "v7"),
)
SUB_STAGES = {"source_planning", "source_planning_repair", "answer", "answer_repair"}


def _run_backfill():
    """SQLite 에 판 표와 구성 표를 만들고 migration 백필 문을 그대로 실행한다."""

    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE chat_profile_revisions (id INTEGER PRIMARY KEY, status TEXT, "
                "generation_model_name TEXT, generation_prompt_version TEXT, "
                "query_rewrite_model_name TEXT, query_rewrite_prompt_version TEXT)"
            )
        )
        connection.execute(
            sa.text(
                f"CREATE TABLE {migration.TABLE} (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "revision_id INTEGER, stage TEXT, model_name TEXT, prompt_key TEXT, "
                "prompt_version TEXT, prompt_sha256 TEXT, params TEXT, recorded_by TEXT, "
                "UNIQUE (revision_id, stage))"
            )
        )
        for row in REVISIONS:
            connection.execute(
                sa.text(
                    "INSERT INTO chat_profile_revisions VALUES "
                    "(:id, :status, :gm, :gv, :rm, :rv)"
                ),
                dict(zip(("id", "status", "gm", "gv", "rm", "rv"), row)),
            )
        for statement, params in migration.backfill_statements():
            # SQLite 는 JSONB 형이 없어 JSON 문자열을 그대로 넣는다.
            statement = statement.replace("CAST(:params AS JSONB)", ":params")
            connection.execute(sa.text(statement).bindparams(**params))
        rows = connection.execute(
            sa.text(
                f"SELECT revision_id, stage, model_name, prompt_key, prompt_version, "
                f"prompt_sha256, params, recorded_by FROM {migration.TABLE}"
            )
        ).all()
    engine.dispose()
    return {(row[0], row[1]): row[2:] for row in rows}


class MigrationConstantsTest(unittest.TestCase):
    def test_hardcoded_fingerprints_match_append_only_table(self) -> None:
        table = load_fingerprint_table()
        expected = {
            f"rewrite@{migration.REWRITE_VERSION}": migration.REWRITE_SHA256,
            f"generation@{migration.GENERATION_VERSION}": migration.GENERATION_SHA256,
            f"judge@{migration.JUDGE_VERSION}": migration.JUDGE_SHA256,
            **{
                f"{component}@{version}": sha256
                for _, component, version, sha256 in migration.GENERATION_SUB_STAGES
            },
        }
        for key, sha256 in expected.items():
            with self.subTest(key=key):
                self.assertEqual(table[key], sha256)

    def test_stage_names_and_prompt_keys_match_orm(self) -> None:
        self.assertEqual(
            tuple(stage.value for stage in ProfileRevisionComponentStage), migration.STAGES
        )
        self.assertEqual(
            tuple(value.value for value in ProfileRevisionComponentRecordedBy),
            migration.RECORDED_BY_VALUES,
        )
        self.assertEqual(PROFILE_REVISION_COMPONENT_PROMPT_KEY_CHECK, migration.PROMPT_KEY_CHECK)
        for stage, component, _, _ in migration.GENERATION_SUB_STAGES:
            with self.subTest(stage=stage):
                self.assertEqual(
                    component,
                    PROFILE_REVISION_COMPONENT_PROMPT_KEYS[ProfileRevisionComponentStage(stage)],
                )

    def test_every_fingerprint_component_has_one_stage(self) -> None:
        # 출력 스키마 키는 같은 버전의 프롬프트 키에 딸려 있어 따로 단계를 두지 않는다.
        components = {
            key.split("@", 1)[0]
            for key in load_fingerprint_table()
            if ".output_schema@" not in key
        }
        self.assertEqual(components, set(PROFILE_REVISION_COMPONENT_PROMPT_KEYS.values()))


class BackfillRulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _run_backfill()
        cls.table = load_fingerprint_table()

    def _stages(self, revision_id: int):
        return {stage for (row_revision, stage) in self.rows if row_revision == revision_id}

    def test_stages_written_per_revision(self) -> None:
        expected = {
            1: {"rewrite", "generation", "judge", *SUB_STAGES},
            2: {"rewrite", "generation", "judge", *SUB_STAGES},
            3: {"rewrite", "generation", *SUB_STAGES},
            4: {"rewrite", "generation"},
            5: {"rewrite", "generation", "judge"},
        }
        for revision_id, stages in expected.items():
            with self.subTest(revision_id=revision_id):
                self.assertEqual(stages, self._stages(revision_id))
        self.assertEqual({"BACKFILL"}, {row[-1] for row in self.rows.values()})

    def test_rewrite_and_generation_copy_columns_and_hash_only_current_versions(self) -> None:
        self.assertEqual(
            ("gpt-5.4-mini", "rewrite@v10", "v10", self.table["rewrite@v10"], None),
            tuple(self.rows[(1, "rewrite")][:5]),
        )
        self.assertEqual(
            ("gpt-5.4-mini", "rewrite@v8", "v8", None, None),
            tuple(self.rows[(2, "rewrite")][:5]),
        )
        self.assertEqual(
            ("gpt-5.6-terra", "generation@v40", "v40", self.table["generation@v40"], None),
            tuple(self.rows[(3, "generation")][:5]),
        )
        self.assertEqual(
            ("gpt-other", "generation@v24", "v24", None, None),
            tuple(self.rows[(5, "generation")][:5]),
        )

    def test_sub_stages_use_generation_model_and_table_hashes(self) -> None:
        for stage in SUB_STAGES:
            with self.subTest(stage=stage):
                model_name, prompt_key, version, sha256, params, _ = self.rows[(1, stage)]
                self.assertEqual("gpt-5.6-terra", model_name)
                self.assertEqual(prompt_key.split("@", 1)[1], version)
                self.assertEqual(self.table[prompt_key], sha256)
                self.assertIsNone(params)

    def test_judge_uses_current_constants_only_for_active_revisions(self) -> None:
        model_name, prompt_key, version, sha256, params, _ = self.rows[(5, "judge")]
        self.assertEqual("gpt-5.6-luna", model_name)
        self.assertEqual("judge@question-grouping-v7-2", prompt_key)
        self.assertEqual("question-grouping-v7-2", version)
        self.assertEqual(self.table[prompt_key], sha256)
        self.assertEqual(
            {"reasoning": {"effort": "low"}, "max_output_tokens": 1024}, json.loads(params)
        )
        self.assertNotIn((3, "judge"), self.rows)
        self.assertNotIn((4, "judge"), self.rows)


if __name__ == "__main__":
    unittest.main()
