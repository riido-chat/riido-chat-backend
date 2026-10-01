"""Add per-stage model and prompt components for chat profile revisions.

판(chat_profile_revisions)마다 단계별 모델, 프롬프트 키와 버전, 지시문 지문을 한 행씩 둔다.
기존 판 칸은 그대로 두고(확장만) 이미 알려진 값만 BACKFILL 로 옮긴다.

백필 규칙:

- rewrite, generation(묶음 판): 모든 판에 기존 칸(모델, 버전)으로 쓴다. 지문은 판의 버전이
  이 마이그레이션 시점의 코드 버전과 같을 때만 지문 표 값으로 채우고 아니면 비운다.
- source_planning, source_planning_repair, answer, answer_repair: 판의 generation 버전이
  이 시점 코드의 묶음 판(v40)과 같을 때만 쓴다. 다른 묶음 판의 하위 버전은 알 수 없다.
- judge: 판에 칸이 없고 코드 상수뿐이라 PUBLISHED, TESTING 판에만 현재 상수로 쓴다.
- params 는 실제로 보낸 값이 확실한 judge 행만 채운다. 나머지 판 칸에는 요청 설정이 없었다.

DEV 판 행은 마이그레이션 12, 15 와 손 수정으로 제자리 갱신된 적이 있어, 백필 행은 판의 현재
칸 값을 옮긴 것일 뿐 그 판이 실제로 서빙한 이력은 아니다. 값은 마이그레이션 시점에 고정한다
(app/core/prompt_fingerprints.json 은 추가만 허용하므로 아래 지문은 표와 계속 같다).

Revision ID: 20261001_17
Revises: 20260918_16
Create Date: 2026-10-01
"""

import json
from typing import Optional, Sequence, Tuple, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20261001_17"
down_revision: Optional[str] = "20260918_16"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


TABLE = "profile_revision_components"
REVISION_FK = "fk_profile_revision_components_revision_id"
STAGE_UNIQUE = "uq_profile_revision_components_revision_id_stage"
# CHECK 이름은 명명 규칙(ck_<table>_<name>)이 앞에 테이블 이름을 붙인다.
STAGE_CONSTRAINT = "component_stage"
RECORDED_BY_CONSTRAINT = "component_recorded_by"

STAGES = (
    "rewrite",
    "judge",
    "generation",
    "source_planning",
    "source_planning_repair",
    "answer",
    "answer_repair",
)
RECORDED_BY_VALUES = ("BACKFILL", "PUBLISH")

# prompt_key 는 지문 표의 키(<component>@<version>)다. 단계마다 component 가 하나로 정해진다.
PROMPT_KEY_CHECK = (
    "prompt_key = (CASE stage"
    " WHEN 'rewrite' THEN 'rewrite'"
    " WHEN 'judge' THEN 'judge'"
    " WHEN 'generation' THEN 'generation'"
    " WHEN 'source_planning' THEN 'generation.source_planning'"
    " WHEN 'source_planning_repair' THEN 'generation.source_planning_repair'"
    " WHEN 'answer' THEN 'generation.answer'"
    " WHEN 'answer_repair' THEN 'generation.answer_repair'"
    " END) || '@' || prompt_version"
)
PROMPT_SHA256_CHECK = "prompt_sha256 IS NULL OR prompt_sha256 ~ '^[0-9a-f]{64}$'"

# 이 마이그레이션 시점의 코드 버전과 지문(app/core/prompt_fingerprints.json 값과 같다).
REWRITE_VERSION = "v10"
REWRITE_SHA256 = "acb11d87f3bf8fa496a978339a64e1998e24f002802224d1a784e28019aa3511"
GENERATION_VERSION = "v40"
GENERATION_SHA256 = "1f6eaf0b3aacfadf795ea309383cd8cf4d7e82144c9617e0634faccca81b97b3"
# (단계, 지문 표 component, 버전, 지문). 묶음 판 v40 을 이루는 하위 프롬프트다.
GENERATION_SUB_STAGES: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "source_planning",
        "generation.source_planning",
        "v34",
        "ee3dc5174145d29cf7dac4576e332462245cc5826530d977ce72291df1a4cbe3",
    ),
    (
        "source_planning_repair",
        "generation.source_planning_repair",
        "v34-repair-1",
        "290ad6c6e8c6f1679c3415ac6e2242afd429de10fd180864b8bf32d2e7a426db",
    ),
    (
        "answer",
        "generation.answer",
        "v23",
        "7cefd4931821158f05581d5a4259a55d6a2b51ae559af69a16fa59b139707591",
    ),
    (
        "answer_repair",
        "generation.answer_repair",
        "v23-repair-1",
        "179ee279b1466e88bd411e3f7839c9751d78e30700a6e4ff14b771c3f3f477ad",
    ),
)
JUDGE_MODEL = "gpt-5.6-luna"
JUDGE_VERSION = "question-grouping-v7-2"
JUDGE_SHA256 = "4e6fb86c0a8a0fb9210e3c4e3bbc1c385a88a887908f437982369316ace1b902"
# judge_client.build_judge_request 가 보내는 요청 설정 중 결과에 영향을 주는 값.
JUDGE_PARAMS = {"reasoning": {"effort": "low"}, "max_output_tokens": 1024}

_INSERT_COLUMNS = (
    f"INSERT INTO {TABLE} "
    "(revision_id, stage, model_name, prompt_key, prompt_version, prompt_sha256, "
    "params, recorded_by) "
)


def backfill_statements() -> Tuple[Tuple[str, dict], ...]:
    """기존 판에 쓸 INSERT … SELECT 문과 인자. 순수 SQL 이라 offline(--sql)에서도 같다."""

    statements = [
        (
            _INSERT_COLUMNS
            + "SELECT id, 'rewrite', query_rewrite_model_name, "
            "'rewrite@' || query_rewrite_prompt_version, query_rewrite_prompt_version, "
            "CASE WHEN query_rewrite_prompt_version = :version THEN :sha256 END, "
            "NULL, 'BACKFILL' FROM chat_profile_revisions",
            {"version": REWRITE_VERSION, "sha256": REWRITE_SHA256},
        ),
        (
            _INSERT_COLUMNS
            + "SELECT id, 'generation', generation_model_name, "
            "'generation@' || generation_prompt_version, generation_prompt_version, "
            "CASE WHEN generation_prompt_version = :version THEN :sha256 END, "
            "NULL, 'BACKFILL' FROM chat_profile_revisions",
            {"version": GENERATION_VERSION, "sha256": GENERATION_SHA256},
        ),
    ]
    for stage, component, version, sha256 in GENERATION_SUB_STAGES:
        statements.append(
            (
                _INSERT_COLUMNS
                + "SELECT id, :stage, generation_model_name, :prompt_key, :version, "
                ":sha256, NULL, 'BACKFILL' FROM chat_profile_revisions "
                "WHERE generation_prompt_version = :generation_version",
                {
                    "stage": stage,
                    "prompt_key": f"{component}@{version}",
                    "version": version,
                    "sha256": sha256,
                    "generation_version": GENERATION_VERSION,
                },
            )
        )
    statements.append(
        (
            _INSERT_COLUMNS
            + "SELECT id, 'judge', :model, :prompt_key, :version, :sha256, "
            "CAST(:params AS JSONB), 'BACKFILL' FROM chat_profile_revisions "
            "WHERE status IN ('PUBLISHED', 'TESTING')",
            {
                "model": JUDGE_MODEL,
                "prompt_key": f"judge@{JUDGE_VERSION}",
                "version": JUDGE_VERSION,
                "sha256": JUDGE_SHA256,
                "params": json.dumps(JUDGE_PARAMS, sort_keys=True),
            },
        )
    )
    return tuple(statements)


def upgrade() -> None:
    op.create_table(
        TABLE,
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("revision_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "stage",
            sa.Enum(
                *STAGES,
                name=STAGE_CONSTRAINT,
                native_enum=False,
                create_constraint=True,
                length=40,
            ),
            nullable=False,
        ),
        sa.Column("model_name", sa.String(length=150), nullable=False),
        sa.Column("prompt_key", sa.String(length=100), nullable=False),
        sa.Column("prompt_version", sa.String(length=50), nullable=False),
        sa.Column("prompt_sha256", sa.String(length=64), nullable=True),
        sa.Column("params", postgresql.JSONB(), nullable=True),
        sa.Column(
            "recorded_by",
            sa.Enum(
                *RECORDED_BY_VALUES,
                name=RECORDED_BY_CONSTRAINT,
                native_enum=False,
                create_constraint=True,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id", name=f"pk_{TABLE}"),
        sa.ForeignKeyConstraint(
            ["revision_id"],
            ["chat_profile_revisions.id"],
            name=REVISION_FK,
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("revision_id", "stage", name=STAGE_UNIQUE),
        sa.CheckConstraint(PROMPT_KEY_CHECK, name="prompt_key"),
        sa.CheckConstraint(PROMPT_SHA256_CHECK, name="prompt_sha256"),
    )
    for statement, params in backfill_statements():
        op.execute(sa.text(statement).bindparams(**params))


def downgrade() -> None:
    op.drop_table(TABLE)
