"""질문 그룹핑과 정본 캐시 테이블을 추가한다.

- 문제 그룹, 세부 문제, 개정 이력, 분류 실행, 질문 연결, 질문 벡터,
  정본, 정본 인용, 캐시 시도 9개 테이블을 만든다.
- model_calls에 classification_run_id와 QUESTION_CLASSIFICATION 용도를 두고
  실행 칸의 허용 조합을 체크 제약으로 건다.
- model_calls에 총량의 부분값인 cached_input_tokens, reasoning_tokens를 둔다.
- document_sources가 원천을 거친 문서 그룹과 직접 적은 문서 그룹이 같은지
  복합 외래키로 강제한다.

Revision ID: 20260915_13
Revises: 20260912_12
Create Date: 2026-09-15
"""

from typing import Optional, Sequence, Union

from alembic import op
from pgvector.sqlalchemy import VECTOR
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260915_13"
down_revision: Optional[str] = "20260912_12"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


EMBEDDING_DIMENSIONS = 1536

MODEL_CALL_PURPOSE_CONSTRAINT = "model_call_purpose"
MODEL_CALL_OWNER_CONSTRAINT = "owner_combination"
MODEL_CALL_CLASSIFICATION_RUN_FK = (
    "fk_model_calls_classification_run_id_classification_runs"
)
MODEL_CALL_CLASSIFICATION_RUN_INDEX = "ix_model_calls_classification_run_id"
MODEL_CALL_CACHED_INPUT_CONSTRAINT = "cached_input_tokens"
MODEL_CALL_REASONING_CONSTRAINT = "reasoning_tokens"

PREVIOUS_MODEL_CALL_PURPOSES = (
    "EMBEDDING",
    "GENERATION",
    "QUERY_EMBEDDING",
    "CHUNK_EMBEDDING",
    "ANSWER_GENERATION",
    "QUERY_REWRITE",
    "CONVERSATION_SUMMARY",
)
MODEL_CALL_PURPOSES = (*PREVIOUS_MODEL_CALL_PURPOSES, "QUESTION_CLASSIFICATION")

# 즉시 판별은 턴과 분류 실행을 모두 채우고, 백필 판정은 분류 실행만 채운다.
# 턴, 색인, 수집 호출은 지금처럼 자기 실행 칸 하나만 채운다.
MODEL_CALL_OWNER_COMBINATION = (
    "(classification_run_id IS NULL"
    " AND purpose <> 'QUESTION_CLASSIFICATION'"
    " AND num_nonnulls(rag_run_id, index_run_id, ingestion_run_id) = 1)"
    " OR (classification_run_id IS NOT NULL"
    " AND index_run_id IS NULL AND ingestion_run_id IS NULL"
    " AND (purpose = 'QUESTION_CLASSIFICATION'"
    " OR (purpose = 'QUERY_EMBEDDING' AND rag_run_id IS NOT NULL)))"
)

PREVIOUS_GROUP_SOURCE_FK = (
    "fk_document_sources_group_source_id_document_group_sources"
)
GROUP_SOURCE_DOCUMENT_GROUP_FK = (
    "fk_document_sources_group_source_id_document_group_id"
)
GROUP_SOURCE_DOCUMENT_GROUP_UNIQUE = (
    "uq_document_group_sources_id_document_group_id"
)

CURRENT_CLASSIFICATION_INDEX = "uq_question_classifications_rag_run_id_current"
OPEN_ONLINE_CLASSIFICATION_RUN_INDEX = "uq_classification_runs_open_online"
APPROVED_CANONICAL_ANSWER_INDEX = "uq_canonical_answers_subproblem_id_approved"

NEW_TABLES_IN_DROP_ORDER = (
    "question_cache_attempts",
    "canonical_answer_citations",
    "canonical_answers",
    "question_embeddings",
    "question_classifications",
    "classification_runs",
    "question_subproblem_revisions",
    "question_subproblems",
    "question_problem_groups",
)


def _enum(name: str, *values: str) -> sa.Enum:
    """VARCHAR + CHECK 제약으로 저장되는 상태 Enum 타입을 만든다."""

    return sa.Enum(
        *values,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=20,
    )


def _allowed_values_condition(column: str, values: Sequence[str]) -> str:
    literals = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({literals})"


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at",
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )


def _check_document_source_groups(connection) -> None:
    """원천의 문서 그룹과 문서의 문서 그룹이 어긋난 행이 없는지 확인한다."""

    mismatched = connection.execute(
        sa.text(
            """
            SELECT ds.id
            FROM document_sources AS ds
            JOIN document_group_sources AS gs ON gs.id = ds.group_source_id
            WHERE gs.document_group_id <> ds.document_group_id
            ORDER BY ds.id
            """
        )
    ).scalars().all()
    if mismatched:
        raise RuntimeError(
            "원천의 문서 그룹과 다른 문서 그룹을 적은 문서가 있습니다: "
            f"{list(mismatched)}"
        )


def _check_model_call_owners(connection) -> None:
    """기존 모델 호출이 새 허용 조합을 만족하는지 확인한다.

    classification_run_id는 아직 없으므로 턴, 색인, 수집 중 정확히 하나를
    채웠는지만 본다. 어긋난 행을 임의로 고치지 않고 멈춘다.
    """

    rows = connection.execute(
        sa.text(
            """
            SELECT id FROM model_calls
            WHERE num_nonnulls(rag_run_id, index_run_id, ingestion_run_id) <> 1
            ORDER BY id
            LIMIT 20
            """
        )
    ).scalars().all()
    if rows:
        raise RuntimeError(
            "실행 칸을 하나만 채우지 않은 모델 호출이 있어 허용 조합 제약을 "
            f"걸 수 없습니다: {list(rows)}"
        )


def _create_question_grouping_tables() -> None:
    op.create_table(
        "question_problem_groups",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "kind",
            _enum("problem_group_kind", "DOCUMENT", "NO_DOCUMENT"),
            nullable=False,
        ),
        sa.Column("document_source_id", sa.BigInteger(), nullable=True),
        sa.Column("document_group_id", sa.BigInteger(), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["document_source_id"],
            ["document_sources.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["document_group_id"],
            ["document_groups.id"],
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "(kind = 'DOCUMENT'"
            " AND document_source_id IS NOT NULL AND document_group_id IS NULL)"
            " OR (kind = 'NO_DOCUMENT'"
            " AND document_source_id IS NULL AND document_group_id IS NOT NULL)",
            name="kind_target",
        ),
        sa.UniqueConstraint("document_source_id"),
        sa.UniqueConstraint("document_group_id"),
    )

    op.create_table(
        "question_subproblems",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("problem_group_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("key", sa.String(length=200), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("inclusion_criteria", sa.Text(), nullable=False),
        sa.Column("exclusion_criteria", sa.Text(), nullable=True),
        sa.Column("current_version", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            _enum("subproblem_status", "DRAFT", "APPROVED", "ARCHIVED"),
            nullable=False,
        ),
        sa.Column(
            "serving_state",
            _enum(
                "subproblem_serving_state",
                "UNUSED",
                "SHADOW",
                "SERVING",
                "STOPPED",
            ),
            nullable=False,
            server_default=sa.text("'UNUSED'"),
        ),
        sa.Column("created_by", sa.String(length=100), nullable=False),
        _created_at(),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["problem_group_id"],
            ["question_problem_groups.id"],
            name="fk_question_subproblems_problem_group_id",
            ondelete="RESTRICT",
        ),
        # 같은 분류 체계를 다른 문서 그룹에 넣을 수 있어 문제 그룹 안에서만 유일하다.
        sa.UniqueConstraint("problem_group_id", "key"),
    )

    op.create_table(
        "question_subproblem_revisions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("subproblem_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("name_snapshot", sa.String(length=200), nullable=False),
        sa.Column("inclusion_snapshot", sa.Text(), nullable=False),
        sa.Column("exclusion_snapshot", sa.Text(), nullable=True),
        sa.Column("change_reason", sa.Text(), nullable=True),
        sa.Column(
            "inclusion_embedding",
            VECTOR(EMBEDDING_DIMENSIONS),
            nullable=True,
        ),
        sa.Column("embedding_config_id", sa.BigInteger(), nullable=True),
        sa.Column("embedding_text_version", sa.String(length=50), nullable=True),
        sa.Column("approved_by", sa.String(length=100), nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["subproblem_id"],
            ["question_subproblems.id"],
            name="fk_question_subproblem_revisions_subproblem_id",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["embedding_config_id"],
            ["embedding_configs.id"],
            name="fk_question_subproblem_revisions_embedding_config_id",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("subproblem_id", "version"),
        # 벡터, 임베딩 설정, 입력 문장 구성 판은 함께 채우거나 함께 비운다.
        sa.CheckConstraint(
            "num_nulls(inclusion_embedding, embedding_config_id,"
            " embedding_text_version) IN (0, 3)",
            name="inclusion_embedding",
        ),
    )

    op.create_table(
        "classification_runs",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("document_group_id", sa.BigInteger(), nullable=False),
        sa.Column("index_version_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "kind",
            _enum(
                "classification_run_kind",
                "ONLINE",
                "BACKFILL",
                "OPERATOR",
                "REJUDGE",
            ),
            nullable=False,
        ),
        sa.Column("model", sa.String(length=150), nullable=False),
        sa.Column("prompt_version", sa.String(length=50), nullable=False),
        sa.Column(
            "row_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("actor", sa.String(length=100), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["document_group_id"],
            ["document_groups.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["index_version_id"],
            ["index_versions.id"],
            ondelete="RESTRICT",
        ),
    )
    op.create_index(
        "ix_classification_runs_document_group_id_started_at",
        "classification_runs",
        ["document_group_id", "started_at"],
    )
    # 동시 턴이 같은 설정의 ONLINE 실행을 둘 열지 못하게 한다.
    op.create_index(
        OPEN_ONLINE_CLASSIFICATION_RUN_INDEX,
        "classification_runs",
        ["document_group_id", "index_version_id", "model", "prompt_version"],
        unique=True,
        postgresql_where=sa.text("kind = 'ONLINE' AND finished_at IS NULL"),
    )

    op.create_table(
        "question_classifications",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("rag_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("subproblem_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("problem_group_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "decision",
            _enum("classification_decision", "CONNECT", "SEPARATE", "UNCLASSIFIED"),
            nullable=False,
        ),
        sa.Column("subproblem_version", sa.Integer(), nullable=True),
        sa.Column(
            "attribution_source",
            _enum("attribution_source", "SUBPROBLEM", "CITATION", "DOCUMENT", "NONE"),
            nullable=False,
        ),
        sa.Column(
            "is_composite",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("confidence", sa.Numeric(), nullable=True),
        sa.Column("judgment_input", postgresql.JSONB(), nullable=True),
        sa.Column(
            "effective_from",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("effective_to", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["rag_run_id"],
            ["rag_runs.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["subproblem_id"],
            ["question_subproblems.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["problem_group_id"],
            ["question_problem_groups.id"],
            name="fk_question_classifications_problem_group_id",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["classification_runs.id"],
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "(decision = 'CONNECT') = (subproblem_id IS NOT NULL)",
            name="connect_subproblem",
        ),
        sa.CheckConstraint(
            "(decision = 'CONNECT') = (attribution_source = 'SUBPROBLEM')",
            name="connect_attribution",
        ),
        sa.CheckConstraint(
            "(subproblem_id IS NULL) = (subproblem_version IS NULL)",
            name="subproblem_version",
        ),
        sa.CheckConstraint(
            "effective_to IS NULL OR effective_to >= effective_from",
            name="effective_period",
        ),
    )
    # 재분류는 이전 행을 닫고 새 행을 추가하므로 질문마다 현재 행은 하나다.
    op.create_index(
        CURRENT_CLASSIFICATION_INDEX,
        "question_classifications",
        ["rag_run_id"],
        unique=True,
        postgresql_where=sa.text("effective_to IS NULL"),
    )
    for column in ("problem_group_id", "subproblem_id", "run_id"):
        op.create_index(
            f"ix_question_classifications_{column}",
            "question_classifications",
            [column],
        )

    op.create_table(
        "question_embeddings",
        sa.Column("rag_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("embedding", VECTOR(EMBEDDING_DIMENSIONS), nullable=False),
        sa.Column("embedding_config_id", sa.BigInteger(), nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint("rag_run_id"),
        sa.ForeignKeyConstraint(
            ["rag_run_id"],
            ["rag_runs.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["embedding_config_id"],
            ["embedding_configs.id"],
            ondelete="RESTRICT",
        ),
    )

    op.create_table(
        "canonical_answers",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("subproblem_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "origin",
            _enum("canonical_answer_origin", "SELECTED", "AUTHORED"),
            nullable=False,
        ),
        sa.Column("source_rag_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("content_markdown", sa.Text(), nullable=False),
        sa.Column("applicability_rules", postgresql.JSONB(), nullable=True),
        sa.Column("subproblem_version", sa.Integer(), nullable=False),
        sa.Column("filter_results", postgresql.JSONB(), nullable=True),
        sa.Column(
            "approval",
            _enum("canonical_answer_approval", "DRAFT", "APPROVED", "REVOKED"),
            nullable=False,
        ),
        sa.Column("approved_by", sa.String(length=100), nullable=True),
        sa.Column("valid_from", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("valid_to", sa.TIMESTAMP(timezone=True), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["subproblem_id"],
            ["question_subproblems.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["source_rag_run_id"],
            ["rag_runs.id"],
            ondelete="SET NULL",
        ),
    )
    op.create_index(
        APPROVED_CANONICAL_ANSWER_INDEX,
        "canonical_answers",
        ["subproblem_id"],
        unique=True,
        postgresql_where=sa.text("approval = 'APPROVED'"),
    )

    op.create_table(
        "canonical_answer_citations",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column(
            "canonical_answer_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("citation_order", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.BigInteger(), nullable=False),
        sa.Column("document_version_id", sa.BigInteger(), nullable=False),
        sa.Column("document_title_snapshot", sa.String(length=500), nullable=True),
        sa.Column("node_path_snapshot", sa.String(length=1000), nullable=True),
        sa.Column("source_uri_snapshot", sa.String(length=1000), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["canonical_answer_id"],
            ["canonical_answers.id"],
            name="fk_canonical_answer_citations_canonical_answer_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["document_chunks.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
            name="fk_canonical_answer_citations_document_version_id",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "canonical_answer_id",
            "citation_order",
            name="uq_canonical_answer_citations_answer_id_citation_order",
        ),
    )
    for column in ("chunk_id", "document_version_id"):
        op.create_index(
            f"ix_canonical_answer_citations_{column}",
            "canonical_answer_citations",
            [column],
        )

    op.create_table(
        "question_cache_attempts",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("rag_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("classification_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "outcome",
            _enum(
                "cache_attempt_outcome",
                "SERVED",
                "SHADOW",
                "GROUP_DISABLED",
                "REJECTED",
                "SKIPPED",
                "FAILED",
            ),
            nullable=False,
        ),
        sa.Column(
            "canonical_answer_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column(
            "rejection_reasons",
            postgresql.ARRAY(sa.String(length=50)),
            nullable=True,
        ),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["rag_run_id"],
            ["rag_runs.id"],
            ondelete="CASCADE",
        ),
        # 턴 삭제가 연결 행과 시도를 함께 지우므로 문장 끝에 검사하는 NO ACTION 을 쓴다.
        sa.ForeignKeyConstraint(
            ["classification_id"],
            ["question_classifications.id"],
            name="fk_question_cache_attempts_classification_id",
            ondelete="NO ACTION",
        ),
        sa.ForeignKeyConstraint(
            ["canonical_answer_id"],
            ["canonical_answers.id"],
            name="fk_question_cache_attempts_canonical_answer_id",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("rag_run_id"),
        sa.UniqueConstraint("classification_id"),
        sa.CheckConstraint(
            "(outcome IN ('SERVED', 'SHADOW', 'GROUP_DISABLED'))"
            " = (canonical_answer_id IS NOT NULL)",
            name="outcome_canonical_answer",
        ),
        sa.CheckConstraint(
            "(outcome = 'REJECTED')"
            " = (COALESCE(cardinality(rejection_reasons), 0) > 0)",
            name="outcome_rejection_reasons",
        ),
    )
    op.create_index(
        "ix_question_cache_attempts_canonical_answer_id",
        "question_cache_attempts",
        ["canonical_answer_id"],
    )


def upgrade() -> None:
    connection = op.get_bind()
    _check_document_source_groups(connection)
    _check_model_call_owners(connection)

    _create_question_grouping_tables()

    op.add_column(
        "model_calls",
        sa.Column("classification_run_id", sa.BigInteger(), nullable=True),
    )
    op.create_foreign_key(
        MODEL_CALL_CLASSIFICATION_RUN_FK,
        "model_calls",
        "classification_runs",
        ["classification_run_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        MODEL_CALL_CLASSIFICATION_RUN_INDEX,
        "model_calls",
        ["classification_run_id"],
    )
    op.drop_constraint(
        MODEL_CALL_PURPOSE_CONSTRAINT,
        "model_calls",
        type_="check",
    )
    op.create_check_constraint(
        MODEL_CALL_PURPOSE_CONSTRAINT,
        "model_calls",
        _allowed_values_condition("purpose", MODEL_CALL_PURPOSES),
    )
    op.create_check_constraint(
        MODEL_CALL_OWNER_CONSTRAINT,
        "model_calls",
        MODEL_CALL_OWNER_COMBINATION,
    )

    # 캐시 입력과 추론 출력은 각 총량에 포함된 부분값이다. 내역이 없으면 비운다.
    op.add_column(
        "model_calls",
        sa.Column("cached_input_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "model_calls",
        sa.Column("reasoning_tokens", sa.Integer(), nullable=True),
    )
    op.create_check_constraint(
        MODEL_CALL_CACHED_INPUT_CONSTRAINT,
        "model_calls",
        "cached_input_tokens IS NULL OR (cached_input_tokens >= 0"
        " AND (input_tokens IS NULL OR cached_input_tokens <= input_tokens))",
    )
    op.create_check_constraint(
        MODEL_CALL_REASONING_CONSTRAINT,
        "model_calls",
        "reasoning_tokens IS NULL OR (reasoning_tokens >= 0"
        " AND (output_tokens IS NULL OR reasoning_tokens <= output_tokens))",
    )

    op.create_unique_constraint(
        GROUP_SOURCE_DOCUMENT_GROUP_UNIQUE,
        "document_group_sources",
        ["id", "document_group_id"],
    )
    op.drop_constraint(
        PREVIOUS_GROUP_SOURCE_FK,
        "document_sources",
        type_="foreignkey",
    )
    # 원천이 비어 있으면 복합 외래키는 검사하지 않아 업로드 문서도 통과한다.
    op.create_foreign_key(
        GROUP_SOURCE_DOCUMENT_GROUP_FK,
        "document_sources",
        "document_group_sources",
        ["group_source_id", "document_group_id"],
        ["id", "document_group_id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint(
        GROUP_SOURCE_DOCUMENT_GROUP_FK,
        "document_sources",
        type_="foreignkey",
    )
    op.create_foreign_key(
        PREVIOUS_GROUP_SOURCE_FK,
        "document_sources",
        "document_group_sources",
        ["group_source_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.drop_constraint(
        GROUP_SOURCE_DOCUMENT_GROUP_UNIQUE,
        "document_group_sources",
        type_="unique",
    )

    op.drop_constraint(
        MODEL_CALL_REASONING_CONSTRAINT,
        "model_calls",
        type_="check",
    )
    op.drop_constraint(
        MODEL_CALL_CACHED_INPUT_CONSTRAINT,
        "model_calls",
        type_="check",
    )
    op.drop_column("model_calls", "reasoning_tokens")
    op.drop_column("model_calls", "cached_input_tokens")

    op.drop_constraint(
        MODEL_CALL_OWNER_CONSTRAINT,
        "model_calls",
        type_="check",
    )
    op.drop_constraint(
        MODEL_CALL_PURPOSE_CONSTRAINT,
        "model_calls",
        type_="check",
    )
    # 판별 호출 행은 임의로 지우지 않고 이전 CHECK 복원 단계에서 실패시킨다.
    op.create_check_constraint(
        MODEL_CALL_PURPOSE_CONSTRAINT,
        "model_calls",
        _allowed_values_condition("purpose", PREVIOUS_MODEL_CALL_PURPOSES),
    )
    op.drop_index(MODEL_CALL_CLASSIFICATION_RUN_INDEX, table_name="model_calls")
    op.drop_constraint(
        MODEL_CALL_CLASSIFICATION_RUN_FK,
        "model_calls",
        type_="foreignkey",
    )
    op.drop_column("model_calls", "classification_run_id")

    for table_name in NEW_TABLES_IN_DROP_ORDER:
        op.drop_table(table_name)
