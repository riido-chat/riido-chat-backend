import unittest

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Enum as SAEnum,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

from app.database.base import Base
from app.database.models import (
    ACTIVE_INDEX_VERSION_CONSTRAINT,
    APPROVED_CANONICAL_ANSWER_CONSTRAINT,
    CHAT_PROFILE_REVISION_FK_CONSTRAINT,
    CURRENT_CLASSIFICATION_CONSTRAINT,
    DOCUMENT_SOURCE_GROUP_SOURCE_FK_CONSTRAINT,
    GROUP_SOURCE_DOCUMENT_GROUP_UNIQUE_CONSTRAINT,
    INDEX_VERSION_NO_CONSTRAINT,
    OPEN_ONLINE_CLASSIFICATION_RUN_CONSTRAINT,
    AnswerCitation,
    AnswerStatus,
    AttributionSource,
    CacheAttemptOutcome,
    CanonicalAnswer,
    CanonicalAnswerApproval,
    CanonicalAnswerCitation,
    CanonicalAnswerOrigin,
    ClassificationDecision,
    ClassificationRun,
    ClassificationRunKind,
    ChatProfile,
    ChatProfileRevision,
    ChatProfileRevisionStatus,
    ChunkEmbedding,
    ContentNode,
    ContextStrategy,
    ConversationChannel,
    DocumentChunk,
    DocumentGroup,
    DocumentGroupSource,
    DocumentSource,
    DocumentVersion,
    IndexOperationType,
    IndexRun,
    IndexRunStage,
    IndexVersion,
    IndexVersionStatus,
    IngestionResultCode,
    IngestionRun,
    IngestionStage,
    ModelCall,
    ModelCallPurpose,
    QuestionCacheAttempt,
    ExactQuestionMatch,
    ExactQuestionMatchSource,
    ExactQuestionMatchState,
    QuestionClassification,
    QuestionEmbedding,
    QuestionProblemGroup,
    QuestionProblemGroupKind,
    QuestionSubproblem,
    QuestionSubproblemRevision,
    QuestionSubproblemServingState,
    QuestionSubproblemStatus,
    RagRun,
    Conversation,
)
from app.retrieval.embedding import OPENAI_EMBEDDING_DIMENSIONS


ERD_TABLE_NAMES = {
    "document_groups",
    "document_group_sources",
    "chat_profiles",
    "chat_profile_revisions",
    "document_sources",
    "ingestion_runs",
    "document_versions",
    "content_nodes",
    "chunking_configs",
    "document_chunks",
    "embedding_configs",
    "chunk_embeddings",
    "index_versions",
    "index_documents",
    "index_runs",
    "conversations",
    "rag_runs",
    "retrieval_results",
    "model_calls",
    "answer_citations",
    "feedbacks",
    "question_problem_groups",
    "question_subproblems",
    "question_subproblem_revisions",
    "classification_runs",
    "question_classifications",
    "question_embeddings",
    "canonical_answers",
    "canonical_answer_citations",
    "question_cache_attempts",
    "exact_question_matches",
}

# 중심벡터는 2차라 아직 만들지 않는다.
REMOVED_TABLE_NAMES = {
    "legacy_document_chunks",
    "legacy_chunk_embeddings",
    "question_group_revisions",
    "question_reviews",
    "question_review_intents",
    "subproblem_centroids",
}


def _application_tables():
    """app.database.models 가 정의한 테이블만 돌려준다.

    같은 Base 를 쓰는 다른 모듈이 먼저 import 되어도 결과가 흔들리지 않게 한다.
    """

    return {
        mapper.local_table
        for mapper in Base.registry.mappers
        if mapper.class_.__module__ == "app.database.models"
    }


class DatabaseModelTest(unittest.TestCase):
    def test_registers_erd_tables_without_legacy_tables(self) -> None:
        table_names = {table.name for table in _application_tables()}

        self.assertTrue(ERD_TABLE_NAMES <= table_names)
        self.assertFalse(REMOVED_TABLE_NAMES & table_names)
        self.assertEqual("document_chunks", DocumentChunk.__tablename__)
        self.assertEqual("chunk_embeddings", ChunkEmbedding.__tablename__)

    def test_embeddings_use_confirmed_vector_dimension(self) -> None:
        for table in (ChunkEmbedding.__table__, QuestionEmbedding.__table__):
            self.assertIsInstance(table.c.embedding.type, VECTOR)
            self.assertEqual(
                OPENAI_EMBEDDING_DIMENSIONS,
                table.c.embedding.type.dim,
            )
            self.assertEqual(1536, table.c.embedding.type.dim)

    def test_erd_document_chunk_shares_primary_key_with_content_node(self) -> None:
        table = DocumentChunk.__table__
        foreign_key = next(iter(table.c.id.foreign_keys))

        self.assertTrue(table.c.id.primary_key)
        self.assertEqual("content_nodes.id", foreign_key.target_fullname)
        self.assertEqual("CASCADE", foreign_key.ondelete)

    def test_content_node_has_nullable_identity_columns(self) -> None:
        table = ContentNode.__table__

        self.assertTrue(table.c.node_identity_hash.nullable)
        self.assertTrue(table.c.node_identity_kind.nullable)
        self.assertFalse(table.c.content_hash.nullable)

    def test_document_version_supports_uri_or_inline_raw_content(self) -> None:
        table = DocumentVersion.__table__
        constraint_names = {
            constraint.name
            for constraint in table.constraints
            if isinstance(constraint, CheckConstraint)
        }

        self.assertTrue(table.c.raw_content_uri.nullable)
        self.assertTrue(table.c.raw_content.nullable)
        self.assertIsInstance(table.c.raw_content.type, Text)
        self.assertIn(
            "ck_document_versions_raw_content_storage",
            constraint_names,
        )

    def test_document_group_defines_console_extension_unit(self) -> None:
        table = DocumentGroup.__table__
        unique_constraints = {
            constraint.name: tuple(constraint.columns.keys())
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        }

        self.assertEqual(
            {
                "id",
                "group_key",
                "name",
                "consumer_key",
                "created_at",
                "updated_at",
            },
            set(table.columns.keys()),
        )
        self.assertEqual(
            ("group_key",),
            unique_constraints["uq_document_groups_group_key"],
        )
        self.assertFalse(table.c.consumer_key.nullable)

    def test_chat_profile_revision_has_single_active_slot_per_status(self) -> None:
        self.assertEqual(
            {
                "id",
                "profile_key",
                "name",
                "created_at",
                "updated_at",
            },
            set(ChatProfile.__table__.columns.keys()),
        )
        table = ChatProfileRevision.__table__
        self.assertFalse(table.c.document_group_id.nullable)
        self.assertFalse(table.c.generation_model_name.nullable)
        self.assertFalse(table.c.query_rewrite_model_name.nullable)
        self.assertFalse(table.c.semantic_cache_enabled.nullable)
        self.assertTrue(table.c.verifier_model_name.nullable)
        partial_unique = {
            index.name: str(index.dialect_options["postgresql"]["where"])
            for index in table.indexes
            if index.unique
        }
        self.assertEqual(
            "status = 'PUBLISHED'",
            partial_unique["uq_chat_profile_revisions_profile_published"],
        )
        self.assertEqual(
            "status = 'TESTING'",
            partial_unique["uq_chat_profile_revisions_profile_testing"],
        )
        check_constraint_names = {
            constraint.name
            for constraint in table.constraints
            if isinstance(constraint, CheckConstraint)
        }
        self.assertTrue(
            any(
                name == "chat_profile_revision_status"
                or (
                    isinstance(name, str)
                    and name.endswith("_chat_profile_revision_status")
                )
                for name in check_constraint_names
            )
        )
        self.assertEqual("PUBLISHED", ChatProfileRevisionStatus.PUBLISHED.value)

    def test_conversation_profile_pin_is_required_and_restricts_delete(self) -> None:
        column = Conversation.__table__.c.chat_profile_revision_id
        self.assertFalse(column.nullable)
        foreign_key = next(iter(column.foreign_keys))
        self.assertEqual(
            "chat_profile_revisions.id",
            foreign_key.target_fullname,
        )
        self.assertEqual(
            CHAT_PROFILE_REVISION_FK_CONSTRAINT,
            foreign_key.constraint.name,
        )
        self.assertLessEqual(len(foreign_key.constraint.name), 63)
        self.assertEqual("RESTRICT", foreign_key.ondelete)
        self.assertFalse(Conversation.__table__.c.channel.nullable)
        self.assertEqual("PUBLIC", ConversationChannel.PUBLIC.value)

    def test_document_source_is_identified_by_group_and_document_key(self) -> None:
        table = DocumentSource.__table__
        unique_constraints = {
            constraint.name: tuple(constraint.columns.keys())
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        }
        # document_group_id 는 원천과의 복합 외래키에도 들어가므로 단일 외래키를 고른다.
        foreign_key = next(
            foreign_key
            for foreign_key in table.c.document_group_id.foreign_keys
            if foreign_key.column.table.name == "document_groups"
        )

        self.assertFalse(table.c.document_group_id.nullable)
        self.assertFalse(table.c.document_key.nullable)
        self.assertIsInstance(table.c.document_key.type, String)
        self.assertEqual(300, table.c.document_key.type.length)
        self.assertEqual("document_groups.id", foreign_key.target_fullname)
        self.assertEqual("RESTRICT", foreign_key.ondelete)
        # 끌어오는 문서는 원천 안에서, 밀어 넣는 문서는 그룹 안에서 유일하다
        partial_unique = {
            index.name: (
                tuple(index.columns.keys()),
                str(index.dialect_options["postgresql"]["where"]),
            )
            for index in table.indexes
            if index.unique
        }
        self.assertEqual(
            (
                ("group_source_id", "document_key"),
                "group_source_id IS NOT NULL",
            ),
            partial_unique["uq_document_sources_group_source_id_document_key"],
        )
        self.assertEqual(
            (
                ("document_group_id", "document_key"),
                "group_source_id IS NULL",
            ),
            partial_unique["uq_document_sources_document_group_id_document_key"],
        )
        self.assertEqual(
            ("document_group_id", "canonical_uri"),
            unique_constraints["uq_document_sources_document_group_id_canonical_uri"],
        )
        # canonical_uri 전역 unique는 그룹 단위 unique로 대체한다.
        self.assertNotIn("uq_document_sources_canonical_uri", unique_constraints)
        self.assertFalse(table.c.canonical_uri.nullable)

    def test_document_version_indexes_normalized_content_hash(self) -> None:
        index_names = {index.name for index in DocumentVersion.__table__.indexes}

        self.assertIn("ix_document_versions_normalized_content_hash", index_names)

    def test_index_version_adds_ready_status_and_group_scoped_numbers(self) -> None:
        table = IndexVersion.__table__
        status_type = table.c.status.type
        indexes = {index.name: index for index in table.indexes}

        self.assertEqual(
            {"BUILDING", "VALIDATING", "READY", "ACTIVE", "FAILED", "INACTIVE"},
            {member.value for member in IndexVersionStatus},
        )
        self.assertIsInstance(status_type, SAEnum)
        self.assertEqual(
            {"BUILDING", "VALIDATING", "READY", "ACTIVE", "FAILED", "INACTIVE"},
            set(status_type.enums),
        )
        self.assertFalse(table.c.document_group_id.nullable)
        self.assertTrue(table.c.version_no.nullable)
        self.assertIsInstance(table.c.version_no.type, Integer)

        active_index = indexes[ACTIVE_INDEX_VERSION_CONSTRAINT]
        self.assertTrue(active_index.unique)
        self.assertEqual(["document_group_id"], list(active_index.columns.keys()))

        numbered_index = indexes[INDEX_VERSION_NO_CONSTRAINT]
        self.assertTrue(numbered_index.unique)
        self.assertEqual(
            ["document_group_id", "version_no"],
            list(numbered_index.columns.keys()),
        )

    def test_index_run_records_stage_and_operation_type(self) -> None:
        table = IndexRun.__table__
        constraint_names = {
            constraint.name
            for constraint in table.constraints
            if isinstance(constraint, CheckConstraint)
        }
        index_names = {index.name for index in table.indexes}

        self.assertEqual(
            {"BUILDING", "VALIDATING", "APPLYING"},
            {member.value for member in IndexRunStage},
        )
        self.assertEqual(
            {"BUILD_AND_APPLY", "BUILD", "APPLY"},
            {member.value for member in IndexOperationType},
        )
        self.assertFalse(table.c.stage.nullable)
        self.assertFalse(table.c.operation_type.nullable)
        self.assertTrue(table.c.error_code.nullable)
        self.assertIn("ck_index_runs_index_run_stage", constraint_names)
        self.assertIn("ck_index_runs_index_operation_type", constraint_names)
        self.assertIn("ix_index_runs_index_version_id_started_at", index_names)

    def test_ingestion_run_records_result_code_stage_and_batch(self) -> None:
        table = IngestionRun.__table__
        constraint_names = {
            constraint.name
            for constraint in table.constraints
            if isinstance(constraint, CheckConstraint)
        }
        duplicate_fk = next(
            iter(table.c.duplicate_of_document_source_id.foreign_keys)
        )

        self.assertEqual(
            {"CREATED", "UPDATED", "NO_CHANGE", "DUPLICATE_CONTENT"},
            {member.value for member in IngestionResultCode},
        )
        self.assertEqual(
            {
                "RECEIVING",
                "VALIDATING",
                "NORMALIZING",
                "PARSING",
                "CHUNKING",
                "EMBEDDING",
                "PERSISTING",
            },
            {member.value for member in IngestionStage},
        )
        self.assertTrue(table.c.result_code.nullable)
        self.assertTrue(table.c.stage.nullable)
        self.assertTrue(table.c.error_code.nullable)
        self.assertTrue(table.c.batch_id.nullable)
        self.assertFalse(table.c.document_source_id.nullable)
        self.assertIn("ck_ingestion_runs_ingestion_result_code", constraint_names)
        self.assertIn("ck_ingestion_runs_ingestion_stage", constraint_names)
        self.assertIn(
            "ix_ingestion_runs_batch_id",
            {index.name for index in table.indexes},
        )
        self.assertEqual("document_sources.id", duplicate_fk.target_fullname)
        self.assertEqual("SET NULL", duplicate_fk.ondelete)

    def test_model_call_links_ingestion_run(self) -> None:
        table = ModelCall.__table__
        foreign_key = next(iter(table.c.ingestion_run_id.foreign_keys))

        self.assertTrue(table.c.ingestion_run_id.nullable)
        self.assertIsInstance(table.c.ingestion_run_id.type, BigInteger)
        self.assertEqual("ingestion_runs.id", foreign_key.target_fullname)
        self.assertEqual("CASCADE", foreign_key.ondelete)

    def test_rag_run_uses_uuid_identifiers_and_answer_status(self) -> None:
        table = RagRun.__table__
        unique_constraints = {
            tuple(constraint.columns.keys())
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        }

        self.assertIsInstance(table.c.id.type, UUID)
        self.assertIsInstance(table.c.conversation_id.type, UUID)
        self.assertIn(("conversation_id", "turn_no"), unique_constraints)
        self.assertEqual(
            {"PROCESSING", "COMPLETED", "WITHHELD", "ERROR", "CANCELLED"},
            {member.value for member in AnswerStatus},
        )
        self.assertTrue(table.c.withheld_reason_code.nullable)
        self.assertTrue(table.c.error_code.nullable)

    def test_expand_model_call_purpose_keeps_legacy_and_new_values(self) -> None:
        purpose_type = ModelCall.__table__.c.purpose.type
        model_call_constraint_names = {
            constraint.name
            for constraint in ModelCall.__table__.constraints
            if isinstance(constraint, CheckConstraint)
        }
        expected = {
            "EMBEDDING",
            "GENERATION",
            "QUERY_EMBEDDING",
            "CHUNK_EMBEDDING",
            "ANSWER_GENERATION",
            "QUERY_REWRITE",
            "CONVERSATION_SUMMARY",
            "QUESTION_CLASSIFICATION",
        }

        self.assertEqual(expected, {member.value for member in ModelCallPurpose})
        self.assertIsInstance(purpose_type, SAEnum)
        self.assertIs(ModelCallPurpose, purpose_type.enum_class)
        self.assertEqual(expected, set(purpose_type.enums))
        self.assertIn(
            "ck_model_calls_model_call_purpose",
            model_call_constraint_names,
        )

    def test_expand_context_strategy_keeps_legacy_and_new_values(self) -> None:
        strategy_type = RagRun.__table__.c.context_strategy.type
        rag_run_constraint_names = {
            constraint.name
            for constraint in RagRun.__table__.constraints
            if isinstance(constraint, CheckConstraint)
        }
        expected = {
            "NEW_TOPIC",
            "FULL",
            "WINDOW",
            "SUMMARY",
            "UNRESOLVED",
            "FOLLOW_UP_FULL",
            "FOLLOW_UP_WINDOW",
            "FOLLOW_UP_SUMMARY",
        }

        self.assertEqual(expected, {member.value for member in ContextStrategy})
        self.assertIsInstance(strategy_type, SAEnum)
        self.assertIs(ContextStrategy, strategy_type.enum_class)
        self.assertEqual(expected, set(strategy_type.enums))
        self.assertIn("ck_rag_runs_context_strategy", rag_run_constraint_names)


    def test_rag_run_drops_sanitized_query_and_keeps_query_hash(self) -> None:
        columns = set(RagRun.__table__.columns.keys())

        self.assertNotIn("sanitized_query", columns)
        self.assertIn("query_hash", columns)

    def test_model_call_links_classification_run_with_owner_combination(self) -> None:
        table = ModelCall.__table__
        foreign_key = next(iter(table.c.classification_run_id.foreign_keys))
        owner_check = next(
            constraint
            for constraint in table.constraints
            if isinstance(constraint, CheckConstraint)
            and constraint.name == "ck_model_calls_owner_combination"
        )

        self.assertTrue(table.c.classification_run_id.nullable)
        self.assertEqual("classification_runs.id", foreign_key.target_fullname)
        self.assertEqual("CASCADE", foreign_key.ondelete)
        self.assertIn("QUESTION_CLASSIFICATION", str(owner_check.sqltext))
        self.assertIn("num_nonnulls", str(owner_check.sqltext))

    def test_model_call_records_cached_and_reasoning_token_subsets(self) -> None:
        table = ModelCall.__table__
        check_names = {
            constraint.name
            for constraint in table.constraints
            if isinstance(constraint, CheckConstraint)
        }

        for column in ("cached_input_tokens", "reasoning_tokens"):
            self.assertTrue(table.c[column].nullable, column)
            self.assertIsInstance(table.c[column].type, Integer)
        self.assertTrue(
            {
                "ck_model_calls_cached_input_tokens",
                "ck_model_calls_reasoning_tokens",
            }
            <= check_names
        )

    def test_document_source_group_must_match_group_source(self) -> None:
        source_table = DocumentGroupSource.__table__
        unique_constraints = {
            constraint.name: tuple(constraint.columns.keys())
            for constraint in source_table.constraints
            if isinstance(constraint, UniqueConstraint)
        }
        self.assertEqual(
            ("id", "document_group_id"),
            unique_constraints[GROUP_SOURCE_DOCUMENT_GROUP_UNIQUE_CONSTRAINT],
        )

        table = DocumentSource.__table__
        foreign_keys = {
            constraint.name: constraint
            for constraint in table.foreign_key_constraints
        }
        composite = foreign_keys[DOCUMENT_SOURCE_GROUP_SOURCE_FK_CONSTRAINT]
        self.assertEqual(
            ["group_source_id", "document_group_id"],
            list(composite.column_keys),
        )
        self.assertEqual(
            ["document_group_sources.id", "document_group_sources.document_group_id"],
            [element.target_fullname for element in composite.elements],
        )
        self.assertEqual("RESTRICT", composite.ondelete)
        self.assertLessEqual(len(composite.name), 63)
        self.assertEqual(1, len(table.c.group_source_id.foreign_keys))
        self.assertTrue(table.c.group_source_id.nullable)

    def test_problem_group_fills_exactly_one_target_matching_kind(self) -> None:
        table = QuestionProblemGroup.__table__
        check_names = {
            constraint.name
            for constraint in table.constraints
            if isinstance(constraint, CheckConstraint)
        }
        unique_constraints = {
            tuple(constraint.columns.keys())
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        }

        self.assertEqual(
            {"DOCUMENT", "NO_DOCUMENT"},
            {member.value for member in QuestionProblemGroupKind},
        )
        self.assertIsInstance(table.c.id.type, UUID)
        self.assertTrue(table.c.document_source_id.nullable)
        self.assertTrue(table.c.document_group_id.nullable)
        self.assertNotIn("title", table.columns.keys())
        self.assertIn("ck_question_problem_groups_kind_target", check_names)
        self.assertIn("ck_question_problem_groups_problem_group_kind", check_names)
        self.assertIn(("document_source_id",), unique_constraints)
        self.assertIn(("document_group_id",), unique_constraints)

    def test_subproblem_has_status_serving_state_and_revisions(self) -> None:
        table = QuestionSubproblem.__table__
        self.assertEqual(
            {"DRAFT", "APPROVED", "ARCHIVED"},
            {member.value for member in QuestionSubproblemStatus},
        )
        self.assertEqual(
            {"UNUSED", "SHADOW", "SERVING", "STOPPED"},
            {member.value for member in QuestionSubproblemServingState},
        )
        self.assertFalse(table.c.problem_group_id.nullable)
        self.assertFalse(table.c.key.nullable)
        self.assertEqual(200, table.c.key.type.length)
        subproblem_unique = {
            tuple(constraint.columns.keys())
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        }
        # 같은 분류 체계를 다른 문서 그룹에 넣을 수 있어 전역 유일이 아니다.
        self.assertEqual({("problem_group_id", "key")}, subproblem_unique)
        self.assertFalse(table.c.serving_state.nullable)
        self.assertEqual("UNUSED", table.c.serving_state.server_default.arg)

        revisions = QuestionSubproblemRevision.__table__
        unique_constraints = {
            tuple(constraint.columns.keys())
            for constraint in revisions.constraints
            if isinstance(constraint, UniqueConstraint)
        }
        self.assertIn(("subproblem_id", "version"), unique_constraints)
        self.assertNotIn("key_snapshot", revisions.columns.keys())
        # 포함 기준 벡터와 그 설정, 입력 문장 구성 판은 함께 채운다.
        self.assertIsInstance(revisions.c.inclusion_embedding.type, VECTOR)
        self.assertEqual(1536, revisions.c.inclusion_embedding.type.dim)
        for column in (
            "inclusion_embedding",
            "embedding_config_id",
            "embedding_text_version",
        ):
            self.assertTrue(revisions.c[column].nullable, column)
        self.assertEqual(50, revisions.c.embedding_text_version.type.length)
        embedding_fk = next(iter(revisions.c.embedding_config_id.foreign_keys))
        self.assertEqual("embedding_configs.id", embedding_fk.target_fullname)
        self.assertEqual("RESTRICT", embedding_fk.ondelete)
        self.assertIn(
            "ck_question_subproblem_revisions_inclusion_embedding",
            {
                constraint.name
                for constraint in revisions.constraints
                if isinstance(constraint, CheckConstraint)
            },
        )
        self.assertFalse(
            [index for index in revisions.indexes if "embedding" in str(index.name)]
        )

    def test_classification_run_records_group_index_and_kind(self) -> None:
        table = ClassificationRun.__table__

        self.assertEqual(
            {"ONLINE", "BACKFILL", "OPERATOR", "REJUDGE"},
            {member.value for member in ClassificationRunKind},
        )
        self.assertFalse(table.c.document_group_id.nullable)
        self.assertFalse(table.c.index_version_id.nullable)
        self.assertFalse(table.c.model.nullable)
        self.assertFalse(table.c.prompt_version.nullable)
        open_online = {index.name: index for index in table.indexes}[
            OPEN_ONLINE_CLASSIFICATION_RUN_CONSTRAINT
        ]
        self.assertTrue(open_online.unique)
        self.assertEqual(
            ["document_group_id", "index_version_id", "model", "prompt_version"],
            list(open_online.columns.keys()),
        )
        self.assertEqual(
            "kind = 'ONLINE' AND finished_at IS NULL",
            str(open_online.dialect_options["postgresql"]["where"]),
        )

    def test_question_classification_is_append_only_with_one_current_row(self) -> None:
        table = QuestionClassification.__table__
        check_names = {
            constraint.name
            for constraint in table.constraints
            if isinstance(constraint, CheckConstraint)
        }
        partial_unique = {
            index.name: (
                tuple(index.columns.keys()),
                str(index.dialect_options["postgresql"]["where"]),
            )
            for index in table.indexes
            if index.unique
        }

        self.assertEqual(
            {"CONNECT", "SEPARATE", "UNCLASSIFIED"},
            {member.value for member in ClassificationDecision},
        )
        self.assertEqual(
            {"SUBPROBLEM", "CITATION", "DOCUMENT", "NONE"},
            {member.value for member in AttributionSource},
        )
        self.assertTrue(table.c.subproblem_id.nullable)
        self.assertFalse(table.c.problem_group_id.nullable)
        self.assertFalse(table.c.run_id.nullable)
        self.assertIsInstance(table.c.judgment_input.type, JSONB)
        self.assertNotIn("verified", table.columns.keys())
        self.assertEqual(
            (("rag_run_id",), "effective_to IS NULL"),
            partial_unique[CURRENT_CLASSIFICATION_CONSTRAINT],
        )
        self.assertTrue(
            {
                "ck_question_classifications_connect_subproblem",
                "ck_question_classifications_connect_attribution",
                "ck_question_classifications_subproblem_version",
                "ck_question_classifications_effective_period",
            }
            <= check_names
        )

    def test_question_embedding_is_keyed_by_rag_run(self) -> None:
        table = QuestionEmbedding.__table__
        foreign_key = next(iter(table.c.rag_run_id.foreign_keys))

        self.assertEqual(["rag_run_id"], [column.name for column in table.primary_key])
        self.assertEqual("rag_runs.id", foreign_key.target_fullname)
        self.assertEqual("CASCADE", foreign_key.ondelete)
        self.assertFalse(table.c.embedding_config_id.nullable)

    def test_canonical_answer_allows_one_approved_row_per_subproblem(self) -> None:
        table = CanonicalAnswer.__table__
        source_fk = next(iter(table.c.source_rag_run_id.foreign_keys))
        partial_unique = {
            index.name: (
                tuple(index.columns.keys()),
                str(index.dialect_options["postgresql"]["where"]),
            )
            for index in table.indexes
            if index.unique
        }

        self.assertEqual(
            {"SELECTED", "AUTHORED"},
            {member.value for member in CanonicalAnswerOrigin},
        )
        self.assertEqual(
            {"DRAFT", "APPROVED", "REVOKED"},
            {member.value for member in CanonicalAnswerApproval},
        )
        self.assertTrue(table.c.source_rag_run_id.nullable)
        self.assertEqual("SET NULL", source_fk.ondelete)
        self.assertEqual(
            (("subproblem_id",), "approval = 'APPROVED'"),
            partial_unique[APPROVED_CANONICAL_ANSWER_CONSTRAINT],
        )

    def test_canonical_answer_citation_restricts_chunk_and_version_delete(self) -> None:
        table = CanonicalAnswerCitation.__table__
        unique_constraints = {
            tuple(constraint.columns.keys())
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        }

        self.assertIn(("canonical_answer_id", "citation_order"), unique_constraints)
        for column, target in (
            ("chunk_id", "document_chunks.id"),
            ("document_version_id", "document_versions.id"),
        ):
            foreign_key = next(iter(table.c[column].foreign_keys))
            self.assertFalse(table.c[column].nullable)
            self.assertEqual(target, foreign_key.target_fullname)
            self.assertEqual("RESTRICT", foreign_key.ondelete)
        self.assertEqual(
            set(AnswerCitation.__table__.columns.keys()) - {"rag_run_id"},
            set(table.columns.keys()) - {"canonical_answer_id"},
        )

    def test_cache_attempt_outcome_constraints(self) -> None:
        table = QuestionCacheAttempt.__table__
        check_names = {
            constraint.name
            for constraint in table.constraints
            if isinstance(constraint, CheckConstraint)
        }
        unique_constraints = {
            tuple(constraint.columns.keys())
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        }

        self.assertEqual(
            {"SERVED", "SHADOW", "GROUP_DISABLED", "REJECTED", "SKIPPED", "FAILED"},
            {member.value for member in CacheAttemptOutcome},
        )
        self.assertIn(("rag_run_id",), unique_constraints)
        self.assertIn(("classification_id",), unique_constraints)
        self.assertTrue(table.c.classification_id.nullable)
        self.assertTrue(table.c.canonical_answer_id.nullable)
        self.assertIsInstance(table.c.rejection_reasons.type, ARRAY)
        self.assertTrue(
            {
                "ck_question_cache_attempts_outcome_canonical_answer",
                "ck_question_cache_attempts_outcome_rejection_reasons",
                "ck_question_cache_attempts_cache_attempt_outcome",
            }
            <= check_names
        )

    def test_exact_question_match_tracks_source_and_historical_provenance(self) -> None:
        table = ExactQuestionMatch.__table__
        self.assertEqual("exact_question_matches", ExactQuestionMatch.__tablename__)
        self.assertEqual({"RECOMMENDED", "HISTORICAL_SERVED"}, {item.value for item in ExactQuestionMatchSource})
        self.assertEqual({"ACTIVE", "CONFLICT"}, {item.value for item in ExactQuestionMatchState})
        self.assertTrue(table.c.canonical_answer_id.nullable)
        self.assertTrue(table.c.source_rag_run_id.nullable)
        self.assertIn(
            "ck_exact_question_matches_source_provenance",
            {constraint.name for constraint in table.constraints if isinstance(constraint, CheckConstraint)},
        )
        self.assertIn(
            ("document_group_id", "normalized_question"),
            {
                tuple(constraint.columns.keys())
                for constraint in table.constraints
                if isinstance(constraint, UniqueConstraint)
            },
        )

    def test_constraint_names_fit_postgres_identifier_limit(self) -> None:
        names = [
            str(item.name)
            for table in _application_tables()
            for item in (*table.constraints, *table.indexes)
        ]
        self.assertEqual([], [name for name in names if len(name) > 63])


if __name__ == "__main__":
    unittest.main()
