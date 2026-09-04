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
from sqlalchemy.dialects.postgresql import ARRAY, UUID

from app.database.base import Base
from app.database.models import (
    ACTIVE_INDEX_VERSION_CONSTRAINT,
    INDEX_VERSION_NO_CONSTRAINT,
    AnswerStatus,
    ChunkEmbedding,
    ContentNode,
    ContextStrategy,
    DocumentChunk,
    DocumentGroup,
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
    LegacyChunkEmbedding,
    LegacyDocumentChunk,
    ModelCall,
    ModelCallPurpose,
    RagRun,
)
from app.retrieval.embedding import OPENAI_EMBEDDING_DIMENSIONS


ERD_TABLE_NAMES = {
    "document_groups",
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
}

LEGACY_TABLE_NAMES = {"legacy_document_chunks", "legacy_chunk_embeddings"}


class DatabaseModelTest(unittest.TestCase):
    def test_registers_erd_and_legacy_tables(self) -> None:
        self.assertEqual(
            ERD_TABLE_NAMES | LEGACY_TABLE_NAMES,
            set(Base.metadata.tables),
        )
        self.assertEqual("legacy_document_chunks", LegacyDocumentChunk.__tablename__)
        self.assertEqual("legacy_chunk_embeddings", LegacyChunkEmbedding.__tablename__)
        self.assertEqual("document_chunks", DocumentChunk.__tablename__)
        self.assertEqual("chunk_embeddings", ChunkEmbedding.__tablename__)

    def test_legacy_document_chunk_matches_retrieval_chunk_fields(self) -> None:
        table = LegacyDocumentChunk.__table__

        self.assertEqual(
            {
                "chunk_id",
                "document_id",
                "section_id",
                "document_title",
                "section_path",
                "source_url",
                "category",
                "content",
            },
            set(table.columns.keys()),
        )
        self.assertTrue(table.c.chunk_id.primary_key)
        self.assertIsInstance(table.c.chunk_id.type, Text)
        self.assertIsInstance(table.c.section_path.type, ARRAY)
        self.assertIsInstance(table.c.section_path.type.item_type, Text)

    def test_legacy_chunk_embedding_keeps_one_to_one_cascade_constraint(self) -> None:
        table = LegacyChunkEmbedding.__table__
        foreign_key = next(iter(table.c.chunk_id.foreign_keys))
        unique_constraints = {
            constraint.name: tuple(constraint.columns.keys())
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        }

        self.assertTrue(table.c.id.primary_key)
        self.assertIsInstance(table.c.id.type, BigInteger)
        self.assertIsNotNone(table.c.id.identity)
        self.assertFalse(table.c.chunk_id.nullable)
        self.assertEqual(
            "legacy_document_chunks.chunk_id",
            foreign_key.target_fullname,
        )
        self.assertEqual("CASCADE", foreign_key.ondelete)
        self.assertEqual(
            ("chunk_id",),
            unique_constraints["uq_chunk_embeddings_chunk_id"],
        )

    def test_embeddings_use_confirmed_vector_dimension(self) -> None:
        for table in (LegacyChunkEmbedding.__table__, ChunkEmbedding.__table__):
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

    def test_document_source_is_identified_by_group_and_document_key(self) -> None:
        table = DocumentSource.__table__
        unique_constraints = {
            constraint.name: tuple(constraint.columns.keys())
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        }
        foreign_key = next(iter(table.c.document_group_id.foreign_keys))

        self.assertFalse(table.c.document_group_id.nullable)
        self.assertFalse(table.c.document_key.nullable)
        self.assertIsInstance(table.c.document_key.type, String)
        self.assertEqual(300, table.c.document_key.type.length)
        self.assertEqual("document_groups.id", foreign_key.target_fullname)
        self.assertEqual("RESTRICT", foreign_key.ondelete)
        self.assertEqual(
            ("document_group_id", "document_key"),
            unique_constraints["uq_document_sources_document_group_id_document_key"],
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


if __name__ == "__main__":
    unittest.main()
