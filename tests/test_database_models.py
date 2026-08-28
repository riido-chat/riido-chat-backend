import unittest

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Enum as SAEnum,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID

from app.database.base import Base
from app.database.models import (
    AnswerStatus,
    ChunkEmbedding,
    ContentNode,
    ContextStrategy,
    DocumentChunk,
    LegacyChunkEmbedding,
    LegacyDocumentChunk,
    ModelCall,
    ModelCallPurpose,
    RagRun,
)
from retrieval.embedding import OPENAI_EMBEDDING_DIMENSIONS


ERD_TABLE_NAMES = {
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
