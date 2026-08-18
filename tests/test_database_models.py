import unittest

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import BigInteger, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY

from app.database.base import Base
from app.database.models import ChunkEmbedding, DocumentChunk
from retrieval.embedding import OPENAI_EMBEDDING_DIMENSIONS


class DatabaseModelTest(unittest.TestCase):
    def test_registers_only_vector_retrieval_tables(self) -> None:
        self.assertEqual(
            {"document_chunks", "chunk_embeddings"},
            set(Base.metadata.tables),
        )
        self.assertEqual("document_chunks", DocumentChunk.__tablename__)
        self.assertEqual("chunk_embeddings", ChunkEmbedding.__tablename__)

    def test_document_chunk_matches_retrieval_chunk_fields(self) -> None:
        table = DocumentChunk.__table__

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
        self.assertFalse(table.c.document_id.nullable)
        self.assertFalse(table.c.section_id.nullable)
        self.assertEqual(0, len(table.c.document_id.foreign_keys))
        self.assertEqual(0, len(table.c.section_id.foreign_keys))
        self.assertIsInstance(table.c.section_path.type, ARRAY)
        self.assertIsInstance(table.c.section_path.type.item_type, Text)
        self.assertTrue(table.c.category.nullable)

    def test_chunk_embedding_has_one_to_one_cascade_constraint(self) -> None:
        table = ChunkEmbedding.__table__
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
            "document_chunks.chunk_id",
            foreign_key.target_fullname,
        )
        self.assertEqual("CASCADE", foreign_key.ondelete)
        self.assertEqual(
            ("chunk_id",),
            unique_constraints["uq_chunk_embeddings_chunk_id"],
        )

    def test_embedding_uses_confirmed_vector_dimension_without_ann_index(self) -> None:
        table = ChunkEmbedding.__table__

        self.assertIsInstance(table.c.embedding.type, VECTOR)
        self.assertEqual(
            OPENAI_EMBEDDING_DIMENSIONS,
            table.c.embedding.type.dim,
        )
        self.assertEqual(1536, table.c.embedding.type.dim)
        self.assertEqual(0, len(table.indexes))


if __name__ == "__main__":
    unittest.main()
