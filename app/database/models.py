"""Vector Retrieval에 필요한 최소 ORM model을 정의한다."""

from typing import List, Optional

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import BigInteger, ForeignKey, Identity, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from retrieval.embedding import OPENAI_EMBEDDING_DIMENSIONS


class DocumentChunk(Base):
    """검색 결과를 RetrievalChunk로 복원하기 위한 Chunk와 metadata."""

    __tablename__ = "document_chunks"

    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(Text, nullable=False)
    section_id: Mapped[str] = mapped_column(Text, nullable=False)
    document_title: Mapped[str] = mapped_column(Text, nullable=False)
    section_path: Mapped[List[str]] = mapped_column(ARRAY(Text), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)


class ChunkEmbedding(Base):
    """Chunk에 1:1로 종속되는 OpenAI embedding."""

    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        UniqueConstraint("chunk_id", name="uq_chunk_embeddings_chunk_id"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
    )
    chunk_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey(
            "document_chunks.chunk_id",
            name="fk_chunk_embeddings_chunk_id_document_chunks",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    embedding: Mapped[List[float]] = mapped_column(
        VECTOR(OPENAI_EMBEDDING_DIMENSIONS),
        nullable=False,
    )
