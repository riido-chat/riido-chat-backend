"""Canonical corpus 전체를 embedding해 pgvector에 저장한다."""

import asyncio
from typing import List, Sequence

from app.database.models import IndexVersion
from app.database.session import dispose_engine, get_session_factory
from retrieval.corpus import build_retrieval_chunks
from retrieval.embedding import OpenAIEmbedder, build_embedding_text
from retrieval.models import RetrievalChunk
from retrieval.pgvector_store import PgVectorStore, StoredEmbedding


def build_index_items(
    chunks: Sequence[RetrievalChunk],
    embedder: OpenAIEmbedder,
) -> List[StoredEmbedding]:
    """모든 Chunk의 embedding이 준비된 저장 목록을 생성한다."""

    if not chunks:
        raise ValueError("Vector indexing할 Chunk가 하나 이상이어야 합니다.")

    embedding_texts = [build_embedding_text(chunk) for chunk in chunks]
    embeddings = embedder.embed_many(embedding_texts)
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Vector indexing embedding 개수가 Chunk 개수와 일치하지 않습니다: "
            f"Chunk {len(chunks)}개, embedding {len(embeddings)}개"
        )

    return list(zip(chunks, embeddings))


async def replace_vector_corpus(
    items: Sequence[StoredEmbedding],
) -> IndexVersion:
    """신규 ERD에 전체 corpus를 적재하고 새 ACTIVE index를 반환한다."""

    try:
        async with get_session_factory()() as session:
            return await PgVectorStore(session).replace_all(items)
    finally:
        await dispose_engine()


def main() -> None:
    chunks = build_retrieval_chunks()
    if not chunks:
        raise ValueError("Vector indexing할 Chunk가 하나 이상이어야 합니다.")

    items = build_index_items(chunks, OpenAIEmbedder())
    index_version = asyncio.run(replace_vector_corpus(items))
    print(
        "Vector corpus indexing 완료: "
        f"{len(items)}개 Chunk, ACTIVE index={index_version.id}"
    )


if __name__ == "__main__":
    main()
