"""Canonical corpus를 수집 이력과 함께 embedding해 pgvector에 저장한다."""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import IndexVersion
from app.database.session import dispose_engine, get_session_factory
from app.document.loader import load_normalized_documents
from app.document.models import NormalizedDocument
from app.retrieval.corpus import build_document_retrieval_chunks
from app.retrieval.embedding import OpenAIEmbedder, build_embedding_text
from app.retrieval.models import RetrievalChunk
from app.document.document_store import DocumentStore
from app.document.ingestion import prepare_chunk_embeddings
from app.indexing.index_writer import IndexWriter
from app.retrieval.models import StoredEmbedding


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReindexResult:
    """전체 재색인 완료 결과."""

    index_version: IndexVersion
    document_count: int
    chunk_count: int


def build_index_items(
    chunks: Sequence[RetrievalChunk],
    embedder: OpenAIEmbedder,
) -> List[StoredEmbedding]:
    """embedding이 없는 Chunk만 채운 저장 목록을 생성한다.

    embedding은 접수 시점에 만들어지므로 보통 빈 목록이 된다.
    """

    if not chunks:
        return []

    embedding_texts = [build_embedding_text(chunk) for chunk in chunks]
    embeddings = embedder.embed_many(embedding_texts)
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Vector indexing embedding 개수가 Chunk 개수와 일치하지 않습니다: "
            f"Chunk {len(chunks)}개, embedding {len(embeddings)}개"
        )

    return list(zip(chunks, embeddings))


async def _record_ingestion_failure(
    session: AsyncSession,
    store: DocumentStore,
    ingestion_run_id: int,
    error: Exception,
) -> None:
    try:
        await store.fail_ingestion(ingestion_run_id, error)
        await session.commit()
    except Exception:
        logger.exception(
            "수집 실패 로그를 마감하지 못했습니다: ingestion_run_id=%s",
            ingestion_run_id,
        )
        await session.rollback()


async def _record_index_failure(
    session: AsyncSession,
    writer: IndexWriter,
    index_run_id: int,
    error: Exception,
    failed_stage: str,
) -> None:
    try:
        await writer.fail_index(
            index_run_id,
            error,
            failed_stage=failed_stage,
        )
        await session.commit()
    except Exception:
        logger.exception(
            "색인 실패 로그를 마감하지 못했습니다: index_run_id=%s",
            index_run_id,
        )
        await session.rollback()


async def run_reindex(
    documents: Sequence[NormalizedDocument],
    embedder: OpenAIEmbedder,
    session: AsyncSession,
) -> ReindexResult:
    """문서별 수집과 전체 색인을 checkpoint transaction으로 실행한다."""

    if not documents:
        raise ValueError("수집할 정제 문서가 하나 이상이어야 합니다.")

    store = DocumentStore(session)
    writer = IndexWriter(session)
    persisted_chunks = []

    for document in documents:
        ingestion_run_id = None
        checkpoint_committed = False
        try:
            ingestion_run = await store.start_ingestion(document)
            ingestion_run_id = ingestion_run.id
            await session.commit()
            checkpoint_committed = True

            chunks = build_document_retrieval_chunks(document)
            embeddings = await prepare_chunk_embeddings(
                session,
                store,
                ingestion_run_id,
                chunks,
                embedder,
            )
            persisted_chunks.extend(
                await store.complete_ingestion(
                    ingestion_run_id,
                    document,
                    chunks,
                    embeddings,
                )
            )
            await session.commit()
        except Exception as error:
            await session.rollback()
            if checkpoint_committed and ingestion_run_id is not None:
                await _record_ingestion_failure(
                    session,
                    store,
                    ingestion_run_id,
                    error,
                )
            raise

    index_run_id = None
    checkpoint_committed = False
    failed_stage = "STARTING"
    try:
        index_run = await writer.start_index(persisted_chunks)
        index_run_id = index_run.id
        await session.commit()
        checkpoint_committed = True

        failed_stage = "EMBEDDING"
        missing = await writer.list_chunks_missing_embedding(
            index_run_id,
            persisted_chunks,
        )
        items = build_index_items(missing, embedder)

        failed_stage = "PERSISTING"
        await writer.store_index_items(index_run_id, persisted_chunks, items)
        await session.commit()

        failed_stage = "VALIDATING"
        await writer.mark_index_ready(index_run_id)
        await session.commit()

        failed_stage = "APPLYING"
        index_version = await writer.apply_index(index_run_id)
        await writer.finish_apply_run(index_run_id)
        await session.commit()
        return ReindexResult(
            index_version=index_version,
            document_count=len(documents),
            chunk_count=len(persisted_chunks),
        )
    except Exception as error:
        await session.rollback()
        if checkpoint_committed and index_run_id is not None:
            await _record_index_failure(
                session,
                writer,
                index_run_id,
                error,
                failed_stage,
            )
        raise


async def reindex_vector_corpus(
    documents: Sequence[NormalizedDocument],
    embedder: OpenAIEmbedder,
) -> ReindexResult:
    """DB session과 engine 수명주기를 포함해 전체 재색인을 실행한다."""

    try:
        async with get_session_factory()() as session:
            return await run_reindex(documents, embedder, session)
    finally:
        await dispose_engine()


def main() -> None:
    documents = load_normalized_documents()
    result = asyncio.run(reindex_vector_corpus(documents, OpenAIEmbedder()))
    print(
        "Vector corpus indexing 완료: "
        f"{result.chunk_count}개 Chunk, "
        f"ACTIVE index={result.index_version.id}"
    )


if __name__ == "__main__":
    main()
