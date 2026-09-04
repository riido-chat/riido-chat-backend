"""문서 접수 시점에 청크 embedding을 준비한다.

embedding API 호출은 store 밖에서 수행한다. store는 DB만 다루고,
호출 사용량과 지연은 호출한 쪽이 정확히 알기 때문이다.
"""

import asyncio
import time
from typing import List, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.error_message import sanitize_error_message
from app.core.hashing import sha256_hex
from app.database.models import ExecutionStatus
from app.document.document_store import DocumentStore
from app.retrieval.embedding import OpenAIEmbedder, build_embedding_text
from app.retrieval.models import RetrievalChunk


async def prepare_chunk_embeddings(
    session: AsyncSession,
    store: DocumentStore,
    ingestion_run_id: int,
    chunks: Sequence[RetrievalChunk],
    embedder: OpenAIEmbedder,
) -> List[List[float]]:
    """Chunk 순서에 맞춘 embedding 목록을 만든다.

    이전 판과 embedding 입력이 같은 Chunk는 저장된 vector를 복사하고
    나머지만 한 번의 요청으로 생성한다. 재사용만으로 채워지면 외부 호출이
    없으므로 남길 model call도 없다.

    호출이 필요하면 대화 경로와 같은 2단계 기록을 쓴다. 호출 직전에
    PROCESSING 행을 checkpoint commit 하고, 성공과 실패 모두 같은 행에서
    마감한다. 그래야 실패로 트랜잭션을 롤백해도 호출 이력이 남는다.
    """

    if not chunks:
        raise ValueError("embedding할 Chunk가 하나 이상이어야 합니다.")

    inputs = [build_embedding_text(chunk) for chunk in chunks]
    input_hashes = [sha256_hex(text) for text in inputs]
    reusable = await store.load_reusable_embeddings(ingestion_run_id, input_hashes)

    missing_positions = [
        position
        for position, input_hash in enumerate(input_hashes)
        if input_hash not in reusable
    ]
    if not missing_positions:
        return [list(reusable[input_hash]) for input_hash in input_hashes]

    call = await store.start_embedding_model_call(ingestion_run_id)
    # commit 이후에는 인스턴스 속성이 만료되므로 식별자를 먼저 붙든다
    model_call_id = call.id
    await session.commit()

    started_at = time.monotonic()
    try:
        response = await asyncio.to_thread(
            embedder.embed_many_with_usage,
            [inputs[position] for position in missing_positions],
        )
    except Exception as error:
        await store.finish_embedding_model_call(
            model_call_id,
            status=ExecutionStatus.FAILED,
            latency_ms=int((time.monotonic() - started_at) * 1000),
            error_message=sanitize_error_message(error),
        )
        await session.commit()
        raise

    await store.finish_embedding_model_call(
        model_call_id,
        status=ExecutionStatus.SUCCESS,
        latency_ms=int((time.monotonic() - started_at) * 1000),
        input_tokens=response.input_tokens,
        retry_count=response.retry_count,
    )

    generated = dict(zip(missing_positions, response.embeddings))
    return [
        list(generated[position])
        if position in generated
        else list(reusable[input_hashes[position]])
        for position in range(len(chunks))
    ]
