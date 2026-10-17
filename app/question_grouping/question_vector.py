"""Shared resolved-query embedding reuse and retry policy."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Sequence, Tuple

from app.core.model_trace import ModelCallTrace
from app.core.openai_error import is_transient_openai_error
from app.database.models import EMBEDDING_DIMENSIONS
from app.retrieval.embedding import OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_PROVIDER
from app.retrieval.models import HybridSearchCall
from app.retrieval.vector_retriever import (
    QUERY_EMBEDDING_INITIAL_RETRY_DELAY_SECONDS,
    QUERY_EMBEDDING_MAX_ATTEMPTS,
    QUERY_EMBEDDING_TIMEOUT_SECONDS,
)


class QuestionVectorDimensionError(ValueError):
    """A question embedding did not match the configured vector dimension."""

    def __init__(self, actual: int) -> None:
        self.actual = actual
        super().__init__(
            f"질문 벡터 차원이 {EMBEDDING_DIMENSIONS}이 아닙니다: {actual}"
        )


@dataclass(frozen=True)
class QuestionVectorResult:
    embedding: Optional[Tuple[float, ...]]
    reused_retrieval_embedding: bool
    trace: Optional[ModelCallTrace] = None
    error: Optional[BaseException] = None


async def resolve_question_vector(
    resolved_query: str,
    search: HybridSearchCall,
    embedder: Any,
    *,
    before_model_call: Optional[Callable[[], Awaitable[None]]] = None,
    clock: Callable[[], float] = time.perf_counter,
) -> QuestionVectorResult:
    """Reuse retrieval's vector when exact; otherwise embed with production retries."""

    if (
        search.retrieval_query is not None
        and search.retrieval_query == resolved_query
        and search.query_embedding
    ):
        embedding = tuple(float(value) for value in search.query_embedding)
        if len(embedding) != EMBEDDING_DIMENSIONS:
            return QuestionVectorResult(
                None,
                True,
                error=QuestionVectorDimensionError(len(embedding)),
            )
        return QuestionVectorResult(embedding, True)

    if before_model_call is not None:
        await before_model_call()
    started = clock()
    attempt = 0
    while True:
        try:
            response = await asyncio.to_thread(
                embedder.embed_many_with_usage,
                [resolved_query],
                sdk_max_retries=0,
                timeout=QUERY_EMBEDDING_TIMEOUT_SECONDS,
            )
            break
        except Exception as error:
            if (
                attempt + 1 < QUERY_EMBEDDING_MAX_ATTEMPTS
                and is_transient_openai_error(error)
            ):
                await asyncio.sleep(
                    QUERY_EMBEDDING_INITIAL_RETRY_DELAY_SECONDS * (2**attempt)
                )
                attempt += 1
                continue
            return QuestionVectorResult(
                None,
                False,
                trace=ModelCallTrace(
                    provider=OPENAI_EMBEDDING_PROVIDER,
                    model_name=OPENAI_EMBEDDING_MODEL,
                    succeeded=False,
                    latency_ms=max(0, int((clock() - started) * 1000)),
                    retry_count=attempt,
                    error_message=(
                        f"질문 임베딩 호출 실패: HTTP {error.status_code}"
                        if isinstance(getattr(error, "status_code", None), int)
                        else f"질문 임베딩 호출 실패: {type(error).__name__}"
                    ),
                ),
                error=error,
            )

    embedding = tuple(float(value) for value in response.embeddings[0])
    trace = ModelCallTrace(
        provider=OPENAI_EMBEDDING_PROVIDER,
        model_name=OPENAI_EMBEDDING_MODEL,
        succeeded=True,
        latency_ms=max(0, int((clock() - started) * 1000)),
        retry_count=attempt + response.retry_count,
        input_tokens=response.input_tokens,
    )
    if len(embedding) != EMBEDDING_DIMENSIONS:
        return QuestionVectorResult(
            None,
            False,
            trace=trace,
            error=QuestionVectorDimensionError(len(embedding)),
        )
    return QuestionVectorResult(embedding, False, trace=trace)
