"""Ports used by the answer orchestration use case.

These protocols are the Python equivalent of Spring application-layer
interfaces: production and evaluation adapters can provide different
implementations without changing the orchestration policy.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, Protocol, TypeVar

from app.answering.models import FinalGenerationResult
from app.retrieval.models import HybridSearchCall

SourceDecision = TypeVar("SourceDecision")


class QueryResolverPort(Protocol):
    def __call__(self) -> Awaitable[Optional[str]]: ...


class RetrievalPort(Protocol):
    def __call__(self, resolved_query: str) -> Awaitable[HybridSearchCall]: ...


class SourceSelectorPort(Protocol[SourceDecision]):
    def __call__(self) -> Awaitable[SourceDecision]: ...


class SourceHandlerPort(Protocol[SourceDecision]):
    def __call__(self, decision: SourceDecision) -> Awaitable[bool]: ...


class SourceSelectionEnabledPort(Protocol):
    def __call__(self) -> Awaitable[bool]: ...


class PreRetrievalCacheInspectorPort(Protocol[SourceDecision]):
    def __call__(self) -> Awaitable[SourceDecision]: ...


class PreRetrievalCacheHandlerPort(Protocol[SourceDecision]):
    def __call__(self, decision: SourceDecision) -> Awaitable[bool]: ...


class AnswerGeneratorPort(Protocol):
    async def generate_answer(
        self, resolved_query: str, candidates: list[Any], **kwargs: Any
    ) -> FinalGenerationResult: ...


# Function aliases remain useful for small lambdas and make callback-based
# adapters read naturally while the Protocols document their ports.
QueryResolver = Callable[[], Awaitable[Optional[str]]]
Retriever = Callable[[str], Awaitable[HybridSearchCall]]
