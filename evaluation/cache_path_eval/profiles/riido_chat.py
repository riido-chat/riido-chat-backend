"""Read-only Riido Chat profile boundary for cache-path evaluation.

The online ``ChatService`` cache callbacks also record operational grouping,
cache-attempt, and turn-log rows. This profile deliberately does not construct
those callbacks or open a database. A caller must inject a read-only selected
service-data adapter and a separate evaluation-owned replay-context provider.

The concrete data adapter is deployment-specific: selected profile/index data,
model-call accounting, and cache policy versions must come from an explicitly
chosen evaluation snapshot. This module does not guess those identifiers or
versions.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from app.answering.trace import ReplayContext
from evaluation.cache_path_eval.execution import CacheDecisionCallbacks
from evaluation.cache_path_eval.run import CachePathEvaluationProfilePort
from evaluation.cache_path_eval.store import CachePathRunMetadata


class RiidoChatReadOnlyCacheDataPort(Protocol):
    """Build callbacks over selected, read-only service data.

    Implementations may compose the shared query/retrieval/cache policy, but
    must not use online ``ChatService`` closures or write operational tables.
    Any judge/model-call accounting belongs to the private evaluation run.
    """

    async def callbacks_for_turn(
        self, selected_turn: Mapping[str, Any]
    ) -> CacheDecisionCallbacks: ...


class RiidoChatEvaluationContextPort(Protocol):
    """Resolve opaque run provenance from explicitly selected evaluation data."""

    def replay_context_for_turn(
        self,
        metadata: CachePathRunMetadata,
        selected_turn: Mapping[str, Any],
    ) -> ReplayContext: ...


class RiidoChatCachePathEvaluationProfile(CachePathEvaluationProfilePort):
    """Compose read-only service callbacks with isolated evaluation provenance.

    This adapter is intentionally a boundary, not a default production profile:
    the repository currently has no declared cache-path evaluation snapshot from
    which service profile IDs, index versions, and policy versions can be chosen.
    """

    def __init__(
        self,
        *,
        cache_data: RiidoChatReadOnlyCacheDataPort,
        evaluation_context: RiidoChatEvaluationContextPort,
    ) -> None:
        self._cache_data = cache_data
        self._evaluation_context = evaluation_context

    async def callbacks_for_turn(
        self, selected_turn: Mapping[str, Any]
    ) -> CacheDecisionCallbacks:
        return await self._cache_data.callbacks_for_turn(selected_turn)

    def replay_context_for_turn(
        self,
        metadata: CachePathRunMetadata,
        selected_turn: Mapping[str, Any],
    ) -> ReplayContext:
        return self._evaluation_context.replay_context_for_turn(
            metadata, selected_turn
        )
