"""Run a selected cache-path manifest and store its private trace record."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from app.answering.trace import ReplayContext
from evaluation.cache_path_eval.execution import (
    CacheDecisionCallbacks,
    CacheDecisionExecutionAdapter,
    CacheDecisionObservation,
)
from evaluation.cache_path_eval.runner import evaluate
from evaluation.cache_path_eval.store import CachePathRunMetadata, CachePathRunStore


class CachePathEvaluationProfilePort(Protocol):
    """Service-specific read-only data and isolated evaluation-call adapter.

    Implementations resolve golden item text from the private registry, read the
    selected service/profile/index, and return callbacks that never write to
    operational chat/grouping tables. Model-call accounting is private to the
    evaluation run.
    """

    async def callbacks_for_turn(
        self, selected_turn: Mapping[str, Any]
    ) -> CacheDecisionCallbacks: ...

    def replay_context_for_turn(
        self,
        metadata: CachePathRunMetadata,
        selected_turn: Mapping[str, Any],
    ) -> ReplayContext: ...


class _TraceCollector:
    def __init__(self) -> None:
        self.traces = []

    def record(self, trace) -> None:
        self.traces.append(trace)


@dataclass(frozen=True)
class CachePathRunOutput:
    result_path: str
    summary: Mapping[str, Any]
    observations: tuple[CacheDecisionObservation, ...]


async def execute_cache_path_run(
    *,
    manifest: Mapping[str, Any],
    metadata: CachePathRunMetadata,
    profile: CachePathEvaluationProfilePort,
    store: CachePathRunStore,
    execution_adapter: CacheDecisionExecutionAdapter,
) -> CachePathRunOutput:
    """Execute every selected turn once and persist it create-only.

    This function does not select a profile, open a DB, or load golden text. Those
    capabilities are supplied by the service-specific profile adapter and the
    private registry caller. The function makes no live calls by itself.
    """

    turns = manifest.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("실행할 cache-path 문항 턴이 없습니다.")
    observations = []
    for selected_turn in turns:
        if not isinstance(selected_turn, dict):
            raise ValueError("cache-path 문항 턴 형식이 잘못되었습니다.")
        callbacks = await profile.callbacks_for_turn(selected_turn)
        replay_context = profile.replay_context_for_turn(metadata, selected_turn)
        trace_collector = _TraceCollector()
        try:
            observation = await execution_adapter.execute(
                item_id=selected_turn["itemId"],
                turn_no=selected_turn["turnNo"],
                callbacks=callbacks,
                replay_context=replay_context,
                trace_recorder=trace_collector,
            )
        except Exception:
            if trace_collector.traces:
                store.write_failed_trace(
                    metadata,
                    item_id=selected_turn["itemId"],
                    turn_no=selected_turn["turnNo"],
                    trace=trace_collector.traces[-1],
                )
            raise
        observations.append(observation)

    rows = [observation.to_row() for observation in observations]
    canonical = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")
    aggregate = evaluate(manifest, rows, hashlib.sha256(canonical).hexdigest())
    destination = store.write_run(
        metadata,
        manifest=manifest,
        observations=observations,
        summary=aggregate["summary"],
        observation_sha256=aggregate["inputs"]["observationsSha256"],
    )
    return CachePathRunOutput(
        result_path=str(destination),
        summary=aggregate["summary"],
        observations=tuple(observations),
    )
