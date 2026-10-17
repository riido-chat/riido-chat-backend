"""Cache-path evaluation task adapter over the generic evaluation lifecycle."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from app.answering.trace import AnswerExecutionTrace, ReplayContext
from evaluation.cache_path_eval.execution import (
    CacheDecisionCallbacks,
    CacheDecisionExecutionAdapter,
    CacheDecisionObservation,
)
from evaluation.cache_path_eval.runner import evaluate
from evaluation.cache_path_eval.store import CachePathRunMetadata, CachePathRunStore
from evaluation.core.run import EvaluationFailure, run_evaluation


class CachePathEvaluationProfilePort(Protocol):
    """Service-specific read-only data and isolated evaluation-call adapter."""

    async def callbacks_for_turn(self, selected_turn: Mapping[str, Any]) -> CacheDecisionCallbacks: ...

    def replay_context_for_turn(
        self, metadata: CachePathRunMetadata, selected_turn: Mapping[str, Any]
    ) -> ReplayContext: ...


@dataclass(frozen=True)
class CachePathRunOutput:
    result_path: str
    summary: Mapping[str, Any]
    observations: tuple[CacheDecisionObservation, ...]


@dataclass(frozen=True)
class _PreparedTurn:
    callbacks: CacheDecisionCallbacks
    replay_context: ReplayContext


class _TraceCollector:
    def __init__(self) -> None:
        self.traces: list[AnswerExecutionTrace] = []

    def record(self, trace: AnswerExecutionTrace) -> None:
        self.traces.append(trace)


def project_reference_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only run provenance and item references to service profiles."""
    projected = {
        "group": manifest["group"],
        "goldenVersion": manifest["goldenVersion"],
        "turns": [_turn_reference(turn) for turn in manifest["turns"]],
    }
    if "sourceCorpusVersion" in manifest:
        projected["sourceCorpusVersion"] = manifest["sourceCorpusVersion"]
    if "inputs" in manifest:
        inputs = manifest["inputs"]
        projected["inputs"] = {
            "draftsSha256": inputs["draftsSha256"],
            "judgeLabelsSha256": inputs["judgeLabelsSha256"],
        }
    return projected


def _turn_reference(turn: Mapping[str, Any]) -> dict[str, Any]:
    return {"itemId": turn["itemId"], "turnNo": turn["turnNo"]}


class _CachePathTask:
    def __init__(
        self,
        *,
        profile: CachePathEvaluationProfilePort,
        store: CachePathRunStore,
        execution_adapter: CacheDecisionExecutionAdapter,
    ) -> None:
        self.profile = profile
        self.store = store
        self.execution_adapter = execution_adapter
        self._trace_collectors: dict[tuple[str, int], _TraceCollector] = {}
        self._contexts: dict[tuple[str, int], ReplayContext] = {}
        self._failure_turn: Mapping[str, Any] | None = None

    def preflight(
        self, manifest: Mapping[str, Any], metadata: CachePathRunMetadata
    ) -> Sequence[Mapping[str, Any]]:
        # The task/store own cache-specific validation. This validates every
        # selected turn and metadata relationship before any service callback.
        return self.store.validate_manifest(metadata, manifest)

    def prepare_run(
        self, metadata: CachePathRunMetadata, turns: Sequence[Mapping[str, Any]]
    ) -> None:
        for turn in turns:
            self._failure_turn = turn
            context = self.profile.replay_context_for_turn(metadata, _turn_reference(turn))
            self._validate_context(context, metadata, turn)
            self._contexts[(turn["itemId"], turn["turnNo"])] = context
        self._failure_turn = None

    async def prepare(
        self, metadata: CachePathRunMetadata, turn: Mapping[str, Any]
    ) -> _PreparedTurn:
        # Context construction and consistency checks precede callbacks, which
        # may bind expensive model or embedding operations.
        replay_context = self._contexts[(turn["itemId"], turn["turnNo"])]
        self._validate_context(replay_context, metadata, turn)
        self._failure_turn = turn
        callbacks = await self.profile.callbacks_for_turn(_turn_reference(turn))
        self._failure_turn = None
        return _PreparedTurn(callbacks, replay_context)

    async def execute(
        self,
        metadata: CachePathRunMetadata,
        turn: Mapping[str, Any],
        prepared: _PreparedTurn,
    ) -> CacheDecisionObservation:
        key = (turn["itemId"], turn["turnNo"])
        collector = _TraceCollector()
        self._trace_collectors[key] = collector
        return await self.execution_adapter.execute(
            item_id=turn["itemId"],
            turn_no=turn["turnNo"],
            callbacks=prepared.callbacks,
            replay_context=prepared.replay_context,
            trace_recorder=collector,
        )

    def finish(
        self,
        manifest: Mapping[str, Any],
        metadata: CachePathRunMetadata,
        observations: Sequence[CacheDecisionObservation],
    ) -> CachePathRunOutput:
        rows = [observation.to_row() for observation in observations]
        canonical = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ).encode("utf-8")
        aggregate = evaluate(manifest, rows, hashlib.sha256(canonical).hexdigest())
        destination = self.store.write_run(
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

    def failure(
        self, stage: str, turn: Mapping[str, Any] | None, error: BaseException
    ) -> EvaluationFailure:
        turn = turn or self._failure_turn
        safe_type = re.sub(r"[^A-Za-z0-9_]", "", type(error).__name__)[:100] or "Exception"
        detail: dict[str, Any] = {}
        if turn is not None:
            detail.update(itemId=turn.get("itemId"), turnNo=turn.get("turnNo"))
            collector = self._trace_collectors.get((turn.get("itemId"), turn.get("turnNo")))
            if collector and collector.traces:
                detail["trace"] = collector.traces[-1]
        return EvaluationFailure(stage=stage, error_type=safe_type, turn=turn, detail=detail)

    @staticmethod
    def _validate_context(
        context: ReplayContext,
        metadata: CachePathRunMetadata,
        turn: Mapping[str, Any],
    ) -> None:
        expected = {
            "run_id": metadata.run_id,
            "item_id": turn["itemId"],
            "turn_no": turn["turnNo"],
            "golden_version": metadata.golden_version,
            "service_version": metadata.service_key,
            "code_revision": metadata.app_build_commit,
            "cache_policy_version": metadata.cache_policy_version,
            "embedding_model_version": metadata.embedding_model_version,
        }
        if any(getattr(context, field) != value for field, value in expected.items()):
            raise ValueError("관측과 실행 context의 재현 메타데이터가 일치하지 않습니다.")


async def execute_cache_path_run(
    *,
    manifest: Mapping[str, Any],
    metadata: CachePathRunMetadata,
    profile: CachePathEvaluationProfilePort,
    store: CachePathRunStore,
    execution_adapter: CacheDecisionExecutionAdapter,
) -> CachePathRunOutput:
    """Compatibility facade for the common evaluation lifecycle."""

    # Callbacks receive isolated turn mappings; scoring always uses this snapshot.
    manifest = deepcopy(manifest)
    task = _CachePathTask(profile=profile, store=store, execution_adapter=execution_adapter)
    return await run_evaluation(manifest=manifest, metadata=metadata, task=task, store=store)
