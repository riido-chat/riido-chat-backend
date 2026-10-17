"""Generic orchestration for evaluation tasks.

The lifecycle knows only how to validate, reserve, prepare, execute, checkpoint,
and finish a run. Task-specific validation and persistence remain behind ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, Sequence, TypeVar


ManifestT = TypeVar("ManifestT")
MetadataT = TypeVar("MetadataT")
TurnT = TypeVar("TurnT")
PreparedT = TypeVar("PreparedT")
ObservationT = TypeVar("ObservationT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class EvaluationFailure:
    stage: str
    error_type: str
    turn: object | None = None
    detail: object | None = None


class EvaluationTask(Protocol, Generic[ManifestT, MetadataT, TurnT, PreparedT, ObservationT, ResultT]):
    def preflight(self, manifest: ManifestT, metadata: MetadataT) -> Sequence[TurnT]: ...
    def prepare_run(self, metadata: MetadataT, turns: Sequence[TurnT]) -> None: ...
    async def prepare(self, metadata: MetadataT, turn: TurnT) -> PreparedT: ...
    async def execute(self, metadata: MetadataT, turn: TurnT, prepared: PreparedT) -> ObservationT: ...
    def finish(self, manifest: ManifestT, metadata: MetadataT, observations: Sequence[ObservationT]) -> ResultT: ...
    def failure(self, stage: str, turn: TurnT | None, error: BaseException) -> EvaluationFailure: ...


class EvaluationRunStore(Protocol, Generic[MetadataT, ObservationT]):
    def reserve(self, metadata: MetadataT) -> None: ...
    def checkpoint(self, metadata: MetadataT, observation: ObservationT) -> None: ...
    def fail(self, metadata: MetadataT, failure: EvaluationFailure) -> None: ...
    def begin_finalization(self, metadata: MetadataT) -> None: ...
    def succeed(self, metadata: MetadataT) -> None: ...


async def run_evaluation(
    *,
    manifest: ManifestT,
    metadata: MetadataT,
    task: EvaluationTask[ManifestT, MetadataT, TurnT, PreparedT, ObservationT, ResultT],
    store: EvaluationRunStore[MetadataT, ObservationT, ResultT],
) -> ResultT:
    """Run a task after complete preflight and an atomic create-only reservation."""

    turns = tuple(task.preflight(manifest, metadata))
    if not turns:
        raise ValueError("선택된 평가 턴이 없습니다.")
    # Reserve after all cheap validation, and before any per-turn service work.
    store.reserve(metadata)
    observations: list[ObservationT] = []
    stage = "prepare"
    current_turn: TurnT | None = None
    try:
        stage = "context_preparation"
        task.prepare_run(metadata, turns)
        for current_turn in turns:
            stage = "prepare"
            prepared = await task.prepare(metadata, current_turn)
            stage = "execute"
            observation = await task.execute(metadata, current_turn, prepared)
            stage = "checkpoint"
            store.checkpoint(metadata, observation)
            observations.append(observation)
        stage = "aggregate"
        store.begin_finalization(metadata)
        result = task.finish(manifest, metadata, observations)
        store.succeed(metadata)
        return result
    except BaseException as error:
        failure = task.failure(stage, current_turn, error)
        try:
            store.fail(metadata, failure)
        except Exception as persistence_error:
            # Keep the execution exception as the raised error while retaining
            # the persistence failure as its explicit cause.
            raise error from persistence_error
        raise
