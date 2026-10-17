"""Text-free execution provenance for inspecting and replaying answer runs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol



class ExecutionStage(str, Enum):
    QUERY_RESOLUTION = "QUERY_RESOLUTION"
    RETRIEVAL = "RETRIEVAL"
    SOURCE_SELECTION = "SOURCE_SELECTION"
    SOURCE_HANDLING = "SOURCE_HANDLING"
    ANSWER_GENERATION = "ANSWER_GENERATION"


class StageOutcome(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class ReplayContext:
    """Identifiers needed to locate private inputs and reproduce a run.

    The values are version labels or opaque IDs, never customer text or source
    document contents. The evaluation repository owns the private input data.
    """

    run_id: str
    item_id: str
    turn_no: int
    golden_version: str
    source_corpus_version: Optional[str]
    service_version: str
    code_revision: str
    cache_policy_version: str
    embedding_model_version: Optional[str] = None
    prompt_version: Optional[str] = None

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "item_id",
            "golden_version",
            "service_version",
            "code_revision",
            "cache_policy_version",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name}은 비어 있을 수 없습니다.")
        if (
            isinstance(self.turn_no, bool)
            or not isinstance(self.turn_no, int)
            or self.turn_no < 1
        ):
            raise ValueError("turn_no는 1 이상의 정수여야 합니다.")


@dataclass(frozen=True)
class ExecutionTraceEvent:
    stage: ExecutionStage
    outcome: StageOutcome
    duration_ms: int = 0
    detail_code: Optional[str] = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or self.duration_ms < 0
        ):
            raise ValueError("duration_ms는 0 이상의 정수여야 합니다.")


@dataclass(frozen=True)
class AnswerExecutionTrace:
    trace_id: uuid.UUID
    target: str
    replay_context: ReplayContext
    events: tuple[ExecutionTraceEvent, ...]


class ExecutionTraceRecorder(Protocol):
    """Output port implemented by an evaluation or observability repository."""

    def record(self, trace: AnswerExecutionTrace) -> None: ...
