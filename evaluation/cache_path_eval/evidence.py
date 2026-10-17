"""Allowlisted, text-free evidence captured by the cache-path evaluator."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from app.core.model_trace import ModelCallTrace

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")
_MODEL_PURPOSES = frozenset(
    {"QUERY_REWRITE", "QUERY_EMBEDDING", "QUESTION_CLASSIFICATION"}
)


def _safe_optional_token(name: str, value: Optional[str]) -> None:
    if value is not None and (
        type(value) is not str or not _SAFE_TOKEN.fullmatch(value)
    ):
        raise ValueError(f"{name} must be a safe identifier or None")


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _optional_nonnegative_int(name: str, value: Optional[int]) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{name} must be a nonnegative integer or None")


@dataclass(frozen=True)
class RetrievalCandidateEvidence:
    chunk_id: int
    document_version_id: int
    index_version_id: int
    rank: int

    def __post_init__(self) -> None:
        for name in ("chunk_id", "document_version_id", "index_version_id", "rank"):
            _positive_int(name, getattr(self, name))

    def to_dict(self) -> dict[str, int]:
        return {
            "chunkId": self.chunk_id,
            "documentVersionId": self.document_version_id,
            "indexVersionId": self.index_version_id,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class SubproblemCandidateEvidence:
    key: str
    subproblem_id: str
    version: int
    canonical_answer_id: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("key", "subproblem_id"):
            value = getattr(self, name)
            if type(value) is not str or not _SAFE_TOKEN.fullmatch(value):
                raise ValueError(f"{name} must be a safe identifier")
        _positive_int("version", self.version)
        _safe_optional_token("canonical_answer_id", self.canonical_answer_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "subproblemId": self.subproblem_id,
            "version": self.version,
            "canonicalAnswerId": self.canonical_answer_id,
        }


@dataclass(frozen=True)
class ModelCallEvidence:
    purpose: str
    provider: str
    model_name: str
    prompt_version: Optional[str]
    succeeded: bool
    latency_ms: int
    retry_count: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cached_input_tokens: Optional[int]
    reasoning_tokens: Optional[int]

    def __post_init__(self) -> None:
        if type(self.purpose) is not str or self.purpose not in _MODEL_PURPOSES:
            raise ValueError("purpose is not a supported model-call purpose")
        for name in ("provider", "model_name"):
            value = getattr(self, name)
            if type(value) is not str or not _SAFE_TOKEN.fullmatch(value):
                raise ValueError(f"{name} must be a safe identifier")
        _safe_optional_token("prompt_version", self.prompt_version)
        if type(self.succeeded) is not bool:
            raise ValueError("succeeded must be a boolean")
        for name in ("latency_ms", "retry_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        ):
            _optional_nonnegative_int(name, getattr(self, name))

    @classmethod
    def from_trace(cls, purpose: str, trace: ModelCallTrace) -> "ModelCallEvidence":
        """Copy only the allowlisted scalar fields; discard trace error text."""
        if not isinstance(trace, ModelCallTrace):
            raise TypeError("trace must be a ModelCallTrace")
        return cls(
            purpose=purpose,
            provider=trace.provider,
            model_name=trace.model_name,
            prompt_version=trace.prompt_version,
            succeeded=trace.succeeded,
            latency_ms=trace.latency_ms,
            retry_count=trace.retry_count,
            input_tokens=trace.input_tokens,
            output_tokens=trace.output_tokens,
            cached_input_tokens=trace.cached_input_tokens,
            reasoning_tokens=trace.reasoning_tokens,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "purpose": self.purpose,
            "provider": self.provider,
            "modelName": self.model_name,
            "promptVersion": self.prompt_version,
            "succeeded": self.succeeded,
            "latencyMs": self.latency_ms,
            "retryCount": self.retry_count,
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
            "cachedInputTokens": self.cached_input_tokens,
            "reasoningTokens": self.reasoning_tokens,
        }


@dataclass(frozen=True)
class CacheRuntimeEvidence:
    seed: Optional[str] = None
    retrieval_candidates: tuple[RetrievalCandidateEvidence, ...] = ()
    subproblem_candidates: tuple[SubproblemCandidateEvidence, ...] = ()
    selected_subproblem_id: Optional[str] = None
    selected_canonical_answer_id: Optional[str] = None
    gate_rejection_reasons: tuple[str, ...] = ()
    exact_source_classification_id: Optional[int] = None
    exact_fallthrough_reasons: tuple[str, ...] = ()
    model_calls: tuple[ModelCallEvidence, ...] = ()

    def __post_init__(self) -> None:
        _safe_optional_token("seed", self.seed)
        _safe_optional_token("selected_subproblem_id", self.selected_subproblem_id)
        _safe_optional_token(
            "selected_canonical_answer_id", self.selected_canonical_answer_id
        )
        if self.exact_source_classification_id is not None:
            _positive_int(
                "exact_source_classification_id", self.exact_source_classification_id
            )
        for name, values, expected_type in (
            ("retrieval_candidates", self.retrieval_candidates, RetrievalCandidateEvidence),
            ("subproblem_candidates", self.subproblem_candidates, SubproblemCandidateEvidence),
            ("model_calls", self.model_calls, ModelCallEvidence),
        ):
            if type(values) is not tuple or any(type(value) is not expected_type for value in values):
                raise ValueError(f"{name} must be a tuple of {expected_type.__name__}")
        for name, reasons in (
            ("gate_rejection_reasons", self.gate_rejection_reasons),
            ("exact_fallthrough_reasons", self.exact_fallthrough_reasons),
        ):
            if type(reasons) is not tuple:
                raise ValueError(f"{name} must be a tuple")
            for reason in reasons:
                if type(reason) is not str or not _SAFE_TOKEN.fullmatch(reason):
                    raise ValueError(f"{name} must contain safe codes")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical allowlisted JSON representation."""
        return {
            "seed": self.seed,
            "retrievalCandidates": [item.to_dict() for item in self.retrieval_candidates],
            "subproblemCandidates": [item.to_dict() for item in self.subproblem_candidates],
            "selectedSubproblemId": self.selected_subproblem_id,
            "selectedCanonicalAnswerId": self.selected_canonical_answer_id,
            "gateRejectionReasons": list(self.gate_rejection_reasons),
            "exactSourceClassificationId": self.exact_source_classification_id,
            "exactFallthroughReasons": list(self.exact_fallthrough_reasons),
            "modelCalls": [item.to_dict() for item in self.model_calls],
        }
