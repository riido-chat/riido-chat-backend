"""Coordinate a set of related evaluation runs.

The plan owns orchestration only. Each executor call is expected to use the
existing single-run lifecycle, so child run state and parent plan state remain
separate and are linked by the store adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_CONFIG_BYTES = 64 * 1024
_MAX_ID_LENGTH = 100
_MAX_CONFIG_STRING_LENGTH = 200
_CREDENTIAL_KEYS = {"apikey", "authorization", "token", "secret", "password"}


def _check_safe_id(value: str, label: str, *, max_length: int = _MAX_ID_LENGTH) -> None:
    if not isinstance(value, str) or len(value) > max_length or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} must be a safe identifier of at most {max_length} characters")


def _check_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")


def _freeze_config(value: Any, path: str = "configuration") -> Any:
    """Validate JSON-only configuration while making a detached immutable copy."""
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} numbers must be finite")
        return value
    if isinstance(value, str):
        _check_safe_id(value, path, max_length=_MAX_CONFIG_STRING_LENGTH)
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            _check_safe_id(key, f"{path} key")
            normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
            if normalized_key in _CREDENTIAL_KEYS:
                raise ValueError(f"{path} contains a credential field")
            frozen[key] = _freeze_config(item, f"{path}.{key}")
        return _FrozenDict(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config(item, f"{path}[]") for item in value)
    raise ValueError(f"{path} must contain only JSON values")


class _FrozenDict(Mapping[str, Any]):
    """Small immutable mapping whose backing data is never shared with callers."""

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]) -> None:
        object.__setattr__(self, "_data", MappingProxyType(dict(data)))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("configuration mappings are immutable")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_config(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class EvaluationVariant:
    variant_id: str
    run_id: str
    configuration: Mapping[str, Any]
    _canonical_configuration: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _check_safe_id(self.variant_id, "variant_id")
        _check_safe_id(self.run_id, "run_id")
        if not isinstance(self.configuration, Mapping):
            raise ValueError("configuration must be a mapping")
        snapshot = _freeze_config(self.configuration)
        canonical = _canonical_config(_thaw(snapshot))
        if len(canonical.encode("utf-8")) > _MAX_CONFIG_BYTES:
            raise ValueError("configuration exceeds 64 KiB")
        object.__setattr__(self, "configuration", snapshot)
        object.__setattr__(self, "_canonical_configuration", canonical)

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(self._canonical_configuration.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "variantId": self.variant_id,
            "runId": self.run_id,
            "configuration": _thaw(self.configuration),
            "configSha256": self.config_sha256,
        }


@dataclass(frozen=True)
class EvaluationPlan:
    plan_id: str
    group: str
    selection_sha256: str
    variants: tuple[EvaluationVariant, ...]

    def __post_init__(self) -> None:
        _check_safe_id(self.plan_id, "plan_id")
        _check_safe_id(self.group, "group")
        _check_sha256(self.selection_sha256, "selection_sha256")
        variants = tuple(self.variants)
        if not variants:
            raise ValueError("variants must not be empty")
        if any(not isinstance(variant, EvaluationVariant) for variant in variants):
            raise ValueError("variants must contain EvaluationVariant values")
        if len({variant.variant_id for variant in variants}) != len(variants):
            raise ValueError("variant_id values must be unique within a plan")
        if len({variant.run_id for variant in variants}) != len(variants):
            raise ValueError("run_id values must be unique within a plan")
        object.__setattr__(self, "variants", variants)

    def to_dict(self) -> dict[str, Any]:
        return {
            "planId": self.plan_id,
            "group": self.group,
            "selectionSha256": self.selection_sha256,
            "variants": [variant.to_dict() for variant in self.variants],
        }


@dataclass(frozen=True)
class PreparedEvaluationInputs:
    input_sha256: str
    value: Any = field(repr=False)
    preparation_evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_sha256(self.input_sha256, "input_sha256")
        if not isinstance(self.preparation_evidence, Mapping):
            raise ValueError("preparation_evidence must be a mapping")
        snapshot = _freeze_config(self.preparation_evidence, "preparation_evidence")
        canonical = _canonical_config(_thaw(snapshot))
        if len(canonical.encode("utf-8")) > _MAX_CONFIG_BYTES:
            raise ValueError("preparation_evidence exceeds 64 KiB")
        object.__setattr__(self, "preparation_evidence", snapshot)

    def preparation_evidence_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible copy of safe preparation metadata."""
        return _thaw(self.preparation_evidence)


@dataclass(frozen=True)
class EvaluationVariantResult:
    variant_id: str
    run_id: str
    result_path: str
    result_sha256: str

    def __post_init__(self) -> None:
        _check_safe_id(self.variant_id, "variant_id")
        _check_safe_id(self.run_id, "run_id")
        _check_sha256(self.result_sha256, "result_sha256")
        if not isinstance(self.result_path, str) or not re.fullmatch(
            r"evals/[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\.json",
            self.result_path,
        ):
            raise ValueError("result_path must be a relative evals/<group>/<runid>.json path")

    def to_dict(self) -> dict[str, str]:
        return {
            "variantId": self.variant_id,
            "runId": self.run_id,
            "resultPath": self.result_path,
            "resultSha256": self.result_sha256,
        }


@dataclass(frozen=True)
class EvaluationPlanOutput:
    result_path: str
    results: tuple[EvaluationVariantResult, ...]

    def __post_init__(self) -> None:
        results = tuple(self.results)
        if any(not isinstance(result, EvaluationVariantResult) for result in results):
            raise ValueError("results must contain EvaluationVariantResult values")
        object.__setattr__(self, "results", results)


class EvaluationPlanExecutor(Protocol):
    def preflight(self, plan: EvaluationPlan) -> None: ...

    async def prepare(self, plan: EvaluationPlan) -> PreparedEvaluationInputs: ...

    async def execute(
        self,
        plan: EvaluationPlan,
        variant: EvaluationVariant,
        prepared: PreparedEvaluationInputs,
    ) -> EvaluationVariantResult: ...

    async def finalize(self, prepared: PreparedEvaluationInputs) -> None: ...


class EvaluationPlanStore(Protocol):
    def reserve(self, plan: EvaluationPlan) -> None: ...

    def prepared(self, plan: EvaluationPlan, prepared: PreparedEvaluationInputs) -> None: ...

    def checkpoint(self, plan: EvaluationPlan, result: EvaluationVariantResult) -> None: ...

    def succeed(
        self,
        plan: EvaluationPlan,
        prepared: PreparedEvaluationInputs,
        results: tuple[EvaluationVariantResult, ...],
    ) -> str: ...

    def fail(
        self,
        plan: EvaluationPlan,
        stage: str,
        variant_id: str | None,
        error_type: str,
    ) -> None: ...


async def run_evaluation_plan(
    *, plan: EvaluationPlan, executor: EvaluationPlanExecutor, store: EvaluationPlanStore
) -> EvaluationPlanOutput:
    """Run each variant sequentially over one prepared input snapshot.

    Preflight runs before the plan is reserved. Once reserved, child runs are
    checkpointed individually; this coordinator does not make those runs
    globally atomic.
    """
    executor.preflight(plan)
    store.reserve(plan)

    prepared: PreparedEvaluationInputs | None = None
    results: list[EvaluationVariantResult] = []
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    stage = "prepare"
    current_variant_id: str | None = None

    try:
        candidate = await executor.prepare(plan)
        if not isinstance(candidate, PreparedEvaluationInputs):
            raise TypeError("executor.prepare must return PreparedEvaluationInputs")
        prepared = candidate
        store.prepared(plan, prepared)
        for variant in plan.variants:
            current_variant_id = variant.variant_id
            stage = "execute"
            result = await executor.execute(plan, variant, prepared)
            if not isinstance(result, EvaluationVariantResult):
                raise TypeError("executor.execute must return EvaluationVariantResult")
            expected_path = f"evals/{plan.group}/{variant.run_id}.json"
            if (
                result.variant_id != variant.variant_id
                or result.run_id != variant.run_id
                or result.result_path != expected_path
            ):
                raise ValueError("executor result identifiers do not match the requested variant")
            stage = "checkpoint"
            store.checkpoint(plan, result)
            results.append(result)
        current_variant_id = None
    except BaseException as exc:
        primary_error = exc

    if prepared is not None:
        try:
            await executor.finalize(prepared)
        except BaseException as exc:
            cleanup_error = exc
            if primary_error is None:
                primary_error = exc
                stage = "finalize"
                current_variant_id = None

    if primary_error is not None:
        try:
            store.fail(plan, stage, current_variant_id, type(primary_error).__name__)
        except BaseException as persistence_error:
            if cleanup_error is not None and persistence_error.__cause__ is None:
                persistence_error.__cause__ = cleanup_error
            raise primary_error from persistence_error
        if cleanup_error is not None and cleanup_error is not primary_error:
            raise primary_error from cleanup_error
        raise primary_error

    assert prepared is not None
    stage = "succeed"
    try:
        result_path = store.succeed(plan, prepared, tuple(results))
    except BaseException as exc:
        try:
            store.fail(plan, stage, None, type(exc).__name__)
        except BaseException as persistence_error:
            raise exc from persistence_error
        raise
    return EvaluationPlanOutput(result_path=result_path, results=tuple(results))
