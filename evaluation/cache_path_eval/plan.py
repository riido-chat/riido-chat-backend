"""Bind the generic evaluation plan lifecycle to existing cache-path runs.

The module deliberately accepts all service preparation and profile construction
as injected ports. It does not bootstrap a database, model provider, or network
transport.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Protocol

from evaluation.cache_path_eval.evidence import CacheRuntimeEvidence
from evaluation.cache_path_eval.execution import (
    CacheDecisionExecutionAdapter,
    CacheDecisionObservation,
)
from evaluation.cache_path_eval.plan_store import CachePathPlanStore
from evaluation.cache_path_eval.run import CachePathEvaluationProfilePort, execute_cache_path_run
from evaluation.cache_path_eval.store import CachePathRunMetadata, CachePathRunStore
from evaluation.core.plan import (
    EvaluationPlan,
    EvaluationPlanOutput,
    EvaluationVariant,
    EvaluationVariantResult,
    PreparedEvaluationInputs,
    run_evaluation_plan,
)


_SAFE_REFERENCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")
_SHARED_IDENTITY_FIELDS = (
    "golden_version",
    "golden_commit",
    "source_corpus_version",
    "app_build_commit",
    "service_key",
    "profile_revision_id",
    "document_group_id",
    "index_version_id",
    "embedding_config_id",
    "cache_policy_version",
    "embedding_model_version",
)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(nested) for key, nested in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(nested) for nested in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(nested) for nested in value]
    return value


def _canonical_json(value: Any, *, label: str) -> str:
    def validate(item: Any) -> None:
        if item is None or type(item) in (bool, int, str):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError(f"{label} numbers must be finite")
            return
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if type(key) is not str:
                    raise ValueError(f"{label} object keys must be strings")
                validate(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                validate(nested)
            return
        raise ValueError(f"{label} must contain only JSON values")

    validate(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite JSON values") from error


def _reject_forbidden_keys(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} keys must be strings")
            normalized = "".join(character for character in key.lower() if character.isalnum())
            if normalized in {"apikey", "authorization", "token", "secret", "password", "instructions", "systemprompt", "developerprompt"}:
                raise ValueError(f"{label} contains a credential or prompt field")
            _reject_forbidden_keys(nested, label=label)


def _reject_expected_label_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str):
                normalized = "".join(character for character in key.lower() if character.isalnum())
                if normalized.startswith("expected") or normalized in {
                    "goldlabel",
                    "groundtruth",
                    "golddecision",
                }:
                    raise ValueError("prepared runtime payload contains expected-label fields")
            _reject_expected_label_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_expected_label_keys(nested)


def _reference_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Project only immutable references; never pass scoring labels or text."""
    projected = {
        "group": manifest["group"],
        "goldenVersion": manifest["goldenVersion"],
        "sourceCorpusVersion": manifest.get("sourceCorpusVersion"),
        "inputs": dict(manifest["inputs"]),
        "turns": [
            {"itemId": turn["itemId"], "turnNo": turn["turnNo"]}
            for turn in manifest["turns"]
        ],
    }
    if "evaluationGroup" in manifest:
        projected["evaluationGroup"] = dict(manifest["evaluationGroup"])
    return projected


@dataclass(frozen=True)
class CachePathPlanRunSpec:
    metadata: CachePathRunMetadata
    profile_config: Mapping[str, Any]
    _profile_config_json: str = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, CachePathRunMetadata):
            raise TypeError("metadata must be CachePathRunMetadata")
        if not isinstance(self.profile_config, Mapping):
            raise TypeError("profile_config must be a mapping")
        canonical = _canonical_json(self.profile_config, label="profile_config")
        _reject_forbidden_keys(self.profile_config, label="profile_config")
        object.__setattr__(self, "profile_config", _freeze_json(json.loads(canonical)))
        object.__setattr__(self, "_profile_config_json", canonical)

    @property
    def profile_config_sha256(self) -> str:
        return hashlib.sha256(self._profile_config_json.encode("utf-8")).hexdigest()

    def profile_config_dict(self) -> dict[str, Any]:
        return _thaw_json(self.profile_config)

    def variant_configuration(self) -> dict[str, Any]:
        """Return the complete safe, reproducible configuration for this run."""
        return {
            "judgeModelVersion": self.metadata.judge_model_version,
            "judgePromptVersion": self.metadata.judge_prompt_version,
            "profileConfigSha256": self.profile_config_sha256,
            "profileConfig": self.profile_config_dict(),
        }


@dataclass(frozen=True)
class _TurnPayloadSnapshot:
    key: tuple[str, int]
    canonical_json: str | None
    payload_sha256: str | None


class PreparedCacheInputs:
    """Detached per-turn prepared values with fresh-copy accessors.

    Payloads are retained internally as canonical JSON strings so callers cannot
    mutate a shared snapshot through nested dictionaries or lists.
    """

    __slots__ = (
        "_snapshots",
        "_input_sha256",
        "_opaque_context",
        "_preparation_evidence_json",
        "_finalizer",
    )

    def __init__(
        self,
        *,
        references: tuple[tuple[str, int], ...],
        payloads: Mapping[tuple[str, int], Any],
        opaque_context: Any = None,
        preparation_evidence: Mapping[str, Any] | None = None,
        finalizer: Callable[["PreparedCacheInputs"], Awaitable[None] | None] | None = None,
    ) -> None:
        normalized_references: list[tuple[str, int]] = []
        for reference in references:
            if not isinstance(reference, tuple) or len(reference) != 2:
                raise ValueError("prepared references must be (item_id, turn_no) tuples")
            item_id, turn_no = reference
            if not isinstance(item_id, str) or not _SAFE_REFERENCE_ID.fullmatch(item_id):
                raise ValueError("prepared reference item_id is invalid")
            if type(turn_no) is not int or turn_no < 1:
                raise ValueError("prepared reference turn_no must be a positive integer")
            normalized_references.append((item_id, turn_no))
        if len(set(normalized_references)) != len(normalized_references):
            raise ValueError("prepared references must be unique")
        references = tuple(normalized_references)
        if set(payloads) != set(references):
            raise ValueError("prepared payloads must map every fixed turn exactly once")
        snapshots: dict[tuple[str, int], _TurnPayloadSnapshot] = {}
        for key in references:
            payload = payloads.get(key)
            if payload is None:
                snapshots[key] = _TurnPayloadSnapshot(key, None, None)
                continue
            if not isinstance(payload, Mapping):
                raise ValueError("each prepared turn payload must be a JSON object or None")
            _reject_expected_label_keys(payload)
            canonical = _canonical_json(payload, label="prepared turn payload")
            snapshots[key] = _TurnPayloadSnapshot(
                key, canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            )
        reference_rows = [
            {"itemId": item_id, "turnNo": turn_no, "payloadSha256": snapshots[(item_id, turn_no)].payload_sha256}
            for item_id, turn_no in references
        ]
        input_hash = hashlib.sha256(_canonical_json(reference_rows, label="prepared input references").encode("utf-8")).hexdigest()
        object.__setattr__(self, "_snapshots", MappingProxyType(snapshots))
        object.__setattr__(self, "_input_sha256", input_hash)
        object.__setattr__(self, "_opaque_context", opaque_context)
        object.__setattr__(
            self,
            "_preparation_evidence_json",
            _canonical_json(preparation_evidence or {}, label="preparation_evidence"),
        )
        object.__setattr__(self, "_finalizer", finalizer)

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("prepared cache inputs are immutable")

    @property
    def input_sha256(self) -> str:
        return self._input_sha256

    @property
    def opaque_context(self) -> Any:
        """Provider-owned resources for injected factories; never persisted by this adapter."""
        return self._opaque_context

    @property
    def preparation_evidence(self) -> dict[str, Any]:
        """Fresh JSON copy of shared preparation measurements for the parent ledger."""
        return json.loads(self._preparation_evidence_json)

    def payload_for_turn(self, item_id: str, turn_no: int) -> Any:
        snapshot = self._snapshots.get((item_id, turn_no))
        if snapshot is None or snapshot.canonical_json is None:
            return None
        return json.loads(snapshot.canonical_json)

    def common_input_sha256_for_turn(self, item_id: str, turn_no: int) -> str | None:
        snapshot = self._snapshots.get((item_id, turn_no))
        return None if snapshot is None else snapshot.payload_sha256


class SharedInputPreparer(Protocol):
    async def __call__(self, reference_manifest: Mapping[str, Any]) -> PreparedCacheInputs: ...


class PreparedProfileFactory(Protocol):
    supports_fixed_prepared_inputs: bool

    def __call__(
        self,
        *,
        variant: EvaluationVariant,
        spec: CachePathPlanRunSpec,
        prepared: PreparedCacheInputs,
        reference_manifest: Mapping[str, Any],
    ) -> CachePathEvaluationProfilePort | Awaitable[CachePathEvaluationProfilePort]: ...


class _EvidenceCheckingExecutionAdapter:
    def __init__(
        self,
        delegate: CacheDecisionExecutionAdapter,
        prepared: PreparedCacheInputs,
        expected_model_name: str,
        expected_prompt_version: str,
    ) -> None:
        self._delegate = delegate
        self._prepared = prepared
        self._expected_model_name = expected_model_name
        self._expected_prompt_version = expected_prompt_version

    async def execute(self, **kwargs: Any) -> CacheDecisionObservation:
        observation = await self._delegate.execute(**kwargs)
        evidence = observation.evidence
        measured = observation.outcome not in {"FAILED", "SKIPPED"}
        if evidence is None:
            if measured or observation.model_calls or observation.embedding_calls:
                raise ValueError("measured cache outcomes require CacheRuntimeEvidence")
            return observation
        if not isinstance(evidence, CacheRuntimeEvidence):
            raise TypeError("cache outcome evidence must be CacheRuntimeEvidence")
        if observation.embedding_calls != 0 or any(
            call.purpose in {"QUERY_REWRITE", "QUERY_EMBEDDING"}
            for call in evidence.model_calls
        ):
            raise ValueError("child runs cannot repeat shared query rewrite or embedding work")
        classification_calls = [
            call for call in evidence.model_calls if call.purpose == "QUESTION_CLASSIFICATION"
        ]
        if observation.model_calls != len(classification_calls):
            raise ValueError("child model call count must match question-classification evidence")
        for call in classification_calls:
            if call.model_name != self._expected_model_name:
                raise ValueError("question-classification model does not match selected judge model")
            if call.prompt_version != self._expected_prompt_version:
                raise ValueError("question-classification prompt does not match selected judge prompt")
        if measured and observation.model_calls == 0 and observation.outcome == "SERVED":
            if evidence.exact_source_classification_id is None:
                raise ValueError("served turns without model calls require exact-source evidence")
        for call in classification_calls:
            expected_hash = self._prepared.common_input_sha256_for_turn(
                observation.item_id, observation.turn_no
            )
            if expected_hash is None:
                raise ValueError("question-classification call has no fixed prepared input")
            request_evidence = call.request_evidence
            if request_evidence is not None:
                if request_evidence.common_input_sha256 != expected_hash:
                    raise ValueError("question-classification evidence does not match fixed prepared input")
            elif call.succeeded or observation.outcome != "FAILED":
                raise ValueError("successful question-classification calls require fixed-input evidence")
        return observation


class CachePathEvaluationPlanExecutor:
    def __init__(
        self,
        *,
        manifest: Mapping[str, Any],
        run_specs: Mapping[str, CachePathPlanRunSpec],
        store: CachePathRunStore,
        execution_adapter: CacheDecisionExecutionAdapter,
        shared_input_preparer: SharedInputPreparer,
        prepared_profile_factory: PreparedProfileFactory,
        finalizer: Callable[[PreparedCacheInputs], Awaitable[None] | None] | None = None,
    ) -> None:
        if not isinstance(manifest, Mapping):
            raise TypeError("manifest must be a mapping")
        self._manifest = json.loads(_canonical_json(manifest, label="manifest"))
        self._run_specs = dict(run_specs)
        self._store = store
        self._execution_adapter = execution_adapter
        self._shared_input_preparer = shared_input_preparer
        self._prepared_profile_factory = prepared_profile_factory
        self._finalizer = finalizer
        self._finalized_bundle_ids: set[int] = set()
        self._reference_manifest_json = _canonical_json(
            _reference_manifest(self._manifest), label="reference manifest"
        )
        reference_manifest = json.loads(self._reference_manifest_json)
        self._expected_references = tuple(
            (turn["itemId"], turn["turnNo"]) for turn in reference_manifest["turns"]
        )

    def _reference_manifest_copy(self) -> dict[str, Any]:
        return json.loads(self._reference_manifest_json)

    def preflight(self, plan: EvaluationPlan) -> None:
        if plan.group != "cache-path":
            raise ValueError("cache-path plans must use group='cache-path'")
        full_manifest_hash = hashlib.sha256(
            _canonical_json(self._manifest, label="manifest").encode("utf-8")
        ).hexdigest()
        if plan.selection_sha256 != full_manifest_hash:
            raise ValueError("plan selection hash does not match the full frozen manifest")
        if set(self._run_specs) != {variant.variant_id for variant in plan.variants}:
            raise ValueError("run spec keys must match plan variant IDs exactly")
        if len({variant.run_id for variant in plan.variants}) != len(plan.variants):
            raise ValueError("child run IDs must be unique")
        baseline: tuple[Any, ...] | None = None
        for variant in plan.variants:
            spec = self._run_specs[variant.variant_id]
            metadata = spec.metadata
            if metadata.run_id != variant.run_id:
                raise ValueError("run metadata run_id must match the plan variant")
            if "sourceCorpusVersion" in self._manifest and metadata.source_corpus_version != self._manifest.get("sourceCorpusVersion"):
                raise ValueError("run metadata source corpus version does not match the frozen manifest")
            if "goldenCommit" in self._manifest and metadata.golden_commit != self._manifest.get("goldenCommit"):
                raise ValueError("run metadata golden commit does not match the frozen manifest")
            identity = tuple(getattr(metadata, field) for field in _SHARED_IDENTITY_FIELDS)
            if baseline is None:
                baseline = identity
            elif identity != baseline:
                raise ValueError("all variants must share golden, target, cache policy, and embedding identity")
            config = dict(variant.configuration)
            expected_config = {
                "judgeModelVersion": metadata.judge_model_version,
                "judgePromptVersion": metadata.judge_prompt_version,
                "profileConfigSha256": spec.profile_config_sha256,
                "profileConfig": spec.profile_config_dict(),
            }
            if config != expected_config:
                raise ValueError("variant configuration must bind judge versions and complete profile config")
            validator = getattr(self._prepared_profile_factory, "validate_variant", None)
            if callable(validator):
                validator(variant, spec)
            self._store.validate_manifest(metadata, self._manifest)
            self._store.check_available(metadata)
        if getattr(self._prepared_profile_factory, "supports_fixed_prepared_inputs", False) is not True:
            raise ValueError("prepared_profile_factory must declare fixed prepared-input support")

    async def prepare(self, _plan: EvaluationPlan) -> PreparedEvaluationInputs:
        reference_manifest = self._reference_manifest_copy()
        expected = self._expected_references
        prepared = await self._shared_input_preparer(reference_manifest)
        if not isinstance(prepared, PreparedCacheInputs):
            raise TypeError("shared_input_preparer must return PreparedCacheInputs")
        try:
            if set(prepared._snapshots) != set(expected) or len(prepared._snapshots) != len(expected):
                raise ValueError("prepared inputs must cover each selected turn exactly once")
            return PreparedEvaluationInputs(
                input_sha256=prepared.input_sha256,
                value=prepared,
                preparation_evidence=prepared.preparation_evidence,
            )
        except BaseException as error:
            try:
                await self._close_prepared_bundle(prepared)
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    async def execute(
        self,
        _plan: EvaluationPlan,
        variant: EvaluationVariant,
        prepared: PreparedEvaluationInputs,
    ) -> EvaluationVariantResult:
        bundle = prepared.value
        if not isinstance(bundle, PreparedCacheInputs):
            raise TypeError("prepared plan value must be PreparedCacheInputs")
        spec = self._run_specs[variant.variant_id]
        profile = self._prepared_profile_factory(
            variant=variant,
            spec=spec,
            prepared=bundle,
            reference_manifest=self._reference_manifest_copy(),
        )
        if inspect.isawaitable(profile):
            profile = await profile
        output = await execute_cache_path_run(
            manifest=self._manifest,
            metadata=spec.metadata,
            profile=profile,
            store=self._store,
            execution_adapter=_EvidenceCheckingExecutionAdapter(
                self._execution_adapter,
                bundle,
                spec.metadata.judge_model_version,
                spec.metadata.judge_prompt_version,
            ),
        )
        root = self._store._registry_root
        result = Path(output.result_path).resolve(strict=True)
        result.relative_to(root)
        relative = PurePosixPath(result.relative_to(root).as_posix()).as_posix()
        content = result.read_bytes()
        return EvaluationVariantResult(
            variant_id=variant.variant_id,
            run_id=variant.run_id,
            result_path=relative,
            result_sha256=hashlib.sha256(content).hexdigest(),
        )

    async def finalize(self, prepared: PreparedEvaluationInputs) -> None:
        bundle = prepared.value
        if not isinstance(bundle, PreparedCacheInputs):
            return
        await self._close_prepared_bundle(bundle)

    async def _close_prepared_bundle(self, bundle: PreparedCacheInputs) -> None:
        identity = id(bundle)
        if identity in self._finalized_bundle_ids:
            return
        self._finalized_bundle_ids.add(identity)
        finalizer = self._finalizer or bundle._finalizer
        if finalizer is None:
            return
        result = finalizer(bundle)
        if inspect.isawaitable(result):
            await result


async def execute_cache_path_plan(
    *,
    plan: EvaluationPlan,
    manifest: Mapping[str, Any],
    run_specs: Mapping[str, CachePathPlanRunSpec],
    execution_adapter: CacheDecisionExecutionAdapter,
    shared_input_preparer: SharedInputPreparer,
    prepared_profile_factory: PreparedProfileFactory,
    registry_root: Path | None = None,
    store: CachePathRunStore | None = None,
    plan_store: Any | None = None,
    finalizer: Callable[[PreparedCacheInputs], Awaitable[None] | None] | None = None,
) -> EvaluationPlanOutput:
    if store is None:
        if registry_root is None:
            raise ValueError("registry_root or an existing CachePathRunStore is required")
        store = CachePathRunStore(registry_root)
    effective_root = registry_root or store._registry_root
    if plan_store is None:
        plan_store = CachePathPlanStore(effective_root)
    executor = CachePathEvaluationPlanExecutor(
        manifest=manifest,
        run_specs=run_specs,
        store=store,
        execution_adapter=execution_adapter,
        shared_input_preparer=shared_input_preparer,
        prepared_profile_factory=prepared_profile_factory,
        finalizer=finalizer,
    )
    return await run_evaluation_plan(plan=plan, executor=executor, store=plan_store)


def compare_cache_path_plan(*, registry_root: Path, plan_id: str) -> dict[str, Any]:
    """Load a completed plan's hashed child summaries for explicit review.

    This is a read-only projection of existing counters. It does not score a
    winner or claim a statistical comparison.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", plan_id):
        raise ValueError("plan_id is not a safe identifier")
    root = Path(registry_root).resolve(strict=True)
    plan_path = root / "evals" / "cache-path" / "plans" / f"{plan_id}.json"
    record = json.loads(plan_path.read_text(encoding="utf-8"))
    if record.get("group") != "cache-path" or record.get("planId") != plan_id:
        raise ValueError("plan record identity does not match the requested cache-path plan")
    input_hash = record.get("inputSha256")
    if not isinstance(input_hash, str) or len(input_hash) != 64:
        raise ValueError("plan record has no fixed prepared-input hash")
    variants = record.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("plan record has no completed variants")
    fingerprint_input = {
        "planId": record.get("planId"),
        "group": record.get("group"),
        "selectionSha256": record.get("selectionSha256"),
        "variants": [
            {
                "variantId": item.get("variantId"),
                "runId": item.get("runId"),
                "configuration": item.get("configuration"),
                "configSha256": item.get("configSha256"),
            }
            for item in variants
            if isinstance(item, Mapping)
        ],
    }
    if len(fingerprint_input["variants"]) != len(variants):
        raise ValueError("plan contains a malformed variant reference")
    for item in variants:
        configuration = item.get("configuration")
        config_digest = hashlib.sha256(
            json.dumps(configuration, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        if config_digest != item.get("configSha256"):
            raise ValueError("variant configuration digest does not match its stored configuration")
    fingerprint = hashlib.sha256(
        (json.dumps(fingerprint_input, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()
    if record.get("planSha256") != fingerprint:
        raise ValueError("plan fingerprint does not match its variant configuration")
    state_path = root / "evals" / "cache-path" / "plans" / "state" / f"{plan_id}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state.get("planId") != plan_id
        or state.get("group") != "cache-path"
        or state.get("status") != "SUCCEEDED"
        or state.get("inputSha256") != input_hash
        or state.get("selectionSha256") != record.get("selectionSha256")
        or state.get("planSha256") != fingerprint
        or state.get("variantIds") != [item.get("variantId") for item in variants]
        or state.get("completedVariants")
        != [
            {
                "variantId": item.get("variantId"),
                "runId": item.get("runId"),
                "resultPath": item.get("resultPath"),
                "resultSha256": item.get("resultSha256"),
            }
            for item in variants
        ]
    ):
        raise ValueError("plan state does not confirm the completed fixed-input plan")
    projections = []
    for item in variants:
        if not isinstance(item, Mapping) or item.get("status") != "SUCCEEDED":
            raise ValueError("plan contains an incomplete variant")
        relative = item.get("resultPath")
        parsed = PurePosixPath(relative) if isinstance(relative, str) else None
        if parsed is None or parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError("variant result path is unsafe")
        expected_path = f"evals/cache-path/{item.get('runId')}.json"
        if parsed.as_posix() != expected_path:
            raise ValueError("variant result path does not match its cache-path run ID")
        child_path = root.joinpath(*parsed.parts).resolve(strict=True)
        child_path.relative_to(root)
        content = child_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != item.get("resultSha256"):
            raise ValueError("child run result hash does not match plan reference")
        child = json.loads(content.decode("utf-8"))
        if child.get("group") != "cache-path" or child.get("runId") != item.get("runId"):
            raise ValueError("child run result identity does not match plan reference")
        projections.append(
            {
                "variantId": item.get("variantId"),
                "runId": item.get("runId"),
                "configSha256": item.get("configSha256"),
                "resultSha256": digest,
                "summary": child.get("summary"),
            }
        )
    return {"planId": plan_id, "inputSha256": input_hash, "variants": projections}
