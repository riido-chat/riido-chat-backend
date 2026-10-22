"""Create-only, text-free persistence for evaluation plans."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from evaluation.cache_path_eval.store import CachePathRunStore

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_SAFE_ERROR = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,99}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_SAFE_CONFIG_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_CREDENTIAL_KEYS = {"apikey", "authorization", "token", "secret", "password"}


class EvaluationPlanStoreError(ValueError):
    """A plan cannot be safely or consistently persisted."""


class CachePathPlanStore:
    """Persist plan lifecycle state and a final index under a private registry."""

    def __init__(self, registry_root: Path) -> None:
        # Reuse the existing guard; keep the resolved root for artifact checks.
        guarded_store = CachePathRunStore(registry_root)
        self._root = guarded_store._registry_root

    def reserve(self, plan: Any) -> None:
        group, plan_id, selection_hash, variants = self._plan_values(plan)
        plan_hash = self._plan_fingerprint(group, plan_id, selection_hash, variants)
        if (self._plan_dir(group) / f"{plan_id}.json").exists():
            raise EvaluationPlanStoreError("plan_id already has a final record")
        directory = self._plan_dir(group) / "state"
        directory.mkdir(parents=True, exist_ok=True)
        state = {
            "schemaVersion": 1,
            "planId": plan_id,
            "group": group,
            "selectionSha256": selection_hash,
            "planSha256": plan_hash,
            "status": "RUNNING",
            "variantIds": [self._variant_values(item)[0] for item in variants],
            "completedVariants": [],
        }
        destination = self._state_path(group, plan_id)
        encoded = self._render(state)
        try:
            descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise EvaluationPlanStoreError("plan_id is already reserved") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    def prepared(self, plan: Any, prepared: Any) -> None:
        group, plan_id, _, _ = self._plan_values(plan)
        input_hash = getattr(prepared, "input_sha256", None)
        self._require_hash(input_hash, "input_sha256")
        state = self._read_running_state(group, plan_id, plan)
        existing = state.get("inputSha256")
        if existing is not None and existing != input_hash:
            raise EvaluationPlanStoreError("prepared input hash cannot be changed")
        state["inputSha256"] = input_hash
        preparation_evidence = self._preparation_evidence(prepared)
        self._validate_safe_json(preparation_evidence)
        state["preparationEvidence"] = self._plain_json(preparation_evidence)
        self._write_atomic(self._state_path(group, plan_id), state)

    def checkpoint(self, plan: Any, result: Any) -> None:
        group, plan_id, _, variants = self._plan_values(plan)
        variant_id, run_id, _, _ = self._result_values(result)
        self._require_plan_variant(variants, variant_id, run_id)
        self._validate_result_file(group, run_id, result)
        state = self._read_running_state(group, plan_id, plan)
        if not state.get("inputSha256"):
            raise EvaluationPlanStoreError("prepared input hash is required before checkpointing")
        completed = state["completedVariants"]
        if any(item.get("variantId") == variant_id for item in completed):
            raise EvaluationPlanStoreError("variant checkpoint already exists")
        completed.append(self._result_reference(result))
        self._write_atomic(self._state_path(group, plan_id), state)

    def succeed(self, plan: Any, prepared: Any, results: Sequence[Any]) -> str:
        group, plan_id, selection_hash, variants = self._plan_values(plan)
        plan_hash = self._plan_fingerprint(group, plan_id, selection_hash, variants)
        input_hash = getattr(prepared, "input_sha256", None)
        self._require_hash(input_hash, "input_sha256")
        state = self._read_running_state(group, plan_id, plan)
        if state.get("inputSha256") != input_hash:
            raise EvaluationPlanStoreError("prepared input hash does not match plan state")
        preparation_evidence = self._preparation_evidence(prepared)
        self._validate_safe_json(preparation_evidence)
        if state.get("preparationEvidence", {}) != self._plain_json(preparation_evidence):
            raise EvaluationPlanStoreError("preparation evidence does not match plan state")
        if len(results) != len(variants):
            raise EvaluationPlanStoreError("every plan variant must have one result")

        expected_refs = [self._result_reference(result) for result in results]
        if state.get("completedVariants") != expected_refs:
            raise EvaluationPlanStoreError("checkpointed results must exactly match final result order")

        variant_rows = []
        for variant, result in zip(variants, results):
            variant_id, run_id, configuration, config_hash = self._variant_values(variant)
            result_id, result_run, result_path, result_hash = self._result_values(result)
            if (result_id, result_run) != (variant_id, run_id):
                raise EvaluationPlanStoreError("result order or identity does not match plan")
            self._require_plan_variant((variant,), result_id, result_run)
            self._validate_result_file(group, run_id, result)
            variant_rows.append(
                {
                    "variantId": variant_id,
                    "runId": run_id,
                    "configuration": self._plain_json(configuration),
                    "configSha256": config_hash,
                    "resultPath": result_path,
                    "resultSha256": result_hash,
                    "status": "SUCCEEDED",
                }
            )

        record = {
            "schemaVersion": 1,
            "planId": plan_id,
            "group": group,
            "selectionSha256": selection_hash,
            "planSha256": plan_hash,
            "inputSha256": input_hash,
            "preparationEvidence": self._plain_json(preparation_evidence),
            "variants": variant_rows,
        }
        output = self._plan_dir(group) / f"{plan_id}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        self._create_only(output, self._render(record))
        try:
            state["status"] = "SUCCEEDED"
            self._write_atomic(self._state_path(group, plan_id), state)
        except Exception:
            # The output is already create-only and must remain visible as an
            # incomplete finalization if its state update could not be saved.
            try:
                state.update(
                    status="FINALIZATION_INCOMPLETE",
                    failureStage="succeed",
                    errorType="FinalizationError",
                )
                self._write_atomic(self._state_path(group, plan_id), state)
            except Exception:
                pass
            raise
        return str(output.resolve())

    def fail(self, plan: Any, stage: str, variant_id: str | None, error_type: str) -> None:
        group, plan_id, _, variants = self._plan_values(plan)
        if not _SAFE_ERROR.fullmatch(stage or ""):
            stage = "unknown"
        if not _SAFE_ERROR.fullmatch(error_type or ""):
            error_type = "Exception"
        if variant_id is not None:
            if not _SAFE_ID.fullmatch(variant_id) or not any(
                self._variant_values(item)[0] == variant_id for item in variants
            ):
                variant_id = None
        state = self._read_state(group, plan_id, plan)
        if state.get("status") == "FINALIZATION_INCOMPLETE":
            return
        if state.get("status") != "RUNNING":
            raise EvaluationPlanStoreError("plan state is not running")
        output = self._plan_dir(group) / f"{plan_id}.json"
        status = "FINALIZATION_INCOMPLETE" if output.exists() else "FAILED"
        state.update(status=status, failureStage=stage, errorType=error_type)
        if variant_id is not None:
            state["failedVariantId"] = variant_id
        self._write_atomic(self._state_path(group, plan_id), state)

    def _plan_values(self, plan: Any) -> tuple[str, str, str, tuple[Any, ...]]:
        group = getattr(plan, "group", None)
        plan_id = getattr(plan, "plan_id", None)
        selection_hash = getattr(plan, "selection_sha256", None)
        variants = getattr(plan, "variants", None)
        if not isinstance(group, str) or not _SAFE_ID.fullmatch(group):
            raise EvaluationPlanStoreError("group is not a safe path component")
        if not isinstance(plan_id, str) or not _SAFE_ID.fullmatch(plan_id):
            raise EvaluationPlanStoreError("plan_id is invalid")
        self._require_hash(selection_hash, "selection_sha256")
        if not isinstance(variants, (tuple, list)) or not variants:
            raise EvaluationPlanStoreError("plan variants are required")
        seen_ids: set[str] = set()
        seen_runs: set[str] = set()
        for variant in variants:
            variant_id, run_id, configuration, config_hash = self._variant_values(variant)
            if variant_id in seen_ids or run_id in seen_runs:
                raise EvaluationPlanStoreError("plan variant identities must be unique")
            seen_ids.add(variant_id)
            seen_runs.add(run_id)
            self._validate_safe_json(configuration)
            self._require_hash(config_hash, "config_sha256")
        return group, plan_id, selection_hash, tuple(variants)

    def _variant_values(self, variant: Any) -> tuple[str, str, Any, str]:
        variant_id = getattr(variant, "variant_id", None)
        run_id = getattr(variant, "run_id", None)
        configuration = getattr(variant, "configuration", None)
        config_hash = getattr(variant, "config_sha256", None)
        if not isinstance(variant_id, str) or not _SAFE_ID.fullmatch(variant_id):
            raise EvaluationPlanStoreError("variant_id is invalid")
        if not isinstance(run_id, str) or not _SAFE_ID.fullmatch(run_id):
            raise EvaluationPlanStoreError("run_id is invalid")
        return variant_id, run_id, configuration, config_hash

    def _result_values(self, result: Any) -> tuple[str, str, str, str]:
        variant_id = getattr(result, "variant_id", None)
        run_id = getattr(result, "run_id", None)
        result_path = getattr(result, "result_path", None)
        result_hash = getattr(result, "result_sha256", None)
        if not isinstance(variant_id, str) or not _SAFE_ID.fullmatch(variant_id):
            raise EvaluationPlanStoreError("result variant_id is invalid")
        if not isinstance(run_id, str) or not _SAFE_ID.fullmatch(run_id):
            raise EvaluationPlanStoreError("result run_id is invalid")
        if not isinstance(result_path, str):
            raise EvaluationPlanStoreError("result_path is invalid")
        self._require_hash(result_hash, "result_sha256")
        return variant_id, run_id, result_path, result_hash

    def _result_reference(self, result: Any) -> dict[str, str]:
        variant_id, run_id, result_path, result_hash = self._result_values(result)
        return {
            "variantId": variant_id,
            "runId": run_id,
            "resultPath": result_path,
            "resultSha256": result_hash,
        }

    @staticmethod
    def _preparation_evidence(prepared: Any) -> Any:
        exporter = getattr(prepared, "preparation_evidence_dict", None)
        return exporter() if callable(exporter) else {}

    def _plan_fingerprint(
        self, group: str, plan_id: str, selection_hash: str, variants: Sequence[Any]
    ) -> str:
        canonical_plan = {
            "planId": plan_id,
            "group": group,
            "selectionSha256": selection_hash,
            "variants": [
                {
                    "variantId": variant_id,
                    "runId": run_id,
                    "configuration": self._plain_json(configuration),
                    "configSha256": config_hash,
                }
                for variant in variants
                for variant_id, run_id, configuration, config_hash in [self._variant_values(variant)]
            ],
        }
        self._validate_safe_json(canonical_plan)
        return hashlib.sha256(self._render(canonical_plan).encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_safe_json(value: Any) -> None:
        def check(item: Any) -> None:
            if isinstance(item, str) and not _SAFE_CONFIG_TEXT.fullmatch(item):
                raise EvaluationPlanStoreError("configuration string is not a safe label")
            if isinstance(item, Mapping):
                for key, nested in item.items():
                    if not isinstance(key, str) or not _SAFE_ID.fullmatch(key):
                        raise EvaluationPlanStoreError("configuration key is unsafe")
                    if re.sub(r"[^a-z0-9]", "", key.lower()) in _CREDENTIAL_KEYS:
                        raise EvaluationPlanStoreError("configuration contains a credential field")
                    check(nested)
            elif isinstance(item, (list, tuple)):
                for nested in item:
                    check(nested)
            elif isinstance(item, float) and not math.isfinite(item):
                raise EvaluationPlanStoreError("configuration numbers must be finite")
            elif item is not None and type(item) not in (str, bool, int, float):
                raise EvaluationPlanStoreError("configuration contains a non-JSON value")
        check(value)

    @classmethod
    def _plain_json(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: cls._plain_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._plain_json(item) for item in value]
        return value

    def _require_plan_variant(self, variants: Sequence[Any], variant_id: str, run_id: str) -> None:
        if not any(self._variant_values(item)[:2] == (variant_id, run_id) for item in variants):
            raise EvaluationPlanStoreError("result does not belong to this plan")

    def _validate_result_file(self, group: str, run_id: str, result: Any) -> None:
        _, result_run, result_path, result_hash = self._result_values(result)
        if result_run != run_id:
            raise EvaluationPlanStoreError("result run_id does not match plan variant")
        parsed = PurePosixPath(result_path)
        expected = f"evals/{group}/{run_id}.json"
        if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != expected:
            raise EvaluationPlanStoreError("result_path must reference this group's run result")
        artifact = self._root.joinpath(*parsed.parts)
        try:
            resolved = artifact.resolve(strict=True)
            resolved.relative_to(self._root)
            content = resolved.read_bytes()
        except (OSError, ValueError) as error:
            raise EvaluationPlanStoreError("child result is missing or outside the registry") from error
        if hashlib.sha256(content).hexdigest() != result_hash:
            raise EvaluationPlanStoreError("child result hash does not match result reference")
        try:
            child = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EvaluationPlanStoreError("child result is not valid JSON") from error
        if not isinstance(child, dict) or child.get("group") != group or child.get("runId") != run_id:
            raise EvaluationPlanStoreError("child result identity does not match reference")

    def _plan_dir(self, group: str) -> Path:
        return self._root / "evals" / group / "plans"

    def _state_path(self, group: str, plan_id: str) -> Path:
        return self._plan_dir(group) / "state" / f"{plan_id}.json"

    def _read_state(self, group: str, plan_id: str, plan: Any) -> dict[str, Any]:
        try:
            value = json.loads(self._state_path(group, plan_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvaluationPlanStoreError("plan state is unavailable") from error
        _, _, selection_hash, variants = self._plan_values(plan)
        expected_fingerprint = self._plan_fingerprint(group, plan_id, selection_hash, variants)
        expected_variant_ids = [self._variant_values(variant)[0] for variant in variants]
        if (
            not isinstance(value, dict)
            or value.get("planId") != plan_id
            or value.get("group") != group
            or value.get("selectionSha256") != selection_hash
            or value.get("planSha256") != expected_fingerprint
            or value.get("variantIds") != expected_variant_ids
            or not isinstance(value.get("completedVariants"), list)
        ):
            raise EvaluationPlanStoreError("plan state does not match the reserved plan")
        return value

    def _read_running_state(self, group: str, plan_id: str, plan: Any) -> dict[str, Any]:
        value = self._read_state(group, plan_id, plan)
        if value.get("status") != "RUNNING":
            raise EvaluationPlanStoreError("plan state is not running")
        return value

    @staticmethod
    def _require_hash(value: Any, label: str) -> None:
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise EvaluationPlanStoreError(f"{label} must be a SHA-256 hash")

    @staticmethod
    def _render(value: Mapping[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"

    @classmethod
    def _write_atomic(cls, destination: Path, value: Mapping[str, Any]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_fd, temp_name = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".tmp", dir=destination.parent)
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as stream:
                stream.write(cls._render(value))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, destination)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    @classmethod
    def _create_only(cls, destination: Path, rendered: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".tmp", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(rendered)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
        except FileExistsError as error:
            raise EvaluationPlanStoreError("plan record already exists") from error
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


EvaluationPlanStore = CachePathPlanStore
PlanStore = CachePathPlanStore
