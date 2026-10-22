"""Create-only writer for text-free cache-path runs in the private eval registry."""

from __future__ import annotations

import json
import os
import re
import tempfile
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.answering.trace import AnswerExecutionTrace
from evaluation.cache_path_eval.execution import CacheDecisionObservation
from evaluation.cache_path_eval.evidence import CacheRuntimeEvidence
from evaluation.core.run import EvaluationFailure

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_MANIFEST_TURN_FIELDS = {
    "itemId",
    "path",
    "turnNo",
    "kind",
    "expectedExactHit",
    "expectedServed",
    "expectedJudgeDecision",
    "expectedSubproblemKeysAny",
    "expectedSelectedTurnNos",
}


class CachePathStoreError(ValueError):
    """The run cannot be safely or consistently written to the eval registry."""


@dataclass(frozen=True)
class CachePathRunMetadata:
    run_id: str
    golden_version: str
    golden_commit: str
    source_corpus_version: str | None
    app_build_commit: str
    service_key: str
    profile_revision_id: int
    document_group_id: int
    index_version_id: int
    embedding_config_id: int
    cache_policy_version: str
    embedding_model_version: str | None
    judge_model_version: str | None
    judge_prompt_version: str | None
    executed_at: str

    def __post_init__(self) -> None:
        if not _SAFE_RUN_ID.fullmatch(self.run_id):
            raise CachePathStoreError("run_id 형식이 잘못되었습니다.")
        for field_name in (
            "golden_version",
            "service_key",
            "cache_policy_version",
            "executed_at",
        ):
            if not getattr(self, field_name):
                raise CachePathStoreError(f"{field_name}은 비어 있을 수 없습니다.")
        for field_name in ("golden_commit", "app_build_commit"):
            if not _SHA40.fullmatch(getattr(self, field_name)):
                raise CachePathStoreError(f"{field_name}은 40자리 커밋 해시여야 합니다.")
        for field_name in (
            "profile_revision_id",
            "document_group_id",
            "index_version_id",
            "embedding_config_id",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CachePathStoreError(f"{field_name}은 1 이상의 정수여야 합니다.")


class CachePathRunStore:
    """Persist observations and traces under ``evals/cache-path``.

    The root must be the checked-out private ``riido-evals`` registry. The writer
    stores only allowlisted IDs, labels, counts, timings, version metadata, and
    stage events. It never writes question, answer, candidate, or document text.
    Each run is one create-only JSON file so a partial trace set cannot look like
    a complete run.
    """

    def __init__(self, registry_root: Path) -> None:
        self._registry_root = registry_root.resolve()
        if not self._is_registry_root(self._registry_root):
            raise CachePathStoreError(
                "결과 저장 위치는 riido-evals 골든 등록부 루트여야 합니다."
            )

    @staticmethod
    def _is_registry_root(path: Path) -> bool:
        readme = path / "README.md"
        git_marker = path / ".git"
        if git_marker.is_dir():
            git_dir = git_marker
        elif git_marker.is_file():
            marker = git_marker.read_text(encoding="utf-8").strip()
            if not marker.startswith("gitdir:"):
                return False
            git_dir = (path / marker.partition(":")[2].strip()).resolve()
        else:
            return False
        config_path = git_dir / "config"
        if not config_path.is_file():
            return False
        git_config = ConfigParser()
        git_config.read(config_path, encoding="utf-8")
        origin = git_config.get('remote "origin"', "url", fallback="")
        repository_name = origin.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        return (
            "riido-evals" in repository_name.lower()
            and (path / "golden" / "manifest.json").is_file()
            and readme.is_file()
            and readme.read_text(encoding="utf-8").startswith("# riido-evals")
        )

    def write_run(
        self,
        metadata: CachePathRunMetadata,
        *,
        manifest: Mapping[str, Any],
        observations: Sequence[CacheDecisionObservation],
        summary: Mapping[str, Any],
        observation_sha256: str,
    ) -> Path:
        turns = self.validate_manifest(metadata, manifest)
        if not _SHA64.fullmatch(observation_sha256):
            raise CachePathStoreError("관측 결과 해시는 64자리 SHA-256이어야 합니다.")
        manifest_inputs = manifest.get("inputs")
        expected = {(turn["itemId"], turn["turnNo"]): turn for turn in turns}

        actual = set()
        serialized_observations = []
        for observation in observations:
            key = (observation.item_id, observation.turn_no)
            if key not in expected or key in actual:
                raise CachePathStoreError("관측 문항이 없거나 중복되었습니다.")
            actual.add(key)
            trace = observation.execution_trace
            if trace is None:
                raise CachePathStoreError("재현용 실행 trace가 없는 관측은 저장하지 않습니다.")
            self._validate_trace(trace, metadata, observation)
            serialized_observations.append(
                {
                    **expected[key],
                    **observation.to_row(),
                    "trace": _trace_to_dict(trace),
                    **_evidence_field(observation.evidence),
                }
            )
        if actual != set(expected):
            raise CachePathStoreError("manifest의 모든 턴 관측이 필요합니다.")
        _validate_summary(summary)

        record = {
            "schemaVersion": 1,
            "group": "cache-path",
            **({"evaluationGroup": dict(manifest["evaluationGroup"])} if "evaluationGroup" in manifest else {}),
            "runId": metadata.run_id,
            "goldenVersion": metadata.golden_version,
            "goldenCommit": metadata.golden_commit,
            "sourceCorpusVersion": metadata.source_corpus_version,
            "inputs": {
                "draftsSha256": manifest_inputs["draftsSha256"],
                "judgeLabelsSha256": manifest_inputs["judgeLabelsSha256"],
                **{
                    key: manifest_inputs[key]
                    for key in ("canonicalItemsSha256", "sourceHtmlSha256")
                    if key in manifest_inputs
                },
                "observationsSha256": observation_sha256,
            },
            "target": {
                "appBuildCommit": metadata.app_build_commit,
                "serviceKey": metadata.service_key,
                "profileRevisionId": metadata.profile_revision_id,
                "documentGroupId": metadata.document_group_id,
                "indexVersionId": metadata.index_version_id,
                "embeddingConfigId": metadata.embedding_config_id,
            },
            "versions": {
                "cachePolicy": metadata.cache_policy_version,
                "embeddingModel": metadata.embedding_model_version,
                "judgeModel": metadata.judge_model_version,
                "judgePrompt": metadata.judge_prompt_version,
            },
            "executedAt": metadata.executed_at,
            "summary": dict(summary),
            "observations": serialized_observations,
        }
        rendered = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        destination = self._registry_root / "evals" / "cache-path" / f"{metadata.run_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._write_create_only(destination, rendered)
        return destination

    def validate_manifest(
        self, metadata: CachePathRunMetadata, manifest: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Validate all cache-path inputs before profiles can perform service work."""
        if manifest.get("group") != "cache-path":
            raise CachePathStoreError("cache-path manifest가 아닙니다.")
        if manifest.get("goldenVersion") != metadata.golden_version:
            raise CachePathStoreError("manifest와 실행 메타데이터의 골든셋 버전이 다릅니다.")
        turns = manifest.get("turns")
        if not isinstance(turns, list) or not turns:
            raise CachePathStoreError("manifest에 문항 턴 목록이 없습니다.")
        manifest_inputs = manifest.get("inputs")
        allowed_input_fields = {
            "draftsSha256", "judgeLabelsSha256", "canonicalItemsSha256", "sourceHtmlSha256"
        }
        if (
            not isinstance(manifest_inputs, dict)
            or not {"draftsSha256", "judgeLabelsSha256"}.issubset(manifest_inputs)
            or set(manifest_inputs) - allowed_input_fields
        ):
            raise CachePathStoreError("manifest 입력 해시가 없거나 형식이 잘못되었습니다.")
        if any(value is not None and (not isinstance(value, str) or not _SHA64.fullmatch(value)) for value in manifest_inputs.values()):
            raise CachePathStoreError("manifest 입력 해시 형식이 잘못되었습니다.")
        group_provenance = manifest.get("evaluationGroup")
        if group_provenance is not None:
            if (
                not isinstance(group_provenance, Mapping)
                or set(group_provenance) != {"groupId", "groupVersion", "groupSha256", "groupFileSha256"}
                or group_provenance.get("groupId") != "cache-path"
                or not isinstance(group_provenance.get("groupVersion"), str)
                or not _SHA64.fullmatch(str(group_provenance.get("groupSha256", "")))
                or not _SHA64.fullmatch(str(group_provenance.get("groupFileSha256", "")))
                or not _SHA64.fullmatch(str(manifest_inputs.get("canonicalItemsSha256", "")))
                or not _SHA64.fullmatch(str(manifest_inputs.get("sourceHtmlSha256", "")))
            ):
                raise CachePathStoreError("정본 평가 그룹 provenance가 잘못되었습니다.")
        expected: set[tuple[str, int]] = set()
        for turn in turns:
            if not isinstance(turn, dict):
                raise CachePathStoreError("manifest 턴 형식이 잘못되었습니다.")
            if set(turn) - _MANIFEST_TURN_FIELDS:
                raise CachePathStoreError("결과 manifest에 텍스트 등 허용되지 않은 필드가 있습니다.")
            item_id, turn_no = turn.get("itemId"), turn.get("turnNo")
            if (
                not isinstance(item_id, str) or not item_id
                or not isinstance(turn_no, int) or isinstance(turn_no, bool) or turn_no < 1
                or turn.get("path") not in {"P1", "P3", "P4", "P6"}
                or turn.get("kind") not in {"exact_hit", "judge", "context_only"}
                or not isinstance(turn.get("expectedExactHit"), bool)
            ):
                raise CachePathStoreError("manifest 턴의 식별자나 분류가 잘못되었습니다.")
            if "expectedServed" in turn and not isinstance(turn["expectedServed"], bool):
                raise CachePathStoreError("manifest 턴의 서빙 기대값이 잘못되었습니다.")
            if turn["kind"] == "context_only" and "expectedServed" in turn:
                raise CachePathStoreError("context-only 턴에는 캐시 기대 결과를 둘 수 없습니다.")
            if turn["kind"] != "context_only" and not isinstance(turn.get("expectedServed"), bool):
                raise CachePathStoreError("캐시 기대 결과가 없습니다.")
            if turn["kind"] == "judge":
                if turn.get("expectedJudgeDecision") not in {"CONNECT", "SEPARATE", "UNCLASSIFIED"}:
                    raise CachePathStoreError("manifest 판별 기대값이 잘못되었습니다.")
                keys = turn.get("expectedSubproblemKeysAny")
                if not isinstance(keys, list) or any(not isinstance(value, str) for value in keys):
                    raise CachePathStoreError("manifest 세부 문제 key가 잘못되었습니다.")
            if "expectedSelectedTurnNos" in turn:
                selected = turn["expectedSelectedTurnNos"]
                if (
                    not isinstance(selected, list)
                    or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in selected)
                    or selected != sorted(set(selected))
                ):
                    raise CachePathStoreError("manifest 문맥 선택 기대값이 잘못되었습니다.")
            key = (item_id, turn_no)
            if key in expected:
                raise CachePathStoreError("manifest에 중복 문항 턴이 있습니다.")
            expected.add(key)
        return turns

    def reserve(self, metadata: CachePathRunMetadata) -> None:
        """Atomically reserve this run ID before any expensive service callback."""
        self.check_available(metadata)
        state_dir = self._registry_root / "evals" / "cache-path" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        destination = self._run_state_path(metadata)
        initial = {"schemaVersion": 1, "runId": metadata.run_id, "status": "RUNNING", "checkpointCount": 0}
        encoded = json.dumps(initial, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise CachePathStoreError(f"run_id가 이미 예약되었습니다: {metadata.run_id}") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    def check_available(self, metadata: CachePathRunMetadata) -> None:
        """Read-only duplicate check used before profile factory construction."""
        result = self._registry_root / "evals" / "cache-path" / f"{metadata.run_id}.json"
        if result.exists() or self._run_state_path(metadata).exists():
            raise CachePathStoreError(f"run_id가 이미 예약되었습니다: {metadata.run_id}")

    def checkpoint(self, metadata: CachePathRunMetadata, observation: CacheDecisionObservation) -> None:
        """Durably retain one successful, allowlisted observation."""
        trace = observation.execution_trace
        if trace is None:
            raise CachePathStoreError("재현용 실행 trace가 없는 관측은 checkpoint하지 않습니다.")
        self._validate_trace(trace, metadata, observation)
        directory = self._registry_root / "evals" / "cache-path" / "checkpoints" / metadata.run_id
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            **observation.to_row(),
            "trace": _trace_to_dict(trace),
            **_evidence_field(observation.evidence),
        }
        destination = directory / f"{_safe_filename(observation.item_id)}-{observation.turn_no}.json"
        rendered = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        self._write_create_only(destination, rendered)
        state = self._read_state(metadata)
        state["checkpointCount"] += 1
        state["lastCheckpoint"] = {"itemId": observation.item_id, "turnNo": observation.turn_no}
        self._write_atomic(self._run_state_path(metadata), state)

    def fail(self, metadata: CachePathRunMetadata, failure: EvaluationFailure) -> None:
        """Persist stage and exception type only; raw exception text is discarded."""
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,99}", failure.stage):
            safe_stage = "unknown"
        else:
            safe_stage = failure.stage
        error_type = failure.error_type if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,99}", failure.error_type) else "Exception"
        detail = failure.detail if isinstance(failure.detail, dict) else {}
        item_id = detail.get("itemId")
        turn_no = detail.get("turnNo")
        final_result = self._registry_root / "evals" / "cache-path" / f"{metadata.run_id}.json"
        finalization_incomplete = final_result.exists()
        run_status = "FINALIZATION_INCOMPLETE" if finalization_incomplete else "FAILED"
        record: dict[str, Any] = {
            "schemaVersion": 1,
            "group": "cache-path-failed-run",
            "runId": metadata.run_id,
            "status": run_status,
            "failureStage": safe_stage,
            "errorType": error_type,
        }
        if isinstance(item_id, str) and item_id and isinstance(turn_no, int) and not isinstance(turn_no, bool) and turn_no >= 1:
            record.update(itemId=item_id, turnNo=turn_no)
        trace = detail.get("trace")
        if isinstance(trace, AnswerExecutionTrace):
            self._validate_trace_context(trace, metadata, item_id, turn_no)
            record["trace"] = _trace_to_dict(trace)
        destination = self._registry_root / "evals" / "cache-path" / "failures" / f"{metadata.run_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._write_create_only(destination, json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        state = self._read_state(metadata)
        state.update(status=run_status, failureStage=safe_stage, errorType=error_type)
        self._write_atomic(self._run_state_path(metadata), state)

    def begin_finalization(self, metadata: CachePathRunMetadata) -> None:
        state = self._read_state(metadata)
        state["status"] = "FINALIZING"
        self._write_atomic(self._run_state_path(metadata), state)

    def succeed(self, metadata: CachePathRunMetadata) -> None:
        state = self._read_state(metadata)
        state["status"] = "SUCCEEDED"
        self._write_atomic(self._run_state_path(metadata), state)

    def _run_state_path(self, metadata: CachePathRunMetadata) -> Path:
        return self._registry_root / "evals" / "cache-path" / "state" / f"{metadata.run_id}.json"

    def _read_state(self, metadata: CachePathRunMetadata) -> dict[str, Any]:
        try:
            value = json.loads(self._run_state_path(metadata).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CachePathStoreError("run 상태를 읽을 수 없습니다.") from error
        if not isinstance(value, dict) or value.get("runId") != metadata.run_id:
            raise CachePathStoreError("run 상태가 올바르지 않습니다.")
        return value

    @staticmethod
    def _write_atomic(destination: Path, value: Mapping[str, Any]) -> None:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        temp_fd, temp_name = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".tmp", dir=destination.parent)
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as stream:
                stream.write(rendered)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, destination)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def write_failed_trace(
        self,
        metadata: CachePathRunMetadata,
        *,
        item_id: str,
        turn_no: int,
        trace: AnswerExecutionTrace,
    ) -> Path:
        """Persist a failed stage trace without pretending the run is complete."""

        self._validate_trace_context(trace, metadata, item_id, turn_no)
        failed_event = next(
            (event for event in reversed(trace.events) if event.outcome.value == "FAILED"),
            None,
        )
        if failed_event is None:
            raise CachePathStoreError("실패 trace에 실패한 단계가 없습니다.")
        record = {
            "schemaVersion": 1,
            "group": "cache-path-failed-trace",
            "runId": metadata.run_id,
            "goldenVersion": metadata.golden_version,
            "appBuildCommit": metadata.app_build_commit,
            "itemId": item_id,
            "turnNo": turn_no,
            "failureStage": failed_event.stage.value,
            "failureCode": failed_event.detail_code,
            "trace": _trace_to_dict(trace),
        }
        destination = (
            self._registry_root
            / "evals"
            / "cache-path"
            / "failures"
            / f"{metadata.run_id}-{_safe_filename(item_id)}-{turn_no}.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        self._write_create_only(destination, rendered)
        return destination

    @staticmethod
    def _write_create_only(destination: Path, rendered: str) -> None:
        temp_fd, temp_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as stream:
                stream.write(rendered)
                stream.flush()
                os.fsync(stream.fileno())
            # A hard link is atomic and fails if this run ID already exists.
            os.link(temp_name, destination)
        except FileExistsError as error:
            raise CachePathStoreError(f"기록은 덮어쓰지 않습니다: {destination.name}") from error
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    @staticmethod
    def _validate_trace(
        trace: AnswerExecutionTrace,
        metadata: CachePathRunMetadata,
        observation: CacheDecisionObservation,
    ) -> None:
        CachePathRunStore._validate_trace_context(
            trace, metadata, observation.item_id, observation.turn_no
        )

    @staticmethod
    def _validate_trace_context(
        trace: AnswerExecutionTrace,
        metadata: CachePathRunMetadata,
        item_id: str,
        turn_no: int,
    ) -> None:
        context = trace.replay_context
        if (
            context.run_id != metadata.run_id
            or context.item_id != item_id
            or context.turn_no != turn_no
            or context.golden_version != metadata.golden_version
            or context.service_version != metadata.service_key
            or context.code_revision != metadata.app_build_commit
            or context.cache_policy_version != metadata.cache_policy_version
            or context.embedding_model_version != metadata.embedding_model_version
        ):
            raise CachePathStoreError("관측과 실행 trace의 재현 메타데이터가 일치하지 않습니다.")


def _trace_to_dict(trace: AnswerExecutionTrace) -> dict[str, Any]:
    if trace.target not in {"FULL_ANSWER", "CACHE_DECISION"}:
        raise CachePathStoreError("trace 실행 대상이 잘못되었습니다.")
    for event in trace.events:
        if event.detail_code is not None and not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]{0,99}", event.detail_code
        ):
            raise CachePathStoreError("trace detailCode에 허용되지 않은 문자가 있습니다.")
    return {
        "traceId": str(trace.trace_id),
        "target": trace.target,
        "events": [
            {
                "stage": event.stage.value,
                "outcome": event.outcome.value,
                "durationMs": event.duration_ms,
                "detailCode": event.detail_code,
            }
            for event in trace.events
        ],
    }


def _evidence_field(evidence: CacheRuntimeEvidence | None) -> dict[str, Any]:
    if evidence is None:
        return {}
    if type(evidence) is not CacheRuntimeEvidence:
        raise CachePathStoreError("runtime evidence 형식이 잘못되었습니다.")
    try:
        return {"evidence": evidence.to_dict()}
    except (AttributeError, TypeError, ValueError) as error:
        raise CachePathStoreError("runtime evidence 형식이 잘못되었습니다.") from error


def _safe_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return sanitized[:80] or "item"


def _validate_summary(summary: Mapping[str, Any]) -> None:
    if set(summary) != {"all", "paths"} or not isinstance(summary.get("paths"), dict):
        raise CachePathStoreError("집계 결과 형식이 올바르지 않습니다.")
    metric_fields = {
        "turns",
        "outcomes",
        "ineligibleServed",
        "nextStage",
        "latencyMs",
        "modelCalls",
        "embeddingCalls",
    }
    nested_metrics = []
    for path_name, path_groups in summary["paths"].items():
        if path_name not in {"P1", "P3", "P4", "P6"}:
            raise CachePathStoreError("집계에 알 수 없는 캐시 경로가 있습니다.")
        if not isinstance(path_groups, dict):
            raise CachePathStoreError("경로 집계 형식이 잘못되었습니다.")
        if set(path_groups) - {"all", "exact_hit", "judge", "context_only"}:
            raise CachePathStoreError("경로 집계에 허용되지 않은 묶음이 있습니다.")
        nested_metrics.extend(path_groups.values())
    for metric in [summary["all"], *nested_metrics]:
        if not isinstance(metric, dict) or set(metric) != metric_fields:
            raise CachePathStoreError("집계 지표에 허용되지 않은 필드가 있습니다.")
        if not isinstance(metric["outcomes"], dict) or any(
            outcome not in {
                "SERVED",
                "SHADOW",
                "GROUP_DISABLED",
                "REJECTED",
                "SKIPPED",
                "FAILED",
            }
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for outcome, count in metric["outcomes"].items()
        ):
            raise CachePathStoreError("집계 결과 건수가 잘못되었습니다.")
        for section, expected in (
            (metric["nextStage"], {"passed", "failed", "notApplicable", "passRate", "denominator"}),
            (metric["latencyMs"], {"total", "mean", "median"}),
        ):
            if not isinstance(section, dict) or set(section) != expected:
                raise CachePathStoreError("집계 결과 세부 형식이 잘못되었습니다.")
        for section in (metric["nextStage"], metric["latencyMs"]):
            for field_name, value in section.items():
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value < 0
                    or (
                        section is metric["nextStage"]
                        and field_name == "passRate"
                        and value > 1
                    )
                ):
                    raise CachePathStoreError("집계 결과 수치가 잘못되었습니다.")
        for field_name in ("turns", "ineligibleServed", "modelCalls", "embeddingCalls"):
            value = metric[field_name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CachePathStoreError("집계 결과 건수가 잘못되었습니다.")
