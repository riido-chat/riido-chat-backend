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
        if manifest.get("group") != "cache-path":
            raise CachePathStoreError("cache-path manifest가 아닙니다.")
        if manifest.get("goldenVersion") != metadata.golden_version:
            raise CachePathStoreError("manifest와 실행 메타데이터의 골든셋 버전이 다릅니다.")
        if not _SHA64.fullmatch(observation_sha256):
            raise CachePathStoreError("관측 결과 해시는 64자리 SHA-256이어야 합니다.")
        turns = manifest.get("turns")
        if not isinstance(turns, list):
            raise CachePathStoreError("manifest에 문항 턴 목록이 없습니다.")
        manifest_inputs = manifest.get("inputs")
        if not isinstance(manifest_inputs, dict) or set(manifest_inputs) != {
            "draftsSha256",
            "judgeLabelsSha256",
        }:
            raise CachePathStoreError("manifest 입력 해시가 없거나 형식이 잘못되었습니다.")
        if any(
            value is not None
            and (not isinstance(value, str) or not _SHA64.fullmatch(value))
            for value in manifest_inputs.values()
        ):
            raise CachePathStoreError("manifest 입력 해시 형식이 잘못되었습니다.")

        expected = {}
        for turn in turns:
            if not isinstance(turn, dict):
                raise CachePathStoreError("manifest 턴 형식이 잘못되었습니다.")
            key = (turn.get("itemId"), turn.get("turnNo"))
            if set(turn) - _MANIFEST_TURN_FIELDS:
                raise CachePathStoreError("결과 manifest에 텍스트 등 허용되지 않은 필드가 있습니다.")
            if (
                not isinstance(key[0], str)
                or not key[0]
                or not isinstance(key[1], int)
                or isinstance(key[1], bool)
                or key[1] < 1
                or turn.get("path") not in {"P1", "P3", "P4", "P6"}
                or turn.get("kind") not in {"exact_hit", "judge", "context_only"}
            ):
                raise CachePathStoreError("manifest 턴의 식별자나 분류가 잘못되었습니다.")
            if not isinstance(turn.get("expectedExactHit"), bool):
                raise CachePathStoreError("manifest 턴의 정확 일치 기대값이 잘못되었습니다.")
            if "expectedServed" in turn and not isinstance(turn["expectedServed"], bool):
                raise CachePathStoreError("manifest 턴의 서빙 기대값이 잘못되었습니다.")
            if turn.get("kind") == "judge":
                if turn.get("expectedJudgeDecision") not in {
                    "CONNECT",
                    "SEPARATE",
                    "UNCLASSIFIED",
                }:
                    raise CachePathStoreError("manifest 판별 기대값이 잘못되었습니다.")
                keys = turn.get("expectedSubproblemKeysAny")
                if not isinstance(keys, list) or any(
                    not isinstance(value, str) for value in keys
                ):
                    raise CachePathStoreError("manifest 세부 문제 key가 잘못되었습니다.")
            if key in expected:
                raise CachePathStoreError("manifest에 중복 문항 턴이 있습니다.")
            expected[key] = turn

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
                }
            )
        if actual != set(expected):
            raise CachePathStoreError("manifest의 모든 턴 관측이 필요합니다.")
        _validate_summary(summary)

        record = {
            "schemaVersion": 1,
            "group": "cache-path",
            "runId": metadata.run_id,
            "goldenVersion": metadata.golden_version,
            "goldenCommit": metadata.golden_commit,
            "sourceCorpusVersion": metadata.source_corpus_version,
            "inputs": {
                "draftsSha256": manifest_inputs["draftsSha256"],
                "judgeLabelsSha256": manifest_inputs["judgeLabelsSha256"],
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
            for value in section.values():
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value < 0
                    or (
                        section is metric["nextStage"]
                        and "passRate" in section
                        and value > 1
                    )
                ):
                    raise CachePathStoreError("집계 결과 수치가 잘못되었습니다.")
        for field_name in ("turns", "ineligibleServed", "modelCalls", "embeddingCalls"):
            value = metric[field_name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CachePathStoreError("집계 결과 건수가 잘못되었습니다.")
