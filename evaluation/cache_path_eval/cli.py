"""Local cache-path run CLI orchestration.

All service behavior is injected through a registered profile factory and an
execution adapter. This module does not create database, model, or embedding
clients.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from evaluation.cache_path_eval.execution import CacheDecisionExecutionAdapter
from evaluation.cache_path_eval.profile_registry import resolve_profile_factory
from evaluation.cache_path_eval.run import execute_cache_path_run
from evaluation.cache_path_eval.store import CachePathRunMetadata, CachePathRunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a selected frozen cache-path manifest using an explicitly registered service profile."
    )
    parser.add_argument("--registry-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path, help="frozen text-free cache-path manifest")
    parser.add_argument("--config", required=True, type=Path, help="explicit run metadata and profile config JSON")
    return parser


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label}을 읽을 수 없습니다: {error}") from error


def load_run_config(path: Path) -> tuple[CachePathRunMetadata, Mapping[str, Any]]:
    raw = _load_json(path, "실행 설정")
    if not isinstance(raw, dict) or set(raw) != {"metadata", "profileConfig"}:
        raise ValueError("실행 설정은 metadata와 profileConfig만 포함해야 합니다.")
    metadata_values = raw["metadata"]
    if not isinstance(metadata_values, dict):
        raise ValueError("metadata는 JSON 객체여야 합니다.")
    required = {field.name for field in fields(CachePathRunMetadata)}
    if set(metadata_values) != required:
        missing = sorted(required - set(metadata_values))
        extra = sorted(set(metadata_values) - required)
        raise ValueError(f"metadata 필드가 맞지 않습니다 (누락={missing}, 허용되지 않음={extra}).")
    profile_config = raw["profileConfig"]
    if not isinstance(profile_config, dict):
        raise ValueError("profileConfig는 JSON 객체여야 합니다.")
    return CachePathRunMetadata(**metadata_values), profile_config


async def run_cache_path_cli(
    *,
    registry_root: Path,
    manifest_path: Path,
    config_path: Path,
    execution_adapter: CacheDecisionExecutionAdapter,
    profile_factory_resolver: Callable[[str], Callable[..., Any]] = resolve_profile_factory,
    store_factory: Callable[[Path], Any] = CachePathRunStore,
) -> Any:
    metadata, profile_config = load_run_config(config_path)
    manifest = _load_json(manifest_path, "frozen manifest")
    if not isinstance(manifest, dict) or manifest.get("group") != "cache-path":
        raise ValueError("선택 manifest가 cache-path 형식이 아닙니다.")
    if manifest.get("goldenVersion") != metadata.golden_version:
        raise ValueError("manifest와 실행 metadata의 goldenVersion이 다릅니다.")
    # Read-only precheck avoids constructing a profile for known duplicate IDs.
    # The factory must remain local and side-effect-free; atomic reservation occurs
    # in the shared lifecycle before any profile callback can run.
    store = store_factory(registry_root)
    store.validate_manifest(metadata, manifest)
    store.check_available(metadata)
    factory = profile_factory_resolver(metadata.service_key)
    profile = factory(config=profile_config, metadata=metadata, manifest=deepcopy(manifest))
    if asyncio.iscoroutine(profile):
        profile = await profile
    return await execute_cache_path_run(
        manifest=manifest,
        metadata=metadata,
        profile=profile,
        store=store,
        execution_adapter=execution_adapter,
    )


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    execution_adapter: Optional[CacheDecisionExecutionAdapter] = None,
) -> int:
    args = build_parser().parse_args(argv)
    if execution_adapter is None:
        print(
            "거부: cache-path 서비스가 제공하는 실행 adapter를 주입해야 합니다.",
            file=sys.stderr,
        )
        return 2
    try:
        output = asyncio.run(
            run_cache_path_cli(
                registry_root=args.registry_root,
                manifest_path=args.manifest,
                config_path=args.config,
                execution_adapter=execution_adapter,
            )
        )
        print(json.dumps({"resultPath": output.result_path, "summary": output.summary}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"거부: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
