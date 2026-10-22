"""Explicit service-profile dispatch for cache-path evaluation.

Service adapters register factories at application startup. This module owns
selection only; it does not import service infrastructure or construct clients.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol

from evaluation.cache_path_eval.store import CachePathRunMetadata


class CachePathProfileFactory(Protocol):
    def __call__(
        self,
        *,
        config: Mapping[str, Any],
        metadata: CachePathRunMetadata,
        manifest: Mapping[str, Any],
    ) -> Any: ...


_FACTORIES: dict[str, CachePathProfileFactory] = {}


def register_profile(service_key: str, factory: CachePathProfileFactory) -> None:
    """Register an explicitly configured adapter factory for one service key."""
    if not isinstance(service_key, str) or not service_key.strip():
        raise ValueError("service_key는 비어 있을 수 없습니다.")
    if not callable(factory):
        raise TypeError("profile factory는 호출 가능해야 합니다.")
    if service_key in _FACTORIES:
        raise ValueError(f"이미 등록된 cache-path 서비스 프로필입니다: {service_key}")
    _FACTORIES[service_key] = factory


def resolve_profile_factory(service_key: str) -> CachePathProfileFactory:
    try:
        return _FACTORIES[service_key]
    except KeyError as error:
        raise ValueError(f"등록되지 않은 cache-path 서비스 프로필입니다: {service_key}") from error


def clear_profiles_for_tests() -> None:
    """Clear process-local registrations; intended for isolated test setup."""
    _FACTORIES.clear()
