"""응답 수명보다 오래 실행되는 애플리케이션 task를 관리한다."""

import asyncio
from typing import Any


def register_pipeline_task(app: Any, task: "asyncio.Task") -> None:
    """shutdown drain 대상으로 등록하고 완료 시 자동으로 제거한다."""

    registry = getattr(app.state, "pipeline_tasks", None)
    if registry is None:
        # lifespan을 대체한 테스트 등 registry가 없는 환경을 방어한다.
        registry = set()
        app.state.pipeline_tasks = registry
    registry.add(task)
    task.add_done_callback(registry.discard)
