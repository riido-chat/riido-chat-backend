"""턴 실행의 진행 단계와 전달 계층 hook 타입을 정의한다.

진행 단계는 화면 표시용 값이며 `rag_runs.status`의 답변 결과 상태와 별개다.
DB에 매핑하지 않고 영속하지도 않는다.

전달 계층(SSE 등)을 걷어내도 코어가 그대로 남도록 이 모듈은 app 하위 어떤
모듈도 import하지 않는다.
"""

import uuid
from collections.abc import Awaitable, Callable
from enum import Enum


class ProgressStage(str, Enum):
    """턴 실행이 실제로 진입한 화면 진행 단계."""

    RETRIEVING = "RETRIEVING"
    GENERATING = "GENERATING"
    VALIDATING = "VALIDATING"


OnTurnStartedHook = Callable[[uuid.UUID, uuid.UUID], Awaitable[None]]
"""턴 생성이 확정된 직후 conversation_id와 rag_run_id를 전달하는 hook."""

OnProgressStageHook = Callable[[ProgressStage], Awaitable[None]]
"""진행 단계에 실제로 진입한 시점에 그 단계를 전달하는 hook."""
