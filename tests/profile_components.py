"""테스트가 프로필 버전 구성 행을 만들 때 쓰는 도우미.

턴은 프로필 버전의 구성 행대로 단계별 모델과 프롬프트를 고르고, 행이 없거나 등록부와 맞지 않으면
답하지 않는다. 서빙되는 프로필 버전이 필요한 테스트는 지금 코드의 기본 구성(code_components)으로
행을 만든다. 모델 이름은 등록부와 무관해 테스트가 바꿀 수 있다.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from app.database.models import (
    ProfileRevisionComponent,
    ProfileRevisionComponentRecordedBy,
    ProfileRevisionComponentStage,
)
from app.profiles.stage_settings import GENERATION_SUB_STAGES, code_components


def _component_values(
    *,
    generation_model: Optional[str] = None,
    rewrite_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    values = []
    for spec in code_components():
        model = spec.model_name
        if generation_model is not None and (
            spec.stage == ProfileRevisionComponentStage.GENERATION
            or spec.stage in GENERATION_SUB_STAGES
        ):
            model = generation_model
        if rewrite_model is not None and spec.stage == ProfileRevisionComponentStage.REWRITE:
            model = rewrite_model
        values.append(
            dict(
                stage=spec.stage,
                model_name=model,
                prompt_key=spec.prompt_key,
                prompt_version=spec.prompt_version,
                prompt_sha256=spec.prompt_sha256,
                params=spec.params,
            )
        )
    return values


def component_rows(**models: Optional[str]) -> List[SimpleNamespace]:
    """DB 없이 쓰는 구성 행. stage 는 DB 에서 읽은 값처럼 문자열이다."""

    return [
        SimpleNamespace(**{**value, "stage": value["stage"].value})
        for value in _component_values(**models)
    ]


def component_models(revision_id: int, **models: Optional[str]) -> List[ProfileRevisionComponent]:
    """revision_id 프로필 버전의 ORM 구성 행. 같은 트랜잭션에서 session.add 한다."""

    return [
        ProfileRevisionComponent(
            revision_id=revision_id,
            recorded_by=ProfileRevisionComponentRecordedBy.PUBLISH,
            **value,
        )
        for value in _component_values(**models)
    ]
