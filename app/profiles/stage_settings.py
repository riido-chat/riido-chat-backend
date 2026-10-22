"""프로필 버전의 구성 행을 단계별 모델 호출 설정으로 바꾼다.

구성 행(profile_revision_components)이 단계마다 모델, 프롬프트 버전, 지문, 요청 설정을 정하고,
지시문 본문은 실행 중인 코드의 등록부(app/prompts)에서 그 프롬프트 버전으로 읽는다. 턴마다
선택한 프로필 버전의 값을 재작성, 판별, 생성 클라이언트에 호출마다 넘긴다.

서빙 조건(하나라도 어기면 그 프로필 버전으로 답하지 않는다):

- 일곱 단계의 구성 행이 모두 있다.
- 프롬프트 단계(rewrite, judge, source_planning, source_planning_repair, answer, answer_repair)는
  등록부에 그 버전 파일이 있고, 실제로 보낼 지시문의 sha256 이 행의 prompt_sha256 과 같다. 재시도
  단계의 지시문은 같은 프로필 버전의 처음 단계 지시문에 재시도 덧붙임을 이은 문자열이다.
- 출력 스키마는 코드가 정한다. 지문 표의 `<component>.output_schema@<버전>` 값이 지금 코드가 보내는
  스키마의 지문과 같아야 한다(재시도 단계는 처음 단계와 같은 스키마를 쓴다).
- 묶음 행(generation@vNN)은 지문 표에 있고, 행의 지문이 표 값과 같으며, 하위 네 단계의 키와 지문,
  두 출력 스키마로 다시 계산한 묶음 지문과도 같다. model_calls 의 ANSWER_GENERATION 행 하나가 생성
  논리 호출 한 건이라 하위 단계 모델은 묶음 행 모델과 같아야 하고, 묶음 행에는 요청 설정이 없다.
- params 가 비어 있으면(NULL) 그 단계의 코드 기본 요청 설정을 쓴다. 값이 있으면 그 값만 보내며
  reasoning, max_output_tokens 외의 키는 받지 않는다.
- 프로필 버전의 옛 칸(생성, 재작성의 모델과 버전)은 구성 행과 같다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import prompts
from app.answering.generator import (
    ANSWER_PROMPT_V17,
    ANSWER_PROMPT_VERSION,
    ANSWER_REPAIR_PROMPT_V17,
    ANSWER_REPAIR_PROMPT_VERSION,
    GENERATION_PROMPT_VERSION,
    OPENAI_GENERATION_MODEL,
    SOURCE_PLANNING_PROMPT_V11,
    SOURCE_PLANNING_PROMPT_VERSION,
    SOURCE_PLANNING_REPAIR_PROMPT_V11,
    SOURCE_PLANNING_REPAIR_PROMPT_VERSION,
    GenerationStageSettings,
    with_repair_instructions,
)
from app.answering.models import GenerationResult, GenerationSourcePlan
from app.chat.query_rewrite import (
    CODE_QUERY_REWRITE_STAGE,
    QUERY_REWRITE_DEFAULT_PARAMS,
    QueryRewriteOutput,
)
from app.core.stage_settings import StageSettings
from app.database.models import (
    PROFILE_REVISION_COMPONENT_PROMPT_KEYS,
    ProfileRevisionComponent,
    ProfileRevisionComponentStage,
)
from app.prompts.fingerprints import (
    composite_fingerprint,
    load_fingerprint_table,
    sha256_json,
    sha256_text,
    text_format_of,
)
from app.question_grouping.judge_client import (
    CODE_JUDGE_STAGE,
    JUDGE_DEFAULT_PARAMS,
    judge_text_format,
)


logger = logging.getLogger(__name__)

Stage = ProfileRevisionComponentStage

# 구성 행을 검사하는 순서. 재시도 단계는 처음 단계 뒤에 둔다(지시문이 처음 단계 본문을 포함한다).
STAGE_ORDER: Tuple[Stage, ...] = (
    Stage.REWRITE,
    Stage.JUDGE,
    Stage.GENERATION,
    Stage.SOURCE_PLANNING,
    Stage.SOURCE_PLANNING_REPAIR,
    Stage.ANSWER,
    Stage.ANSWER_REPAIR,
)
GENERATION_SUB_STAGES: Tuple[Stage, ...] = (
    Stage.SOURCE_PLANNING,
    Stage.SOURCE_PLANNING_REPAIR,
    Stage.ANSWER,
    Stage.ANSWER_REPAIR,
)
# 재시도 단계 → 지시문 앞부분과 출력 스키마를 함께 쓰는 처음 단계.
REPAIR_BASE_STAGE: Mapping[Stage, Stage] = {
    Stage.SOURCE_PLANNING_REPAIR: Stage.SOURCE_PLANNING,
    Stage.ANSWER_REPAIR: Stage.ANSWER,
}
# params 가 비었을 때 보내는 단계별 요청 설정. 생성 호출은 요청 설정을 보내지 않는다.
DEFAULT_STAGE_PARAMS: Mapping[Stage, Mapping[str, Any]] = {
    Stage.REWRITE: QUERY_REWRITE_DEFAULT_PARAMS,
    Stage.JUDGE: JUDGE_DEFAULT_PARAMS,
    Stage.SOURCE_PLANNING: {},
    Stage.SOURCE_PLANNING_REPAIR: {},
    Stage.ANSWER: {},
    Stage.ANSWER_REPAIR: {},
}
ALLOWED_PARAM_KEYS = frozenset({"reasoning", "max_output_tokens"})


class ProfileStageSettingsError(ValueError):
    """실행 중인 코드가 프로필 버전의 구성 행대로 서빙할 수 없을 때."""

    def __init__(self, revision_id: Any, problems: Sequence[str]) -> None:
        self.revision_id = revision_id
        self.problems = tuple(problems)
        super().__init__(
            f"프로필 버전 {revision_id} 을 서빙할 수 없습니다: " + "; ".join(self.problems)
        )


@dataclass(frozen=True)
class ProfileStageSettings:
    """턴 하나가 쓰는 프로필 버전의 단계별 설정."""

    revision_id: int
    rewrite: StageSettings
    judge: StageSettings
    generation: GenerationStageSettings


@dataclass(frozen=True)
class ComponentSpec:
    """구성 행 하나의 값. 코드 기본 구성으로 새 프로필 버전을 만들 때 쓴다."""

    stage: Stage
    model_name: str
    prompt_key: str
    prompt_version: str
    prompt_sha256: str
    params: Optional[Dict[str, Any]]


@lru_cache(maxsize=1)
def _fingerprint_table() -> Mapping[str, str]:
    """지문 표. 배포된 파일이라 프로세스 안에서 한 번만 읽는다."""

    return dict(load_fingerprint_table())


@lru_cache(maxsize=1)
def _code_output_schemas() -> Mapping[Stage, str]:
    """지금 코드가 단계마다 보내는 출력 스키마의 지문(app/prompts/fingerprints.py 규칙)."""

    return {
        Stage.REWRITE: sha256_json(text_format_of(QueryRewriteOutput)),
        Stage.JUDGE: sha256_json(judge_text_format()),
        Stage.SOURCE_PLANNING: sha256_json(text_format_of(GenerationSourcePlan)),
        Stage.ANSWER: sha256_json(text_format_of(GenerationResult)),
    }


def _component(stage: Stage) -> str:
    return PROFILE_REVISION_COMPONENT_PROMPT_KEYS[stage]


def _key(stage: Stage, version: str) -> str:
    return f"{_component(stage)}@{version}"


def _schema_key(stage: Stage, version: str) -> str:
    return f"{_component(stage)}.output_schema@{version}"


def _short(value: Optional[str]) -> str:
    return "없음" if value is None else value[:12]


def _stage_of(value: Any) -> Optional[Stage]:
    try:
        return Stage(value)
    except ValueError:
        return None


def _params_problems(raw: Any) -> List[str]:
    if not isinstance(raw, dict):
        return ["요청 설정이 객체가 아님"]
    problems = []
    unknown = sorted(set(raw) - ALLOWED_PARAM_KEYS)
    if unknown:
        problems.append(f"지원하지 않는 요청 설정 {', '.join(unknown)}")
    if "max_output_tokens" in raw:
        tokens = raw["max_output_tokens"]
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            problems.append("max_output_tokens 가 양의 정수가 아님")
    if "reasoning" in raw:
        reasoning = raw["reasoning"]
        if not isinstance(reasoning, dict) or not all(
            isinstance(name, str) for name in reasoning
        ):
            problems.append("reasoning 이 객체가 아님")
    return problems


def _resolve(
    revision: Any,
    rows: Sequence[Any],
) -> Tuple[Optional[ProfileStageSettings], Tuple[str, ...]]:
    """구성 행을 단계별 설정으로 바꾼다. 문제가 있으면 설정 없이 문제 목록을 돌려준다."""

    table = _fingerprint_table()
    schemas = _code_output_schemas()
    problems: List[str] = []

    by_stage: Dict[Stage, Any] = {}
    for row in rows:
        stage = _stage_of(getattr(row, "stage", None))
        if stage is None:
            problems.append(f"알 수 없는 단계 {getattr(row, 'stage', None)!r}")
        else:
            by_stage[stage] = row
    for stage in STAGE_ORDER:
        if stage not in by_stage:
            problems.append(f"{stage.value}: 구성 행 없음")

    def report(stage: Stage, message: str) -> None:
        problems.append(f"{stage.value}: {message}")

    instructions: Dict[Stage, str] = {}
    settings: Dict[Stage, StageSettings] = {}
    for stage in STAGE_ORDER:
        row = by_stage.get(stage)
        if row is None or stage == Stage.GENERATION:
            continue
        version = row.prompt_version
        key = _key(stage, version)
        before = len(problems)
        if not row.model_name:
            report(stage, "모델 없음")
        if row.prompt_key != key:
            report(stage, f"프롬프트 키 {row.prompt_key} 가 {key} 와 다름")
        try:
            body = prompts.get(_component(stage), version)
        except prompts.PromptNotFoundError:
            report(stage, f"실행 중인 코드에 없는 프롬프트 버전 {key}")
            continue
        base_stage = REPAIR_BASE_STAGE.get(stage)
        if base_stage is None:
            text = body
        else:
            base = instructions.get(base_stage)
            if base is None:
                report(stage, f"처음 단계 {base_stage.value} 지시문을 만들지 못함")
                continue
            text = with_repair_instructions(base, body)
        instructions[stage] = text
        digest = sha256_text(text)
        if row.prompt_sha256 is None:
            report(stage, f"{key} 지문 없음")
        elif row.prompt_sha256 != digest:
            report(
                stage,
                f"{key} 지문 행 {_short(row.prompt_sha256)}, 코드 {_short(digest)}",
            )
        schema_stage = stage if base_stage is None else base_stage
        schema_version = (
            version if base_stage is None else by_stage[base_stage].prompt_version
        )
        schema_key = _schema_key(schema_stage, schema_version)
        registered_schema = table.get(schema_key)
        if registered_schema is None:
            report(stage, f"지문 표에 없는 출력 스키마 {schema_key}")
        elif registered_schema != schemas[schema_stage]:
            report(
                stage,
                f"출력 스키마 {schema_key} 표 {_short(registered_schema)}, "
                f"코드 {_short(schemas[schema_stage])}",
            )
        if row.params is None:
            params: Mapping[str, Any] = DEFAULT_STAGE_PARAMS[stage]
        else:
            params = row.params
            for message in _params_problems(row.params):
                report(stage, message)
        if len(problems) == before:
            settings[stage] = StageSettings(
                model_name=row.model_name,
                prompt_version=version,
                instructions=text,
                params=params,
            )

    generation = by_stage.get(Stage.GENERATION)
    if generation is not None:
        _check_composite(generation, by_stage, table, report)

    for column, stage, field in (
        ("generation_model_name", Stage.GENERATION, "model_name"),
        ("generation_prompt_version", Stage.GENERATION, "prompt_version"),
        ("query_rewrite_model_name", Stage.REWRITE, "model_name"),
        ("query_rewrite_prompt_version", Stage.REWRITE, "prompt_version"),
    ):
        row = by_stage.get(stage)
        if row is None:
            continue
        legacy = getattr(revision, column, None)
        if legacy != getattr(row, field):
            report(stage, f"프로필 버전 칸 {column}={legacy} 가 구성 행과 다름")

    if problems:
        return None, tuple(problems)
    assert generation is not None
    return (
        ProfileStageSettings(
            revision_id=revision.id,
            rewrite=settings[Stage.REWRITE],
            judge=settings[Stage.JUDGE],
            generation=GenerationStageSettings(
                model_name=generation.model_name,
                prompt_version=generation.prompt_version,
                source_planning=settings[Stage.SOURCE_PLANNING],
                source_planning_repair=settings[Stage.SOURCE_PLANNING_REPAIR],
                answer=settings[Stage.ANSWER],
                answer_repair=settings[Stage.ANSWER_REPAIR],
            ),
        ),
        (),
    )


def _check_composite(
    generation: Any,
    by_stage: Mapping[Stage, Any],
    table: Mapping[str, str],
    report: Any,
) -> None:
    """묶음 행이 지문 표의 묶음 버전이고 하위 단계 구성과 맞는지 확인한다."""

    stage = Stage.GENERATION
    key = _key(stage, generation.prompt_version)
    if not generation.model_name:
        report(stage, "모델 없음")
    if generation.prompt_key != key:
        report(stage, f"프롬프트 키 {generation.prompt_key} 가 {key} 와 다름")
    if generation.params not in (None, {}):
        report(stage, "묶음 행에는 요청 설정을 두지 않음")
    for sub_stage in GENERATION_SUB_STAGES:
        sub = by_stage.get(sub_stage)
        if sub is not None and sub.model_name != generation.model_name:
            report(
                sub_stage,
                f"모델 {sub.model_name} 가 묶음 행 모델 {generation.model_name} 와 다름",
            )

    registered = table.get(key)
    if registered is None:
        report(stage, f"실행 중인 코드에 없는 묶음 버전 {key}")
        return
    if generation.prompt_sha256 != registered:
        report(
            stage,
            f"{key} 지문 행 {_short(generation.prompt_sha256)}, 표 {_short(registered)}",
        )
        return
    subs = [by_stage.get(sub_stage) for sub_stage in GENERATION_SUB_STAGES]
    if any(sub is None or sub.prompt_sha256 is None for sub in subs):
        return
    parts = {
        _key(sub_stage, sub.prompt_version): sub.prompt_sha256
        for sub_stage, sub in zip(GENERATION_SUB_STAGES, subs)
    }
    for schema_stage in (Stage.SOURCE_PLANNING, Stage.ANSWER):
        schema_key = _schema_key(schema_stage, by_stage[schema_stage].prompt_version)
        parts[schema_key] = table.get(schema_key, "")
    try:
        rebuilt = composite_fingerprint(parts)
    except ValueError as error:
        report(stage, f"묶음 지문을 다시 계산하지 못함: {error}")
        return
    if rebuilt != registered:
        report(stage, f"{key} 가 하위 단계 구성 행과 맞지 않음")


def component_problems(revision: Any, rows: Sequence[Any]) -> Tuple[str, ...]:
    """프로필 버전을 실행 중인 코드로 서빙할 수 없는 이유. 서빙할 수 있으면 빈 튜플이다."""

    return _resolve(revision, rows)[1]


def stage_settings_from_components(
    revision: Any,
    rows: Sequence[Any],
) -> ProfileStageSettings:
    """구성 행대로 단계별 설정을 만든다. 서빙할 수 없으면 ProfileStageSettingsError."""

    settings, problems = _resolve(revision, rows)
    if settings is None:
        raise ProfileStageSettingsError(getattr(revision, "id", None), problems)
    return settings


async def read_profile_components(
    session: AsyncSession,
    revision_id: int,
) -> List[ProfileRevisionComponent]:
    """프로필 버전 하나의 구성 행."""

    return list(
        (
            await session.scalars(
                select(ProfileRevisionComponent)
                .where(ProfileRevisionComponent.revision_id == revision_id)
                .order_by(ProfileRevisionComponent.id)
            )
        ).all()
    )


async def load_profile_stage_settings(
    session: AsyncSession,
    revision: Any,
) -> ProfileStageSettings:
    """프로필 버전의 구성 행을 읽어 단계별 설정을 만든다. 서빙할 수 없으면 ProfileStageSettingsError."""

    rows = await read_profile_components(session, revision.id)
    return stage_settings_from_components(revision, rows)


def code_components() -> Tuple[ComponentSpec, ...]:
    """지금 코드의 기본 구성을 구성 행 값으로. 새 프로필 버전을 코드 기본 구성으로 만들 때 쓴다."""

    def spec(
        stage: Stage,
        model_name: str,
        version: str,
        sha256: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> ComponentSpec:
        return ComponentSpec(
            stage=stage,
            model_name=model_name,
            prompt_key=_key(stage, version),
            prompt_version=version,
            prompt_sha256=sha256,
            params=None if params is None else dict(params),
        )

    rewrite = CODE_QUERY_REWRITE_STAGE
    judge = CODE_JUDGE_STAGE
    model = OPENAI_GENERATION_MODEL
    return (
        spec(
            Stage.REWRITE,
            rewrite.model_name,
            rewrite.prompt_version,
            sha256_text(rewrite.instructions),
            rewrite.params,
        ),
        spec(
            Stage.JUDGE,
            judge.model_name,
            judge.prompt_version,
            sha256_text(judge.instructions),
            judge.params,
        ),
        spec(
            Stage.GENERATION,
            model,
            GENERATION_PROMPT_VERSION,
            _fingerprint_table()[_key(Stage.GENERATION, GENERATION_PROMPT_VERSION)],
        ),
        spec(
            Stage.SOURCE_PLANNING,
            model,
            SOURCE_PLANNING_PROMPT_VERSION,
            sha256_text(SOURCE_PLANNING_PROMPT_V11),
        ),
        spec(
            Stage.SOURCE_PLANNING_REPAIR,
            model,
            SOURCE_PLANNING_REPAIR_PROMPT_VERSION,
            sha256_text(SOURCE_PLANNING_REPAIR_PROMPT_V11),
        ),
        spec(Stage.ANSWER, model, ANSWER_PROMPT_VERSION, sha256_text(ANSWER_PROMPT_V17)),
        spec(
            Stage.ANSWER_REPAIR,
            model,
            ANSWER_REPAIR_PROMPT_VERSION,
            sha256_text(ANSWER_REPAIR_PROMPT_V17),
        ),
    )
