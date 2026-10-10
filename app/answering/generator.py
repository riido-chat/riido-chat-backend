"""Hybrid 검색 결과를 근거로 OpenAI 답변을 생성한다."""

import time
from dataclasses import dataclass, replace
from typing import Any, List, Optional, Sequence

from openai import AsyncOpenAI
from pydantic import ValidationError

from app import prompts
from app.core.config import get_settings
from app.core.model_trace import ModelCallTrace
from app.core.openai_error import is_transient_openai_error
from app.core.openai_usage import cached_input_tokens as usage_cached_input_tokens
from app.core.openai_usage import reasoning_tokens as usage_reasoning_tokens
from app.answering.models import (
    GenerationCall,
    GenerationAnswerType,
    GenerationContextSource,
    GenerationEvidenceRequirement,
    GenerationPlanningStatus,
    GenerationResult,
    GenerationSourcePlan,
    GenerationStageTrace,
    GenerationStatus,
    GenerationWithheldReason,
)
from app.retrieval.models import HybridRetrievalResult


OPENAI_GENERATION_PROVIDER = "openai"
OPENAI_GENERATION_MODEL = "gpt-5.6-terra"
GENERATION_PROMPT_VERSION = "v40"
SOURCE_PLANNING_PROMPT_VERSION = "v34"
SOURCE_PLANNING_REPAIR_PROMPT_VERSION = "v34-repair-1"
ANSWER_PROMPT_VERSION = "v23"
ANSWER_REPAIR_PROMPT_VERSION = "v23-repair-1"
MAX_CONTEXT_SOURCES = 5
MAX_GENERATION_ATTEMPTS = 2
MAX_SOURCE_PLANNING_REGENERATIONS = 1

PROCEDURE_EVIDENCE_RULES_VERSION = "v1"
# 재시도 지시문은 처음 지시문 뒤에 이 구분자와 재시도 덧붙임을 이어 만든다.
REPAIR_INSTRUCTIONS_SEPARATOR = "\n\n"


def with_repair_instructions(instructions: str, repair: str) -> str:
    """처음 지시문에 재시도 덧붙임을 이은 재시도 지시문."""

    return instructions + REPAIR_INSTRUCTIONS_SEPARATOR + repair


# 지시문 본문은 app/prompts/<component>/<version>.md 에 있다. 아래 상수 이름은 기존 import 를
# 위한 별칭이며 현재 버전 상수가 가리키는 본문이다.
SOURCE_PLANNING_PROMPT_V11 = prompts.get(
    "generation.source_planning", SOURCE_PLANNING_PROMPT_VERSION
)
SOURCE_PLANNING_REPAIR_PROMPT_V11 = with_repair_instructions(
    SOURCE_PLANNING_PROMPT_V11,
    prompts.get("generation.source_planning_repair", SOURCE_PLANNING_REPAIR_PROMPT_VERSION),
)
ANSWER_PROMPT_V17 = prompts.get("generation.answer", ANSWER_PROMPT_VERSION)
ANSWER_REPAIR_PROMPT_V17 = with_repair_instructions(
    ANSWER_PROMPT_V17,
    prompts.get("generation.answer_repair", ANSWER_REPAIR_PROMPT_VERSION),
)
# 소스 계획과 답변 지시문이 같은 문장으로 포함하는 절차 근거 규칙. 모델에 따로 보내지 않고,
# 두 지시문이 이 문단을 그대로 담고 있는지 테스트가 확인하는 기준으로만 쓴다.
PROCEDURE_EVIDENCE_RULES = prompts.get(
    "generation.procedure_evidence_rules", PROCEDURE_EVIDENCE_RULES_VERSION
)


def build_generation_context(
    results: Sequence[HybridRetrievalResult],
) -> List[GenerationContextSource]:
    """Hybrid 순서를 유지하며 SOURCE_1부터 SOURCE_5까지 부여한다."""

    if len(results) > MAX_CONTEXT_SOURCES:
        raise ValueError(
            f"Generation Context는 최대 {MAX_CONTEXT_SOURCES}개까지 사용할 수 있습니다."
        )

    return [
        GenerationContextSource(
            source_id=f"SOURCE_{index}",
            chunk=result.chunk,
        )
        for index, result in enumerate(results, start=1)
    ]


def build_generation_input(
    question: str,
    sources: Sequence[GenerationContextSource],
) -> str:
    """LLM에 노출할 Context와 사용자 질문을 조립한다."""

    context_parts = []
    for source in sources:
        section_path = " > ".join(source.chunk.section_path)
        content = source.chunk.content.replace(r"\[", "[")
        context_parts.append(
            "\n".join(
                (
                    f"### {source.source_id}",
                    f"Document Title: {source.chunk.document_title}",
                    f"Section Path: {section_path}",
                    "Content:",
                    content,
                )
            )
        )

    context = "\n\n".join(context_parts) or "제공된 Context가 없습니다."
    return f"## Top-5 Context\n\n{context}\n\n## User Question\n\n{question}"


def build_answer_input(
    question: str,
    sources: Sequence[GenerationContextSource],
    plan: GenerationSourcePlan,
) -> str:
    """선택된 Context와 Planner가 정한 작성 계약을 함께 조립한다."""

    def format_requirements(
        requirements: Sequence[GenerationEvidenceRequirement],
    ) -> str:
        return "\n".join(
            f"- {requirement.information_unit}: "
            f"{', '.join(requirement.source_ids)}"
            for requirement in requirements
        ) or "- 없음"

    required_coverage = format_requirements(plan.evidence_requirements)
    optional_context = format_requirements(plan.optional_context)
    related_guidance = format_requirements(plan.related_guidance)
    unanswered_information = "\n".join(
        (
            f"- {information}"
            for information in plan.unanswered_information
        )
    ) or "- 없음"
    return (
        f"{build_generation_input(question, sources)}"
        f"\n\n## Answer Contract\n\n"
        f"- Plan Status: {plan.status.value}\n"
        f"- Answer Type: {plan.answer_type.value}\n"
        f"- Answer Scope: {plan.answer_scope.value}"
        f"\n\n## Required Answer Coverage\n\n{required_coverage}"
        f"\n\n## Optional Context\n\n{optional_context}"
        f"\n\n## Unanswered Information\n\n{unanswered_information}"
        f"\n\n## Related Guidance\n\n{related_guidance}"
    )


def build_source_planning_repair_input(
    question: str,
    sources: Sequence[GenerationContextSource],
    validation_error: str,
) -> str:
    """같은 질문과 후보 Source로 구조 오류만 교정하도록 입력을 만든다."""

    return (
        f"{build_generation_input(question, sources)}"
        f"\n\n## Validation Failure\n\n{validation_error}"
    )


def build_answer_repair_input(
    question: str,
    sources: Sequence[GenerationContextSource],
    plan: GenerationSourcePlan,
    previous_result: GenerationResult,
    validation_error: str,
) -> str:
    """같은 근거와 Coverage에 직전 검증 실패 정보만 추가한다."""

    previous_limitation = (
        previous_result.limitation_markdown or "미확인 고지 없음"
    )
    previous_answer = previous_result.answer_markdown or "답변 본문 없음"
    return (
        f"{build_answer_input(question, sources, plan)}"
        f"\n\n## Validation Failure\n\n{validation_error}"
        f"\n\n## Previous Invalid Limitation\n\n{previous_limitation}"
        f"\n\n## Previous Invalid Answer\n\n{previous_answer}"
    )


def select_required_sources(
    plan: GenerationSourcePlan,
    sources: Sequence[GenerationContextSource],
) -> List[GenerationContextSource]:
    """계획이 선택한 Source를 중복 없이 모으고 존재 여부를 검증한다."""

    if plan.status == GenerationPlanningStatus.ANSWERABLE:
        requirements = [
            *plan.evidence_requirements,
            *plan.optional_context,
        ]
    elif plan.status == GenerationPlanningStatus.RELATED_GUIDANCE:
        requirements = plan.related_guidance
    else:
        raise RuntimeError("WITHHELD Source Plan에는 선택할 Source가 없습니다.")
    required_source_ids = list(
        dict.fromkeys(
            source_id
            for requirement in requirements
            for source_id in requirement.source_ids
        )
    )
    source_by_id = {source.source_id: source for source in sources}
    invalid_source_ids = [
        source_id
        for source_id in required_source_ids
        if source_id not in source_by_id
    ]
    if invalid_source_ids:
        invalid_sources = ", ".join(invalid_source_ids)
        raise RuntimeError(
            "Source Plan에 존재하지 않는 Source가 있습니다: "
            f"{invalid_sources}"
        )

    return [source_by_id[source_id] for source_id in required_source_ids]


def count_distinct_citations(
    sources: Sequence[GenerationContextSource],
) -> int:
    """Backend Citation 병합 기준과 동일하게 고유 출처 수를 센다."""

    return len(
        {
            (source.chunk.source_url, source.chunk.section_path)
            for source in sources
        }
    )


@dataclass(frozen=True)
class _ParsedResponseCall:
    """Responses API 한 단계의 결과와 실제 재시도 횟수."""

    response: Any = None
    retry_count: int = 0
    error: Optional[Exception] = None


def _sum_tokens(current: Optional[int], added: Optional[int]) -> Optional[int]:
    if added is None:
        return current
    return (current or 0) + added


def _generation_trace(
    started: float,
    *,
    model_name: str = OPENAI_GENERATION_MODEL,
    retry_count: int,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cached_input_tokens: Optional[int] = None,
    reasoning_tokens: Optional[int] = None,
    error: Optional[Exception] = None,
    prompt_version: str = GENERATION_PROMPT_VERSION,
) -> ModelCallTrace:
    """Source 선택과 답변 생성을 한 논리 호출의 관측값으로 합친다."""

    return ModelCallTrace(
        provider=OPENAI_GENERATION_PROVIDER,
        model_name=model_name,
        succeeded=error is None,
        latency_ms=int((time.perf_counter() - started) * 1000),
        retry_count=retry_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        reasoning_tokens=reasoning_tokens,
        prompt_version=prompt_version,
        error_message=None if error is None else str(error),
    )


class OpenAIGenerator:
    """필요 Source를 확정한 뒤 해당 Context만으로 답변을 생성한다."""

    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        *,
        model_name: str = OPENAI_GENERATION_MODEL,
    ) -> None:
        if not model_name.strip():
            raise ValueError("Generation model_name은 비어 있을 수 없습니다.")
        if client is None:
            api_key = get_settings().openai_api_key
            if not api_key:
                raise ValueError("OPENAI_API_KEY 환경변수가 필요합니다.")
            client = AsyncOpenAI(api_key=api_key, max_retries=0, timeout=30.0)

        self._client = client
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate(
        self,
        question: str,
        sources: Sequence[GenerationContextSource],
    ) -> GenerationResult:
        """같은 호출에서 답변 생성과 answerability 판단을 수행한다."""

        call = await self.generate_with_trace(question, sources)
        if call.error is not None:
            raise call.error
        return call.result

    async def generate_with_trace(
        self,
        question: str,
        sources: Sequence[GenerationContextSource],
    ) -> GenerationCall:
        """Source 선택과 답변 생성의 합산 관측값을 반환한다.

        두 단계는 Generation 논리 호출 1건이며 retry_count에는 정상적인 두 번째
        호출이 아니라 일시 오류로 실제 재시도한 횟수만 합산한다.
        """

        started = time.perf_counter()
        total_retry_count = 0
        total_input_tokens: Optional[int] = None
        total_output_tokens: Optional[int] = None
        total_cached_input_tokens: Optional[int] = None
        total_reasoning_tokens: Optional[int] = None

        generation_input = build_generation_input(question, sources)
        plan_call = await self._parse_with_retry(
            instructions=SOURCE_PLANNING_PROMPT_V11,
            input_text=generation_input,
            text_format=GenerationSourcePlan,
        )
        total_retry_count += plan_call.retry_count
        planning_attempt_count = plan_call.retry_count + 1
        planning_regeneration_count = 0
        planning_regeneration_trace: Optional[ModelCallTrace] = None
        planning_regeneration_result: Optional[GenerationSourcePlan] = None

        if (
            MAX_SOURCE_PLANNING_REGENERATIONS
            and isinstance(plan_call.error, ValidationError)
        ):
            repair_started = time.perf_counter()
            repair_call = await self._parse_with_retry(
                instructions=SOURCE_PLANNING_REPAIR_PROMPT_V11,
                input_text=build_source_planning_repair_input(
                    question,
                    sources,
                    str(plan_call.error),
                ),
                text_format=GenerationSourcePlan,
            )
            total_retry_count += repair_call.retry_count
            planning_attempt_count += repair_call.retry_count + 1
            planning_regeneration_count = 1
            repair_usage = getattr(repair_call.response, "usage", None)
            planning_regeneration_trace = _generation_trace(
                repair_started,
                model_name=self._model_name,
                retry_count=repair_call.retry_count,
                input_tokens=getattr(repair_usage, "input_tokens", None),
                output_tokens=getattr(repair_usage, "output_tokens", None),
                cached_input_tokens=usage_cached_input_tokens(repair_usage),
                reasoning_tokens=usage_reasoning_tokens(repair_usage),
                error=repair_call.error,
                prompt_version=SOURCE_PLANNING_REPAIR_PROMPT_VERSION,
            )
            plan_call = repair_call

        if plan_call.error is not None:
            return self._error_call(
                started,
                plan_call.error,
                total_retry_count,
                total_input_tokens,
                total_output_tokens,
                cached_input_tokens=total_cached_input_tokens,
                reasoning_tokens=total_reasoning_tokens,
                stage_trace=GenerationStageTrace(
                    planning_attempt_count=planning_attempt_count,
                    planning_regeneration_count=planning_regeneration_count,
                    planning_regeneration_model_call=(
                        planning_regeneration_trace
                    ),
                ),
            )

        plan_response = plan_call.response
        plan_usage = getattr(plan_response, "usage", None)
        total_input_tokens = _sum_tokens(
            total_input_tokens,
            getattr(plan_usage, "input_tokens", None),
        )
        total_output_tokens = _sum_tokens(
            total_output_tokens,
            getattr(plan_usage, "output_tokens", None),
        )
        total_cached_input_tokens = _sum_tokens(
            total_cached_input_tokens,
            usage_cached_input_tokens(plan_usage),
        )
        total_reasoning_tokens = _sum_tokens(
            total_reasoning_tokens,
            usage_reasoning_tokens(plan_usage),
        )
        plan = plan_response.output_parsed
        if planning_regeneration_count:
            planning_regeneration_result = plan
        planning_trace_fields = {
            "planning_attempt_count": planning_attempt_count,
            "planning_regeneration_count": planning_regeneration_count,
            "planning_regeneration_model_call": planning_regeneration_trace,
            "planning_regeneration_result": planning_regeneration_result,
        }

        if plan.status == GenerationPlanningStatus.WITHHELD:
            result = GenerationResult(
                status=GenerationStatus.WITHHELD,
                limitation_markdown=None,
                answer_markdown=None,
                withheld_reason=plan.withheld_reason,
            )
            return GenerationCall(
                trace=_generation_trace(
                    started,
                    model_name=self._model_name,
                    retry_count=total_retry_count,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cached_input_tokens=total_cached_input_tokens,
                    reasoning_tokens=total_reasoning_tokens,
                ),
                result=result,
                stage_trace=GenerationStageTrace(
                    source_plan=plan,
                    **planning_trace_fields,
                ),
            )

        try:
            selected_sources = select_required_sources(plan, sources)
        except Exception as error:
            return self._error_call(
                started,
                error,
                total_retry_count,
                total_input_tokens,
                total_output_tokens,
                cached_input_tokens=total_cached_input_tokens,
                reasoning_tokens=total_reasoning_tokens,
                stage_trace=GenerationStageTrace(
                    source_plan=plan,
                    **planning_trace_fields,
                ),
            )

        answer_call = await self._parse_with_retry(
            instructions=ANSWER_PROMPT_V17,
            input_text=build_answer_input(question, selected_sources, plan),
            text_format=GenerationResult,
        )
        total_retry_count += answer_call.retry_count
        if answer_call.error is not None:
            return self._error_call(
                started,
                answer_call.error,
                total_retry_count,
                total_input_tokens,
                total_output_tokens,
                cached_input_tokens=total_cached_input_tokens,
                reasoning_tokens=total_reasoning_tokens,
                stage_trace=GenerationStageTrace(
                    source_plan=plan,
                    selected_sources=tuple(selected_sources),
                    answer_attempt_count=answer_call.retry_count + 1,
                    **planning_trace_fields,
                ),
            )

        answer_response = answer_call.response
        answer_usage = getattr(answer_response, "usage", None)
        total_input_tokens = _sum_tokens(
            total_input_tokens,
            getattr(answer_usage, "input_tokens", None),
        )
        total_output_tokens = _sum_tokens(
            total_output_tokens,
            getattr(answer_usage, "output_tokens", None),
        )
        total_cached_input_tokens = _sum_tokens(
            total_cached_input_tokens,
            usage_cached_input_tokens(answer_usage),
        )
        total_reasoning_tokens = _sum_tokens(
            total_reasoning_tokens,
            usage_reasoning_tokens(answer_usage),
        )
        result = answer_response.output_parsed
        return GenerationCall(
            trace=_generation_trace(
                started,
                model_name=self._model_name,
                retry_count=total_retry_count,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cached_input_tokens=total_cached_input_tokens,
                reasoning_tokens=total_reasoning_tokens,
            ),
            result=result,
            stage_trace=GenerationStageTrace(
                source_plan=plan,
                selected_sources=tuple(selected_sources),
                pre_validation_result=result,
                answer_attempt_count=answer_call.retry_count + 1,
                **planning_trace_fields,
            ),
        )

    async def regenerate_answer_with_trace(
        self,
        question: str,
        stage_trace: GenerationStageTrace,
        validation_error: str,
    ) -> GenerationCall:
        """같은 계획과 Source로 answer 단계만 한 번 다시 생성한다."""

        plan = stage_trace.source_plan
        previous_result = stage_trace.pre_validation_result
        if plan is None or previous_result is None:
            error = RuntimeError("답변 재생성에 필요한 Generation 단계 trace가 없습니다.")
            return GenerationCall(
                trace=_generation_trace(
                    time.perf_counter(),
                    model_name=self._model_name,
                    retry_count=0,
                    error=error,
                    prompt_version=ANSWER_REPAIR_PROMPT_VERSION,
                ),
                error=error,
                stage_trace=stage_trace,
            )

        started = time.perf_counter()
        answer_call = await self._parse_with_retry(
            instructions=ANSWER_REPAIR_PROMPT_V17,
            input_text=build_answer_repair_input(
                question,
                stage_trace.selected_sources,
                plan,
                previous_result,
                validation_error,
            ),
            text_format=GenerationResult,
        )
        usage = getattr(answer_call.response, "usage", None)
        repair_trace = _generation_trace(
            started,
            model_name=self._model_name,
            retry_count=answer_call.retry_count,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            cached_input_tokens=usage_cached_input_tokens(usage),
            reasoning_tokens=usage_reasoning_tokens(usage),
            error=answer_call.error,
            prompt_version=ANSWER_REPAIR_PROMPT_VERSION,
        )
        result = (
            None
            if answer_call.response is None
            else answer_call.response.output_parsed
        )
        updated_stage_trace = replace(
            stage_trace,
            answer_attempt_count=(
                stage_trace.answer_attempt_count + answer_call.retry_count + 1
            ),
            validation_regeneration_count=(
                stage_trace.validation_regeneration_count + 1
            ),
            validation_regeneration_model_call=repair_trace,
            validation_regeneration_result=result,
        )
        return GenerationCall(
            trace=repair_trace,
            result=result,
            error=answer_call.error,
            stage_trace=updated_stage_trace,
        )

    async def _parse_with_retry(
        self,
        *,
        instructions: str,
        input_text: str,
        text_format: Any,
    ) -> _ParsedResponseCall:
        """한 단계의 Structured Output 호출만 일시 오류에 한해 재시도한다."""

        for attempt in range(MAX_GENERATION_ATTEMPTS):
            try:
                response = await self._client.responses.parse(
                    model=self._model_name,
                    instructions=instructions,
                    input=input_text,
                    text_format=text_format,
                )
                if response.output_parsed is None:
                    raise RuntimeError(
                        "OpenAI Generation 응답에 Structured Output이 없습니다."
                    )
                return _ParsedResponseCall(response=response, retry_count=attempt)
            except Exception as error:
                if attempt == 0 and is_transient_openai_error(error):
                    continue
                return _ParsedResponseCall(error=error, retry_count=attempt)

        error = RuntimeError("OpenAI Generation 호출이 완료되지 않았습니다.")
        return _ParsedResponseCall(
            error=error,
            retry_count=MAX_GENERATION_ATTEMPTS - 1,
        )

    def _error_call(
        self,
        started: float,
        error: Exception,
        retry_count: int,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        stage_trace: Optional[GenerationStageTrace] = None,
        *,
        cached_input_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
    ) -> GenerationCall:
        return GenerationCall(
            trace=_generation_trace(
                started,
                model_name=self._model_name,
                retry_count=retry_count,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                reasoning_tokens=reasoning_tokens,
                error=error,
            ),
            error=error,
            stage_trace=stage_trace,
        )
