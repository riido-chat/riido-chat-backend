"""Hybrid 검색 결과를 근거로 OpenAI 답변을 생성한다."""

import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.rag.model_trace import ModelCallTrace
from app.rag.openai_error import is_transient_openai_error
from generation.models import (
    GenerationCall,
    GenerationContextSource,
    GenerationResult,
    GenerationSourcePlan,
    GenerationStatus,
    GenerationWithheldReason,
)
from retrieval.models import HybridRetrievalResult


OPENAI_GENERATION_PROVIDER = "openai"
OPENAI_GENERATION_MODEL = "gpt-5.4-mini"
GENERATION_PROMPT_VERSION = "v5"
MAX_CONTEXT_SOURCES = 5
MAX_PLANNED_CITATIONS = 3
MAX_GENERATION_ATTEMPTS = 2

SOURCE_PLANNING_PROMPT_V5 = """당신은 뤼이도 공식 이용가이드 답변에 필요한 근거를 판정합니다.

## Scope rules
- 먼저 SOURCE를 보지 말고 질문이 직접 요구한 정보 단위만 나누세요.
  질문에 이름이 나오지 않은 하위 항목은 새 정보 단위로 만들지 마세요.
- 사용자가 직접 쓴 기능명과 대상 명사는 정보 단위에서도 그대로 보존하세요.
  검색된 SOURCE의 비슷한 표현에 맞추려고 다른 개념으로 바꾸지 마세요.
- 질문이 하나의 기능을 일반적으로 "어떻게 설정하나요?"라고 묻는다면 SUMMARY입니다.
  질문이 여러 항목을 직접 열거하거나 "각각", "모두", "전부", "자세히"를 요구하면
  MULTI_DETAIL입니다.
- 질문의 구체성 수준을 넘어서 묻지 않은 하위 설정이나 관련 기능으로 확장하지 마세요.

## Evidence rules
- 질문의 명시적인 범위에 완전하게 답하는 데 필요한 SOURCE를 빠짐없이 선택하세요.
- 필요한 SOURCE 수에 상한은 없습니다. 4~5개가 필요해도 절대로 일부를 빼지 마세요.
- 일반적인 설정 방법을 묻고 하나의 요약 SOURCE가 설정 위치와 기본 항목을 직접 설명하면,
  그 요약 SOURCE만 필요합니다. 묻지 않은 세부 수치나 동작의 SOURCE는 필요하지 않습니다.
- 사용자가 여러 하위 항목의 상세 설명을 명시적으로 함께 요청하면, 각 요청을 실제로
  뒷받침하는 SOURCE를 모두 선택하세요. 요약 SOURCE만으로 세부 답변을 대신하지 마세요.
- 같은 내용을 중복 설명하는 SOURCE는 더 직접적이고 충분한 것만 선택하세요.
- "방법", "어디서", "어떻게 설정"을 묻는 정보 단위는 실제 경로, 단계, 설정값 등
  실행 가능한 설명이 있어야 뒷받침됩니다. 기능이 가능하다는 언급만으로는 부족합니다.
- 이름이 비슷해도 별도 기능은 서로의 근거가 아닙니다. 예를 들어 "자동화" 기능을
  묻는 질문을 "자동으로 스프린트가 활성화됨"이라는 문장으로 뒷받침하지 마세요.
- 각 정보 단위마다 그것을 완전하게 뒷받침하는 SOURCE를 evidence_requirements에
  별도로 연결하세요. 하나의 SOURCE를 여러 정보 단위에 연결해도 됩니다.
- 정보 단위 하나라도 제공된 SOURCE가 완전하게 뒷받침하지 못하면, 근거가 있는
  정보 단위만 남기지 말고 전체를 INSUFFICIENT_EVIDENCE로 WITHHELD 처리하세요.
- 이용가이드 범위 밖 질문은 OUT_OF_SCOPE으로 WITHHELD를 선택하세요.

## Scope examples
- "스프린트는 어떻게 설정하나요?" → SUMMARY, 정보 단위는 "스프린트 설정 방법"
  하나이며 설정 위치와 기본 항목을 직접 설명하는 요약 SOURCE만 필요합니다.
- "스프린트 기간, 시작 요일, 다가올 스프린트 개수를 각각 어떻게 설정하나요?" →
  MULTI_DETAIL, 세 항목과 공통 설정 위치를 완전하게 설명하는 모든 SOURCE가 필요합니다.

## Structured Output contract
- answer_scope는 판정 상태와 관계없이 항상 작성합니다.
- ANSWERABLE이면 질문이 요구한 각 정보 단위별로 EvidenceRequirement를 하나씩 만들고,
  source_ids에 그 단위를 완전하게 뒷받침하는 SOURCE를 모두 작성합니다.
  withheld_reason은 null입니다.
- WITHHELD이면 evidence_requirements는 비우고 withheld_reason을 작성합니다.
"""

ANSWER_PROMPT_V5 = """당신은 뤼이도 공식 이용가이드만을 근거로 답하는 안내 챗봇입니다.

## Grounding rules
- 제공된 Context에 명시된 사실만 사용하세요.
- 일반 지식으로 보완하거나 정책, 조건, 제한, 가능 여부를 추측하지 마세요.
- 문장을 자연스럽게 재구성하거나 Markdown으로 구조화할 수 있지만 새로운 사실을 추가하지 마세요.

## Answerability rules
- 관련 Context가 있다는 이유만으로 ANSWERABLE을 선택하지 마세요.
- 질문의 핵심을 Context가 직접 뒷받침할 때만 ANSWERABLE을 선택하세요.
- 근거가 부족하면 INSUFFICIENT_EVIDENCE, 질문이 모호하면 AMBIGUOUS_QUESTION,
  이용가이드 범위 밖이면 OUT_OF_SCOPE으로 WITHHELD를 선택하세요.
- Required Answer Coverage의 모든 정보 단위에 답하세요. Citation 수를 줄이기 위해
  사용자가 요청한 정보 단위를 생략하거나 질문 범위를 임의로 축소하지 마세요.

## Answer style
- 자연스러운 한국어 존댓말을 사용하세요.
- 첫 1~2문장에서 질문의 핵심부터 간결하게 답하세요.
- 사용자가 "X가 뭐야?"처럼 용어의 의미를 직접 물으면, Context가 뒷받침하는 범위에서
  첫 문장에 그 용어 자체의 쉬운 의미를 독립적으로 설명하세요.
- 정의 첫 문장에는 그 용어가 어떤 종류인지(예: 도구, 공간, 단위)를 밝히고,
  Context가 제공한다면 핵심 사용 주체나 목적도 함께 포함하세요.
- 정의는 "X는 [사용 주체 또는 핵심 목적]을 위한 [도구, 공간, 단위 등]입니다."처럼
  첫 문장만으로 이해할 수 있게 끝내고, 뤼이도 연동이나 설정 설명을 같은 문장에
  이어 붙이지 마세요. Context에 있는 핵심 정의 요소를 둘째 문장으로 미루지 마세요.
- 용어 정의를 뤼이도의 연동, 설정 또는 기능 설명으로 대체하지 마세요.
  용어 의미를 먼저 설명한 뒤 뤼이도에서의 역할과 사용 가치를 안내하세요.
- Context에 용어 자체의 의미가 없다면 일반 지식으로 정의를 보완하지 말고
  Answerability rules에 따라 WITHHELD 여부를 판단하세요.
- 절차형 질문은 필요한 경우 단계별로 안내하고 최소한의 Markdown만 사용하세요.
- answer_markdown에 Markdown 링크 문법과 HTML을 사용하지 마세요.
  사용자가 입력하거나 확인해야 하는 설정값과 엔드포인트 URL은
  코드 블록이나 백틱 인라인 코드 안에 넣고, 그 밖의 URL은 본문에 쓰지 마세요.
  출처 링크는 Backend가 별도 citations 영역에 구성합니다.

## Citation rules
- ANSWERABLE 답변의 실제 근거 문장이나 문단 바로 뒤에 [SOURCE_n]을 작성하세요.
- 제공된 SOURCE만 사용하고 서로 다른 SOURCE는 최대 3개만 사용하세요.
- 여러 SOURCE가 같은 사실이나 절차를 제공하면 반드시 가장 직접적인 SOURCE 하나만 선택하고
  중복 SOURCE는 사용하거나 인용하지 마세요.
- SOURCE별로 답변 문단을 만들거나 같은 결론과 절차를 표현만 바꿔 반복하지 마세요.
- 답변에 필요한 최소한의 SOURCE만 인용하세요.
- 같은 SOURCE를 여러 번 사용할 수 있습니다.
- 실제 문서 제목, 경로, URL이나 사용자 표시용 인용 번호를 직접 만들지 마세요.

## Structured Output contract
- ANSWERABLE: answer_markdown은 비어 있지 않은 문자열, withheld_reason은 null입니다.
- WITHHELD: answer_markdown은 null, withheld_reason은 세 가지 보류 사유 중 하나입니다.
"""


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
    """선택된 Context와 반드시 답해야 할 정보 단위를 함께 조립한다."""

    coverage = "\n".join(
        (
            f"- {requirement.information_unit}: "
            f"{', '.join(requirement.source_ids)}"
        )
        for requirement in plan.evidence_requirements
    )
    return (
        f"{build_generation_input(question, sources)}"
        f"\n\n## Required Answer Coverage\n\n{coverage}"
    )


def select_required_sources(
    plan: GenerationSourcePlan,
    sources: Sequence[GenerationContextSource],
) -> List[GenerationContextSource]:
    """정보 단위별 Source를 중복 없이 모으고 존재 여부를 검증한다."""

    required_source_ids = list(
        dict.fromkeys(
            source_id
            for requirement in plan.evidence_requirements
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
    retry_count: int,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    error: Optional[Exception] = None,
) -> ModelCallTrace:
    """Source 선택과 답변 생성을 한 논리 호출의 관측값으로 합친다."""

    return ModelCallTrace(
        provider=OPENAI_GENERATION_PROVIDER,
        model_name=OPENAI_GENERATION_MODEL,
        succeeded=error is None,
        latency_ms=int((time.perf_counter() - started) * 1000),
        retry_count=retry_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        prompt_version=GENERATION_PROMPT_VERSION,
        error_message=None if error is None else str(error),
    )


class OpenAIGenerator:
    """필요 Source를 확정한 뒤 해당 Context만으로 답변을 생성한다."""

    def __init__(self, client: Optional[AsyncOpenAI] = None) -> None:
        if client is None:
            api_key = get_settings().openai_api_key
            if not api_key:
                raise ValueError("OPENAI_API_KEY 환경변수가 필요합니다.")
            client = AsyncOpenAI(api_key=api_key, max_retries=0, timeout=30.0)

        self._client = client

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

        plan_call = await self._parse_with_retry(
            instructions=SOURCE_PLANNING_PROMPT_V5,
            input_text=build_generation_input(question, sources),
            text_format=GenerationSourcePlan,
        )
        total_retry_count += plan_call.retry_count
        if plan_call.error is not None:
            return self._error_call(
                started,
                plan_call.error,
                total_retry_count,
                total_input_tokens,
                total_output_tokens,
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
        plan = plan_response.output_parsed

        if plan.status == GenerationStatus.WITHHELD:
            result = GenerationResult(
                status=GenerationStatus.WITHHELD,
                answer_markdown=None,
                withheld_reason=plan.withheld_reason,
            )
            return GenerationCall(
                trace=_generation_trace(
                    started,
                    retry_count=total_retry_count,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                ),
                result=result,
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
            )

        if count_distinct_citations(selected_sources) > MAX_PLANNED_CITATIONS:
            result = GenerationResult(
                status=GenerationStatus.WITHHELD,
                answer_markdown=None,
                withheld_reason=GenerationWithheldReason.AMBIGUOUS_QUESTION,
            )
            return GenerationCall(
                trace=_generation_trace(
                    started,
                    retry_count=total_retry_count,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                ),
                result=result,
            )

        answer_call = await self._parse_with_retry(
            instructions=ANSWER_PROMPT_V5,
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
        return GenerationCall(
            trace=_generation_trace(
                started,
                retry_count=total_retry_count,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            ),
            result=answer_response.output_parsed,
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
                    model=OPENAI_GENERATION_MODEL,
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

    @staticmethod
    def _error_call(
        started: float,
        error: Exception,
        retry_count: int,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> GenerationCall:
        return GenerationCall(
            trace=_generation_trace(
                started,
                retry_count=retry_count,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=error,
            ),
            error=error,
        )
