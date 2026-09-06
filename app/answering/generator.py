"""Hybrid 검색 결과를 근거로 OpenAI 답변을 생성한다."""

import time
from dataclasses import dataclass, replace
from typing import Any, List, Optional, Sequence

from openai import AsyncOpenAI
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.model_trace import ModelCallTrace
from app.core.openai_error import is_transient_openai_error
from app.answering.models import (
    GenerationCall,
    GenerationAnswerType,
    GenerationContextSource,
    GenerationResult,
    GenerationSourcePlan,
    GenerationStageTrace,
    GenerationStatus,
    GenerationWithheldReason,
)
from app.retrieval.models import HybridRetrievalResult


OPENAI_GENERATION_PROVIDER = "openai"
OPENAI_GENERATION_MODEL = "gpt-5.4-mini"
GENERATION_PROMPT_VERSION = "v19"
SOURCE_PLANNING_PROMPT_VERSION = "v10"
SOURCE_PLANNING_REPAIR_PROMPT_VERSION = "v10-repair-1"
ANSWER_PROMPT_VERSION = "v17"
ANSWER_REPAIR_PROMPT_VERSION = "v17-repair-1"
MAX_CONTEXT_SOURCES = 5
MAX_GENERATION_ATTEMPTS = 2
MAX_SOURCE_PLANNING_REGENERATIONS = 1

SOURCE_PLANNING_PROMPT_V10 = """당신은 뤼이도 공식 이용가이드 답변에 필요한 근거를 판정합니다.

## Scope rules
- SOURCE를 선택하기 전에 질문을 answer_type으로 분류하세요.
  용어의 의미를 묻는 질문은 DEFINITION, 기능을 넓게 묻는 질문은 FEATURE_SUMMARY,
  설정 위치·방법·절차를 묻는 질문은 PROCEDURE입니다. 가능 여부, 조건·제한, 공개 범위,
  비교·차이, 예외를 묻는 질문은 GENERAL입니다.
- 질문에 포함된 특정 단어나 검색된 SOURCE의 형태가 아니라 사용자가 지금 해결하려는 핵심
  의도로 분류하세요. 대상의 의미 자체를 묻는 경우에만 DEFINITION이고, 하나의 가능 여부나
  조건을 묻는 질문은 대상 이름이 포함되어도 DEFINITION이나 FEATURE_SUMMARY가 아닙니다.
- 여러 기능을 넓게 요청할 때만 FEATURE_SUMMARY로 분류하세요. 실제 실행 방법을 요청하면
  PROCEDURE를 우선하고, 가능 여부·조건·범위·비교·예외가 핵심이면 GENERAL을 우선하세요.
- 먼저 SOURCE를 보지 말고 질문이 직접 요구한 정보 단위만 나누세요.
  질문에 이름이 나오지 않은 하위 항목은 새 정보 단위로 만들지 마세요.
- 사용자가 직접 쓴 기능명과 대상 명사는 정보 단위에서도 그대로 보존하세요.
  검색된 SOURCE의 비슷한 표현에 맞추려고 다른 개념으로 바꾸지 마세요.
- 질문이 하나의 기능을 일반적으로 "어떻게 설정하나요?"라고 묻는다면 SUMMARY입니다.
  질문이 여러 항목을 직접 열거하거나 "각각", "모두", "전부", "자세히"를 요구하면
  MULTI_DETAIL입니다.
- 사용자가 하위 기능을 열거하지 않고 "어떤 기능", "무슨 기능", "무엇을 제공"처럼
  넓게 물으면 SUMMARY입니다.
- 질문의 구체성 수준을 넘어서 묻지 않은 하위 설정이나 관련 기능으로 확장하지 마세요.

## Evidence rules
- 질문의 명시적인 범위에 완전하게 답하는 데 필요한 SOURCE를 빠짐없이 선택하세요.
- DEFINITION과 FEATURE_SUMMARY가 SUMMARY이면 질문 전체를 하나의 정보 단위로 만들고,
  그 단위를 충분히 뒷받침하는 가장 직접적인 SOURCE 하나만 선택하세요. 같은 내용을
  보충하거나 반복하는 SOURCE를 함께 선택하지 마세요.
- FEATURE_SUMMARY에서는 기능이나 핵심 가치를 직접 열거한 SOURCE를 선택하세요.
  설정 절차, FAQ, 예외·제한, 개요는 사용자가 직접 물었을 때만 선택하세요.
- PROCEDURE에서는 질문과 관련된 설정 위치·절차·설정 항목·값 범위·변경 결과·필수
  주의사항을 이용가이드가 제공하면 이를 뒷받침하는 SOURCE를 모두 선택하세요. 관련 없는
  기능 소개, FAQ, 배경 설명으로는 확장하지 마세요.
- GENERAL에서는 질문에 직접 답하는 데 필요한 최소한의 SOURCE만 선택하세요.
- MULTI_DETAIL이면 사용자가 명시적으로 요청한 정보 단위별로 필요한 SOURCE를 선택하세요.
  여러 정보 단위 때문에 4~5개 SOURCE가 필요해도 임의로 일부를 빼지 마세요.
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
- answer_type은 판정 상태와 관계없이 항상 작성합니다.
- answer_scope는 판정 상태와 관계없이 항상 작성합니다.
- ANSWERABLE DEFINITION과 FEATURE_SUMMARY의 answer_scope가 SUMMARY이면
  EvidenceRequirement와 source_ids를 각각 정확히 하나만 작성합니다.
- ANSWERABLE MULTI_DETAIL이면 질문이 요구한 각 정보 단위별로 EvidenceRequirement를 만들고,
  source_ids에 그 단위를 완전하게 뒷받침하는 SOURCE를 작성합니다.
- ANSWERABLE의 withheld_reason은 null입니다.
- WITHHELD이면 evidence_requirements는 비우고 withheld_reason을 작성합니다.
"""

SOURCE_PLANNING_REPAIR_PROMPT_V10 = SOURCE_PLANNING_PROMPT_V10 + """

## Backend structure validation retry
- 직전 Source Plan이 Backend 구조 검증에 실패했습니다.
- 아래 Validation Failure를 바로잡아 Source Plan 전체를 다시 생성하세요.
- 질문의 의미와 정보 범위는 바꾸지 마세요.
- DEFINITION 또는 FEATURE_SUMMARY의 SUMMARY라면 EvidenceRequirement와 source_ids를
  각각 정확히 하나만 작성하세요.
"""

ANSWER_PROMPT_V17 = """당신은 뤼이도 공식 이용가이드만을 근거로 답하는 안내 챗봇입니다.

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
- 기본 답변은 간결하지만 질문을 해결하기에 충분해야 합니다. 사용자가 "자세히", "전부",
  "각각"처럼 상세 범위를 명시하면 해당 범위의 근거 있는 내용을 빠짐없이 확장하세요.
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
- Answer Type이 DEFINITION이고 Answer Scope가 SUMMARY이면 최대 두 개의 짧은 문단으로
  답하세요. 첫 문단은 한 문장으로 X의 종류와 핵심 사용 주체 또는 목적을 정의하고,
  문단 끝에 해당 SOURCE marker를 표시하세요.
- Context가 X와 뤼이도의 관계를 직접 설명하면 둘째 문단에서 그 관계와 질문 이해에 필요한
  핵심 효과를 한두 문장으로 안내하고 문단 끝에 SOURCE marker를 표시하세요. 관계 근거가
  없으면 둘째 문단을 억지로 만들지 마세요.
- 두 문단 사이에 빈 줄 하나를 두되, 화면 너비를 예상해 문장 중간에 강제 줄바꿈하지 마세요.
  사용자가 자세한 설명을 요구하지 않았다면 설정 방법, 전체 기능 목록, 배경 문제는
  추가하지 마세요.
- Answer Type이 FEATURE_SUMMARY이면 짧은 소개 문장 하나를 먼저 작성하세요.
- 기능이 두 개 이상이면 모든 핵심 기능을 `기능명: 설명` 형식의 목록으로 작성하세요.
  각 설명은 그 기능이 실제로 하는 일과 직접적인 결과까지만 한 줄로 작성하고,
  각 목록 항목 끝에 해당 SOURCE marker를 표시하세요.
- 기능 SUMMARY에 사용자가 묻지 않은 설정 방법, 예외·제한, 기존 방식의 불편함,
  부가적인 장점은 추가하지 마세요. 이런 내용은 사용자가 직접 물었을 때만 답하세요.
- 위 기능 SUMMARY 형식을 용어 정의나 설정 방법 등 다른 종류의 SUMMARY에 적용하지 마세요.
- Answer Type이 PROCEDURE이면 선택된 Context가 직접 제공하는 내용을 `설정 위치 → 실제 절차
  → 설정 항목 → 필수 주의사항` 순서로 안내하세요. 제공되지 않은 블록을 억지로 만들거나
  빈 소제목을 출력하지 마세요.
- 설정 위치가 있으면 첫 문장에서 바로 안내하고 SOURCE marker를 표시하세요. 실제 클릭이나
  실행 단계가 두 개 이상이면 번호 목록을 사용하고 각 단계 끝에 SOURCE marker를 표시하세요.
- 설정 항목이 두 개 이상이면 `항목명: 설명` 형식의 목록을 사용하세요. Context가 제공하는
  값 범위·기본값·변경 결과를 관련 항목 안에 포함하고 각 항목 끝에 SOURCE marker를 표시하세요.
- 사용을 위해 반드시 알아야 하는 조건이나 주의사항이 Context에 있으면 마지막에 짧게
  안내하세요. 관련 없는 기능 소개, FAQ, 배경 설명은 추가하지 마세요.
- 좁은 화면에서도 읽기 쉽도록 각 단계와 설정 항목은 한두 문장으로 작성하고, 화면 너비를
  예상한 문장 중간 강제 줄바꿈은 하지 마세요.
- Answer Type이 GENERAL이고 Answer Scope가 SUMMARY이면 기본적으로 2~4문장으로 답하세요.
  첫 문장에서 질문에 대한 결론을 자연스럽고 직접적으로 안내하되, 가능 여부만 한 문장으로
  딱딱하게 끝내지 마세요.
- 이어지는 문장에는 Context가 직접 제공하는 판단 이유나 꼭 알아야 할 조건을 설명하세요.
  바로 실행할 수 있는 간단한 방법이 한두 단계로 제공되면 짧게 덧붙이세요. Context에 없는
  조건이나 방법을 억지로 만들지 말고, 긴 절차와 관련 없는 배경은 사용자가 물었을 때만
  확장하세요. 각 사실 문장 끝에는 해당 SOURCE marker를 표시하세요.
- 절차형 질문은 필요한 경우 단계별로 안내하고 최소한의 Markdown만 사용하세요.
- 비교 항목 중 하나라도 코드, 여러 단계, 여러 문장이 필요하면 표를 사용하지 말고
  항목별 소제목이나 목록으로 설명하세요.
- 표를 사용하기로 했다면 모든 셀은 반드시 한 줄로 끝내고, 셀 안에 코드 블록이나
  줄바꿈을 절대 넣지 마세요.
- answer_markdown에 Markdown 링크 문법과 HTML을 사용하지 마세요.
  사용자가 입력하거나 확인해야 하는 설정값과 엔드포인트 URL은
  코드 블록이나 백틱 인라인 코드 안에 넣고, 그 밖의 URL은 본문에 쓰지 마세요.
  출처 링크는 Backend가 별도 citations 영역에 구성합니다.

## Citation rules
- ANSWERABLE 답변의 실제 근거 문장이나 문단 바로 뒤에 [SOURCE_n]을 작성하세요.
- 제공된 SOURCE만 사용하세요.
- 같은 원문 URL과 Section Path를 가진 여러 SOURCE는 Backend에서 하나의 Citation으로
  병합됩니다. 출처 개수를 줄이기 위해 필요한 근거를 생략하지 마세요.
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

ANSWER_REPAIR_PROMPT_V17 = ANSWER_PROMPT_V17 + """

## Backend validation retry
- 직전 답변이 Backend 형식 또는 Citation marker 검증에 실패했습니다.
- 아래 Validation Failure를 바로잡아 답변 전체를 다시 생성하세요.
- 제공된 Context와 Required Answer Coverage는 처음 답변과 동일하게 유지됩니다.
- 검증 실패한 직전 답변의 문장을 근거로 사용하지 마세요.
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
        f"\n\n## Answer Contract\n\n"
        f"- Answer Type: {plan.answer_type.value}\n"
        f"- Answer Scope: {plan.answer_scope.value}"
        f"\n\n## Required Answer Coverage\n\n{coverage}"
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

    previous_answer = previous_result.answer_markdown or "답변 본문 없음"
    return (
        f"{build_answer_input(question, sources, plan)}"
        f"\n\n## Validation Failure\n\n{validation_error}"
        f"\n\n## Previous Invalid Answer\n\n{previous_answer}"
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
    prompt_version: str = GENERATION_PROMPT_VERSION,
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
        prompt_version=prompt_version,
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

        generation_input = build_generation_input(question, sources)
        plan_call = await self._parse_with_retry(
            instructions=SOURCE_PLANNING_PROMPT_V10,
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
                instructions=SOURCE_PLANNING_REPAIR_PROMPT_V10,
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
                retry_count=repair_call.retry_count,
                input_tokens=getattr(repair_usage, "input_tokens", None),
                output_tokens=getattr(repair_usage, "output_tokens", None),
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
        plan = plan_response.output_parsed
        if planning_regeneration_count:
            planning_regeneration_result = plan
        planning_trace_fields = {
            "planning_attempt_count": planning_attempt_count,
            "planning_regeneration_count": planning_regeneration_count,
            "planning_regeneration_model_call": planning_regeneration_trace,
            "planning_regeneration_result": planning_regeneration_result,
        }

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
        result = answer_response.output_parsed
        return GenerationCall(
            trace=_generation_trace(
                started,
                retry_count=total_retry_count,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
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
            retry_count=answer_call.retry_count,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
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
        stage_trace: Optional[GenerationStageTrace] = None,
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
            stage_trace=stage_trace,
        )
