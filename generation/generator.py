"""Hybrid 검색 결과를 근거로 OpenAI 답변을 생성한다."""

from typing import List, Optional, Sequence

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from app.core.config import get_settings
from generation.models import (
    GenerationContextSource,
    GenerationResult,
)
from retrieval.models import HybridRetrievalResult


OPENAI_GENERATION_MODEL = "gpt-5.4-mini"
MAX_CONTEXT_SOURCES = 5

PROMPT_V1 = """당신은 뤼이도 공식 이용가이드만을 근거로 답하는 안내 챗봇입니다.

## Grounding rules
- 제공된 Context에 명시된 사실만 사용하세요.
- 일반 지식으로 보완하거나 정책, 조건, 제한, 가능 여부를 추측하지 마세요.
- 문장을 자연스럽게 재구성하거나 Markdown으로 구조화할 수 있지만 새로운 사실을 추가하지 마세요.

## Answerability rules
- 관련 Context가 있다는 이유만으로 ANSWERABLE을 선택하지 마세요.
- 질문의 핵심을 Context가 직접 뒷받침할 때만 ANSWERABLE을 선택하세요.
- 근거가 부족하면 INSUFFICIENT_EVIDENCE, 질문이 모호하면 AMBIGUOUS_QUESTION,
  이용가이드 범위 밖이면 OUT_OF_SCOPE으로 WITHHELD를 선택하세요.

## Answer style
- 자연스러운 한국어 존댓말을 사용하세요.
- 첫 1~2문장에서 질문의 핵심부터 간결하게 답하세요.
- 절차형 질문은 필요한 경우 단계별로 안내하고 최소한의 Markdown만 사용하세요.

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


def _is_transient_openai_error(error: Exception) -> bool:
    """연결 문제, timeout, rate limit과 서버 오류만 일시적 오류로 본다."""

    if isinstance(error, APIConnectionError):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in (408, 409, 429) or error.status_code >= 500
    return False


class OpenAIGenerator:
    """OpenAI Responses API로 single-pass 답변을 생성한다."""

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

        generation_input = build_generation_input(question, sources)

        for attempt in range(2):
            try:
                response = await self._client.responses.parse(
                    model=OPENAI_GENERATION_MODEL,
                    instructions=PROMPT_V1,
                    input=generation_input,
                    text_format=GenerationResult,
                )
                result = response.output_parsed
                if result is None:
                    raise RuntimeError(
                        "OpenAI Generation 응답에 Structured Output이 없습니다."
                    )
                return result
            except Exception as error:
                if attempt == 0 and _is_transient_openai_error(error):
                    continue
                raise

        raise RuntimeError("OpenAI Generation 호출이 완료되지 않았습니다.")
