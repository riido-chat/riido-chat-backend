"""Chat HTTP DTO와 기존 RAG 파이프라인을 연결한다."""

from app.api.chat_schema import (
    ChatAnswer,
    ChatCitation,
    ChatCompletedResponse,
    ChatError,
    ChatErrorCode,
    ChatErrorResponse,
    ChatResponse,
    ChatResponseStatus,
    ChatWithheld,
    ChatWithheldReasonCode,
    ChatWithheldResponse,
)
from app.rag.generation_service import GenerationService
from generation.models import FinalAnswerStatus, FinalGenerationResult
from retrieval.hybrid_retriever import HybridRetriever


INTERNAL_ERROR_MESSAGE = "답변을 생성하는 중 오류가 발생했습니다."


def _internal_error_response() -> ChatErrorResponse:
    return ChatErrorResponse(
        status=ChatResponseStatus.ERROR,
        answer=None,
        error=ChatError(
            code=ChatErrorCode.INTERNAL_ERROR,
            message=INTERNAL_ERROR_MESSAGE,
        ),
        citations=[],
    )


def _to_chat_response(result: FinalGenerationResult) -> ChatResponse:
    if result.status == FinalAnswerStatus.COMPLETED:
        if result.answer_markdown is None:
            raise ValueError("COMPLETED 결과에 answer_markdown이 없습니다.")

        return ChatCompletedResponse(
            status=ChatResponseStatus.COMPLETED,
            answer=ChatAnswer(answer_markdown=result.answer_markdown),
            citations=[
                ChatCitation(
                    citation_number=citation.citation_number,
                    document_title=citation.document_title,
                    section_path=list(citation.section_path),
                    source_url=citation.source_url,
                )
                for citation in result.citations
            ],
        )

    if result.status == FinalAnswerStatus.WITHHELD:
        if result.withheld_reason is None or result.answer_markdown is None:
            raise ValueError("WITHHELD 결과에 보류 사유 또는 안내 문구가 없습니다.")

        return ChatWithheldResponse(
            status=ChatResponseStatus.WITHHELD,
            answer=None,
            withheld=ChatWithheld(
                reason_code=ChatWithheldReasonCode(result.withheld_reason.value),
                message=result.answer_markdown,
            ),
            citations=[],
        )

    if result.status == FinalAnswerStatus.ERROR:
        return _internal_error_response()

    raise ValueError(f"지원하지 않는 최종 답변 상태입니다: {result.status}")


class ChatService:
    """Hybrid Retrieval부터 Chat HTTP 응답 변환까지 연결한다."""

    def __init__(
        self,
        retriever: HybridRetriever,
        generation_service: GenerationService,
    ) -> None:
        self._retriever = retriever
        self._generation_service = generation_service

    async def answer_question(self, question: str) -> ChatResponse:
        """질문을 검색·생성 파이프라인에 전달하고 외부 DTO로 변환한다."""

        try:
            retrieval_results = await self._retriever.search(question)
            generation_result = await self._generation_service.generate_answer(
                question,
                retrieval_results,
            )
            return _to_chat_response(generation_result)
        except Exception:
            return _internal_error_response()
