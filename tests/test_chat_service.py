import unittest
from unittest.mock import AsyncMock

from app.api.chat_schema import (
    ChatCompletedResponse,
    ChatErrorCode,
    ChatErrorResponse,
    ChatResponseStatus,
    ChatWithheldReasonCode,
    ChatWithheldResponse,
)
from app.rag.chat_service import ChatService
from app.rag.generation_service import WITHHELD_RESPONSES, GenerationService
from generation.models import (
    Citation,
    FinalAnswerStatus,
    FinalGenerationResult,
    FinalWithheldReason,
)
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import HybridRetrievalResult, RetrievalChunk


class ChatServiceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.retriever = AsyncMock(spec=HybridRetriever)
        self.generation_service = AsyncMock(spec=GenerationService)
        self.service = ChatService(
            retriever=self.retriever,
            generation_service=self.generation_service,
        )

    async def test_maps_completed_result_and_preserves_citation_order(self) -> None:
        retrieval_results = [self._retrieval_result(1), self._retrieval_result(2)]
        self.retriever.search.return_value = retrieval_results
        self.generation_service.generate_answer.return_value = FinalGenerationResult(
            status=FinalAnswerStatus.COMPLETED,
            answer_markdown="두 번째 근거입니다. [1] 첫 번째 근거입니다. [2]",
            citations=(
                self._citation(2, citation_number=1),
                self._citation(1, citation_number=2),
            ),
        )

        response = await self.service.answer_question("멤버를 어떻게 초대하나요?")

        self.assertIsInstance(response, ChatCompletedResponse)
        self.assertEqual(ChatResponseStatus.COMPLETED, response.status)
        self.assertEqual(
            "두 번째 근거입니다. [1] 첫 번째 근거입니다. [2]",
            response.answer.answer_markdown,
        )
        self.assertEqual(
            [1, 2],
            [citation.citation_number for citation in response.citations],
        )
        self.assertEqual(
            ["문서 2", "문서 1"],
            [citation.document_title for citation in response.citations],
        )
        self.assertEqual(
            [["문서 2", "섹션 2"], ["문서 1", "섹션 1"]],
            [citation.section_path for citation in response.citations],
        )
        self.assertEqual(
            ["https://docs.riido.io/2", "https://docs.riido.io/1"],
            [citation.source_url for citation in response.citations],
        )
        self.retriever.search.assert_awaited_once_with(
            "멤버를 어떻게 초대하나요?"
        )
        self.generation_service.generate_answer.assert_awaited_once_with(
            "멤버를 어떻게 초대하나요?",
            retrieval_results,
        )

    async def test_maps_all_withheld_reasons_and_messages(self) -> None:
        self.retriever.search.return_value = []

        for reason in FinalWithheldReason:
            with self.subTest(reason=reason):
                self.generation_service.generate_answer.return_value = (
                    FinalGenerationResult(
                        status=FinalAnswerStatus.WITHHELD,
                        answer_markdown=WITHHELD_RESPONSES[reason],
                        citations=(),
                        withheld_reason=reason,
                    )
                )

                response = await self.service.answer_question("질문")

                self.assertIsInstance(response, ChatWithheldResponse)
                self.assertEqual(ChatResponseStatus.WITHHELD, response.status)
                self.assertIsNone(response.answer)
                self.assertEqual(
                    ChatWithheldReasonCode(reason.value),
                    response.withheld.reason_code,
                )
                self.assertEqual(
                    WITHHELD_RESPONSES[reason],
                    response.withheld.message,
                )
                self.assertEqual([], response.citations)

    async def test_maps_internal_error_without_exposing_internal_code(self) -> None:
        self.retriever.search.return_value = []
        self.generation_service.generate_answer.return_value = FinalGenerationResult(
            status=FinalAnswerStatus.ERROR,
            answer_markdown=None,
            citations=(),
            error_code="UPSTREAM_ERROR",
        )

        response = await self.service.answer_question("질문")

        self.assertIsInstance(response, ChatErrorResponse)
        self.assertEqual(ChatResponseStatus.ERROR, response.status)
        self.assertEqual(ChatErrorCode.INTERNAL_ERROR, response.error.code)
        self.assertEqual(
            "답변을 생성하는 중 오류가 발생했습니다.",
            response.error.message,
        )
        self.assertNotIn(
            "UPSTREAM_ERROR",
            str(response.model_dump(mode="json", by_alias=True)),
        )
        self.assertEqual([], response.citations)

    async def test_returns_safe_error_when_retriever_raises(self) -> None:
        self.retriever.search.side_effect = RuntimeError(
            "postgresql connection detail"
        )

        response = await self.service.answer_question("질문")

        self.assert_safe_error(response, "postgresql connection detail")
        self.generation_service.generate_answer.assert_not_awaited()

    async def test_returns_safe_error_when_generation_service_raises(self) -> None:
        self.retriever.search.return_value = []
        self.generation_service.generate_answer.side_effect = RuntimeError(
            "provider secret detail"
        )

        response = await self.service.answer_question("질문")

        self.assert_safe_error(response, "provider secret detail")

    def assert_safe_error(self, response: object, private_detail: str) -> None:
        self.assertIsInstance(response, ChatErrorResponse)
        assert isinstance(response, ChatErrorResponse)
        payload = response.model_dump(mode="json", by_alias=True)
        self.assertEqual(
            {
                "status": "ERROR",
                "answer": None,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "답변을 생성하는 중 오류가 발생했습니다.",
                },
                "citations": [],
            },
            payload,
        )
        self.assertNotIn(private_detail, str(payload))

    @staticmethod
    def _citation(index: int, citation_number: int) -> Citation:
        return Citation(
            citation_number=citation_number,
            document_title=f"문서 {index}",
            section_path=(f"문서 {index}", f"섹션 {index}"),
            source_url=f"https://docs.riido.io/{index}",
        )

    @staticmethod
    def _retrieval_result(index: int) -> HybridRetrievalResult:
        chunk = RetrievalChunk(
            document_id=f"document-{index}",
            section_id=f"section-{index}",
            document_title=f"문서 {index}",
            section_path=(f"문서 {index}", f"섹션 {index}"),
            source_url=f"https://docs.riido.io/{index}",
            category="guide",
            content=f"본문 {index}",
            chunk_id=index,
            document_version_id=100 + index,
            index_version_id=1,
        )
        return HybridRetrievalResult(
            chunk=chunk,
            rrf_score=0.1,
            final_rank=index,
            bm25_rank=index,
            vector_rank=index,
        )


if __name__ == "__main__":
    unittest.main()
