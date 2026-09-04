import unittest
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.chat.schema import (
    ChatAnswer,
    ChatCitation,
    ChatCompletedResponse,
    ChatResponseStatus,
)
from app.database.models import (
    AnswerCitation,
    AnswerStatus,
    ExecutionStatus,
    Feedback,
    FeedbackRating,
    ModelCall,
    ModelCallPurpose,
    RagRun,
    RetrievalResultRow,
    RetrieverType,
)
from app.main import create_app
from app.chat.dependencies import get_rag_log_store
from app.answering.service import WITHHELD_RESPONSES
from app.chat.log_store import RagLogStore, RagRunDetail
from app.answering.models import FinalWithheldReason


VIEW_LOGGER = "app.chat.rag_run_view"

ANSWER_MARKDOWN = "멤버를 초대할 수 있습니다. [1]"
DOCUMENT_TITLE = "멤버 관리"
SOURCE_URL = "https://docs.riido.io/member/invite"
NODE_PATH = "멤버 관리 > 워크스페이스 > 멤버 초대"


@asynccontextmanager
async def test_lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


class RagRunApiTest(unittest.TestCase):
    """저장된 턴 결과 조회(polling) endpoint를 검증한다."""

    def setUp(self) -> None:
        self.conversation_id = uuid.uuid4()
        self.rag_run_id = uuid.uuid4()
        self.log_store = AsyncMock(spec=RagLogStore)

        with patch("app.main.lifespan", test_lifespan):
            self.app = create_app()

        self.app.dependency_overrides[get_rag_log_store] = lambda: self.log_store

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()

    # ------------------------------------------------------------------
    # 상태별 응답
    # ------------------------------------------------------------------

    def test_processing_returns_status_and_ids_only(self) -> None:
        self._detail_returns(self._run(AnswerStatus.PROCESSING))

        response = self._get()

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "status": "PROCESSING",
                "conversationId": str(self.conversation_id),
                "ragRunId": str(self.rag_run_id),
            },
            response.json(),
        )

    def test_completed_matches_sync_response_shape(self) -> None:
        self._detail_returns(
            self._completed_run(),
            citations=[self._citation()],
        )

        response = self._get()

        expected = ChatCompletedResponse(
            status=ChatResponseStatus.COMPLETED,
            conversation_id=self.conversation_id,
            rag_run_id=self.rag_run_id,
            answer=ChatAnswer(answer_markdown=ANSWER_MARKDOWN),
            citations=[
                ChatCitation(
                    citation_number=1,
                    document_title=DOCUMENT_TITLE,
                    section_path=["워크스페이스", "멤버 초대"],
                    source_url=SOURCE_URL,
                )
            ],
        ).model_dump(mode="json", by_alias=True)

        self.assertEqual(200, response.status_code)
        self.assertEqual(expected, response.json())

    def test_completed_strips_document_title_prefix_from_section_path(
        self,
    ) -> None:
        self._detail_returns(
            self._completed_run(),
            citations=[self._citation(node_path=NODE_PATH)],
        )

        response = self._get()

        self.assertEqual(
            ["워크스페이스", "멤버 초대"],
            response.json()["citations"][0]["sectionPath"],
        )

    def test_completed_keeps_section_path_without_title_prefix(self) -> None:
        self._detail_returns(
            self._completed_run(),
            citations=[self._citation(node_path="워크스페이스 > 멤버 초대")],
        )

        response = self._get()

        self.assertEqual(
            ["워크스페이스", "멤버 초대"],
            response.json()["citations"][0]["sectionPath"],
        )

    def test_withheld_returns_reason_message_from_lookup(self) -> None:
        for reason in FinalWithheldReason:
            with self.subTest(reason=reason.value):
                self.log_store.reset_mock()
                self._detail_returns(
                    self._run(
                        AnswerStatus.WITHHELD,
                        withheld_reason_code=reason.value,
                    )
                )

                body = self._get().json()

                self.assertEqual("WITHHELD", body["status"])
                self.assertEqual(reason.value, body["withheld"]["reasonCode"])
                self.assertEqual(
                    WITHHELD_RESPONSES[reason],
                    body["withheld"]["message"],
                )
                self.assertEqual([], body["citations"])
                self.assertIsNone(body["answer"])

    def test_stored_error_returns_200_with_error_body(self) -> None:
        self._detail_returns(
            self._run(AnswerStatus.ERROR, error_code="UPSTREAM_ERROR")
        )

        response = self._get()

        self.assertEqual(200, response.status_code)
        self.assertEqual("ERROR", response.json()["status"])
        self.assertEqual("INTERNAL_ERROR", response.json()["error"]["code"])
        self.assertEqual(str(self.rag_run_id), response.json()["ragRunId"])

    # ------------------------------------------------------------------
    # 오류
    # ------------------------------------------------------------------

    def test_unknown_rag_run_returns_404(self) -> None:
        self.log_store.get_rag_run_detail.return_value = None

        response = self._get()

        self.assertEqual(404, response.status_code)
        self.assertEqual(
            {"code": "NOT_FOUND", "message": "존재하지 않는 답변입니다."},
            response.json(),
        )

    def test_malformed_rag_run_id_returns_422(self) -> None:
        with TestClient(self.app) as client:
            response = client.get("/api/chat/not-a-uuid")

        self.assertEqual(422, response.status_code)
        self.log_store.get_rag_run_detail.assert_not_awaited()

    # ------------------------------------------------------------------
    # 방어 — 미지 상태와 빈 스냅샷
    # ------------------------------------------------------------------

    def test_cancelled_status_returns_internal_error(self) -> None:
        self._detail_returns(self._run(AnswerStatus.CANCELLED))

        with self.assertLogs(VIEW_LOGGER, level="WARNING") as logs:
            response = self._get()

        self.assertEqual(200, response.status_code)
        self.assertEqual("ERROR", response.json()["status"])
        self.assertEqual("INTERNAL_ERROR", response.json()["error"]["code"])
        self.assertIn("외부 표현이 없는 답변 상태", logs.output[0])

    def test_null_snapshot_fields_fall_back_to_empty_string(self) -> None:
        self._detail_returns(
            self._completed_run(),
            citations=[
                self._citation(
                    title=None,
                    node_path=None,
                    url=None,
                )
            ],
        )

        with self.assertLogs(VIEW_LOGGER, level="WARNING") as logs:
            response = self._get()

        citation = response.json()["citations"][0]
        self.assertEqual(200, response.status_code)
        self.assertEqual("", citation["documentTitle"])
        self.assertEqual("", citation["sourceUrl"])
        self.assertEqual([], citation["sectionPath"])
        # 번호 연속성을 지키려고 인용 행 자체는 남긴다
        self.assertEqual(1, citation["citationNumber"])
        self.assertIn("인용 스냅샷이 비어 있어", logs.output[0])

    def test_completed_without_answer_content_degrades_to_internal_error(
        self,
    ) -> None:
        self._detail_returns(
            self._run(AnswerStatus.COMPLETED, answer_content=None),
            citations=[self._citation()],
        )

        with self.assertLogs(VIEW_LOGGER, level="WARNING") as logs:
            response = self._get()

        self.assertEqual(200, response.status_code)
        self.assertEqual("INTERNAL_ERROR", response.json()["error"]["code"])
        self.assertIn("답변 본문이 없어", logs.output[0])

    def test_withheld_with_unknown_reason_degrades_to_internal_error(
        self,
    ) -> None:
        self._detail_returns(
            self._run(AnswerStatus.WITHHELD, withheld_reason_code=None)
        )

        with self.assertLogs(VIEW_LOGGER, level="WARNING") as logs:
            response = self._get()

        self.assertEqual(200, response.status_code)
        self.assertEqual("INTERNAL_ERROR", response.json()["error"]["code"])
        self.assertIn("외부 표현이 없는 보류 사유", logs.output[0])

    # ------------------------------------------------------------------
    # 계약 보호
    # ------------------------------------------------------------------

    def test_internal_records_are_not_exposed(self) -> None:
        self._detail_returns(
            self._run(
                AnswerStatus.ERROR,
                error_code="CITATION_VALIDATION_ERROR",
            ),
            retrieval_results=[
                RetrievalResultRow(
                    rag_run_id=self.rag_run_id,
                    chunk_id=41,
                    retriever_type=RetrieverType.BM25,
                    raw_score=7.5,
                    retriever_rank=1,
                    fused_rank=1,
                    fused_score=0.0328,
                    selected_as_evidence=True,
                    latency_ms=987654,
                )
            ],
            model_calls=[
                ModelCall(
                    rag_run_id=self.rag_run_id,
                    purpose=ModelCallPurpose.GENERATION,
                    provider="openai",
                    model_name="secret-model-name",
                    status=ExecutionStatus.SUCCESS,
                    latency_ms=987654,
                )
            ],
            feedback=Feedback(
                rag_run_id=self.rag_run_id,
                rating=FeedbackRating.GOOD,
            ),
        )

        body = self._get().text

        for leaked in (
            "CITATION_VALIDATION_ERROR",
            "BM25",
            "secret-model-name",
            "987654",
            "GOOD",
            "selected",
            "fused",
            "retriever",
            "modelCall",
            "feedback",
        ):
            self.assertNotIn(leaked, body)

    def test_sync_chat_schema_is_unchanged(self) -> None:
        spec = self.app.openapi()
        sync_schema = spec["paths"]["/api/chat"]["post"]["responses"]["200"][
            "content"
        ]["application/json"]["schema"]

        # 동기 응답은 여전히 마감된 3개 상태만 노출해야 한다
        self.assertEqual(
            {
                "COMPLETED": "#/components/schemas/ChatCompletedResponse",
                "WITHHELD": "#/components/schemas/ChatWithheldResponse",
                "ERROR": "#/components/schemas/ChatErrorResponse",
            },
            sync_schema["discriminator"]["mapping"],
        )
        # 공유 enum에 PROCESSING이 새로 들어가지 않았는지 확인한다
        self.assertEqual(
            ["COMPLETED", "WITHHELD", "ERROR"],
            [member.value for member in ChatResponseStatus],
        )

    # ------------------------------------------------------------------

    def _run(self, status: AnswerStatus, **overrides) -> RagRun:
        fields = {
            "id": self.rag_run_id,
            "conversation_id": self.conversation_id,
            "turn_no": 1,
            "index_version_id": 1,
            "user_query": "멤버를 어떻게 초대하나요?",
            "context_strategy": "NEW_TOPIC",
            "status": status,
        }
        fields.update(overrides)
        return RagRun(**fields)

    def _completed_run(self) -> RagRun:
        return self._run(
            AnswerStatus.COMPLETED,
            answer_content=ANSWER_MARKDOWN,
            citation_validated=True,
        )

    def _citation(
        self,
        order: int = 1,
        title=DOCUMENT_TITLE,
        node_path=NODE_PATH,
        url=SOURCE_URL,
    ) -> AnswerCitation:
        return AnswerCitation(
            rag_run_id=self.rag_run_id,
            chunk_id=41,
            document_version_id=7,
            citation_order=order,
            document_title_snapshot=title,
            node_path_snapshot=node_path,
            source_uri_snapshot=url,
        )

    def _detail_returns(self, run: RagRun, **parts) -> None:
        self.log_store.get_rag_run_detail.return_value = RagRunDetail(
            run=run,
            retrieval_results=list(parts.get("retrieval_results", [])),
            model_calls=list(parts.get("model_calls", [])),
            citations=list(parts.get("citations", [])),
            feedback=parts.get("feedback"),
        )

    def _get(self):
        with TestClient(self.app) as client:
            return client.get(f"/api/chat/{self.rag_run_id}")


if __name__ == "__main__":
    unittest.main()
