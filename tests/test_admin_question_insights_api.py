"""질문 로그 콘솔 조회 HTTP 계약 테스트.

집계 값의 규칙은 test_admin_question_insights(_db) 가 고정한다. 여기서는 경로, 쿼리 이름,
기본값, camelCase 응답 모양, 404·422 본문만 본다. 서비스는 가짜로 바꾼다.
끝의 DB 테스트는 실제 서비스와 세션을 거친 응답 직렬화(askedAt, uuid)와 BIGINT 범위 밖 id 만 본다.
"""

import asyncio
import unittest
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.admin.dependencies import get_question_insight_service
from app.admin.question_insights.schema import (
    AnswerStatus,
    ApplyStatus,
    SubproblemPresence,
    WithheldReason,
)
from app.admin.question_insights.service import (
    CanonicalAnswerView,
    DocumentDetail,
    DocumentList,
    DocumentRef,
    DocumentRow,
    DocumentSummary,
    FrequentSubproblem,
    QuestionDashboard,
    QuestionInsightService,
    QuestionListFilters,
    QuestionPage,
    QuestionRow,
    SubproblemDetail,
    SubproblemNotFoundError,
    SubproblemRow,
    WithheldDocument,
    WithheldReasonCounts,
)
from app.chat.log_store import RagLogStore
from app.core.config import get_settings
from app.database.models import (
    AnswerStatus as RunStatus,
)
from app.database.models import (
    AttributionSource,
    CanonicalAnswer,
    CanonicalAnswerApproval,
    CanonicalAnswerOrigin,
    ClassificationDecision,
    ClassificationRun,
    ClassificationRunKind,
    ContextStrategy,
    QuestionClassification,
    QuestionSubproblemStatus,
    RagRun,
)
from app.database.session import get_db_session
from app.document.ingestion_service import (
    DocumentGroupNotFoundError,
    DocumentNotFoundError,
)
from app.main import create_app
from tests.test_question_grouping_readers_db import _available, _Seed


GROUP_ID = 3
DOCUMENT_ID = 12
SUBPROBLEM_ID = uuid.UUID("5b0c1d2e-3f40-4a51-8b62-7c83d94ea5f6")
RAG_RUN_ID = uuid.UUID("0e1f2a3b-4c5d-4e6f-8a9b-0c1d2e3f4a5b")
ASKED_AT = datetime(2026, 9, 15, 3, 12, tzinfo=timezone.utc)
BASE_PATH = f"/api/admin/document-groups/{GROUP_ID}/question-log"
OUT_OF_BIGINT = "9223372036854775808"


@asynccontextmanager
async def test_lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


class QuestionLogApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = AsyncMock(spec=QuestionInsightService)
        with patch("app.main.lifespan", test_lifespan):
            self.app = create_app()
        self.app.dependency_overrides[get_question_insight_service] = (
            lambda: self.service
        )
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()
        self.client.close()

    def assertAdminError(self, response, status_code: int, code: str) -> None:
        self.assertEqual(status_code, response.status_code, response.text)
        body = response.json()
        self.assertEqual({"code", "message"}, set(body))
        self.assertEqual(code, body["code"])
        self.assertTrue(body["message"])

    # ------------------------------------------------------------------
    # 응답 모양
    # ------------------------------------------------------------------

    def test_dashboard_shape(self) -> None:
        self.service.get_dashboard.return_value = QuestionDashboard(
            question_count=96,
            unanswerable_count=14,
            withheld_reason_counts=WithheldReasonCounts(
                insufficient_evidence=5,
                ambiguous_question=4,
                out_of_scope=3,
                unverifiable_answer=2,
            ),
            frequent_subproblems=[
                FrequentSubproblem(SUBPROBLEM_ID, "세부 문제 가", DOCUMENT_ID, "문서 가", 8),
                FrequentSubproblem(SUBPROBLEM_ID, "문서 밖 세부 문제", None, None, 1),
            ],
            withheld_documents=[WithheldDocument(7, "문서 나", 3)],
        )

        response = self.client.get(f"{BASE_PATH}/dashboard")

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "questionCount": 96,
                "unanswerableCount": 14,
                "withheldReasonCounts": {
                    "insufficientEvidence": 5,
                    "ambiguousQuestion": 4,
                    "outOfScope": 3,
                    "unverifiableAnswer": 2,
                },
                "frequentSubproblems": [
                    {
                        "subproblemId": str(SUBPROBLEM_ID),
                        "name": "세부 문제 가",
                        "documentId": DOCUMENT_ID,
                        "documentTitle": "문서 가",
                        "questionCount": 8,
                    },
                    {
                        "subproblemId": str(SUBPROBLEM_ID),
                        "name": "문서 밖 세부 문제",
                        "documentId": None,
                        "documentTitle": None,
                        "questionCount": 1,
                    },
                ],
                "withheldDocuments": [
                    {"documentId": 7, "documentTitle": "문서 나", "insufficientEvidenceCount": 3}
                ],
            },
            response.json(),
        )
        self.service.get_dashboard.assert_awaited_once_with(GROUP_ID)

    def test_documents_shape(self) -> None:
        self.service.list_documents.return_value = DocumentList(
            items=[DocumentRow(7, "문서 나", 20, 3, 3)],
            no_document_question_count=2,
            unclassified_question_count=5,
        )

        response = self.client.get(f"{BASE_PATH}/documents")

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "items": [
                    {
                        "documentId": 7,
                        "documentTitle": "문서 나",
                        "questionCount": 20,
                        "withheldCount": 3,
                        "subproblemCount": 3,
                    }
                ],
                "noDocumentQuestionCount": 2,
                "unclassifiedQuestionCount": 5,
            },
            response.json(),
        )
        self.service.list_documents.assert_awaited_once_with(GROUP_ID)

    def test_document_detail_shape(self) -> None:
        self.service.get_document_detail.return_value = DocumentDetail(
            document=DocumentRef(DOCUMENT_ID, "문서 가"),
            summary=DocumentSummary(16, 2, 5, 2),
            subproblems=[
                SubproblemRow(SUBPROBLEM_ID, "세부 문제 가", 8, "문서 가 > 절 하나", ApplyStatus.APPLIED),
                SubproblemRow(SUBPROBLEM_ID, "세부 문제 나", 0, None, ApplyStatus.NEEDS_CANONICAL),
            ],
        )

        response = self.client.get(f"{BASE_PATH}/documents/{DOCUMENT_ID}")

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "document": {"documentId": DOCUMENT_ID, "documentTitle": "문서 가"},
                "summary": {
                    "questionCount": 16,
                    "insufficientEvidenceCount": 2,
                    "cachedAnswerCount": 5,
                    "subproblemCount": 2,
                },
                "subproblems": [
                    {
                        "subproblemId": str(SUBPROBLEM_ID),
                        "name": "세부 문제 가",
                        "questionCount": 8,
                        "sourceSection": "문서 가 > 절 하나",
                        "applyStatus": "APPLIED",
                    },
                    {
                        "subproblemId": str(SUBPROBLEM_ID),
                        "name": "세부 문제 나",
                        "questionCount": 0,
                        "sourceSection": None,
                        "applyStatus": "NEEDS_CANONICAL",
                    },
                ],
            },
            response.json(),
        )
        self.service.get_document_detail.assert_awaited_once_with(GROUP_ID, DOCUMENT_ID)

    def test_document_full_detail_shape(self) -> None:
        self.service.get_document_full_detail.return_value = DocumentDetail(
            document=DocumentRef(DOCUMENT_ID, "문서 가"),
            summary=DocumentSummary(16, 2, 5, 2),
            subproblems=[
                SubproblemRow(
                    SUBPROBLEM_ID,
                    "세부 문제 가",
                    8,
                    "문서 가 > 절 하나",
                    ApplyStatus.APPLIED,
                    ("질문 범위 하나", "질문 범위 둘"),
                    ("다른 의도",),
                    CanonicalAnswerView("정본 본문 [1]", ["규칙 하나"]),
                ),
                SubproblemRow(
                    SUBPROBLEM_ID,
                    "세부 문제 나",
                    0,
                    None,
                    ApplyStatus.NEEDS_CANONICAL,
                    ("질문 범위 셋",),
                    (),
                    None,
                ),
            ],
        )

        response = self.client.get(f"{BASE_PATH}/documents/{DOCUMENT_ID}/full")

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "document": {"documentId": DOCUMENT_ID, "documentTitle": "문서 가"},
                "summary": {
                    "questionCount": 16,
                    "insufficientEvidenceCount": 2,
                    "cachedAnswerCount": 5,
                    "subproblemCount": 2,
                },
                "subproblems": [
                    {
                        "subproblemId": str(SUBPROBLEM_ID),
                        "name": "세부 문제 가",
                        "questionCount": 8,
                        "sourceSection": "문서 가 > 절 하나",
                        "applyStatus": "APPLIED",
                        "inclusionCriteria": ["질문 범위 하나", "질문 범위 둘"],
                        "exclusionCriteria": ["다른 의도"],
                        "canonicalAnswer": {
                            "contentMarkdown": "정본 본문 [1]",
                            "applicabilityRules": ["규칙 하나"],
                        },
                    },
                    {
                        "subproblemId": str(SUBPROBLEM_ID),
                        "name": "세부 문제 나",
                        "questionCount": 0,
                        "sourceSection": None,
                        "applyStatus": "NEEDS_CANONICAL",
                        "inclusionCriteria": ["질문 범위 셋"],
                        "exclusionCriteria": [],
                        "canonicalAnswer": None,
                    },
                ],
            },
            response.json(),
        )
        self.service.get_document_full_detail.assert_awaited_once_with(
            GROUP_ID, DOCUMENT_ID
        )

    def test_subproblem_detail_shape(self) -> None:
        path = f"{BASE_PATH}/documents/{DOCUMENT_ID}/subproblems/{SUBPROBLEM_ID}"
        self.service.get_subproblem_detail.return_value = SubproblemDetail(
            SUBPROBLEM_ID,
            "세부 문제 가",
            ApplyStatus.APPLIED,
            CanonicalAnswerView("정본 본문 [1]", ["규칙 하나"]),
        )

        response = self.client.get(path)

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "subproblemId": str(SUBPROBLEM_ID),
                "name": "세부 문제 가",
                "applyStatus": "APPLIED",
                "canonicalAnswer": {
                    "contentMarkdown": "정본 본문 [1]",
                    "applicabilityRules": ["규칙 하나"],
                },
            },
            response.json(),
        )
        self.service.get_subproblem_detail.assert_awaited_once_with(
            GROUP_ID, DOCUMENT_ID, SUBPROBLEM_ID
        )

        self.service.get_subproblem_detail.return_value = SubproblemDetail(
            SUBPROBLEM_ID, "세부 문제 가", ApplyStatus.NEEDS_CANONICAL, None
        )
        body = self.client.get(path).json()
        self.assertEqual("NEEDS_CANONICAL", body["applyStatus"])
        self.assertIsNone(body["canonicalAnswer"])

    def test_questions_shape(self) -> None:
        self.service.list_questions.return_value = QuestionPage(
            items=[
                QuestionRow(
                    RAG_RUN_ID,
                    "합성 질문 하나",
                    DOCUMENT_ID,
                    "문서 가",
                    SUBPROBLEM_ID,
                    "세부 문제 가",
                    ASKED_AT,
                    AnswerStatus.CACHED_ANSWER,
                    None,
                ),
                QuestionRow(
                    RAG_RUN_ID,
                    "합성 질문 둘",
                    None,
                    None,
                    None,
                    None,
                    ASKED_AT,
                    AnswerStatus.WITHHELD,
                    WithheldReason.INSUFFICIENT_EVIDENCE,
                ),
            ],
            page=1,
            size=20,
            total_count=96,
        )

        response = self.client.get(f"{BASE_PATH}/questions")

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "items": [
                    {
                        "ragRunId": str(RAG_RUN_ID),
                        "question": "합성 질문 하나",
                        "documentId": DOCUMENT_ID,
                        "documentTitle": "문서 가",
                        "subproblemId": str(SUBPROBLEM_ID),
                        "subproblemName": "세부 문제 가",
                        "askedAt": "2026-09-15T03:12:00Z",
                        "answerStatus": "CACHED_ANSWER",
                        "withheldReason": None,
                    },
                    {
                        "ragRunId": str(RAG_RUN_ID),
                        "question": "합성 질문 둘",
                        "documentId": None,
                        "documentTitle": None,
                        "subproblemId": None,
                        "subproblemName": None,
                        "askedAt": "2026-09-15T03:12:00Z",
                        "answerStatus": "WITHHELD",
                        "withheldReason": "INSUFFICIENT_EVIDENCE",
                    },
                ],
                "page": 1,
                "size": 20,
                "totalCount": 96,
            },
            response.json(),
        )

    # ------------------------------------------------------------------
    # 질문 목록 쿼리
    # ------------------------------------------------------------------

    def _empty_page(self, page: int = 1, size: int = 20) -> QuestionPage:
        return QuestionPage(items=[], page=page, size=size, total_count=0)

    def _filters_of_last_call(self) -> QuestionListFilters:
        args = self.service.list_questions.await_args.args
        self.assertEqual(GROUP_ID, args[0])
        return args[1]

    def test_questions_defaults(self) -> None:
        self.service.list_questions.return_value = self._empty_page()

        body = self.client.get(f"{BASE_PATH}/questions").json()

        self.assertEqual(QuestionListFilters(), self._filters_of_last_call())
        self.assertEqual({"items": [], "page": 1, "size": 20, "totalCount": 0}, body)

    def test_questions_query_aliases(self) -> None:
        self.service.list_questions.return_value = self._empty_page(3, 50)

        response = self.client.get(
            f"{BASE_PATH}/questions",
            params={
                "answerStatus": "WITHHELD",
                "documentId": "7",
                "subproblemPresence": "PRESENT",
                "subproblemId": str(SUBPROBLEM_ID),
                "q": "  합성 검색어 ",
                "page": "3",
                "size": "50",
            },
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            QuestionListFilters(
                answer_status=AnswerStatus.WITHHELD,
                document_id=7,
                subproblem_presence=SubproblemPresence.PRESENT,
                subproblem_id=SUBPROBLEM_ID,
                q="합성 검색어",
                page=3,
                size=50,
            ),
            self._filters_of_last_call(),
        )

    def test_questions_ignores_snake_case_names(self) -> None:
        self.service.list_questions.return_value = self._empty_page()

        response = self.client.get(
            f"{BASE_PATH}/questions",
            params={"answer_status": "HELD", "document_id": "x", "subproblem_presence": "MAYBE"},
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(QuestionListFilters(), self._filters_of_last_call())

    def test_questions_blank_query_means_no_search(self) -> None:
        self.service.list_questions.return_value = self._empty_page()

        for q in ("", "   "):
            with self.subTest(q=q):
                response = self.client.get(f"{BASE_PATH}/questions", params={"q": q})
                self.assertEqual(200, response.status_code)
                self.assertIsNone(self._filters_of_last_call().q)

    def test_questions_invalid_query_returns_422(self) -> None:
        cases = [
            {"page": "abc"},
            {"page": ""},
            {"page": "1.5"},
            {"page": " 2"},
            {"page": "0"},
            {"page": OUT_OF_BIGINT},
            {"size": "0"},
            {"size": "101"},
            {"size": "+5"},
            {"documentId": "abc"},
            {"documentId": OUT_OF_BIGINT},
            {"answerStatus": "HELD"},
            {"answerStatus": "withheld"},
            {"answerStatus": ""},
            {"subproblemPresence": "MAYBE"},
            {"subproblemId": "not-a-uuid"},
            {"q": "가" * 101},
            {"subproblemPresence": "ABSENT", "subproblemId": str(SUBPROBLEM_ID)},
        ]
        for params in cases:
            with self.subTest(**params):
                response = self.client.get(f"{BASE_PATH}/questions", params=params)
                self.assertAdminError(response, 422, "INVALID_REQUEST")
        self.service.list_questions.assert_not_awaited()

    def test_questions_invalid_query_message(self) -> None:
        body = self.client.get(f"{BASE_PATH}/questions", params={"page": "abc"}).json()
        self.assertEqual("page 는 정수여야 합니다.", body["message"])

        body = self.client.get(f"{BASE_PATH}/questions", params={"size": "101"}).json()
        self.assertEqual("size 는 1 이상 100 이하여야 합니다.", body["message"])

    # ------------------------------------------------------------------
    # 경로와 오류
    # ------------------------------------------------------------------

    def test_malformed_path_values_return_422(self) -> None:
        paths = [
            "/api/admin/document-groups/abc/question-log/dashboard",
            "/api/admin/document-groups/abc/question-log/documents",
            "/api/admin/document-groups/1.5/question-log/questions",
            f"{BASE_PATH}/documents/abc",
            f"{BASE_PATH}/documents/abc/full",
            f"{BASE_PATH}/documents/abc/subproblems/{SUBPROBLEM_ID}",
            f"{BASE_PATH}/documents/{DOCUMENT_ID}/subproblems/not-a-uuid",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.assertAdminError(self.client.get(path), 422, "INVALID_REQUEST")

    def test_out_of_bigint_path_ids_return_404_without_query(self) -> None:
        paths = [
            f"/api/admin/document-groups/{OUT_OF_BIGINT}/question-log/dashboard",
            f"/api/admin/document-groups/{OUT_OF_BIGINT}/question-log/documents",
            f"/api/admin/document-groups/{OUT_OF_BIGINT}/question-log/questions",
            f"{BASE_PATH}/documents/{OUT_OF_BIGINT}",
            f"{BASE_PATH}/documents/{OUT_OF_BIGINT}/full",
            f"{BASE_PATH}/documents/{OUT_OF_BIGINT}/subproblems/{SUBPROBLEM_ID}",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.assertAdminError(self.client.get(path), 404, "NOT_FOUND")
        self.assertEqual([], self.service.method_calls)

    def test_not_found_errors_use_admin_body(self) -> None:
        self.service.get_dashboard.side_effect = DocumentGroupNotFoundError()
        self.service.list_documents.side_effect = DocumentGroupNotFoundError()
        self.service.list_questions.side_effect = DocumentGroupNotFoundError()
        self.service.get_document_detail.side_effect = DocumentNotFoundError()
        self.service.get_document_full_detail.side_effect = DocumentNotFoundError()
        self.service.get_subproblem_detail.side_effect = SubproblemNotFoundError()

        cases = [
            (f"{BASE_PATH}/dashboard", "존재하지 않는 문서 그룹입니다."),
            (f"{BASE_PATH}/documents", "존재하지 않는 문서 그룹입니다."),
            (f"{BASE_PATH}/questions", "존재하지 않는 문서 그룹입니다."),
            (f"{BASE_PATH}/documents/{DOCUMENT_ID}", "존재하지 않는 문서입니다."),
            (f"{BASE_PATH}/documents/{DOCUMENT_ID}/full", "존재하지 않는 문서입니다."),
            (
                f"{BASE_PATH}/documents/{DOCUMENT_ID}/subproblems/{SUBPROBLEM_ID}",
                "존재하지 않는 세부 문제입니다.",
            ),
        ]
        for path, message in cases:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertAdminError(response, 404, "NOT_FOUND")
                self.assertEqual(message, response.json()["message"])

    def test_openapi_lists_query_aliases(self) -> None:
        operation = self.client.get("/openapi.json").json()["paths"][
            "/api/admin/document-groups/{group_id}/question-log/questions"
        ]["get"]
        names = {param["name"] for param in operation["parameters"] if param["in"] == "query"}
        self.assertEqual(
            {"answerStatus", "documentId", "subproblemPresence", "subproblemId", "q", "page", "size"},
            names,
        )
        self.assertIn("404", operation["responses"])
        self.assertIn("422", operation["responses"])

        full_operation = self.client.get("/openapi.json").json()["paths"][
            "/api/admin/document-groups/{group_id}/question-log/documents/{document_id}/full"
        ]["get"]
        self.assertIn("404", full_operation["responses"])
        self.assertIn("422", full_operation["responses"])


class QuestionLogApiDbTest(unittest.IsolatedAsyncioTestCase):
    """실제 서비스와 세션을 거친 응답. 한 턴만 시드하고 rollback 한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = get_settings().database_url
        if not asyncio.run(_available(cls.database_url)):
            raise unittest.SkipTest("로컬 DB에 연결할 수 없어 통합 테스트를 건너뜁니다.")

    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine(self.database_url)
        self.connection = await self.engine.connect()
        self.transaction = await self.connection.begin()
        self.session = AsyncSession(
            bind=self.connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        await self._seed()

        self.app = create_app()

        async def override_db_session() -> AsyncIterator[AsyncSession]:
            yield self.session

        self.app.dependency_overrides[get_db_session] = override_db_session
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app), base_url="http://testserver"
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        self.app.dependency_overrides.clear()
        await self.session.close()
        await self.transaction.rollback()
        await self.connection.close()
        await self.engine.dispose()

    async def _seed(self) -> None:
        seed = _Seed(self.session)
        chunking = await seed.chunking()
        embedding = await seed.embedding()
        self.group = await seed.group()
        self.document = await seed.source(self.group, "api", "합성 문서")
        index = await seed.index(self.group, chunking, embedding, [])
        problem_group = await seed.document_group(self.group, self.document)
        self.subproblem = await seed.subproblem(problem_group, f"api-{seed.suffix}")
        self.archived_subproblem = await seed.subproblem(
            problem_group,
            f"archived-{seed.suffix}",
            status=QuestionSubproblemStatus.ARCHIVED,
        )
        await seed.add(
            CanonicalAnswer(
                subproblem_id=self.subproblem.id,
                origin=CanonicalAnswerOrigin.AUTHORED,
                content_markdown="승인 정본 본문 [1]",
                applicability_rules={"rules": ["이 정본이 적용되는 범위"]},
                subproblem_version=1,
                approval=CanonicalAnswerApproval.APPROVED,
                approved_by="test",
            )
        )
        run = ClassificationRun(
            document_group_id=self.group.id,
            index_version_id=index.index_version_id,
            kind=ClassificationRunKind.ONLINE,
            model="test-model",
            prompt_version=f"test-{seed.suffix}",
            started_at=ASKED_AT,
            finished_at=ASKED_AT,
        )
        await seed.add(run)
        conversation = await RagLogStore(self.session).create_conversation()
        self.turn = RagRun(
            conversation_id=conversation.id,
            turn_no=1,
            index_version_id=index.index_version_id,
            user_query="합성 질문",
            context_strategy=ContextStrategy.NEW_TOPIC,
            status=RunStatus.COMPLETED,
            created_at=ASKED_AT,
        )
        await seed.add(self.turn)
        await seed.add(
            QuestionClassification(
                rag_run_id=self.turn.id,
                subproblem_id=self.subproblem.id,
                subproblem_version=1,
                problem_group_id=problem_group.id,
                run_id=run.id,
                decision=ClassificationDecision.CONNECT,
                attribution_source=AttributionSource.SUBPROBLEM,
                effective_from=ASKED_AT,
            )
        )

    def _path(self, suffix: str, group_id: object = None) -> str:
        group = self.group.id if group_id is None else group_id
        return f"/api/admin/document-groups/{group}/question-log{suffix}"

    async def test_endpoints_serialize_service_results(self) -> None:
        questions = await self.client.get(self._path("/questions"))
        self.assertEqual(200, questions.status_code, questions.text)
        self.assertEqual(
            {
                "items": [
                    {
                        "ragRunId": str(self.turn.id),
                        "question": "합성 질문",
                        "documentId": self.document.id,
                        "documentTitle": "합성 문서",
                        "subproblemId": str(self.subproblem.id),
                        "subproblemName": self.subproblem.name,
                        "askedAt": "2026-09-15T03:12:00Z",
                        "answerStatus": "ANSWERED",
                        "withheldReason": None,
                    }
                ],
                "page": 1,
                "size": 20,
                "totalCount": 1,
            },
            questions.json(),
        )

        dashboard = (await self.client.get(self._path("/dashboard"))).json()
        self.assertEqual(1, dashboard["questionCount"])
        self.assertEqual(str(self.subproblem.id), dashboard["frequentSubproblems"][0]["subproblemId"])

        documents = (await self.client.get(self._path("/documents"))).json()
        self.assertEqual([self.document.id], [item["documentId"] for item in documents["items"]])

        detail = (await self.client.get(self._path(f"/documents/{self.document.id}"))).json()
        self.assertEqual(
            [{"subproblemId": str(self.subproblem.id), "name": self.subproblem.name, "questionCount": 1, "sourceSection": None, "applyStatus": "APPLIED"}],
            detail["subproblems"],
        )

        full = await self.client.get(
            self._path(f"/documents/{self.document.id}/full")
        )
        self.assertEqual(200, full.status_code, full.text)
        self.assertEqual(
            [{
                "subproblemId": str(self.subproblem.id),
                "name": self.subproblem.name,
                "questionCount": 1,
                "sourceSection": None,
                "applyStatus": "APPLIED",
                "inclusionCriteria": [
                    f"{self.subproblem.key} 기준 하나",
                    f"{self.subproblem.key} 기준 둘",
                ],
                "exclusionCriteria": [f"{self.subproblem.key} 제외"],
                "canonicalAnswer": {
                    "contentMarkdown": "승인 정본 본문 [1]",
                    "applicabilityRules": ["이 정본이 적용되는 범위"],
                },
            }],
            full.json()["subproblems"],
        )

        expanded = await self.client.get(
            self._path(f"/documents/{self.document.id}/subproblems/{self.subproblem.id}")
        )
        self.assertEqual(200, expanded.status_code)
        self.assertEqual(
            "승인 정본 본문 [1]", expanded.json()["canonicalAnswer"]["contentMarkdown"]
        )

    async def test_full_endpoint_checks_group_document_ownership(self) -> None:
        other_group = await _Seed(self.session).group()
        other_document = await _Seed(self.session).source(
            other_group, "other-api", "다른 그룹 문서"
        )

        wrong_document = await self.client.get(
            self._path(f"/documents/{other_document.id}/full")
        )
        self.assertEqual(404, wrong_document.status_code, wrong_document.text)
        self.assertEqual("NOT_FOUND", wrong_document.json()["code"])

        wrong_group = await self.client.get(
            self._path(f"/documents/{self.document.id}/full", group_id=other_group.id)
        )
        self.assertEqual(404, wrong_group.status_code, wrong_group.text)
        self.assertEqual("NOT_FOUND", wrong_group.json()["code"])

    async def test_boundary_ids_do_not_reach_driver_errors(self) -> None:
        bigint_max = "9223372036854775807"
        response = await self.client.get(self._path("/questions"), params={"documentId": bigint_max})
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(0, response.json()["totalCount"])

        response = await self.client.get(
            self._path("/questions"), params={"page": bigint_max, "size": "100"}
        )
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual([], response.json()["items"])

        for path in (
            self._path("/dashboard", group_id=bigint_max),
            self._path("/dashboard", group_id=OUT_OF_BIGINT),
            self._path(f"/documents/{bigint_max}"),
            self._path(f"/documents/{OUT_OF_BIGINT}"),
        ):
            with self.subTest(path=path):
                response = await self.client.get(path)
                self.assertEqual(404, response.status_code, response.text)
                self.assertEqual("NOT_FOUND", response.json()["code"])


if __name__ == "__main__":
    unittest.main()
