import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.admin.dependencies import get_document_group_service
from app.admin.group_service import (
    ActiveIndexVersion,
    DocumentGroupService,
    GroupDetail,
    GroupDocument,
    GroupSummary,
)
from app.document.ingestion_service import DocumentGroupNotFoundError
from app.document.group_source import GroupSourceView
from app.main import create_app


STARTED_AT = datetime(2026, 9, 4, 8, 0, 3, tzinfo=timezone.utc)
FINISHED_AT = datetime(2026, 9, 4, 8, 0, 41, tzinfo=timezone.utc)


@asynccontextmanager
async def test_lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


class AdminDocumentGroupApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = AsyncMock(spec=DocumentGroupService)
        with patch("app.main.lifespan", test_lifespan):
            self.app = create_app()
        self.app.dependency_overrides[get_document_group_service] = (
            lambda: self.service
        )
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()
        self.client.close()

    def _detail(self, **changes) -> GroupDetail:
        base = {
            "group_id": 1,
            "group_key": "HELP_CHATBOT",
            "name": "도움말 챗봇 이용가이드",
            "consumer_key": "HELP_CHATBOT",
            "sources": [
                GroupSourceView(
                    group_source_id=1,
                    provider="GITBOOK",
                    root_url="https://docs.riido.io",
                    enabled=True,
                    document_count=39,
                )
            ],
            "active_index_version": ActiveIndexVersion(
                index_version_id=57,
                version_no=12,
            ),
            "pending_count": 0,
            "search_status": "UP_TO_DATE",
            "documents": [],
            "job_in_progress": False,
        }
        base.update(changes)
        return GroupDetail(**base)

    def test_list_returns_group_rows(self) -> None:
        self.service.list_groups.return_value = [
            GroupSummary(
                group_id=1,
                group_key="HELP_CHATBOT",
                name="도움말 챗봇 이용가이드",
                consumer_key="HELP_CHATBOT",
                document_count=41,
                active_index_version_no=12,
                search_status="UP_TO_DATE",
            )
        ]

        body = self.client.get("/api/admin/document-groups").json()

        self.assertEqual(
            [
                {
                    "groupId": 1,
                    "groupKey": "HELP_CHATBOT",
                    "name": "도움말 챗봇 이용가이드",
                    "consumerKey": "HELP_CHATBOT",
                    "documentCount": 41,
                    "activeIndexVersionNo": 12,
                    "searchStatus": "UP_TO_DATE",
                }
            ],
            body["groups"],
        )

    def test_detail_returns_summary_and_documents(self) -> None:
        self.service.get_group_detail.return_value = self._detail(
            pending_count=1,
            search_status="REINDEX_REQUIRED",
            documents=[
                GroupDocument(
                    document_id=101,
                    document_key="upload/자주-묻는-질문",
                    title="자주 묻는 질문",
                    source_type="UPLOAD",
                    group_source_id=None,
                    document_version_no=4,
                    applied_version_no=3,
                    applied_status="UNAPPLIED",
                )
            ],
        )

        body = self.client.get("/api/admin/document-groups/1").json()

        self.assertEqual(12, body["summary"]["activeIndexVersion"]["versionNo"])
        self.assertEqual(1, body["summary"]["pendingCount"])
        self.assertEqual("REINDEX_REQUIRED", body["summary"]["searchStatus"])
        source = body["sources"][0]
        self.assertEqual("https://docs.riido.io", source["rootUrl"])
        self.assertEqual(39, source["documentCount"])
        document = body["documents"][0]
        self.assertEqual("UPLOAD", document["sourceType"])
        self.assertEqual(4, document["documentVersionNo"])
        self.assertEqual(3, document["appliedVersionNo"])
        self.assertEqual("UNAPPLIED", document["appliedStatus"])
        # 콘솔 업로드 문서는 수집 원천이 없다
        self.assertIsNone(document["groupSourceId"])
        self.assertFalse(body["jobInProgress"])

    def test_detail_drops_polling_fields(self) -> None:
        """실행 조회가 사라져 복원용 필드를 내려주지 않는다."""

        self.service.get_group_detail.return_value = self._detail()

        body = self.client.get("/api/admin/document-groups/1").json()

        for gone in ("runningJob", "latestIndexRun"):
            self.assertNotIn(gone, body)
        self.assertNotIn("pendingDocuments", body["summary"])
        self.assertNotIn("activatedAt", body["summary"]["activeIndexVersion"])

    def test_detail_reports_job_in_progress(self) -> None:
        """작업 종류를 구분하지 않고 boolean 하나로 내려준다."""

        self.service.get_group_detail.return_value = self._detail(
            search_status="IN_PROGRESS",
            job_in_progress=True,
        )

        body = self.client.get("/api/admin/document-groups/1").json()

        self.assertTrue(body["jobInProgress"])
        self.assertEqual("IN_PROGRESS", body["summary"]["searchStatus"])

    def test_detail_reports_failed_search_status(self) -> None:
        """직전 반영이 실패해도 검색에 반영하기는 활성이다."""

        self.service.get_group_detail.return_value = self._detail(
            pending_count=2,
            search_status="FAILED",
        )

        body = self.client.get("/api/admin/document-groups/1").json()

        self.assertEqual("FAILED", body["summary"]["searchStatus"])
        self.assertEqual(2, body["summary"]["pendingCount"])

    def test_unknown_group_returns_404(self) -> None:
        self.service.get_group_detail.side_effect = DocumentGroupNotFoundError()

        response = self.client.get("/api/admin/document-groups/999")

        self.assertEqual(404, response.status_code)
        self.assertEqual("NOT_FOUND", response.json()["code"])

    def test_malformed_group_id_returns_422(self) -> None:
        response = self.client.get("/api/admin/document-groups/not-a-number")

        self.assertEqual(422, response.status_code)


if __name__ == "__main__":
    unittest.main()
