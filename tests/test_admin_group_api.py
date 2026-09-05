import unittest
import uuid
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
    LatestIndexRun,
    PendingDocumentView,
)
from app.database.models import ExecutionStatus
from app.document.ingestion_service import DocumentGroupNotFoundError
from app.document.job_gate import RunningJob
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
            "active_index_version": ActiveIndexVersion(
                index_version_id=57,
                version_no=12,
                activated_at=STARTED_AT,
            ),
            "pending_documents": [],
            "search_status": "UP_TO_DATE",
            "documents": [],
            "running_job": None,
            "latest_index_run": None,
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
            pending_documents=[
                PendingDocumentView(
                    document_id=140,
                    title="임시 공지 9월",
                    change_type="NEW",
                )
            ],
            search_status="REINDEX_REQUIRED",
            documents=[
                GroupDocument(
                    document_id=101,
                    document_key="upload/자주-묻는-질문",
                    title="자주 묻는 질문",
                    source_type="UPLOAD",
                    document_version_no=4,
                    applied_version_no=3,
                    processing_status="READY",
                )
            ],
        )

        body = self.client.get("/api/admin/document-groups/1").json()

        self.assertEqual(12, body["summary"]["activeIndexVersion"]["versionNo"])
        self.assertEqual(1, body["summary"]["pendingCount"])
        self.assertEqual("NEW", body["summary"]["pendingDocuments"][0]["changeType"])
        self.assertEqual("REINDEX_REQUIRED", body["summary"]["searchStatus"])
        document = body["documents"][0]
        self.assertEqual("UPLOAD", document["sourceType"])
        self.assertEqual(4, document["documentVersionNo"])
        self.assertEqual(3, document["appliedVersionNo"])
        self.assertIsNone(body["runningJob"])
        self.assertIsNone(body["latestIndexRun"])

    def test_detail_reports_running_recollect_batch(self) -> None:
        batch_id = uuid.uuid4()
        self.service.get_group_detail.return_value = self._detail(
            search_status="IN_PROGRESS",
            running_job=RunningJob(
                job_type="RECOLLECT",
                stage="PROCESSING",
                batch_id=batch_id,
            ),
        )

        body = self.client.get("/api/admin/document-groups/1").json()

        running = body["runningJob"]
        self.assertEqual("RECOLLECT", running["jobType"])
        self.assertEqual(str(batch_id), running["batchId"])
        self.assertIsNone(running["ingestionRunId"])

    def test_detail_reports_failed_apply_for_modal_restore(self) -> None:
        self.service.get_group_detail.return_value = self._detail(
            search_status="REINDEX_REQUIRED",
            latest_index_run=LatestIndexRun(
                index_run_id=310,
                index_version_id=58,
                operation_type="BUILD_AND_APPLY",
                status=ExecutionStatus.FAILED,
                stage="APPLYING",
                error_code="CORPUS_RELOAD_FAILED",
                started_at=STARTED_AT,
                finished_at=FINISHED_AT,
            ),
        )

        body = self.client.get("/api/admin/document-groups/1").json()

        latest = body["latestIndexRun"]
        # FE 는 이 조합으로 4-4 모달을 복원한다
        self.assertEqual("FAILED", latest["status"])
        self.assertEqual("APPLYING", latest["stage"])
        self.assertEqual("CORPUS_RELOAD_FAILED", latest["errorCode"])

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
