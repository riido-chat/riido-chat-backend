import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.admin.dependencies import get_index_reindex_service
from app.chat.dependencies import get_corpus_state
from app.database.models import ExecutionStatus
from app.indexing.index_service import (
    AcceptedIndexRun,
    IndexReindexService,
    IndexRunDetail,
    IndexRunNotFoundError,
    IndexVersionSummary,
    NoReadyDocumentsError,
    ReindexNotRequiredError,
    RetryNotAllowedError,
)
from app.main import create_app


STARTED_AT = datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc)
FINISHED_AT = datetime(2026, 9, 5, 10, 0, 41, tzinfo=timezone.utc)


@asynccontextmanager
async def test_lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


class AdminIndexRunApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = AsyncMock(spec=IndexReindexService)
        with patch("app.main.lifespan", test_lifespan):
            self.app = create_app()
        self.app.dependency_overrides[get_index_reindex_service] = (
            lambda: self.service
        )
        self.app.dependency_overrides[get_corpus_state] = lambda: object()
        self.client = TestClient(self.app)
        self.job = patch("app.admin.router.run_admin_index_job", new=AsyncMock())
        self.job.start()

    def tearDown(self) -> None:
        self.job.stop()
        self.app.dependency_overrides.clear()
        self.client.close()

    def _processing_detail(self, **changes) -> IndexRunDetail:
        base = {
            "index_run_id": 311,
            "group_id": 1,
            "index_version_id": 59,
            "operation_type": "BUILD_AND_APPLY",
            "trigger_type": "MANUAL",
            "status": ExecutionStatus.PROCESSING,
            "stage": "VALIDATING",
            "started_at": STARTED_AT,
        }
        base.update(changes)
        return IndexRunDetail(**base)

    def test_reindex_accepts_and_starts_background_job(self) -> None:
        self.service.start_reindex.return_value = AcceptedIndexRun(
            index_run_id=311,
            index_version_id=59,
            group_id=1,
            operation_type="BUILD_AND_APPLY",
            trigger_type="MANUAL",
            stage="BUILDING",
        )

        response = self.client.post("/api/admin/document-groups/1/reindex")

        self.assertEqual(202, response.status_code)
        self.assertEqual(
            {
                "indexRunId": 311,
                "indexVersionId": 59,
                "groupId": 1,
                "operationType": "BUILD_AND_APPLY",
                "triggerType": "MANUAL",
                "status": "PROCESSING",
                "stage": "BUILDING",
                "retryOfIndexRunId": None,
            },
            response.json(),
        )

    def test_reindex_returns_409_when_nothing_changed(self) -> None:
        self.service.start_reindex.side_effect = ReindexNotRequiredError()

        response = self.client.post("/api/admin/document-groups/1/reindex")

        self.assertEqual(409, response.status_code)
        self.assertEqual("REINDEX_NOT_REQUIRED", response.json()["code"])

    def test_reindex_returns_409_without_ready_documents(self) -> None:
        self.service.start_reindex.side_effect = NoReadyDocumentsError()

        response = self.client.post("/api/admin/document-groups/1/reindex")

        self.assertEqual(409, response.status_code)
        self.assertEqual("NO_READY_DOCUMENTS", response.json()["code"])

    def test_retry_apply_returns_retry_lineage(self) -> None:
        self.service.start_retry_apply.return_value = AcceptedIndexRun(
            index_run_id=312,
            index_version_id=58,
            group_id=1,
            operation_type="APPLY",
            trigger_type="RETRY",
            stage="APPLYING",
            retry_of_index_run_id=310,
        )

        response = self.client.post("/api/admin/index-runs/310/retry-apply")

        self.assertEqual(202, response.status_code)
        body = response.json()
        self.assertEqual("APPLY", body["operationType"])
        self.assertEqual("RETRY", body["triggerType"])
        self.assertEqual("APPLYING", body["stage"])
        self.assertEqual(310, body["retryOfIndexRunId"])

    def test_retry_apply_returns_409_when_not_allowed(self) -> None:
        self.service.start_retry_apply.side_effect = RetryNotAllowedError()

        response = self.client.post("/api/admin/index-runs/310/retry-apply")

        self.assertEqual(409, response.status_code)
        self.assertEqual("RETRY_NOT_ALLOWED", response.json()["code"])

    def test_processing_returns_stage_without_result_fields(self) -> None:
        self.service.get_index_run.return_value = self._processing_detail()

        response = self.client.get("/api/admin/index-runs/311")

        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual("PROCESSING", body["status"])
        self.assertEqual("VALIDATING", body["stage"])
        self.assertNotIn("indexVersion", body)
        self.assertNotIn("error", body)

    def test_success_returns_version_transition(self) -> None:
        self.service.get_index_run.return_value = self._processing_detail(
            status=ExecutionStatus.SUCCESS,
            stage="APPLYING",
            finished_at=FINISHED_AT,
            index_version=IndexVersionSummary(
                index_version_id=59,
                version_no=13,
                status="ACTIVE",
                activated_at=FINISHED_AT,
            ),
            previous_index_version=IndexVersionSummary(
                index_version_id=57,
                version_no=12,
                status="INACTIVE",
                activated_at=STARTED_AT,
            ),
            document_count=41,
            chunk_count=1274,
        )

        response = self.client.get("/api/admin/index-runs/311")

        body = response.json()
        self.assertEqual("SUCCESS", body["status"])
        # 화면 4-3 의 "검색 버전 #12 → #13"
        self.assertEqual(12, body["previousIndexVersion"]["versionNo"])
        self.assertEqual(13, body["indexVersion"]["versionNo"])
        self.assertEqual(41, body["documentCount"])
        self.assertEqual(1274, body["chunkCount"])

    def test_failed_apply_is_retryable(self) -> None:
        self.service.get_index_run.return_value = self._processing_detail(
            index_run_id=310,
            index_version_id=58,
            status=ExecutionStatus.FAILED,
            stage="APPLYING",
            finished_at=FINISHED_AT,
            index_version=IndexVersionSummary(
                index_version_id=58,
                version_no=13,
                status="READY",
                activated_at=None,
            ),
            error_code="CORPUS_RELOAD_FAILED",
            error_message="corpus 교체에 실패했습니다.",
            retryable=True,
        )

        response = self.client.get("/api/admin/index-runs/310")

        body = response.json()
        self.assertEqual("FAILED", body["status"])
        self.assertEqual("APPLYING", body["stage"])
        self.assertEqual("CORPUS_RELOAD_FAILED", body["error"]["code"])
        self.assertTrue(body["retryable"])
        self.assertEqual("READY", body["indexVersion"]["status"])

    def test_failed_build_is_not_retryable(self) -> None:
        self.service.get_index_run.return_value = self._processing_detail(
            status=ExecutionStatus.FAILED,
            stage="BUILDING",
            finished_at=FINISHED_AT,
            index_version=IndexVersionSummary(
                index_version_id=59,
                version_no=None,
                status="FAILED",
                activated_at=None,
            ),
            error_code=None,
            error_message="예상하지 못한 오류",
            retryable=False,
        )

        response = self.client.get("/api/admin/index-runs/311")

        body = response.json()
        # 기록되지 않은 코드는 내부 오류로 내린다
        self.assertEqual("INTERNAL_ERROR", body["error"]["code"])
        self.assertFalse(body["retryable"])
        self.assertIsNone(body["indexVersion"]["versionNo"])

    def test_unknown_index_run_returns_404(self) -> None:
        self.service.get_index_run.side_effect = IndexRunNotFoundError()

        response = self.client.get("/api/admin/index-runs/999")

        self.assertEqual(404, response.status_code)
        self.assertEqual("NOT_FOUND", response.json()["code"])


if __name__ == "__main__":
    unittest.main()
