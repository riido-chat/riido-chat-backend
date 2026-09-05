import unittest
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.admin.dependencies import get_recollect_service
from app.database.models import ExecutionStatus
from app.document.recollect import AcceptedRecollect
from app.document.recollect_service import (
    RecollectBatchDetail,
    RecollectBatchNotFoundError,
    RecollectFailure,
    RecollectService,
    SourceListFailedError,
)
from app.main import create_app


STARTED_AT = datetime(2026, 9, 5, 11, 0, tzinfo=timezone.utc)
FINISHED_AT = datetime(2026, 9, 5, 11, 3, 12, tzinfo=timezone.utc)
BATCH_ID = uuid.UUID("6f1d0c7e-2b7c-4c1a-9c0e-3a1b2c3d4e5f")


@asynccontextmanager
async def test_lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


class AdminGitBookSyncApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = AsyncMock(spec=RecollectService)
        with patch("app.main.lifespan", test_lifespan):
            self.app = create_app()
        self.app.dependency_overrides[get_recollect_service] = lambda: self.service
        self.client = TestClient(self.app)
        self.job = patch("app.admin.router.run_recollect_batch", new=AsyncMock())
        self.job.start()

    def tearDown(self) -> None:
        self.job.stop()
        self.app.dependency_overrides.clear()
        self.client.close()

    def _sync(self, source_url: str = "https://docs.riido.io"):
        return self.client.post(
            "/api/admin/document-groups/1/gitbook-sync",
            json={"sourceUrl": source_url},
        )

    def test_accepts_sync_and_returns_batch_id(self) -> None:
        self.service.start_sync.return_value = AcceptedRecollect(
            batch_id=BATCH_ID,
            group_id=1,
            page_count=41,
        )

        response = self._sync()

        self.assertEqual(202, response.status_code)
        self.assertEqual(
            {
                "batchId": str(BATCH_ID),
                "groupId": 1,
                "status": "PROCESSING",
                "stage": "PROCESSING",
                "pageCount": 41,
            },
            response.json(),
        )
        self.service.start_sync.assert_awaited_once_with(1, "https://docs.riido.io")

    def test_rejects_non_https_root(self) -> None:
        response = self._sync("http://docs.riido.io")

        self.assertEqual(422, response.status_code)
        self.service.start_sync.assert_not_awaited()

    def test_source_list_failure_returns_502(self) -> None:
        self.service.start_sync.side_effect = SourceListFailedError("timeout")

        response = self._sync()

        self.assertEqual(502, response.status_code)
        self.assertEqual("SOURCE_LIST_FAILED", response.json()["code"])

    def test_processing_batch_returns_progress(self) -> None:
        self.service.get_batch.return_value = RecollectBatchDetail(
            batch_id=BATCH_ID,
            group_id=1,
            status=ExecutionStatus.PROCESSING,
            total=41,
            processed=17,
            started_at=STARTED_AT,
        )

        body = self.client.get(f"/api/admin/recollect-batches/{BATCH_ID}").json()

        self.assertEqual("PROCESSING", body["status"])
        self.assertEqual({"total": 41, "processed": 17}, body["progress"])
        self.assertNotIn("counts", body)

    def test_finished_batch_returns_counts_and_failures(self) -> None:
        self.service.get_batch.return_value = RecollectBatchDetail(
            batch_id=BATCH_ID,
            group_id=1,
            status=ExecutionStatus.SUCCESS,
            total=41,
            processed=41,
            started_at=STARTED_AT,
            finished_at=FINISHED_AT,
            counts={
                "total": 41,
                "created": 1,
                "updated": 3,
                "no_change": 35,
                "removed": 1,
                "failed": 1,
            },
            failures=(
                RecollectFailure(
                    document_key="sprints/automations",
                    title="스프린트 자동화",
                    ingestion_run_id=950,
                    stage="EMBEDDING",
                    error_code="UPSTREAM_ERROR",
                ),
            ),
        )

        body = self.client.get(f"/api/admin/recollect-batches/{BATCH_ID}").json()

        self.assertEqual("SUCCESS", body["status"])
        self.assertEqual(35, body["counts"]["noChange"])
        self.assertEqual(1, body["counts"]["removed"])
        failure = body["failures"][0]
        # 목록에 사람이 읽을 제목과 원인 단계가 함께 나온다
        self.assertEqual("스프린트 자동화", failure["title"])
        self.assertEqual("EMBEDDING", failure["stage"])
        self.assertEqual("UPSTREAM_ERROR", failure["errorCode"])
        # 상세는 기존 업로드 실행 조회로 이어진다
        self.assertEqual(950, failure["ingestionRunId"])

    def test_unknown_batch_returns_404(self) -> None:
        self.service.get_batch.side_effect = RecollectBatchNotFoundError()

        response = self.client.get(f"/api/admin/recollect-batches/{uuid.uuid4()}")

        self.assertEqual(404, response.status_code)
        self.assertEqual("NOT_FOUND", response.json()["code"])

    def test_malformed_batch_id_returns_422(self) -> None:
        response = self.client.get("/api/admin/recollect-batches/not-a-uuid")

        self.assertEqual(422, response.status_code)


if __name__ == "__main__":
    unittest.main()
