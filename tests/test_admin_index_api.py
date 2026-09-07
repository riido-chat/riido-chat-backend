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
from app.document.ingestion_service import DocumentGroupNotFoundError
from app.indexing.index_service import (
    AcceptedIndexRun,
    IndexReindexService,
    IndexRunDetail,
    IndexVersionSummary,
    NoReadyDocumentsError,
    ReindexNotRequiredError,
)
from app.main import create_app


STARTED_AT = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)


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

    def _accepted(self) -> AcceptedIndexRun:
        return AcceptedIndexRun(
            index_run_id=311,
            index_version_id=59,
            group_id=1,
            operation_type="BUILD_AND_APPLY",
            trigger_type="MANUAL",
            stage="BUILDING",
        )

    def _detail(self, **changes) -> IndexRunDetail:
        base = {
            "index_run_id": 311,
            "group_id": 1,
            "index_version_id": 59,
            "operation_type": "BUILD_AND_APPLY",
            "trigger_type": "MANUAL",
            "status": ExecutionStatus.SUCCESS,
            "stage": "APPLYING",
            "started_at": STARTED_AT,
            "finished_at": STARTED_AT,
            "index_version": IndexVersionSummary(
                index_version_id=59,
                version_no=13,
            ),
            "previous_index_version": IndexVersionSummary(
                index_version_id=57,
                version_no=12,
            ),
        }
        base.update(changes)
        return IndexRunDetail(**base)

    def test_reindex_returns_version_transition(self) -> None:
        self.service.start_reindex.return_value = self._accepted()
        self.service.read_finished_run.return_value = self._detail()

        with patch(
            "app.admin.router.run_admin_index_job",
            new=AsyncMock(),
        ) as job:
            response = self.client.post("/api/admin/document-groups/1/reindex")

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "indexRunId": 311,
                "indexVersion": {"indexVersionId": 59, "versionNo": 13},
                "previousIndexVersion": {
                    "indexVersionId": 57,
                    "versionNo": 12,
                },
            },
            response.json(),
        )
        # background task 가 아니라 요청 안에서 끝난다
        job.assert_awaited_once()

    def test_first_reindex_has_no_previous_version(self) -> None:
        self.service.start_reindex.return_value = self._accepted()
        self.service.read_finished_run.return_value = self._detail(
            previous_index_version=None,
        )

        with patch("app.admin.router.run_admin_index_job", new=AsyncMock()):
            response = self.client.post("/api/admin/document-groups/1/reindex")

        self.assertIsNone(response.json()["previousIndexVersion"])

    def test_reindex_failure_returns_500_internal_error(self) -> None:
        """처리 중 실패는 원인을 가리지 않고 코드 하나로 내린다."""

        self.service.start_reindex.return_value = self._accepted()
        self.service.read_finished_run.return_value = self._detail(
            status=ExecutionStatus.FAILED,
            error_code="CORPUS_RELOAD_FAILED",
            error_message="corpus 재적재에 실패했습니다.",
        )

        with patch("app.admin.router.run_admin_index_job", new=AsyncMock()):
            response = self.client.post("/api/admin/document-groups/1/reindex")

        self.assertEqual(500, response.status_code)
        body = response.json()
        self.assertEqual("INTERNAL_ERROR", body["code"])
        self.assertEqual({"code", "message"}, set(body))

    def test_reindex_returns_404_for_unknown_group(self) -> None:
        self.service.start_reindex.side_effect = DocumentGroupNotFoundError()

        response = self.client.post("/api/admin/document-groups/999/reindex")

        self.assertEqual(404, response.status_code)
        self.assertEqual("NOT_FOUND", response.json()["code"])

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

    def test_retry_apply_route_is_gone(self) -> None:
        """재시도는 폐기됐다. 검색 반영 시작을 다시 호출한다."""

        response = self.client.post("/api/admin/index-runs/310/retry-apply")

        self.assertEqual(404, response.status_code)

    def test_index_run_query_route_is_gone(self) -> None:
        """동기 전환으로 실행 조회가 사라졌다."""

        response = self.client.get("/api/admin/index-runs/311")

        self.assertEqual(404, response.status_code)


if __name__ == "__main__":
    unittest.main()
