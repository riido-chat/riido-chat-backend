import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.admin.dependencies import get_index_reindex_service
from app.chat.dependencies import get_corpus_state
from app.indexing.index_service import (
    AcceptedIndexRun,
    IndexReindexService,
    NoReadyDocumentsError,
    ReindexNotRequiredError,
)
from app.main import create_app


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
