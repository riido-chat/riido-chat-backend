import unittest
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.admin.dependencies import get_recollect_service
from app.admin.schema import AdminGitBookSyncRequest
from app.document.recollect import AcceptedRecollect
from app.document.recollect_service import (
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
            group_source_id=1,
            root_url="https://docs.riido.io",
        )

        response = self._sync()

        self.assertEqual(202, response.status_code)
        self.assertEqual(
            {
                "batchId": str(BATCH_ID),
                "groupId": 1,
                "groupSourceId": 1,
                "rootUrl": "https://docs.riido.io",
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

    def test_batch_query_route_is_gone(self) -> None:
        """동기 전환으로 배치 조회가 사라졌다."""

        response = self.client.get(f"/api/admin/recollect-batches/{BATCH_ID}")

        self.assertEqual(404, response.status_code)




class AdminValidationErrorFormatTest(unittest.TestCase):
    """422 도 다른 오류와 같은 {code, message} 형식으로 나가야 한다."""

    def setUp(self) -> None:
        app = create_app()
        app.router.lifespan_context = test_lifespan
        self.client = TestClient(app)

    def test_admin_422_uses_error_response_shape(self) -> None:
        response = self.client.post(
            "/api/admin/document-groups/1/gitbook-sync",
            json={"sourceUrl": "http://docs.riido.io"},
        )

        self.assertEqual(422, response.status_code)
        body = response.json()
        self.assertEqual("INVALID_REQUEST", body["code"])
        self.assertIn("message", body)
        self.assertNotIn("detail", body)

    def test_admin_422_on_missing_field(self) -> None:
        response = self.client.post(
            "/api/admin/document-groups/1/gitbook-sync",
            json={},
        )

        self.assertEqual(422, response.status_code)
        self.assertEqual("INVALID_REQUEST", response.json()["code"])

    def test_non_admin_path_keeps_default_shape(self) -> None:
        """콘솔 밖 경로는 FastAPI 기본 형식을 유지한다."""

        app = create_app()
        app.router.lifespan_context = test_lifespan

        @app.post("/probe")
        def probe(payload: AdminGitBookSyncRequest) -> dict:
            return {}

        response = TestClient(app).post("/probe", json={})

        self.assertEqual(422, response.status_code)
        self.assertIn("detail", response.json())


if __name__ == "__main__":
    unittest.main()
