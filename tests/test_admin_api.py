import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.admin.dependencies import get_admin_ingestion_service
from app.admin.ingestion_service import (
    AcceptedIngestion,
    AdminIngestionService,
    DocumentAlreadyExistsError,
    IngestionRunDetail,
)
from app.database.models import ExecutionStatus
from app.main import create_app


@asynccontextmanager
async def test_lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


class AdminDocumentApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = AsyncMock(spec=AdminIngestionService)
        with patch("app.main.lifespan", test_lifespan):
            self.app = create_app()
        self.app.dependency_overrides[get_admin_ingestion_service] = (
            lambda: self.service
        )
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()
        self.client.close()

    def test_accepts_markdown_upload_and_starts_background_ingestion(self) -> None:
        self.service.start_new_document.return_value = AcceptedIngestion(
            ingestion_run_id=101,
            document_source_id=42,
        )

        with patch(
            "app.api.admin_documents.run_admin_ingestion",
            new=AsyncMock(),
        ) as run:
            response = self._upload(
                "guide.md",
                "# 문서\n\n## 안내\n\n본문".encode("utf-8"),
            )

        self.assertEqual(202, response.status_code)
        self.assertEqual(
            {
                "ingestionRunId": 101,
                "documentId": 42,
                "status": "PROCESSING",
            },
            response.json(),
        )
        self.service.start_new_document.assert_awaited_once_with(
            title="문서 제목",
            source_url="https://docs.riido.io/new-guide",
            category="guide",
            filename="guide.md",
        )
        run.assert_awaited_once_with(
            101,
            "# 문서\n\n## 안내\n\n본문",
        )

    def test_accepts_upload_without_source_url(self) -> None:
        self.service.start_new_document.return_value = AcceptedIngestion(
            ingestion_run_id=102,
            document_source_id=43,
        )

        with patch(
            "app.api.admin_documents.run_admin_ingestion",
            new=AsyncMock(),
        ):
            response = self.client.post(
                "/api/admin/documents",
                data={"title": "문서 제목", "category": "guide"},
                files={"file": ("guide.md", b"# guide", "text/markdown")},
            )

        self.assertEqual(202, response.status_code)
        self.service.start_new_document.assert_awaited_once_with(
            title="문서 제목",
            source_url=None,
            category="guide",
            filename="guide.md",
        )

    def test_rejects_non_markdown_extension(self) -> None:
        response = self._upload("guide.html", b"<h1>guide</h1>")

        self.assertEqual(422, response.status_code)
        self.assertEqual("INVALID_FILE", response.json()["code"])
        self.service.start_new_document.assert_not_awaited()

    def test_rejects_non_utf8_file(self) -> None:
        response = self._upload("guide.md", b"\xff\xfe")

        self.assertEqual(422, response.status_code)
        self.assertEqual("INVALID_FILE", response.json()["code"])
        self.service.start_new_document.assert_not_awaited()

    def test_rejects_empty_markdown(self) -> None:
        response = self._upload("guide.md", b" \n\t")

        self.assertEqual(422, response.status_code)
        self.assertEqual("INVALID_FILE", response.json()["code"])
        self.service.start_new_document.assert_not_awaited()

    def test_rejects_file_larger_than_five_megabytes(self) -> None:
        response = self._upload("guide.md", b"a" * (5 * 1024 * 1024 + 1))

        self.assertEqual(413, response.status_code)
        self.assertEqual("FILE_TOO_LARGE", response.json()["code"])
        self.service.start_new_document.assert_not_awaited()

    def test_rejects_extra_multipart_field(self) -> None:
        response = self.client.post(
            "/api/admin/documents",
            data={
                "title": "문서 제목",
                "sourceUrl": "https://docs.riido.io/new-guide",
                "category": "guide",
                "extra": "not-allowed",
            },
            files={"file": ("guide.md", b"# guide", "text/markdown")},
        )

        self.assertEqual(422, response.status_code)
        self.service.start_new_document.assert_not_awaited()

    def test_returns_admin_error_contract_for_duplicate_document(self) -> None:
        self.service.start_new_document.side_effect = DocumentAlreadyExistsError()

        response = self._upload("guide.md", b"# guide")

        self.assertEqual(409, response.status_code)
        self.assertEqual(
            {
                "code": "DOCUMENT_ALREADY_EXISTS",
                "message": "같은 문서명의 문서가 이미 존재합니다.",
            },
            response.json(),
        )

    def test_returns_processing_ingestion_run(self) -> None:
        started_at = datetime(2026, 8, 31, tzinfo=timezone.utc)
        self.service.get_ingestion_run.return_value = IngestionRunDetail(
            ingestion_run_id=101,
            document_source_id=42,
            status=ExecutionStatus.PROCESSING,
            document_version_id=None,
            version_no=None,
            section_count=None,
            chunk_count=None,
            error_code=None,
            error_message=None,
            started_at=started_at,
            finished_at=None,
        )

        response = self.client.get("/api/admin/ingestion-runs/101")

        self.assertEqual(200, response.status_code)
        self.assertEqual("PROCESSING", response.json()["status"])
        self.assertNotIn("documentVersionId", response.json())

    def test_returns_successful_ingestion_run(self) -> None:
        started_at = datetime(2026, 8, 31, tzinfo=timezone.utc)
        finished_at = datetime(2026, 8, 31, 0, 0, 1, tzinfo=timezone.utc)
        self.service.get_ingestion_run.return_value = IngestionRunDetail(
            ingestion_run_id=101,
            document_source_id=42,
            status=ExecutionStatus.SUCCESS,
            document_version_id=77,
            version_no=1,
            section_count=2,
            chunk_count=2,
            error_code=None,
            error_message=None,
            started_at=started_at,
            finished_at=finished_at,
        )

        body = self.client.get("/api/admin/ingestion-runs/101").json()

        self.assertEqual("SUCCESS", body["status"])
        self.assertEqual(77, body["documentVersionId"])
        self.assertEqual(1, body["versionNo"])
        self.assertTrue(body["changed"])
        self.assertEqual(2, body["chunkCount"])

    def test_returns_failed_ingestion_run(self) -> None:
        started_at = datetime(2026, 8, 31, tzinfo=timezone.utc)
        finished_at = datetime(2026, 8, 31, 0, 0, 1, tzinfo=timezone.utc)
        self.service.get_ingestion_run.return_value = IngestionRunDetail(
            ingestion_run_id=101,
            document_source_id=42,
            status=ExecutionStatus.FAILED,
            document_version_id=None,
            version_no=None,
            section_count=None,
            chunk_count=None,
            error_code="INVALID_FILE",
            error_message="정제·청킹 후 유효한 본문이 없습니다.",
            started_at=started_at,
            finished_at=finished_at,
        )

        body = self.client.get("/api/admin/ingestion-runs/101").json()

        self.assertEqual("FAILED", body["status"])
        self.assertEqual("INVALID_FILE", body["error"]["code"])
        self.assertIn("유효한 본문", body["error"]["message"])

    def test_openapi_documents_multipart_request_and_accepted_response(self) -> None:
        operation = self.app.openapi()["paths"]["/api/admin/documents"]["post"]

        self.assertIn("multipart/form-data", operation["requestBody"]["content"])
        self.assertIn("202", operation["responses"])
        self.assertIn("409", operation["responses"])
        self.assertIn("413", operation["responses"])
        self.assertIn("FILE_TOO_LARGE", operation["responses"]["413"]["description"])
        self.assertIn(
            "DOCUMENT_ALREADY_EXISTS",
            operation["responses"]["409"]["description"],
        )
        self.assertIn("INVALID_FILE", operation["responses"]["422"]["description"])

        status_operation = self.app.openapi()["paths"][
            "/api/admin/ingestion-runs/{ingestion_run_id}"
        ]["get"]
        self.assertIn("NOT_FOUND", status_operation["responses"]["404"]["description"])

    def _upload(self, filename: str, content: bytes):
        return self.client.post(
            "/api/admin/documents",
            data={
                "title": " 문서 제목 ",
                "sourceUrl": "https://docs.riido.io/new-guide",
                "category": " guide ",
            },
            files={"file": (filename, content, "application/octet-stream")},
        )


if __name__ == "__main__":
    unittest.main()
