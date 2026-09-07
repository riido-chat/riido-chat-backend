import unittest
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from unittest.mock import ANY, AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.admin.dependencies import get_admin_ingestion_service
from app.database.models import ExecutionStatus
from app.document.ingestion_service import (
    AcceptedIngestion,
    AdminIngestionService,
    DocumentNotRevisableError,
    IngestionRunDetail,
)
from app.main import create_app


STARTED_AT = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)


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

    def _success_detail(self, **changes) -> IngestionRunDetail:
        base = {
            "ingestion_run_id": 101,
            "document_source_id": 42,
            "status": ExecutionStatus.SUCCESS,
            "stage": "PERSISTING",
            "document_version_id": 5001,
            "version_no": 1,
            "section_count": 2,
            "chunk_count": 2,
            "error_code": None,
            "error_message": None,
            "started_at": STARTED_AT,
            "finished_at": STARTED_AT,
            "result_code": "CREATED",
            "chunk_stats": {
                "added": 2,
                "changed": 0,
                "deleted": 0,
                "reused": 0,
            },
            "duplicate_of": None,
        }
        base.update(changes)
        return IngestionRunDetail(**base)

    def test_upload_returns_result_synchronously(self) -> None:
        self.service.start_new_document.return_value = AcceptedIngestion(
            ingestion_run_id=101,
            document_source_id=42,
        )
        self.service.read_finished_run.return_value = self._success_detail()

        with patch(
            "app.admin.router.run_admin_ingestion",
            new=AsyncMock(),
        ) as run:
            response = self._upload(
                "guide.md",
                "# 문서\n\n## 안내\n\n본문".encode("utf-8"),
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "ingestionRunId": 101,
                "documentId": 42,
                "resultCode": "CREATED",
                "documentVersionId": 5001,
                "versionNo": 1,
                "sectionCount": 2,
                "chunkCount": 2,
                "chunkStats": {
                    "added": 2,
                    "changed": 0,
                    "deleted": 0,
                    "reused": 0,
                },
                "duplicateOf": None,
            },
            response.json(),
        )
        # 요청 필드는 file 과 title 둘뿐이다
        self.service.start_new_document.assert_awaited_once_with(
            group_id=1,
            title="문서 제목",
            filename="guide.md",
        )
        # background task 가 아니라 요청 안에서 끝난다
        run.assert_awaited_once_with(
            101,
            "# 문서\n\n## 안내\n\n본문",
            ANY,
        )

    def test_upload_failure_becomes_500_without_stage(self) -> None:
        self.service.start_new_document.return_value = AcceptedIngestion(
            ingestion_run_id=101,
            document_source_id=42,
        )
        self.service.read_finished_run.return_value = self._success_detail(
            status=ExecutionStatus.FAILED,
            error_code="INVALID_FILE",
            error_message="문서 구조를 분석하지 못했습니다.",
        )

        with patch("app.admin.router.run_admin_ingestion", new=AsyncMock()):
            response = self._upload("guide.md", b"# guide")

        self.assertEqual(500, response.status_code)
        body = response.json()
        self.assertEqual("INVALID_FILE", body["code"])
        self.assertEqual("문서 구조를 분석하지 못했습니다.", body["message"])
        # 오류 본문은 code 와 message 둘뿐이다
        self.assertEqual({"code", "message"}, set(body))

    def test_upload_rejects_category_field(self) -> None:
        """분류는 요청 필드가 아니다."""

        response = self.client.post(
            "/api/admin/document-groups/1/documents",
            data={"title": "문서 제목", "category": "guide"},
            files={"file": ("guide.md", b"# guide", "text/markdown")},
        )

        self.assertEqual(422, response.status_code)
        self.service.start_new_document.assert_not_awaited()

    def test_rejects_source_url_field(self) -> None:
        # 콘솔 문서의 canonical_uri 는 서버가 만든다. 입력으로 받지 않는다
        response = self.client.post(
            "/api/admin/document-groups/1/documents",
            data={
                "title": "문서 제목",
                "sourceUrl": "https://docs.riido.io/new-guide",
            },
            files={"file": ("guide.md", b"# guide", "text/markdown")},
        )

        self.assertEqual(422, response.status_code)

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
            "/api/admin/document-groups/1/documents",
            data={
                "title": "문서 제목",
                "extra": "not-allowed",
            },
            files={"file": ("guide.md", b"# guide", "text/markdown")},
        )

        self.assertEqual(422, response.status_code)
        self.service.start_new_document.assert_not_awaited()

    def test_revision_upload_rejects_gitbook_document(self) -> None:
        self.service.start_document_revision.side_effect = (
            DocumentNotRevisableError()
        )

        response = self.client.post(
            "/api/admin/documents/42/versions",
            files={"file": ("guide.md", b"# guide", "text/markdown")},
        )

        self.assertEqual(409, response.status_code)
        self.assertEqual("DOCUMENT_NOT_REVISABLE", response.json()["code"])

    def test_ingestion_run_query_route_is_gone(self) -> None:
        """동기 전환으로 실행 조회가 사라졌다."""

        response = self.client.get("/api/admin/ingestion-runs/101")

        self.assertEqual(404, response.status_code)

    def test_openapi_documents_multipart_request_and_result_response(self) -> None:
        operation = self.app.openapi()["paths"][
            "/api/admin/document-groups/{group_id}/documents"
        ]["post"]

        self.assertIn("multipart/form-data", operation["requestBody"]["content"])
        self.assertIn("200", operation["responses"])
        self.assertNotIn("202", operation["responses"])
        self.assertIn("500", operation["responses"])
        self.assertIn("409", operation["responses"])
        self.assertIn("413", operation["responses"])
        self.assertIn("FILE_TOO_LARGE", operation["responses"]["413"]["description"])
        self.assertIn(
            "JOB_IN_PROGRESS",
            operation["responses"]["409"]["description"],
        )
        self.assertIn("INVALID_FILE", operation["responses"]["422"]["description"])
        self.assertNotIn(
            "/api/admin/ingestion-runs/{ingestion_run_id}",
            self.app.openapi()["paths"],
        )

    def _upload(self, filename: str, content: bytes):
        return self.client.post(
            "/api/admin/document-groups/1/documents",
            data={"title": " 문서 제목 "},
            files={"file": (filename, content, "application/octet-stream")},
        )


if __name__ == "__main__":
    unittest.main()
