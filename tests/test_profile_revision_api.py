"""프로필 버전 API의 기본 비활성, 요청 제한과 오류 계약. 모델은 호출하지 않는다."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.admin.profile_revisions.router import get_profile_revision_service
from app.admin.profile_revisions.schema import ComponentDTO
from app.core.config import Settings
from app.main import create_app
from app.profiles.lifecycle import ProfileRevisionError, RevisionDetail
from app.profiles.stage_settings import code_components


class ProfileRevisionApiTest(TestCase):
    def setUp(self):
        with patch("app.main.get_settings", return_value=Settings(_env_file=None, profile_revision_admin_enabled=True)):
            self.app = create_app()
        self.service = SimpleNamespace(**{
            name: AsyncMock() for name in ("list", "detail", "create", "testing", "publish", "rollback", "delete")
        })
        self.app.dependency_overrides[get_profile_revision_service] = lambda: self.service
        self.client = TestClient(self.app)
        self.base = "/api/admin/profiles/SYNTHETIC_A/revisions"
        self.revision = SimpleNamespace(
            id=17, profile_id=1, document_group_id=2, version=3, status="DRAFT",
            semantic_cache_enabled=False, exact_cache_enabled=True,
            created_at=datetime(2026, 10, 10, tzinfo=timezone.utc),
        )
        self.detail = RevisionDetail(self.revision, code_components())

    def test_router_is_disabled_by_default(self):
        with patch("app.main.get_settings", return_value=Settings(_env_file=None)):
            app = create_app()
        for route in app.routes:
            self.assertNotIn("/profiles/{profile_key}/revisions", getattr(route, "path", ""))
        self.assertEqual(404, TestClient(app).get(self.base).status_code)

    def test_list_detail_and_default_creation(self):
        self.service.list.return_value = [self.revision]
        self.service.detail.return_value = self.detail
        self.service.create.return_value = self.detail
        self.assertEqual(3, self.client.get(self.base).json()["revisions"][0]["version"])
        response = self.client.get(self.base + "/17")
        self.assertEqual(7, len(response.json()["components"]))
        self.assertNotIn("instructions", response.text)
        response = self.client.post(self.base, json={})
        self.assertEqual(201, response.status_code)
        self.service.create.assert_awaited_once_with(
            "SYNTHETIC_A", components=None, semantic_cache_enabled=None, exact_cache_enabled=None,
        )

    def test_full_components_and_cache_overrides(self):
        self.service.create.return_value = self.detail
        payload = [ComponentDTO.model_validate(spec).model_dump(mode="json", by_alias=True) for spec in code_components()]
        response = self.client.post(self.base, json={"components": payload, "exactCacheEnabled": False})
        self.assertEqual(201, response.status_code)
        kwargs = self.service.create.await_args.kwargs
        self.assertEqual(code_components(), tuple(kwargs["components"]))
        self.assertFalse(kwargs["exact_cache_enabled"])

    def test_rejects_mutable_or_cross_service_fields(self):
        for field in ("documentGroupId", "profileId", "status", "version", "instructions"):
            response = self.client.post(self.base, json={field: 10})
            self.assertEqual(422, response.status_code)
            self.assertEqual("INVALID_REQUEST", response.json()["code"])
        self.service.create.assert_not_awaited()
        self.assertEqual(405, self.client.patch(self.base + "/17", json={}).status_code)

    def test_transitions_and_delete_forward_service_scope(self):
        for method in ("testing", "publish", "rollback"):
            getattr(self.service, method).return_value = self.detail
            self.assertEqual(200, self.client.post(self.base + f"/17/{method}").status_code)
            getattr(self.service, method).assert_awaited_once_with("SYNTHETIC_A", 17)
        self.assertEqual(204, self.client.delete(self.base + "/17").status_code)
        self.service.delete.assert_awaited_once_with("SYNTHETIC_A", 17)

    def test_consistent_domain_errors(self):
        for code, status in (("NOT_FOUND", 404), ("INVALID_PROFILE_CONFIGURATION", 400), ("PROFILE_REVISION_CONFLICT", 409)):
            self.service.publish.side_effect = ProfileRevisionError(code, "synthetic error", status)
            response = self.client.post(self.base + "/17/publish")
            self.assertEqual(status, response.status_code)
            self.assertEqual({"code": code, "message": "synthetic error"}, response.json())
