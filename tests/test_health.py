import unittest
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.core.system_version import SYSTEM_VERSION
from app.database.session import get_db_session
from app.main import create_app
from app.profiles.readiness import ServiceReadiness


class HealthApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.session = AsyncMock()
        self.app = create_app()

        async def override_db_session() -> AsyncIterator[AsyncMock]:
            yield self.session

        self.app.dependency_overrides[get_db_session] = override_db_session
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.app.dependency_overrides.clear()

    def test_health_check_returns_ok(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {"status": "ok", "version": SYSTEM_VERSION},
            response.json(),
        )

    def test_database_health_check_returns_connected(self) -> None:
        response = self.client.get("/health/db")

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {"status": "ok", "database": "connected"},
            response.json(),
        )
        self.session.execute.assert_awaited_once()

    def test_database_health_check_hides_connection_error(self) -> None:
        self.session.execute.side_effect = SQLAlchemyError(
            "secret connection details"
        )

        response = self.client.get("/health/db")

        self.assertEqual(503, response.status_code)
        self.assertEqual(
            {"status": "error", "database": "unavailable"},
            response.json(),
        )
        self.assertNotIn("secret", response.text)


    def test_profile_versions_check_passes_when_every_service_is_servable(self) -> None:
        results = (
            ServiceReadiness("HELP_CHATBOT", 1, ()),
            ServiceReadiness("HELP_CHATBOT_TEST", 2, ()),
        )
        with patch(
            "app.api.health.check_published_profile_versions",
            new=AsyncMock(return_value=results),
        ) as check:
            response = self.client.get("/health/profiles")

        self.assertEqual(200, response.status_code)
        self.assertEqual({"status": "ok", "profiles": "servable"}, response.json())
        self.assertIs(self.session, check.await_args.args[0])

    def test_profile_versions_check_fails_and_logs_reasons_only(self) -> None:
        results = (
            ServiceReadiness("HELP_CHATBOT", 1, ()),
            ServiceReadiness(
                "HELP_CHATBOT_TEST",
                2,
                ("rewrite: 실행 중인 코드에 없는 프롬프트 버전 rewrite@v9",),
            ),
        )
        with patch(
            "app.api.health.check_published_profile_versions",
            new=AsyncMock(return_value=results),
        ), self.assertLogs("app.api.health", level="ERROR") as logs:
            response = self.client.get("/health/profiles")

        self.assertEqual(503, response.status_code)
        # 공개 주소로도 열려 있어 사유는 응답에 싣지 않고 로그에만 남긴다.
        self.assertEqual({"status": "error", "profiles": "unservable"}, response.json())
        self.assertNotIn("rewrite", response.text)
        self.assertIn("service=HELP_CHATBOT_TEST, revision_id=2", logs.output[0])
        self.assertIn("rewrite@v9", logs.output[0])

    def test_profile_versions_check_hides_database_error(self) -> None:
        with patch(
            "app.api.health.check_published_profile_versions",
            new=AsyncMock(side_effect=SQLAlchemyError("secret connection details")),
        ), self.assertLogs("app.api.health", level="ERROR"):
            response = self.client.get("/health/profiles")

        self.assertEqual(503, response.status_code)
        self.assertEqual({"status": "error", "database": "unavailable"}, response.json())
        self.assertNotIn("secret", response.text)


if __name__ == "__main__":
    unittest.main()
