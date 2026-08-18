import unittest
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.database.session import get_db_session
from app.main import create_app


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
        self.assertEqual({"status": "ok"}, response.json())

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


if __name__ == "__main__":
    unittest.main()
