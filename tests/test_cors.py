import os
import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app


@asynccontextmanager
async def test_lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


ALLOWED_ORIGIN = "http://localhost:3000"
SECOND_ORIGIN = "http://localhost:5173"
UNKNOWN_ORIGIN = "http://unknown.example.com"


class CorsTest(unittest.TestCase):
    def setUp(self) -> None:
        environment = patch.dict(
            os.environ,
            {"CORS_ORIGINS": f"{ALLOWED_ORIGIN}, {SECOND_ORIGIN}"},
        )
        environment.start()
        self.addCleanup(environment.stop)

        get_settings.cache_clear()
        self.addCleanup(get_settings.cache_clear)

        with patch("app.main.lifespan", test_lifespan):
            self.app = create_app()
        self.client = TestClient(self.app)

    def test_parses_comma_separated_origins(self) -> None:
        self.assertEqual(
            [ALLOWED_ORIGIN, SECOND_ORIGIN],
            get_settings().cors_origin_list,
        )

    def test_preflight_allows_configured_origin(self) -> None:
        response = self._preflight(ALLOWED_ORIGIN)

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            ALLOWED_ORIGIN,
            response.headers["access-control-allow-origin"],
        )
        self.assertIn("POST", response.headers["access-control-allow-methods"])

    def test_preflight_rejects_unknown_origin(self) -> None:
        response = self._preflight(UNKNOWN_ORIGIN)

        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_response_includes_allow_origin_for_configured_origin(self) -> None:
        response = self.client.get("/health", headers={"Origin": SECOND_ORIGIN})

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            SECOND_ORIGIN,
            response.headers["access-control-allow-origin"],
        )

    def test_response_omits_allow_origin_for_unknown_origin(self) -> None:
        response = self.client.get("/health", headers={"Origin": UNKNOWN_ORIGIN})

        self.assertEqual(200, response.status_code)
        self.assertNotIn("access-control-allow-origin", response.headers)

    def _preflight(self, origin: str):
        return self.client.options(
            "/api/chat",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )


if __name__ == "__main__":
    unittest.main()
