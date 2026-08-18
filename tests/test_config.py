import os
import unittest
from unittest.mock import patch

from app.core.config import Settings, get_settings


class SettingsTest(unittest.TestCase):
    def tearDown(self) -> None:
        get_settings.cache_clear()

    def test_database_url_can_select_local_database(self) -> None:
        local_url = "postgresql+asyncpg://riido:riido@localhost:5432/riido"

        with patch.dict(os.environ, {"DATABASE_URL": local_url}):
            get_settings.cache_clear()
            settings = get_settings()

        self.assertEqual(local_url, settings.database_url)

    def test_database_url_can_select_shared_test_database(self) -> None:
        shared_url = "postgresql+asyncpg://riido:secret@shared-db:5432/riido"

        with patch.dict(os.environ, {"DATABASE_URL": shared_url}):
            get_settings.cache_clear()
            settings = get_settings()

        self.assertEqual(shared_url, settings.database_url)

    def test_settings_can_be_created_without_a_dotenv_file(self) -> None:
        settings = Settings(_env_file=None)

        self.assertEqual("local", settings.app_env)
        self.assertTrue(settings.database_url.startswith("postgresql+asyncpg://"))


if __name__ == "__main__":
    unittest.main()
