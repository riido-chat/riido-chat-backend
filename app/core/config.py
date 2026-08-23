from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    database_url: str = "postgresql+asyncpg://riido:riido@localhost:5432/riido"
    openai_api_key: Optional[str] = None
    corpus_dir: str = "data"
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> List[str]:
        """쉼표로 구분한 CORS 허용 오리진을 목록으로 변환한다."""

        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
