from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    database_url: str = "postgresql+asyncpg://riido:riido@localhost:5432/riido"
    openai_api_key: Optional[str] = None
    corpus_dir: str = "data"
    cors_origins: str = "http://localhost:3000"
    # 질문 판별·정본 캐시 서빙 전체 스위치(R19). 꺼져 있으면 판별 행·캐시 시도·질문 임베딩
    # 저장 없이 기존 턴 흐름과 같다.
    question_grouping_enabled: bool = False

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
