import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(BASE_DIR)

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = Field(default="postgresql+asyncpg://postgres:postgrespassword@localhost:5432/fake_news_detection")
    DATABASE_SYNC_URL: str = Field(default="postgresql://postgres:postgrespassword@localhost:5432/fake_news_detection")

    # LLM
    LLM_PROVIDER: str = Field(default="openai")
    LLM_ENDPOINT: str = Field(default="https://chinh-de--qwen3-4b-t4-production-v2-serve.modal.run/v1")
    LLM_API_KEY: str = Field(default="EMPTY")
    LLM_MODEL: str = Field(default="Qwen/Qwen3-4B-AWQ")
    GOOGLE_API_KEY: str = Field(default="")

    # SLM
    SLM_MODEL_PATH: str = Field(default=os.path.join(BACKEND_DIR, "model"))
    SLM_REPO_ID: str = Field(default="chinhde/fake-news-detection-slm")
    HF_TOKEN: str = Field(default="")

    # Embedding Model
    EMBEDDING_MODEL_CACHE_DIR: str = Field(default=os.path.join(BACKEND_DIR, "model", "embeddings"))

    # CORS
    CORS_ORIGINS: str = Field(default="http://localhost:3000,http://localhost:5173,chrome-extension://*")

    # Security
    JWT_SECRET: str = Field(default="super-secret-jwt-key-change-in-production")
    ADMIN_USERNAME: str = Field(default="admin")
    ADMIN_PASSWORD: str = Field(default="adminpassword")

    # Seeding
    SEED_CORPUS_CSV: str = Field(default=os.path.join(BASE_DIR, "seed_data", "vifactcheck_all.csv"))
    # Logging
    ENABLE_ANALYSIS_LOG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_file=os.path.join(BACKEND_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

settings = Settings()
