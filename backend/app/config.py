"""Environment-driven settings. PRD Appendix B."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "anthropic"  # "anthropic" or "groq" -- see services/llm_client.py

    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-5"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_ttl_seconds: int = 86400

    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50

    chroma_persist_dir: str = "./chroma_db"  # reserved, currently unused — see README

    app_env: str = "development"
    log_level: str = "INFO"

    frontend_origin: str = "http://localhost:5173"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
