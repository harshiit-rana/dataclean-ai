"""Environment-driven settings. PRD Appendix B."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

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

    # Uploaded CSVs, cleaned copies and audit reports are deleted this long after a session
    # was last touched. Matches redis_ttl_seconds by default so the files and the checkpoint
    # that refers to them expire together -- a session whose state has already gone can't be
    # resumed, and keeping the user's data on disk past that point is storage cost and
    # privacy exposure for nothing. Set to 0 to keep files indefinitely.
    upload_retention_seconds: int = 86400
    # How often the sweep runs. Not a precision job; the point is that it happens at all.
    cleanup_interval_seconds: int = 3600

    chroma_persist_dir: str = "./chroma_db"  # reserved, currently unused — see README

    app_env: str = "development"
    log_level: str = "INFO"

    # Require the token issued at upload on every request scoped to that session. Left off in
    # development so `docker compose up` and a browser is still all it takes to try the app;
    # on by default anywhere else, because a session id alone should not be enough to read
    # someone's uploaded data. See services/session_access.py.
    require_session_tokens: Optional[bool] = None

    frontend_origin: str = "http://localhost:5173"

    @property
    def session_tokens_required(self) -> bool:
        """Explicit setting wins; otherwise on everywhere but development."""
        if self.require_session_tokens is not None:
            return self.require_session_tokens
        return self.app_env.lower() != "development"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
