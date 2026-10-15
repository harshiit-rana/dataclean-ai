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

    # The most work one training request may be estimated to do, in "row-fits": one row fitted by the
    # cheapest model counts as one, and the forests and boosted trees count as many (see
    # modeling.ModelSpec.cost_weight). Past it the request is reduced rather than left to run for as long
    # as it takes -- model choice is made on a sample of the training rows, then the most expensive
    # families are skipped, then the request is refused with a message saying what would fit. Every
    # reduction is stated in the result. 0 turns the limit off. The default lets a file of about a
    # hundred thousand rows through unreduced.
    training_cost_limit: int = 30_000_000

    # Trained models kept per session, each in its own folder with its fitted pipeline, so an earlier one can
    # be compared or brought back. The oldest beyond this are pruned; the one in use never is. A fitted forest
    # can run to tens of megabytes, so this is the disk a session can take for models. At least 1.
    max_stored_runs: int = 10

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
