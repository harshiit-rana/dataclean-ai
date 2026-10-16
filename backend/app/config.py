"""Environment-driven settings. PRD Appendix B."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "anthropic"  # "anthropic" or "groq" -- see services/llm_client.py

    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-5"

    groq_api_key: str = ""
    # Groq retired llama-3.3-70b-versatile: on 16 October 2026 the API answered model_not_found for it (#114). A
    # model name is an external dependency like any other; check `groq.Groq().models.list()` when this fails.
    groq_model: str = "openai/gpt-oss-120b"

    redis_host: str = "localhost"
    redis_port: int = 6379
    # REDIS_TTL_SECONDS is gone (#120): checkpoints no longer expire on their own; the sweep deletes a session's
    # checkpoint with its files, and never a kept project's. An old .env that still sets it is ignored.

    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50
    # The byte limit alone lets a 50 MB file hold hundreds of thousands of columns or one enormous cell, either of
    # which can exhaust memory in parsing, profiling (a loop per column) or one-hot encoding. Checked on every
    # uploaded file: the columns from the header before the file is parsed, the rest straight after.
    max_columns: int = 1000
    max_rows: int = 2_000_000
    # The longest a single cell may be, in characters. Excel's own limit for a cell.
    max_cell_chars: int = 32_767

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

    # Uploaded CSVs, cleaned copies and audit reports are deleted this long after a session was last touched,
    # together with the session's checkpoint in Redis -- a session whose state has gone can't be resumed, and
    # keeping the user's data on disk past that point is storage cost and privacy exposure for nothing. A project
    # its owner chose to keep (#120) is never deleted by this. Set to 0 to keep every session indefinitely.
    upload_retention_seconds: int = 86400
    # How often the sweep runs. Not a precision job; the point is that it happens at all.
    cleanup_interval_seconds: int = 3600

    # How long work runs (#103): "inline" runs training, searches and applications in a thread of the API process,
    # recorded as jobs on disk; "worker" only queues them, for `python -m app.worker` (a container of its own under
    # Docker Compose) to run, so restarting the API does not lose a search. Both read progress from the job's
    # events file. See services/jobs.py and docs/decisions/0004-platform-architecture.md.
    job_runner: str = "inline"

    chroma_persist_dir: str = "./chroma_db"  # reserved, currently unused — see README

    # "production" unless set. It used to default to "development", which turned session tokens off -- so a hosted
    # deployment that forgot to set it served every session's data to anyone holding a session id. Development is
    # now something a setup says out loud: .env.example and docker-compose.yml set it for the local setup.
    app_env: str = "production"
    log_level: str = "INFO"
    # "text" (the default) or "json": one JSON object per line, with the request id, for a log collector.
    log_format: str = "text"

    # Require the token issued at upload on every request scoped to that session. On unless development is
    # explicitly enabled (APP_ENV=development) or this is set to false, because a session id alone should not be
    # enough to read someone's uploaded data; the app logs a warning at startup whenever it is off. See
    # services/session_access.py.
    require_session_tokens: Optional[bool] = None

    @field_validator("require_session_tokens", mode="before")
    @classmethod
    def _blank_means_unset(cls, value):
        """`REQUIRE_SESSION_TOKENS=` left blank, as .env.example ships it, means "follow APP_ENV". It used to fail
        to parse as a boolean and stop the app at startup -- for anyone who copied .env.example as the README
        says, which the compose job in CI was the first thing to do."""
        return None if isinstance(value, str) and not value.strip() else value

    frontend_origin: str = "http://localhost:5173"

    @property
    def session_tokens_required(self) -> bool:
        """Explicit setting wins; otherwise on everywhere but an explicitly development setup."""
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
