"""Optional limits for a hosted deployment: requests per client, storage per project, unfinished jobs per session
(#123, docs/security/threat-model.md risk 9).

One upload and one training request were already bounded (size and shape limits, the processing limit). Nothing
bounded how many a client started, and a kept project never expires. That is fine for the documented single-user
setup, so every limit here is off (0) by default; a hosted instance turns on what it needs.

- `RATE_LIMIT_PER_MINUTE`: uploads, trainings, searches and applications one client may start per minute. A client
  is its session token when it sends one, its address otherwise. Counted in this process's memory, so with several
  API processes each counts on its own -- put a proxy's rate limiting in front for a hard limit across them.
- `PROJECT_QUOTA_MB`: the most disk a session or project may use. Over it, new uploads into it, trainings and
  applications are refused with a message saying so; nothing is deleted.
- `MAX_UNFINISHED_JOBS_PER_SESSION`: queued or running jobs one session may have at once.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.core.errors import AppError
from app.services import jobs

WINDOW_SECONDS = 60.0


class TooManyRequestsError(AppError):
    status_code = 429

    def __init__(self, detail: str, code: str):
        super().__init__("Too many requests", detail, code=code)


class QuotaExceededError(AppError):
    status_code = 413

    def __init__(self, detail: str):
        super().__init__("Storage quota reached", detail, code="quota_exceeded")


_lock = threading.Lock()
_recent: dict[str, deque[float]] = defaultdict(deque)


def _client(request: Request) -> str:
    token = request.headers.get("x-session-token")
    if token:
        return f"token:{token}"
    return f"address:{request.client.host if request.client else 'unknown'}"


def _check_rate(settings: Settings, request: Request, *, now: float | None = None) -> None:
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return
    now = time.monotonic() if now is None else now
    key = _client(request)
    with _lock:
        stamps = _recent[key]
        while stamps and now - stamps[0] >= WINDOW_SECONDS:
            stamps.popleft()
        if len(stamps) >= limit:
            wait = int(WINDOW_SECONDS - (now - stamps[0])) + 1
            raise TooManyRequestsError(
                f"This client has started {limit} uploads, trainings or applications in the last minute, the most this "
                f"server allows. Try again in about {wait} seconds.",
                "rate_limited",
            )
        stamps.append(now)


def folder_size(directory: Path) -> int:
    return sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())


def _check_quota(settings: Settings, directory: Path) -> None:
    quota_mb = settings.project_quota_mb
    if quota_mb <= 0 or not directory.is_dir():
        return
    used = folder_size(directory)
    if used >= quota_mb * 1024 * 1024:
        raise QuotaExceededError(
            f"This project uses {used / 1024 / 1024:,.1f} MB, at or over the {quota_mb:,} MB this server allows per "
            f"project. Delete earlier runs or applications, or start a new project; nothing has been deleted."
        )


def _check_jobs(settings: Settings, directory: Path) -> None:
    cap = settings.max_unfinished_jobs_per_session
    if cap <= 0:
        return
    unfinished = sum(1 for folder in jobs.session_jobs(directory)
                     if jobs.read_state(folder).get("state") not in jobs.FINISHED)
    if unfinished >= cap:
        raise TooManyRequestsError(
            f"This project already has {unfinished} trainings or searches waiting or running, the most this server "
            f"allows at once. Wait for one to finish.",
            "too_many_jobs",
        )


def limit_uploads(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """For POST /api/upload: the rate limit. A new upload is a new session, so there is no quota to check yet."""
    _check_rate(settings, request)


def limit_work(session_id: str, request: Request, settings: Settings = Depends(get_settings)) -> None:
    """For routes that start work on a session: the rate limit, its storage quota and its unfinished jobs."""
    _check_rate(settings, request)
    directory = Path(settings.upload_dir) / session_id
    _check_quota(settings, directory)
    _check_jobs(settings, directory)


def reset() -> None:
    """Forget every client's recent requests (tests)."""
    with _lock:
        _recent.clear()
