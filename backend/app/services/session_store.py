"""Isolated, UUID-scoped session storage for uploaded files. PRD FR-08, NFR-12, NFR-13."""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def new_session_id() -> str:
    """Session IDs must be server-generated UUIDs, never user-supplied (NFR-13)."""
    return str(uuid.uuid4())


def session_dir(upload_dir: str, session_id: str) -> Path:
    path = Path(upload_dir) / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_uploaded_csv(upload_dir: str, session_id: str, raw_bytes: bytes) -> Path:
    """Stores the upload under a session-scoped filename, never the original filename
    (NFR-12), inside an isolated per-session directory (FR-08)."""
    directory = session_dir(upload_dir, session_id)
    file_path = directory / "original.csv"
    file_path.write_bytes(raw_bytes)
    return file_path


def _meta_path(upload_dir: str, session_id: str) -> Path:
    return session_dir(upload_dir, session_id) / "meta.json"


def write_session_meta(upload_dir: str, session_id: str, meta: dict) -> None:
    """Small JSON sidecar holding what routes_upload/routes_goal collect (file info,
    goal context) before the graph exists for this session -- the graph/checkpointer only
    gets a state to track once POST /api/profile/{id} actually starts it."""
    _meta_path(upload_dir, session_id).write_text(json.dumps(meta))


def read_session_meta(upload_dir: str, session_id: str) -> Optional[dict]:
    path = _meta_path(upload_dir, session_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _last_touched(directory: Path) -> float:
    """Newest mtime anywhere in the session folder. The directory's own mtime is not enough:
    on some filesystems rewriting cleaned.csv in place doesn't bump it, and a session someone
    is actively working in would look idle."""
    newest = directory.stat().st_mtime
    for child in directory.rglob("*"):
        try:
            newest = max(newest, child.stat().st_mtime)
        except OSError:  # vanished mid-scan; nothing to weigh in
            continue
    return newest


def purge_expired_sessions(upload_dir: str, retention_seconds: int, *, now: float | None = None) -> int:
    """Deletes session folders untouched for longer than retention_seconds. Returns how many
    were removed.

    Never raises: this runs unattended in the background, and a permission error on one
    folder must not stop the rest being swept or take the app down with it. A retention of 0
    disables expiry entirely.
    """
    if retention_seconds <= 0:
        return 0

    root = Path(upload_dir)
    if not root.exists():
        return 0

    cutoff = (time.time() if now is None else now) - retention_seconds
    removed = 0

    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        try:
            if _last_touched(directory) >= cutoff:
                continue
            shutil.rmtree(directory)
            removed += 1
            logger.info("Purged expired session directory %s", directory.name)
        except OSError:
            logger.warning("Could not purge session directory %s", directory.name, exc_info=True)

    return removed
