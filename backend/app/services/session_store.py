"""Isolated, UUID-scoped session storage for uploaded files. PRD FR-08, NFR-12, NFR-13."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional


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
