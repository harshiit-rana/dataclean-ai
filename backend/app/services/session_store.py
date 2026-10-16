"""Isolated, UUID-scoped session storage for uploaded files. PRD FR-08, NFR-12, NFR-13."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.services.session_access import META_KEY as TOKEN_HASH_KEY

logger = logging.getLogger(__name__)

# Serialises read-modify-write of the meta sidecar *within this process*. The deployment is a
# single process (see STATUS.md), and what this guards against is two requests of one session
# -- an undo and a chat instruction -- each reading the list, appending, and writing back, so
# that one version vanishes. The atomic write below stops a reader seeing half a file; this
# stops the lost update.
_META_LOCK = threading.Lock()

# A long editing session could otherwise grow the list without bound. Version 1 (the upload)
# is always kept, with the newest versions after it; `dataset_versions_dropped` says how many
# fell off so the gap is stated rather than hidden.
MAX_DATASET_VERSIONS = 200

# Set in the sidecar of a session its owner chose to keep as a project (#120). See docs/decisions/0004.
KEPT_KEY = "kept"


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
    _write_meta_file(_meta_path(upload_dir, session_id), meta)


def _write_meta_file(path: Path, meta: dict) -> None:
    """Writes to a temporary file and renames it over the real one, so a reader -- or a crash
    mid-write -- never sees half a file. The sidecar holds the goal, the rules, the access-token
    hash and the dataset versions; a truncated one would lose all of them."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(meta))
    os.replace(temporary, path)


def read_session_meta(upload_dir: str, session_id: str) -> Optional[dict]:
    path = _meta_path(upload_dir, session_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _current_data_path(directory: Path) -> Path:
    """The file the model path reads: the cleaned data once cleaning has run, the upload
    before. Kept identical to `routes_model._load_session` so a dataset version and a run's
    fingerprint describe the same bytes."""
    cleaned = directory / "cleaned.csv"
    return cleaned if cleaned.exists() else directory / "original.csv"


def record_dataset_version_in(directory: Path, event: str, frame=None, actions=None) -> Optional[dict]:
    """Appends a dataset version to the session's meta when its working data has changed.

    The session's data is rewritten in place -- by the cleaning run, by undo, by a chat
    instruction -- so nothing stable existed to attach a model, a test-set look or an export
    to. Each entry is `{version, fingerprint, rows, columns, at, event}`; the fingerprint is the
    full content hash (model_store.full_fingerprint) of the data as the model path will read it.

    Appended only when the fingerprint differs from the *latest* version. An undo that restores
    an earlier state is therefore recorded as a new version whose fingerprint equals an older
    one: the history stays chronological, which is the point of it. A step that changed nothing
    adds nothing.

    Never raises. A version is bookkeeping about work already done, and failing the cleaning
    run, the undo or the chat instruction over it would lose something that exists. Returns the
    entry appended, or None.

    `actions` is the action log as it stands after the change. Its applied entries are kept on the
    version, so the version can be rebuilt from the upload by replaying them (version_diff.py) and
    compared with another.
    """
    try:
        from app.services import model_store

        meta_file = directory / "meta.json"
        if not meta_file.exists():
            return None  # not a session this app created; nothing to attach a version to

        if frame is None:
            import pandas as pd

            frame = pd.read_csv(_current_data_path(directory))
        fingerprint = model_store.full_fingerprint(frame)

        with _META_LOCK:
            meta = json.loads(meta_file.read_text())
            versions = list(meta.get("dataset_versions") or [])

            if not versions and event != "upload":
                # A session that predates versioning. The upload is still on disk, so version
                # 1 can be stated truthfully rather than guessed at.
                import pandas as pd

                original = pd.read_csv(directory / "original.csv")
                versions.append(_version_entry(1, original, model_store, "upload"))

            if versions and versions[-1]["fingerprint"] == fingerprint:
                if meta.get("dataset_versions") != versions:
                    meta["dataset_versions"] = versions
                    _write_meta_file(meta_file, meta)
                return None

            entry = _version_entry(
                (versions[-1]["version"] + 1) if versions else 1, frame, model_store, event
            )
            entry["fingerprint"] = fingerprint
            if actions is not None:
                from app.services.run_metadata import applied_actions

                entry["actions"] = applied_actions(list(actions))
            versions.append(entry)

            if len(versions) > MAX_DATASET_VERSIONS:
                overflow = len(versions) - MAX_DATASET_VERSIONS
                versions = [versions[0], *versions[1 + overflow :]]
                meta["dataset_versions_dropped"] = (
                    int(meta.get("dataset_versions_dropped", 0)) + overflow
                )

            meta["dataset_versions"] = versions
            _write_meta_file(meta_file, meta)
            return entry
    except Exception:  # noqa: BLE001 -- bookkeeping must never fail the work it describes
        logger.warning("Could not record a dataset version (%s)", event, exc_info=True)
        return None


def _version_entry(version: int, frame, model_store, event: str) -> dict:
    return {
        "version": version,
        "fingerprint": model_store.full_fingerprint(frame),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
    }


def record_dataset_version(upload_dir: str, session_id: str, event: str, frame=None, actions=None):
    """`record_dataset_version_in` for a session addressed the way the routes address it."""
    return record_dataset_version_in(Path(upload_dir) / session_id, event, frame, actions)


def dataset_versions(upload_dir: str, session_id: str) -> list[dict]:
    """The session's versions, oldest first. Empty for a session that predates versioning and
    has not been touched since."""
    meta = read_session_meta(upload_dir, session_id) or {}
    return list(meta.get("dataset_versions") or [])


def current_dataset_version(upload_dir: str, session_id: str) -> Optional[dict]:
    versions = dataset_versions(upload_dir, session_id)
    return versions[-1] if versions else None


def current_dataset_version_in(directory: Path, *, ensure: bool = False) -> Optional[dict]:
    """The latest dataset version of the session in `directory`.

    `ensure=True` is for the moment something is about to be attached to a version -- a look
    at the held-out rows, say. A session that predates versioning has none, and attaching to
    "no version" would leave that look unable to say what data it was taken on, so the list is
    started first (see `record_dataset_version_in`) and read again. Never raises.
    """

    def latest() -> Optional[dict]:
        try:
            meta = json.loads((directory / "meta.json").read_text())
        except (OSError, ValueError):
            return None
        versions = meta.get("dataset_versions") or []
        return versions[-1] if versions else None

    version = latest()
    if version is None and ensure:
        record_dataset_version_in(directory, "first_seen")
        version = latest()
    return version


# Reading a session counts as using it: the guard on every session route touches this marker (session_access), so
# a session someone is only viewing keeps its files for the retention period, in step with its Redis checkpoint,
# whose expiry is refreshed on every read. Touched at most once a minute, not on every request.
ACTIVITY_MARKER = ".last_active"
ACTIVITY_GRANULARITY_SECONDS = 60


def touch_session(directory: Path) -> None:
    """Records that the session in `directory` was used now. Never raises: bookkeeping must not fail a request."""
    try:
        marker = Path(directory) / ACTIVITY_MARKER
        if marker.exists() and time.time() - marker.stat().st_mtime < ACTIVITY_GRANULARITY_SECONDS:
            return
        marker.touch()
    except OSError:
        logger.debug("Could not record activity for %s", directory, exc_info=True)


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


def is_kept(directory: Path) -> bool:
    """Whether the session in `directory` is a project its owner chose to keep (#120). Kept projects are never
    swept. An unreadable sidecar reads as not kept only when there is no sidecar at all: a sidecar that exists but
    cannot be parsed is treated as kept, because deleting someone's project over a parse error is the one mistake
    the sweep must not make."""
    path = Path(directory) / "meta.json"
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text()).get(KEPT_KEY))
    except (OSError, ValueError):
        return True


def purge_expired_session_ids(upload_dir: str, retention_seconds: int, *, now: float | None = None) -> list[str]:
    """Deletes session folders untouched for longer than retention_seconds, except kept projects. Returns the ids
    removed, so the caller can delete their checkpoints too (the checkpoint no longer expires on its own, #120).

    Never raises: this runs unattended in the background, and a permission error on one
    folder must not stop the rest being swept or take the app down with it. A retention of 0
    disables expiry entirely.
    """
    if retention_seconds <= 0:
        return []

    root = Path(upload_dir)
    if not root.exists():
        return []

    cutoff = (time.time() if now is None else now) - retention_seconds
    removed: list[str] = []

    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        try:
            if is_kept(directory) or _last_touched(directory) >= cutoff:
                continue
            shutil.rmtree(directory)
            removed.append(directory.name)
            logger.info("Purged expired session directory %s", directory.name)
        except OSError:
            logger.warning("Could not purge session directory %s", directory.name, exc_info=True)

    return removed


def purge_expired_sessions(upload_dir: str, retention_seconds: int, *, now: float | None = None) -> int:
    """`purge_expired_session_ids`, counted."""
    return len(purge_expired_session_ids(upload_dir, retention_seconds, now=now))


def delete_session_files(upload_dir: str, session_id: str) -> dict:
    """Removes the session's folder and everything in it: the upload, cleaned versions, run folders and fitted
    models, job records, reports and the sidecar (#120). Returns what was there, for the caller to report."""
    directory = Path(upload_dir) / session_id
    if not directory.is_dir():
        return {"files": 0, "bytes": 0}
    files = [p for p in directory.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    with _META_LOCK:
        shutil.rmtree(directory)
    return {"files": len(files), "bytes": size}


def keep_session(upload_dir: str, session_id: str, name: str, *, token_hash: str | None = None) -> dict:
    """Marks the session as a project kept until deleted, under `name`. `token_hash` replaces the stored hash when
    given (a session from before tokens had none). Returns the updated sidecar."""
    with _META_LOCK:
        meta = read_session_meta(upload_dir, session_id) or {}
        if not meta.get(KEPT_KEY):
            meta["kept_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta[KEPT_KEY] = True
        meta["project_name"] = name
        if token_hash:
            meta[TOKEN_HASH_KEY] = token_hash
        _write_meta_file(_meta_path(upload_dir, session_id), meta)
    return meta


def replace_token_hash(upload_dir: str, session_id: str, token_hash: str) -> None:
    with _META_LOCK:
        meta = read_session_meta(upload_dir, session_id) or {}
        meta[TOKEN_HASH_KEY] = token_hash
        _write_meta_file(_meta_path(upload_dir, session_id), meta)
