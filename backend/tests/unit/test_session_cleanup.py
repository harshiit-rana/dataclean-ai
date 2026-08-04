"""Retention sweep for uploaded files (NFR-12/13).

Without this, every CSV anyone ever uploaded stays on disk forever -- a storage problem
immediately and a privacy problem the moment the app is hosted.
"""

from __future__ import annotations

import os
import time

from app.services.session_store import (
    purge_expired_sessions,
    save_uploaded_csv,
    session_dir,
    write_session_meta,
)

RETENTION = 3600


def _make_session(upload_dir, session_id: str, *, age_seconds: float = 0.0) -> None:
    save_uploaded_csv(str(upload_dir), session_id, b"a,b\n1,2\n")
    write_session_meta(str(upload_dir), session_id, {"file_name": "x.csv"})
    if age_seconds:
        stale = time.time() - age_seconds
        directory = session_dir(str(upload_dir), session_id)
        for path in [*directory.rglob("*"), directory]:
            os.utime(path, (stale, stale))


def test_expired_sessions_are_removed(tmp_path):
    _make_session(tmp_path, "old", age_seconds=RETENTION * 2)

    assert purge_expired_sessions(str(tmp_path), RETENTION) == 1
    assert not (tmp_path / "old").exists()


def test_recent_sessions_are_kept(tmp_path):
    _make_session(tmp_path, "fresh")

    assert purge_expired_sessions(str(tmp_path), RETENTION) == 0
    assert (tmp_path / "fresh" / "original.csv").exists()


def test_a_session_being_worked_in_is_not_swept(tmp_path):
    """The folder is old but a file inside it was just written -- someone is mid-session.
    Judging by the directory's own mtime alone would delete their data underneath them."""
    _make_session(tmp_path, "active", age_seconds=RETENTION * 2)
    (tmp_path / "active" / "cleaned.csv").write_bytes(b"a,b\n1,2\n")

    assert purge_expired_sessions(str(tmp_path), RETENTION) == 0
    assert (tmp_path / "active").exists()


def test_retention_of_zero_disables_expiry(tmp_path):
    _make_session(tmp_path, "ancient", age_seconds=RETENTION * 1000)

    assert purge_expired_sessions(str(tmp_path), 0) == 0
    assert (tmp_path / "ancient").exists()


def test_only_expired_sessions_are_touched(tmp_path):
    _make_session(tmp_path, "old", age_seconds=RETENTION * 2)
    _make_session(tmp_path, "fresh")

    assert purge_expired_sessions(str(tmp_path), RETENTION) == 1
    assert not (tmp_path / "old").exists()
    assert (tmp_path / "fresh").exists()


def test_missing_upload_dir_is_survivable(tmp_path):
    assert purge_expired_sessions(str(tmp_path / "does-not-exist"), RETENTION) == 0


def test_stray_files_alongside_sessions_are_left_alone(tmp_path):
    """Only session directories are ours to delete."""
    stray = tmp_path / "notes.txt"
    stray.write_text("keep me")
    old = time.time() - RETENTION * 2
    os.utime(stray, (old, old))

    assert purge_expired_sessions(str(tmp_path), RETENTION) == 0
    assert stray.exists()
