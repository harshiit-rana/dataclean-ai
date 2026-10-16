"""One retention sweep at a time per deployment (#105, docs/decisions/0004-platform-architecture.md).

The sweep used to run in every API process, so two API workers swept the same folder at once, each deleting what
the other was walking. It now takes a lock file in `UPLOAD_DIR` for each pass. The lock is a file created
exclusively, holding its owner and the time it was taken; a lock older than its time-to-live is taken to belong to a
sweeper that died, and is broken by renaming it aside -- an atomic step only one process can win -- before trying
again. Released only by its owner.

A file on the shared volume rather than a Redis lock: the job runner keeps Redis out of its path for the same
reasons (see services/jobs.py), and the volume is the thing being swept.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

LOCK_FILE = ".sweep.lock"
# A pass over thousands of sessions takes seconds; a lock this old belongs to a sweeper that is gone.
TTL_SECONDS = 600.0


def _path(upload_dir: str) -> Path:
    return Path(upload_dir) / LOCK_FILE


def acquire(upload_dir: str, owner: str, *, now: float | None = None, ttl: float = TTL_SECONDS) -> bool:
    """Takes the lock for `owner`. False when another live sweeper holds it."""
    path = _path(upload_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time() if now is None else now
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not _stale(path, now, ttl):
                return False
            # Break it by renaming it aside: of several processes that all saw it stale, one rename succeeds.
            try:
                os.replace(path, path.with_name(f"{LOCK_FILE}.stale.{uuid.uuid4().hex[:8]}"))
            except FileNotFoundError:
                pass
            for leftover in path.parent.glob(f"{LOCK_FILE}.stale.*"):
                leftover.unlink(missing_ok=True)
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"owner": owner, "at": now}, handle)
        return True
    return False


def _stale(path: Path, now: float, ttl: float) -> bool:
    try:
        taken = float(json.loads(path.read_text(encoding="utf-8")).get("at", 0))
    except (OSError, ValueError):
        # Unreadable: half-written by a process that died between create and write, or damaged. Judge by its age.
        try:
            taken = path.stat().st_mtime
        except OSError:
            return True
    return now - taken > ttl


def release(upload_dir: str, owner: str) -> None:
    """Removes the lock if `owner` holds it; leaves someone else's alone."""
    path = _path(upload_dir)
    try:
        if json.loads(path.read_text(encoding="utf-8")).get("owner") == owner:
            path.unlink(missing_ok=True)
    except (OSError, ValueError):
        pass


@contextmanager
def held(upload_dir: str, owner: str) -> Iterator[bool]:
    """`with held(dir, me) as mine:` -- `mine` says whether this process got the lock; released on the way out."""
    mine = acquire(upload_dir, owner)
    try:
        yield mine
    finally:
        if mine:
            release(upload_dir, owner)
