"""A record that a search is running, so a search the process never finished is not lost without trace.

A streamed search whose browser goes away still finishes and is saved, by design. A search killed with the process --
a restart, a deploy, an out-of-memory kill -- left nothing at all: no result, no history entry, no sign it ever ran.
Each search now writes a small marker when it starts, adds each round to it as the round finishes, and removes it when
it ends, whether it succeeded or failed (a failure is recorded by the route, #64). Only a process that died mid-search
leaves one behind.

Telling a dead search from a slow one: while a search runs, a background thread touches its marker every
`HEARTBEAT_SECONDS`. A round can take minutes, but the heartbeat cannot stop unless the process has. A marker silent for
longer than `STALE_AFTER_SECONDS` becomes an `interrupted` history entry the next time the history is read, listing the
rounds that had finished. Markers are keyed by search, so two searches in one session never share one.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SEARCHES_DIR = "searches"
HEARTBEAT_SECONDS = 30.0
STALE_AFTER_SECONDS = 120.0


def _write(path: Path, payload: dict) -> None:
    """Whole-file replace, so a reader never sees half a marker."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, default=str), encoding="utf-8")
    os.replace(temporary, path)


class SearchMarker:
    """The marker of one running search. Use as a context manager around the search: the marker is removed on the
    way out, however the search ended, and the heartbeat runs while it is open. Never raises: a search must not fail
    because its bookkeeping did."""

    def __init__(self, directory: Path, *, route: str, request: dict | None, target: str | None, task: str | None):
        self.search_id = uuid.uuid4().hex[:12]
        self.path = Path(directory) / SEARCHES_DIR / f"{self.search_id}.json"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        now = time.time()
        self._payload: dict[str, Any] = {
            "search_id": self.search_id,
            "route": route,
            "request": request,
            "target_column": target,
            "task": task,
            "started_at": now,
            "heartbeat_at": now,
            "rounds": [],
        }

    def _save(self) -> None:
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                _write(self.path, self._payload)
        except OSError:
            logger.warning("Search marker %s could not be written", self.path, exc_info=True)

    def round_finished(self, entry: dict) -> None:
        """Adds a finished round: what it was, and whether it was kept."""
        with self._lock:
            self._payload["rounds"].append(
                {"name": entry.get("name"), "label": entry.get("label"), "accepted": entry.get("accepted")}
            )
            self._payload["heartbeat_at"] = time.time()
        self._save()

    def _beat(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            with self._lock:
                self._payload["heartbeat_at"] = time.time()
            self._save()

    def __enter__(self) -> "SearchMarker":
        self._save()
        self._thread = threading.Thread(target=self._beat, name=f"search-heartbeat-{self.search_id}", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        try:
            with self._lock:
                self.path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Search marker %s could not be removed", self.path, exc_info=True)


def stale_markers(directory: Path, *, now: float | None = None) -> list[tuple[Path, dict]]:
    """Markers whose heartbeat stopped: searches whose process is gone. An unreadable marker counts as stale -- it
    can only be left by a write that died."""
    root = Path(directory) / SEARCHES_DIR
    if not root.is_dir():
        return []
    now = time.time() if now is None else now
    found = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {"search_id": path.stem, "rounds": [], "heartbeat_at": 0}
        if now - float(payload.get("heartbeat_at") or 0) > STALE_AFTER_SECONDS:
            found.append((path, payload))
    return found


def interrupted_reason(payload: dict) -> str:
    rounds = payload.get("rounds") or []
    finished = (
        f" {len(rounds)} round{'s' if len(rounds) != 1 else ''} had finished: "
        + ", ".join(str(r.get("label") or r.get("name")) for r in rounds)
        + "."
        if rounds
        else " No round had finished."
    )
    return (
        "The search stopped before it finished: the server restarted or the process was stopped, so nothing was "
        "saved." + finished + " Run it again to get a model."
    )
