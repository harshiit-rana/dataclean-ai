"""Long work as jobs recorded on disk, run inline or by a separate worker (#103, docs/decisions/0004).

Training, the improvement search and applying a model used to run in a thread inside the API request, so a deploy
or a crash mid-search lost the search. Each now runs as a job: a folder under the session,

    <session>/jobs/<job_id>/
        job.json      kind, payload, state, timestamps, the error or the result's place
        events.jsonl  progress events in order, one JSON object per line; the last is `result` or `error`
        claim         created exclusively by whoever runs the job, so two workers never take the same one
        heartbeat     touched while the job runs; a stale one means the runner died

Two runners, chosen by `JOB_RUNNER`. `inline` (the default, and what the tests use) runs the job in a thread of the
API process, as before, but through the same record. `worker` only writes the job as queued; `python -m app.worker`
claims and runs it, so restarting the API does not touch it. Either way the API reads progress from
`events.jsonl`, which is what lets a page that reconnects, or an API that restarted, pick the stream up.

No queue library and no Redis in this path: the shared volume is already required for the data the jobs read and
write, a folder per job is something a person can read when a job goes wrong, and claiming by exclusive create is
atomic on one filesystem. The decision record says why RQ, Arq and Celery were not used.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JOBS_DIR = "jobs"
STATE_FILE = "job.json"
EVENTS_FILE = "events.jsonl"
CLAIM_FILE = "claim"
HEARTBEAT_FILE = "heartbeat"

QUEUED, RUNNING, DONE, FAILED, INTERRUPTED, CANCELLED = (
    "queued", "running", "done", "failed", "interrupted", "cancelled",
)
FINISHED = {DONE, FAILED, INTERRUPTED, CANCELLED}

# How often a running job touches its heartbeat, and how long without one before it counts as dead. Generous: a
# fit can hold the GIL for a while, and declaring a live job dead would run it twice.
HEARTBEAT_SECONDS = 5.0
STALE_AFTER_SECONDS = 120.0

# The kinds of job and what runs each: `handler(job_dir, payload, emit) -> result`. Registered by the modules that
# own the work (routes_model, routes_apply), so this module knows nothing about training.
Handler = Callable[[Path, dict, Callable[[dict], None]], dict]
HANDLERS: dict[str, Handler] = {}


class JobError(Exception):
    """A job's failure as the user should read it: a title, a message, a stable code and the HTTP status a caller
    waiting on the job answers with. Raised by a handler to fail its job; raised by `result_of` for a job that
    failed. Anything else a handler raises fails the job with a generic message."""

    def __init__(self, detail: str, code: str | None = None, status: int = 422, error: str = "This didn't finish"):
        super().__init__(detail)
        self.detail = detail
        self.code = code
        self.status = status
        self.error = error


def register(kind: str, handler: Handler) -> None:
    HANDLERS[kind] = handler


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, default=str), encoding="utf-8")
    os.replace(temporary, path)


def job_dir(session_directory: Path, job_id: str) -> Path:
    return Path(session_directory) / JOBS_DIR / job_id


def create(session_directory: Path, kind: str, payload: dict, files: dict[str, bytes] | None = None) -> Path:
    """Writes a new job as queued and returns its folder. `files` (name -> bytes) are written first: a job is
    visible to a worker only once its state file exists, so the worker never finds it without them."""
    job_id = uuid.uuid4().hex[:12]
    folder = job_dir(session_directory, job_id)
    folder.mkdir(parents=True, exist_ok=False)
    for name, content in (files or {}).items():
        (folder / name).write_bytes(content)
    (folder / EVENTS_FILE).touch()
    _write_json(folder / STATE_FILE, {
        "job_id": job_id, "kind": kind, "payload": payload, "state": QUEUED, "created_at": _now(),
        "created_epoch": time.time(),
    })
    return folder


def read_state(folder: Path) -> dict:
    try:
        return json.loads((Path(folder) / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def update_state(folder: Path, **fields: Any) -> dict:
    state = read_state(folder)
    state.update(fields)
    _write_json(Path(folder) / STATE_FILE, state)
    return state


def emit_to(folder: Path) -> Callable[[dict], None]:
    """An `emit(event)` that appends to the job's events file. Thread-safe within a process; one writer per job."""
    lock = threading.Lock()
    path = Path(folder) / EVENTS_FILE

    def emit(event: dict) -> None:
        line = json.dumps(event, default=str)
        with lock, path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    return emit


def read_events(folder: Path, offset: int = 0) -> tuple[list[dict], int]:
    """The complete events after byte `offset`, and the offset to read from next. A line still being written (no
    newline yet) is left for the next read."""
    path = Path(folder) / EVENTS_FILE
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
    except OSError:
        return [], offset
    events = []
    consumed = 0
    for raw in data.splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            break
        consumed += len(raw)
        text = raw.decode("utf-8").strip()
        if text:
            events.append(json.loads(text))
    return events, offset + consumed


def claim(folder: Path, owner: str) -> bool:
    """Takes the job for `owner`. Exactly one caller gets True: the claim file is created exclusively."""
    try:
        descriptor = os.open(Path(folder) / CLAIM_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(owner)
    return True


def beat(folder: Path) -> None:
    try:
        (Path(folder) / HEARTBEAT_FILE).touch()
    except OSError:
        logger.debug("Could not touch the heartbeat of %s", folder, exc_info=True)


def is_stale(folder: Path, *, now: float | None = None, after: float = STALE_AFTER_SECONDS) -> bool:
    """A running job whose heartbeat (or, before its first beat, its claim) is older than `after`."""
    now = time.time() if now is None else now
    for name in (HEARTBEAT_FILE, CLAIM_FILE):
        path = Path(folder) / name
        if path.exists():
            return now - path.stat().st_mtime > after
    return False


def run(folder: Path, owner: str = "inline") -> dict:
    """Runs a claimed job to the end: running, the handler, then done or failed, with the last event saying which.
    Never raises; a caller that needs the outcome reads the state."""
    folder = Path(folder)
    state = update_state(folder, state=RUNNING, started_at=_now(), owner=owner)
    emit = emit_to(folder)
    handler = HANDLERS.get(state.get("kind", ""))
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            beat(folder)

    beat(folder)
    beating = threading.Thread(target=heartbeat, daemon=True, name=f"heartbeat-{folder.name}")
    beating.start()
    try:
        if handler is None:
            raise JobError(f"Unknown job kind {state.get('kind')!r}.", code="unknown_job", status=500)
        result = handler(folder, state.get("payload") or {}, emit)
        emit({"type": "result", "result": result})
        return update_state(folder, state=DONE, finished_at=_now())
    except JobError as exc:
        emit({"type": "error", "detail": exc.detail, **({"code": exc.code} if exc.code else {})})
        return update_state(folder, state=FAILED, finished_at=_now(), detail=exc.detail, code=exc.code,
                            status=exc.status, error=exc.error)
    except Exception:  # noqa: BLE001 -- a job's failure is recorded, never raised into the runner
        logger.exception("Job %s failed", folder.name)
        detail = "Something went wrong while running this. The details are in the server log."
        emit({"type": "error", "detail": detail, "code": "job_error"})
        return update_state(folder, state=FAILED, finished_at=_now(), detail=detail, code="job_error", status=500)
    finally:
        stop.set()


def result_of(folder: Path) -> dict:
    """The result a finished job ended with, or JobError with its message. Read from the last event."""
    state = read_state(folder)
    if state.get("state") == DONE:
        events, _ = read_events(folder)
        for event in reversed(events):
            if event.get("type") == "result":
                return event["result"]
    raise JobError(
        state.get("detail") or "This did not finish.",
        state.get("code"),
        int(state.get("status") or 422),
        state.get("error") or "This didn't finish",
    )


def session_jobs(session_directory: Path) -> Iterator[Path]:
    root = Path(session_directory) / JOBS_DIR
    if not root.is_dir():
        return iter(())
    return iter(sorted((p for p in root.iterdir() if (p / STATE_FILE).exists()),
                       key=lambda p: read_state(p).get("created_epoch", 0)))


def all_jobs(upload_dir: str) -> Iterator[Path]:
    root = Path(upload_dir)
    if not root.is_dir():
        return
    for session in root.iterdir():
        if session.is_dir():
            yield from session_jobs(session)


def queued(upload_dir: str) -> list[Path]:
    """Jobs waiting for a worker, oldest first."""
    waiting = [p for p in all_jobs(upload_dir)
               if read_state(p).get("state") == QUEUED and not (p / CLAIM_FILE).exists()]
    return sorted(waiting, key=lambda p: read_state(p).get("created_epoch", 0))


def mark_interrupted(folder: Path, detail: str) -> None:
    emit_to(folder)({"type": "error", "detail": detail, "code": "job_interrupted"})
    update_state(folder, state=INTERRUPTED, finished_at=_now(), detail=detail, code="job_interrupted", status=503)


def reap_stale(upload_dir: str, *, now: float | None = None) -> list[Path]:
    """Running jobs whose runner died: marked interrupted, with an event saying so, so a stream waiting on one ends.
    Returns them, for a caller that resumes what can be resumed (#104)."""
    found = []
    for folder in all_jobs(upload_dir):
        if read_state(folder).get("state") == RUNNING and is_stale(folder, now=now):
            mark_interrupted(folder, "The process running this stopped before it finished. Start it again.")
            found.append(folder)
    return found


def interrupt_orphans(upload_dir: str, owner: str = "inline") -> list[Path]:
    """At API start-up under the inline runner: every job this process kind was running died with the previous
    process, so it is marked interrupted at once rather than after its heartbeat goes stale."""
    found = []
    for folder in all_jobs(upload_dir):
        state = read_state(folder)
        if state.get("state") in (RUNNING, QUEUED) and state.get("owner", owner) == owner and (folder / CLAIM_FILE).exists():
            mark_interrupted(folder, "The server restarted before this finished. Start it again.")
            found.append(folder)
    return found


def cancel_session_jobs(session_directory: Path) -> int:
    """Marks the session's unfinished jobs cancelled (before the session is deleted). A job a worker is running
    cannot be stopped mid-fit; it finds its folder gone when it tries to save, and fails."""
    cancelled = 0
    for folder in session_jobs(session_directory):
        if read_state(folder).get("state") not in FINISHED:
            update_state(folder, state=CANCELLED, finished_at=_now())
            cancelled += 1
    return cancelled
