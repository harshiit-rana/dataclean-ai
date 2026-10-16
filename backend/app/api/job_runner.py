"""Starting jobs, waiting on them and streaming their progress, from the API side (#103).

`start` writes the job and, under the inline runner, runs it in a thread of this process; under the worker runner
it only queues it. `wait` returns the result or raises the job's failure as the route's own error. `stream` turns
the job's events file into server-sent events, from any point, so a page that reconnects picks up where it was.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

from app.config import Settings
from app.core.errors import AppError
from app.services import jobs

# Seconds between reads of a job's state or events while waiting on a worker, and of silence before a stream sends
# a comment line so proxies do not close it.
POLL_SECONDS = 0.25
KEEPALIVE_SECONDS = 15.0

# Inline jobs running in this process, so a waiter awaits the thread instead of polling, and the task is not
# garbage-collected mid-run.
_INLINE: dict[str, asyncio.Task] = {}


class JobFailedError(AppError):
    """A job's failure, answered with the status and code its handler gave."""

    def __init__(self, failure: jobs.JobError):
        super().__init__(failure.error, failure.detail, code=failure.code)
        self.status_code = failure.status


def start(
    settings: Settings, session_directory: Path, kind: str, payload: dict, files: dict[str, bytes] | None = None
) -> Path:
    """Writes the job (with any `files` it needs, such as an uploaded CSV) and launches it."""
    folder = jobs.create(session_directory, kind, payload, files)
    launch(settings, folder)
    return folder


def launch(settings: Settings, folder: Path) -> None:
    """Under the inline runner, runs the job in a thread of this process; under the worker runner, leaves it queued
    for the worker."""
    if settings.job_runner == "worker" or not jobs.claim(folder, "inline"):
        return
    task = asyncio.get_running_loop().create_task(asyncio.to_thread(jobs.run, folder, "inline"))
    _INLINE[str(folder)] = task
    task.add_done_callback(lambda _: _INLINE.pop(str(folder), None))


async def wait(folder: Path) -> dict:
    """The job's result, once it has finished; its failure as a `JobFailedError`."""
    task = _INLINE.get(str(folder))
    if task is not None:
        await asyncio.shield(task)
    while jobs.read_state(folder).get("state") not in jobs.FINISHED:
        await asyncio.sleep(POLL_SECONDS)
    try:
        return jobs.result_of(folder)
    except jobs.JobError as failure:
        raise JobFailedError(failure) from failure


def frame(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


async def stream(folder: Path, *, after: int = 0) -> AsyncIterator[str]:
    """The job's events as server-sent events, skipping the first `after` (those a reconnecting page already has),
    until its `result` or `error`. The first frame names the job, so the page can reconnect to it."""
    yield frame({"type": "job", "job_id": folder.name})
    offset = 0
    seen = 0
    quiet = 0.0
    finished = False
    while True:
        events, offset = jobs.read_events(folder, offset)
        for event in events:
            seen += 1
            if seen <= after:
                continue
            yield frame(event)
            if event.get("type") in ("result", "error"):
                return
        if events:
            quiet = 0.0
            continue
        if finished:
            # Finished without a closing event (cancelled, or the file was cut short): say so rather than hang.
            state = jobs.read_state(folder).get("state")
            yield frame({"type": "error", "detail": "This job ended without a result.", "code": f"job_{state}"})
            return
        if jobs.read_state(folder).get("state") in jobs.FINISHED:
            finished = True  # one more read, for events written just before the state changed
            continue
        await asyncio.sleep(POLL_SECONDS)
        quiet += POLL_SECONDS
        if quiet >= KEEPALIVE_SECONDS:
            quiet = 0.0
            yield ": still working\n\n"
