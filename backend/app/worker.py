"""The job worker: `python -m app.worker` (#103, docs/decisions/0004-platform-architecture.md).

Runs the jobs the API queues under `JOB_RUNNER=worker` -- training, improvement searches, applications -- one at a
time, from the shared `UPLOAD_DIR`. Restarting the API does not touch a job a worker is running; a worker that dies
leaves a stale heartbeat, and the next worker to look marks the job interrupted so the page is told. Several workers
can share one volume: a job is claimed by creating its claim file exclusively, so only one of them runs it.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import time

from app.config import get_settings
from app.core.logging import configure_logging
from app.services import jobs

logger = logging.getLogger("app.worker")

# Seconds between looks for queued jobs when there are none.
IDLE_SECONDS = 1.0


def _register_handlers() -> None:
    """The modules that own the work register their job kinds when imported."""
    import app.api.routes_apply  # noqa: F401
    import app.api.routes_model  # noqa: F401


def run_once(upload_dir: str, owner: str) -> bool:
    """Marks dead jobs interrupted, then runs the oldest queued job, if any. Returns whether it ran one."""
    for folder in jobs.reap_stale(upload_dir):
        logger.warning("Job %s was interrupted: its runner stopped beating", folder.name)
    for folder in jobs.queued(upload_dir):
        if jobs.claim(folder, owner):
            logger.info("Running job %s (%s)", folder.name, jobs.read_state(folder).get("kind"))
            final = jobs.run(folder, owner)
            logger.info("Job %s %s", folder.name, final.get("state"))
            return True
    return False


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    _register_handlers()
    owner = f"{socket.gethostname()}:{os.getpid()}"
    stopping = False

    def stop(signum, _frame) -> None:
        nonlocal stopping
        stopping = True
        logger.info("Stopping after the current job (signal %s)", signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    logger.info("Worker %s watching %s", owner, settings.upload_dir)
    while not stopping:
        try:
            ran = run_once(settings.upload_dir, owner)
        except Exception:  # noqa: BLE001 -- a broken pass must not end the worker
            logger.exception("Worker pass failed; trying again")
            ran = False
        if not ran:
            time.sleep(IDLE_SECONDS)


if __name__ == "__main__":
    main()
