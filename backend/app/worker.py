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


def sweep(settings, owner: str) -> list[str] | None:
    """One retention pass from the worker (#105), under the same lock as the API's, with the Redis checkpointer for
    deleting the removed sessions' checkpoints. Never raises."""
    import asyncio

    from redis.exceptions import ConnectionError as RedisConnectionError

    from app.graph.checkpointer import build_checkpointer
    from app.main import sweep_once

    async def one_pass():
        try:
            async with build_checkpointer(settings) as checkpointer:
                return await sweep_once(settings, checkpointer, owner=owner)
        except (OSError, RedisConnectionError):
            # Redis unreachable: no pass at all. Removing the files without their checkpoints would orphan the
            # checkpoints for good, since nothing else ever expires them (#120); the next pass catches up.
            logger.warning("Skipping the retention sweep: Redis is unreachable", exc_info=True)
            return None

    try:
        return asyncio.run(one_pass())
    except Exception:  # noqa: BLE001 -- a failed sweep must not stop the worker
        logger.exception("Retention sweep failed; trying again next interval")
        return None


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
    next_sweep = 0.0
    while not stopping:
        if settings.upload_retention_seconds > 0 and time.monotonic() >= next_sweep:
            sweep(settings, owner)
            next_sweep = time.monotonic() + settings.cleanup_interval_seconds
        try:
            ran = run_once(settings.upload_dir, owner)
        except Exception:  # noqa: BLE001 -- a broken pass must not end the worker
            logger.exception("Worker pass failed; trying again")
            ran = False
        if not ran:
            time.sleep(IDLE_SECONDS)


if __name__ == "__main__":
    main()
