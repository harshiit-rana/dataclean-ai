"""Redis checkpointer factory. PRD §13.3: key prefix `dataclean_session`. No TTL since #120: the sweep deletes
checkpoints with the session's files.

Uses `langgraph-checkpoint-redis` (redis-developer/langgraph-redis on PyPI as
`langgraph-checkpoint-redis`), which persists checkpoints via RedisJSON/RediSearch --
built into Redis 8.0+, hence docker-compose pins `redis:8-alpine` rather than 7.x.

NOTE: this package manages its own internal Redis key naming; it does not expose a
constructor parameter for a literal `dataclean_session:{session_id}` key prefix. The
PRD's key-format spec is satisfied at the thread_id level -- every checkpoint call in
this app is scoped by `configurable.thread_id = session_id` -- but the literal Redis key
bytes on disk are whatever this library generates internally. Flagging this as a known gap
between the PRD's literal spec and the library's actual API, rather than silently
implementing a custom prefix that doesn't exist.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from redisvl.exceptions import RedisSearchError

from app.config import Settings

logger = logging.getLogger(__name__)

_SETUP_ATTEMPTS = 3


@asynccontextmanager
async def build_checkpointer(settings: Settings) -> AsyncIterator[AsyncRedisSaver]:
    # No TTL (#120). Checkpoints used to expire on their own after REDIS_TTL_SECONDS while the sweep deleted the
    # files separately, which cannot express a project kept until deleted. The sweep is now the one place that
    # decides: it deletes a session's checkpoint thread when it removes the session's folder, and never touches a
    # kept project. See docs/decisions/0004-platform-architecture.md.
    #
    # Entering the saver creates its search indexes, and the library checks for an index and then creates it. On an
    # empty Redis the backend and the worker do this at the same moment, so one of them can be told the index already
    # exists and fail startup (#141). Only the entry is retried, and only for that error; a second pass finds the
    # indexes and skips them. The body below the yield is never retried.
    stack = AsyncExitStack()
    for attempt in range(1, _SETUP_ATTEMPTS + 1):
        try:
            checkpointer = await stack.enter_async_context(AsyncRedisSaver.from_conn_string(settings.redis_url))
            break
        except RedisSearchError as exc:
            await stack.aclose()
            if "already exists" not in str(exc).lower() or attempt == _SETUP_ATTEMPTS:
                raise
            logger.info("Another process created the checkpoint index first; setting up again (%s)", exc)
    async with stack:
        yield checkpointer


async def delete_thread(checkpointer, session_id: str) -> bool:
    """Deletes every checkpoint of the session. Returns whether it worked; never raises, because the caller is a
    deletion or a sweep that must carry on with the files whatever Redis says."""
    if checkpointer is None:
        return False
    try:
        await checkpointer.adelete_thread(session_id)
        return True
    except Exception:  # noqa: BLE001 -- logged; the files are still removed
        logger.warning("Could not delete the checkpoint of session %s", session_id, exc_info=True)
        return False


async def persist_thread(checkpointer, session_id: str) -> int:
    """Removes any TTL from the session's existing checkpoint keys, which were written before checkpoints stopped
    expiring and would otherwise still lapse. Returns how many keys it persisted; never raises."""
    client = getattr(checkpointer, "_redis", None)
    if client is None:
        return 0
    persisted = 0
    try:
        async for key in client.scan_iter(match=f"*{session_id}*", count=500):
            persisted += int(bool(await client.persist(key)))
    except Exception:  # noqa: BLE001 -- best effort; the project is kept either way
        logger.warning("Could not persist the checkpoint keys of session %s", session_id, exc_info=True)
    return persisted


def thread_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}
