"""Redis checkpointer factory. PRD §13.3: key prefix `dataclean_session`, 24h default TTL.

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

from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.config import Settings


@asynccontextmanager
async def build_checkpointer(settings: Settings) -> AsyncIterator[AsyncRedisSaver]:
    ttl_minutes = max(1, settings.redis_ttl_seconds // 60)
    async with AsyncRedisSaver.from_conn_string(
        settings.redis_url,
        ttl={"default_ttl": ttl_minutes, "refresh_on_read": True},
    ) as checkpointer:
        await checkpointer.asetup()
        yield checkpointer


def thread_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}
