"""GET /health and GET /health/ready.

`/health` is liveness: the process is up and answering. It checks nothing else, so a restart loop is not triggered
by a dependency being slow.

`/health/ready` is readiness: it checks that Redis, which holds every session's state, answers, and that the
cleaning graph was built at startup. It answers 503 naming what is missing. docker-compose.yml's backend healthcheck
uses it, and so does the CI job that starts the whole stack.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import get_settings

router = APIRouter()

# How long a dependency may take to answer before it counts as down.
READY_TIMEOUT_SECONDS = 2.0


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


async def _redis_answers(url: str) -> bool:
    import redis.asyncio as redis

    client = redis.from_url(url, socket_connect_timeout=READY_TIMEOUT_SECONDS)
    try:
        # The asyncio client's ping is always awaitable; redis-py types it for both clients at once.
        ping = cast(Awaitable[bool], client.ping())
        return bool(await asyncio.wait_for(ping, timeout=READY_TIMEOUT_SECONDS))
    except Exception:  # noqa: BLE001 -- any failure to answer is "not ready"
        return False
    finally:
        await client.aclose()


@router.get("/health/ready")
async def ready(request: Request) -> JSONResponse:
    checks = {
        "redis": "ok" if await _redis_answers(get_settings().redis_url) else "unreachable",
        "graph": "ok" if getattr(request.app.state, "graph", None) is not None else "not built",
    }
    ready_now = all(v == "ok" for v in checks.values())
    return JSONResponse(status_code=200 if ready_now else 503,
                        content={"status": "ready" if ready_now else "not ready", **checks})
