"""GET /health -- REAL. Liveness check, used by docker-compose's healthcheck."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
