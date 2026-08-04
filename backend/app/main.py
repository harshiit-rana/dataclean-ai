"""FastAPI app entrypoint. PRD §8.1, §8.2, NFR-14 (CORS locked to the frontend origin)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    health,
    routes_chat,
    routes_export,
    routes_goal,
    routes_model,
    routes_preview,
    routes_profile,
    routes_readiness,
    routes_resume,
    routes_session,
    routes_undo,
    routes_upload,
)
from app.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.graph.checkpointer import build_checkpointer
from app.graph.workflow import build_graph
from app.services.session_store import purge_expired_sessions

logger = logging.getLogger(__name__)


async def _sweep_expired_sessions(settings) -> None:
    """Deletes uploaded files whose session has aged out (NFR-12/13).

    Runs in-process rather than as a cron job or sidecar so a plain `docker compose up`
    doesn't quietly accumulate every CSV anyone ever uploaded. The sweep is blocking file
    I/O, so it goes to a thread; failures are logged and the loop continues, because losing
    cleanup is survivable and taking the API down over it is not.
    """
    while True:
        try:
            removed = await asyncio.to_thread(
                purge_expired_sessions, settings.upload_dir, settings.upload_retention_seconds
            )
            if removed:
                logger.info("Session cleanup removed %d expired session(s)", removed)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- a broken sweep must not end the process
            logger.exception("Session cleanup pass failed; will retry next interval")
        await asyncio.sleep(settings.cleanup_interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    async with build_checkpointer(settings) as checkpointer:
        app.state.graph = build_graph(checkpointer)

        cleanup_task: asyncio.Task | None = None
        if settings.upload_retention_seconds > 0:
            cleanup_task = asyncio.create_task(_sweep_expired_sessions(settings))
        else:
            logger.warning(
                "UPLOAD_RETENTION_SECONDS is 0 -- uploaded files will be kept indefinitely"
            )

        try:
            yield
        finally:
            if cleanup_task:
                cleanup_task.cancel()
                try:
                    await cleanup_task
                except asyncio.CancelledError:
                    pass


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="DataClean AI", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(routes_upload.router)
    app.include_router(routes_goal.router)
    app.include_router(routes_preview.router)
    app.include_router(routes_profile.router)
    app.include_router(routes_session.router)
    app.include_router(routes_resume.router)
    app.include_router(routes_chat.router)
    app.include_router(routes_undo.router)
    app.include_router(routes_export.router)
    app.include_router(routes_readiness.router)
    app.include_router(routes_model.router)

    return app


app = create_app()
