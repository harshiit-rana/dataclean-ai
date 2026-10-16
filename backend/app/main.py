"""FastAPI app entrypoint. PRD §8.1, §8.2, NFR-14 (CORS locked to the frontend origin)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    health,
    routes_apply,
    routes_chat,
    routes_export,
    routes_goal,
    routes_model,
    routes_preview,
    routes_profile,
    routes_promotion,
    routes_readiness,
    routes_rules,
    routes_resume,
    routes_session,
    routes_undo,
    routes_upload,
)
from app.core.request_context import RequestIdMiddleware
from app.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.dependencies import verify_session_access
from app.graph.checkpointer import build_checkpointer, delete_thread
from app.graph.workflow import build_graph
from app.services.session_store import purge_expired_session_ids

logger = logging.getLogger(__name__)


async def _sweep_expired_sessions(settings, checkpointer=None) -> None:
    """Deletes uploaded files whose session has aged out (NFR-12/13).

    Runs in-process rather than as a cron job or sidecar so a plain `docker compose up`
    doesn't quietly accumulate every CSV anyone ever uploaded. The sweep is blocking file
    I/O, so it goes to a thread; failures are logged and the loop continues, because losing
    cleanup is survivable and taking the API down over it is not.
    """
    while True:
        try:
            removed = await asyncio.to_thread(
                purge_expired_session_ids, settings.upload_dir, settings.upload_retention_seconds
            )
            # Checkpoints no longer expire on their own (#120): the sweep removes them with the files.
            for session_id in removed:
                await delete_thread(checkpointer, session_id)
            if removed:
                logger.info("Session cleanup removed %d expired session(s)", len(removed))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- a broken sweep must not end the process
            logger.exception("Session cleanup pass failed; will retry next interval")
        await asyncio.sleep(settings.cleanup_interval_seconds)


def warn_if_tokens_are_off(settings) -> bool:
    """Says loudly, at every start, when session tokens are off: then knowing a session id is enough to read its
    data. Returns whether it warned."""
    if settings.session_tokens_required:
        return False
    logger.warning(
        "SESSION TOKENS ARE OFF (APP_ENV=%s, REQUIRE_SESSION_TOKENS=%s): anyone who knows a session id can read "
        "that session's data and exports. Fine for local development; never for a hosted instance.",
        settings.app_env, settings.require_session_tokens,
    )
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    warn_if_tokens_are_off(settings)

    async with build_checkpointer(settings) as checkpointer:
        app.state.graph = build_graph(checkpointer)

        cleanup_task: asyncio.Task | None = None
        if settings.upload_retention_seconds > 0:
            cleanup_task = asyncio.create_task(_sweep_expired_sessions(settings, checkpointer))
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
    # Outermost, so the id is set before anything else runs and reaches every response, errors included.
    app.add_middleware(RequestIdMiddleware)

    # No session yet, so nothing to check against.
    app.include_router(health.router)
    app.include_router(routes_upload.router)

    # Everything below is scoped to one session. The access check is applied here, at the
    # router, rather than on each route: every route in these modules takes a {session_id},
    # so attaching it once means a route added later cannot quietly ship without it.
    # A no-op unless session tokens are required -- see Settings.session_tokens_required.
    guarded = [Depends(verify_session_access)]
    for router in (
        routes_goal.router,
        routes_preview.router,
        routes_profile.router,
        routes_session.router,
        routes_resume.router,
        routes_chat.router,
        routes_undo.router,
        routes_export.router,
        routes_readiness.router,
        routes_rules.router,
        routes_model.router,
        routes_apply.router,
        routes_promotion.router,
    ):
        app.include_router(router, dependencies=guarded)

    return app


app = create_app()
