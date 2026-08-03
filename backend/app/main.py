"""FastAPI app entrypoint. PRD §8.1, §8.2, NFR-14 (CORS locked to the frontend origin)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    health,
    routes_chat,
    routes_export,
    routes_goal,
    routes_preview,
    routes_profile,
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    async with build_checkpointer(settings) as checkpointer:
        app.state.graph = build_graph(checkpointer)
        yield


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

    return app


app = create_app()
