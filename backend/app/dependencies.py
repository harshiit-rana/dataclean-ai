"""FastAPI dependency providers."""

from __future__ import annotations

from functools import lru_cache

import redis.asyncio as redis_asyncio
from fastapi import Depends, Request
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings, get_settings
from app.services.llm_client import get_llm_client as _get_llm_client
from app.services.session_store import session_dir


@lru_cache
def _redis_pool(redis_url: str) -> redis_asyncio.ConnectionPool:
    return redis_asyncio.ConnectionPool.from_url(redis_url)


def get_redis(settings: Settings = Depends(get_settings)) -> redis_asyncio.Redis:
    pool = _redis_pool(settings.redis_url)
    return redis_asyncio.Redis(connection_pool=pool)


def get_llm_client(settings: Settings = Depends(get_settings)) -> BaseChatModel:
    return _get_llm_client(settings, streaming=True)


def get_session_dir(session_id: str, settings: Settings = Depends(get_settings)):
    return session_dir(settings.upload_dir, session_id)


def get_graph(request: Request):
    """The compiled LangGraph workflow, built once at startup (see main.py's lifespan)
    with a Redis checkpointer whose lifecycle spans the whole app -- not per-request."""
    return request.app.state.graph
