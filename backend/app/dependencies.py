"""FastAPI dependency providers."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

import redis.asyncio as redis_asyncio
from fastapi import Depends, Header, Request
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings, get_settings
from app.core.errors import SessionAccessDeniedError, SessionNotFoundError
from app.services.llm_client import get_llm_client as _get_llm_client
from app.services.session_access import META_KEY, token_matches
from app.services.session_store import read_session_meta, session_dir


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


def verify_session_access(
    session_id: str,
    x_session_token: Annotated[str | None, Header()] = None,
    settings: Settings = Depends(get_settings),
) -> str:
    """Guards every route scoped to a single session.

    Knowing the session id must not be enough to read the data behind it -- see
    services/session_access.py. Attached at the router in main.py rather than per-route, so a
    session route added later cannot ship without it.

    A session created before tokens were switched on has no stored hash. Those are allowed
    through rather than locked out, so enabling the setting doesn't strand work in progress.
    """
    if not settings.session_tokens_required:
        return session_id

    meta = read_session_meta(settings.upload_dir, session_id)
    if meta is None:
        raise SessionNotFoundError(session_id)

    expected = meta.get(META_KEY)
    if not expected:
        return session_id
    if not token_matches(x_session_token, expected):
        raise SessionAccessDeniedError()
    return session_id
