"""A request id on every log line and every error response, so a user's report can be matched to the server's log.

When a run failed in a hosted deployment there was no way to connect what the user saw to the log lines of that
request. The middleware gives each request an id -- the caller's `X-Request-Id` when it sends a sensible one, a new
one otherwise -- keeps it in a context variable for the length of the request (worker threads started with
`asyncio.to_thread` inherit it), stamps it on every log record through `RequestIdFilter`, returns it in the
`X-Request-Id` response header, and the error handlers put it in the body.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from contextvars import ContextVar

HEADER = "X-Request-Id"
_request_id: ContextVar[str] = ContextVar("request_id", default="-")
# A caller's id is used only when it looks like one: no spaces, newlines or markup to inject into a log line.
_ACCEPTABLE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def current_request_id(request=None) -> str:
    """The id of the request being handled, or "-" outside one. Given the request, read from it first."""
    if request is not None:
        found = getattr(request.state, "request_id", None)
        if found:
            return found
    return _request_id.get()


def unexpected_detail(what: str) -> str:
    """What a user is told when something failed for a reason nobody anticipated: a stable sentence and the id to quote,
    never the exception's own text, which can be meaningless to them and can leak internals. The cause is in the
    server log under the same id."""
    return (
        f"{what} stopped unexpectedly. The cause is in the server log under request id {current_request_id()}; "
        f"quote it when reporting the problem."
    )


class RequestIdFilter(logging.Filter):
    """Adds `request_id` to every record, so a format can print it."""

    def filter(self, record: logging.LogRecord) -> bool:
        # An id passed explicitly (`extra={"request_id": ...}`) wins: the error handler logs outside the request.
        record.request_id = getattr(record, "request_id", None) or current_request_id()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for a log collector."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class RequestIdMiddleware:
    """Pure ASGI, so it wraps streamed responses as well as ordinary ones."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        sent = dict(scope.get("headers") or []).get(HEADER.lower().encode(), b"").decode("latin-1")
        request_id = sent if _ACCEPTABLE.match(sent) else uuid.uuid4().hex[:16]
        token = _request_id.set(request_id)
        # Also on the request itself: an unexpected error is answered by Starlette's outermost middleware, after
        # this one has returned and the context variable is gone, but the request's state is still there.
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.append((HEADER.lower().encode(), request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            _request_id.reset(token)
