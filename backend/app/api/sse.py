"""Shared SSE StreamingResponse helper. PRD §10.2 payload shapes are defined in
app.models.sse_events; this module just wires an async string generator up as a proper
`text/event-stream` HTTP response with the headers needed to stop proxies/browsers from
buffering it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse


def sse_response(generator: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
