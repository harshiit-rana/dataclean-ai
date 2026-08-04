"""Bridges LangGraph's custom stream events to an HTTP SSE response. PRD §8.2, §13.2.

Nodes call LangGraph's `get_stream_writer()` and invoke it with a dict payload matching one
of the event models in app/models/sse_events.py. This module consumes
`graph.astream(..., stream_mode=["custom"])` and re-emits each payload as a wire-format SSE
frame, then checks whether the graph is now paused at an interrupt node and, if so, emits
interrupt_reached.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from app.models.sse_events import ErrorEvent, InterruptReachedEvent, encode_sse

logger = logging.getLogger(__name__)


def _readable_failure(exc: Exception) -> str:
    """Turns a provider exception into something a user can act on.

    Rate limits are the one that matters in practice -- both supported providers have free
    tiers with daily caps, and hitting one mid-run is normal rather than exceptional.
    """
    text = str(exc)
    name = type(exc).__name__

    if "RateLimit" in name or "429" in text or "rate_limit" in text:
        return (
            "The AI provider's rate limit was reached, so the remaining suggestions couldn't "
            "be written. Your data is safe and nothing has been changed. Wait for the limit "
            "to reset and start the review again, or switch provider in .env."
        )
    if "Authentication" in name or "401" in text or "api_key" in text.lower():
        return (
            "The AI provider rejected the API key. Check LLM_PROVIDER and the matching key "
            "in .env."
        )
    if "Connection" in name or "Timeout" in name:
        return "Couldn't reach the AI provider. Check your connection and try again."
    return f"The AI provider returned an error: {text[:300]}"


async def stream_graph_as_sse(
    graph: Any, input_state: Any, config: dict, session_id: str
) -> AsyncIterator[str]:
    # A provider failure part-way through -- a rate limit, a dropped connection -- used to
    # propagate out of the generator, which closes the HTTP response mid-chunk. The browser
    # sees a truncated stream and reports a generic network error, with no way to tell a rate
    # limit from a bug. Caught here so the client is told what happened in words.
    try:
        async for stream_mode, payload in graph.astream(
            input_state, config, stream_mode=["custom"]
        ):
            if stream_mode != "custom":
                continue
            event_name = payload.get("event", "message")
            yield f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"
    except Exception as exc:  # noqa: BLE001 -- the response must end cleanly whatever happened
        logger.exception("Graph run failed for session %s", session_id)
        yield encode_sse(
            ErrorEvent(error="Couldn't finish this run", detail=_readable_failure(exc))
        )
        return

    try:
        state = await graph.aget_state(config)
        reached_interrupt = bool(state.next)
    except Exception:
        # The custom-event loop above already finished, meaning the graph itself completed
        # its run (there's nothing after explanation_agent but the human_approval interrupt
        # in the current workflow) -- a failure here is in re-reading that already-persisted
        # state back from the checkpointer, not in the run itself. Don't let that crash the
        # response out from under a client that just watched every card render successfully.
        logger.exception(
            "Failed to read back graph state for session %s after streaming; "
            "still signaling interrupt_reached since explanation_agent completed",
            session_id,
        )
        reached_interrupt = True

    if reached_interrupt:
        yield encode_sse(InterruptReachedEvent(session_id=session_id))
