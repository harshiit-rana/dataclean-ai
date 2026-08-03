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

from app.models.sse_events import InterruptReachedEvent, encode_sse

logger = logging.getLogger(__name__)


async def stream_graph_as_sse(
    graph: Any, input_state: Any, config: dict, session_id: str
) -> AsyncIterator[str]:
    async for stream_mode, payload in graph.astream(input_state, config, stream_mode=["custom"]):
        if stream_mode != "custom":
            continue
        event_name = payload.get("event", "message")
        yield f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"

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
