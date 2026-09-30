"""conversational_loop node. PRD §13.1 / FR-37..40 -- deliberately unwired.

The PRD's graph has a node here for plain-English cleaning instructions after the card pass.
That feature exists, as POST /api/chat/{session_id} (routes_chat.py), not as this node: by the
time anyone is typing instructions the graph has already run to completion, and re-entering it
per message would re-run profiling and explanation for nothing. It also maps instructions
onto the whitelisted operations rather than generating pandas code, as the PRD had it.

It is not added to the graph (see workflow.py) and still raises if called. Wire it in only if
chat ever needs to happen *before* cleaning finishes.
"""

from __future__ import annotations

from app.models.graph_state import GraphState


async def conversational_loop(state: GraphState) -> dict:
    raise NotImplementedError("conversational_loop is not yet implemented -- see PRD §13.1")
