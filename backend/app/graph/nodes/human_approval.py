"""human_approval node. PRD §13.2 -- REAL implementation.

Pauses the graph with LangGraph's native `interrupt()`. Execution stops here and the
current state is checkpointed to Redis; the FastAPI layer surfaces the pending suggestion
cards to the frontend and later resumes the graph via `Command(resume=user_selections)`
from POST /api/resume/{session_id} (PRD §10.1). Whatever value is passed to `resume` is
exactly what `interrupt()` returns below -- there is no separate "read resume payload"
step.
"""

from __future__ import annotations

from langgraph.types import interrupt

from app.models.graph_state import GraphState


def human_approval(state: GraphState) -> dict:
    user_selections = interrupt(
        {
            "suggestion_cards": [card.model_dump() for card in state.suggestion_cards],
        }
    )

    return {
        "approved_action_ids": user_selections.get("approved_action_ids", []),
        "skipped_action_ids": user_selections.get("skipped_action_ids", []),
        "skip_reasons": user_selections.get("skip_reasons", {}),
        "current_agent": "human_approval",
    }
