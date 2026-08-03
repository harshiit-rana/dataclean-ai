"""goal_intake node. PRD §13.1: "Receives and validates goal context."

By the time the graph starts, GoalContext was already collected via POST /api/goal/{id}
(PRD §10.1) and passed into the initial GraphState by routes_profile.py. This node's job
is purely to validate that it is present and well-formed before profiling spends any time
or tokens -- it does not collect the goal itself.
"""

from __future__ import annotations

from app.models.graph_state import GraphState


def goal_intake(state: GraphState) -> dict:
    if state.goal_context is None:
        return {
            "error_message": (
                "No goal context was provided. Submit POST /api/goal/{session_id} "
                "before starting profiling."
            ),
            "current_agent": "goal_intake",
        }
    return {"current_agent": "goal_intake", "error_message": None}
