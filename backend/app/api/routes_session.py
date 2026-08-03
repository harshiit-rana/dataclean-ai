"""GET /api/session/{session_id} -- REAL. Reads current graph state from the checkpointer."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.errors import SessionNotFoundError
from app.dependencies import get_graph
from app.graph.checkpointer import thread_config
from app.models.api_schemas import SessionStatusResponse

router = APIRouter()


@router.get("/api/session/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(session_id: str, graph=Depends(get_graph)) -> SessionStatusResponse:
    state = await graph.aget_state(thread_config(session_id))
    if not state.values:
        raise SessionNotFoundError(session_id)

    values = state.values
    issues = values.get("issues", [])
    approved = values.get("approved_action_ids", [])
    skipped = values.get("skipped_action_ids", [])

    return SessionStatusResponse(
        session_id=session_id,
        current_agent=values.get("current_agent"),
        error_message=values.get("error_message"),
        issue_count=len(issues),
        approved_count=len(approved),
        rejected_count=len(skipped),
    )
