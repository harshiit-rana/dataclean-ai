"""POST /api/goal/{session_id} -- REAL. PRD §10.1, US-01."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.core.errors import SessionNotFoundError
from app.models.api_schemas import GoalContextRequest, GoalContextResponse
from app.services.session_store import read_session_meta, write_session_meta

router = APIRouter()


@router.post("/api/goal/{session_id}", response_model=GoalContextResponse)
async def submit_goal_context(
    session_id: str,
    goal: GoalContextRequest,
    settings: Settings = Depends(get_settings),
) -> GoalContextResponse:
    meta = read_session_meta(settings.upload_dir, session_id)
    if meta is None:
        raise SessionNotFoundError(session_id)

    meta["goal_context"] = goal.model_dump()
    write_session_meta(settings.upload_dir, session_id, meta)

    return GoalContextResponse(session_id=session_id, accepted=True)
