"""GET /api/readiness/{session_id} -- REAL.

Answers "can this data support the model I said I wanted?" against the file as it stands
right now: the cleaned copy if cleaning has run, the original otherwise.

Computed on demand rather than stored in graph state, deliberately. Undo and conversational
edits both rewrite cleaned.csv after the graph has finished, so a cached verdict would go
stale in exactly the situations where the user is most likely to be looking at it.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.core.errors import SessionNotFoundError, UnparseableCSVError
from app.models.api_schemas import ReadinessResponse
from app.models.graph_state import GoalContext
from app.services.ml_readiness import assess_readiness
from app.services.session_store import read_session_meta, session_dir

router = APIRouter()


@router.get("/api/readiness/{session_id}", response_model=ReadinessResponse)
async def get_readiness(
    session_id: str,
    settings: Settings = Depends(get_settings),
) -> ReadinessResponse:
    meta = read_session_meta(settings.upload_dir, session_id)
    if meta is None:
        raise SessionNotFoundError(session_id)

    directory = session_dir(settings.upload_dir, session_id)
    cleaned = directory / "cleaned.csv"
    file_path = cleaned if cleaned.exists() else directory / "original.csv"

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise UnparseableCSVError() from exc

    raw_goal = meta.get("goal_context")
    goal = GoalContext.model_validate(raw_goal) if raw_goal else None

    report = assess_readiness(df, goal)
    return ReadinessResponse(session_id=session_id, **report)
