"""POST /api/undo/{session_id} -- STUB, correct signature. PRD US-10.

Will revert the most recent (or a specified) applied ActionLogEntry by re-deriving the
working DataFrame from state.action_log, marking that entry status="undone".

TODO: implement per PRD US-10, once cleaning_agent (§12.3) is implemented.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.api_schemas import UndoRequest

router = APIRouter()


@router.post("/api/undo/{session_id}")
async def undo_action(session_id: str, undo: UndoRequest):
    raise HTTPException(status_code=501, detail="Undo is not yet implemented -- see PRD US-10")
