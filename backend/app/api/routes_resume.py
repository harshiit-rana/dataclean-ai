"""POST /api/resume/{session_id} -- STUB, correct signature. PRD §13.2.

Will resume the paused graph with `graph.astream(Command(resume=payload), config, ...)`,
where `payload` becomes exactly what `interrupt()` returns inside human_approval (see
graph/nodes/human_approval.py), then stream the cleaning_agent/conversational_loop/
audit_agent events the same way routes_profile.py streams the pre-interrupt agents.

TODO: implement per PRD §13.2, once cleaning_agent (§12.3) is implemented.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.api_schemas import ResumeRequest

router = APIRouter()


@router.post("/api/resume/{session_id}")
async def resume_session(session_id: str, resume: ResumeRequest):
    raise HTTPException(status_code=501, detail="Resume is not yet implemented -- see PRD §13.2")
