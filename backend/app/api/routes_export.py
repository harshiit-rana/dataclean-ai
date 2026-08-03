"""GET /api/export/{session_id}/{artifact} -- STUB, correct signature. PRD §8.3, FR-41..44.

Will serve the cleaned CSV (state.cleaned_file_path) and the WeasyPrint-rendered PDF audit
report produced by graph/nodes/audit_agent.py.

TODO: implement per PRD §8.3, FR-41..44, once audit_agent (§12.4) is implemented.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/api/export/{session_id}/{artifact}")
async def export_artifact(session_id: str, artifact: Literal["csv", "pdf"]):
    raise HTTPException(status_code=501, detail="Export is not yet implemented -- see PRD §8.3")
