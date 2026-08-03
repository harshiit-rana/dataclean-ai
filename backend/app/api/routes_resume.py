"""POST /api/resume/{session_id} -- REAL, SSE stream. PRD §13.2.

Resumes the graph paused at the human_approval interrupt. The payload becomes exactly what
`interrupt()` returns inside that node (see graph/nodes/human_approval.py), which the
cleaning_agent then acts on. Streams the same custom events as routes_profile.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from langgraph.types import Command

from app.api.sse import sse_response
from app.core.errors import SessionNotFoundError
from app.dependencies import get_graph
from app.graph.checkpointer import thread_config
from app.models.api_schemas import ResumeRequest
from app.services.sse_broker import stream_graph_as_sse

router = APIRouter()


@router.post("/api/resume/{session_id}")
async def resume_session(
    session_id: str,
    resume: ResumeRequest,
    graph=Depends(get_graph),
):
    config = thread_config(session_id)
    state = await graph.aget_state(config)
    if not state.values:
        raise SessionNotFoundError(session_id)

    command = Command(resume=resume.model_dump())
    return sse_response(stream_graph_as_sse(graph, command, config, session_id))
