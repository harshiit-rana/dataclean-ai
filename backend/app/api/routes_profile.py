"""POST /api/profile/{session_id} -- REAL, SSE stream.

Starts the graph (goal_intake -> profiling_agent -> explanation_agent -> human_approval)
and streams progress as SSE. The response ends once the graph pauses at the human_approval
interrupt (an `interrupt_reached` event, emitted by stream_graph_as_sse) or completes.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.sse import sse_response
from app.config import Settings, get_settings
from app.core.errors import AppError, SessionNotFoundError
from app.dependencies import get_graph
from app.graph.checkpointer import thread_config
from app.services.session_store import read_session_meta, session_dir
from app.services.sse_broker import stream_graph_as_sse

router = APIRouter()


class GoalContextMissingError(AppError):
    def __init__(self):
        super().__init__(
            "Goal context missing",
            "Submit POST /api/goal/{session_id} before starting profiling.",
        )


@router.post("/api/profile/{session_id}")
async def start_profiling(
    session_id: str,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_graph),
):
    meta = read_session_meta(settings.upload_dir, session_id)
    if meta is None:
        raise SessionNotFoundError(session_id)
    if meta.get("goal_context") is None:
        raise GoalContextMissingError()

    file_path = session_dir(settings.upload_dir, session_id) / "original.csv"
    input_state = {
        "session_id": session_id,
        "file_path": str(file_path),
        "file_name": meta["file_name"],
        "file_size_bytes": meta["file_size_bytes"],
        "goal_context": meta["goal_context"],
    }
    config = thread_config(session_id)

    return sse_response(stream_graph_as_sse(graph, input_state, config, session_id))
