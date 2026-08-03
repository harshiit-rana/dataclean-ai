"""Action log and undo. PRD US-10.

- GET  /api/actions/{session_id}  -- the persisted action log
- POST /api/undo/{session_id}     -- revert one applied action

Undo re-derives the cleaned data from the original file (see services/undo.py for why
replay beats stored snapshots) and regenerates the audit report, since a report describing
a change that no longer exists would be worse than no report at all.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.core.errors import SessionNotFoundError
from app.dependencies import get_graph
from app.graph.checkpointer import thread_config
from app.graph.nodes.audit_agent import generate_audit_artifacts
from app.models.api_schemas import UndoRequest
from app.models.graph_state import GraphState
from app.services.session_store import session_dir
from app.services.undo import ActionNotFoundError, undo_action

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/actions/{session_id}")
async def get_action_log(session_id: str, graph=Depends(get_graph)) -> dict:
    state = await graph.aget_state(thread_config(session_id))
    if not state.values:
        raise SessionNotFoundError(session_id)
    return {
        "session_id": session_id,
        "actions": state.values.get("action_log", []),
        "cleaned_file_path": state.values.get("cleaned_file_path"),
    }


@router.post("/api/undo/{session_id}")
async def undo(
    session_id: str,
    request: UndoRequest,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_graph),
) -> dict:
    config = thread_config(session_id)
    state = await graph.aget_state(config)
    if not state.values:
        raise SessionNotFoundError(session_id)

    action_log = state.values.get("action_log", [])
    action_id = request.action_id
    if action_id is None:
        # Default to the most recent applied action (US-10: "undo the last change").
        applied = [entry for entry in action_log if entry["status"] == "applied"]
        if not applied:
            raise ActionNotFoundError("<most recent>")
        action_id = applied[-1]["action_id"]

    directory = session_dir(settings.upload_dir, session_id)
    updated_log = undo_action(
        original_path=directory / "original.csv",
        cleaned_path=directory / "cleaned.csv",
        action_log=action_log,
        action_id=action_id,
    )

    await graph.aupdate_state(config, {"action_log": updated_log})

    refreshed = await graph.aget_state(config)
    values = dict(refreshed.values)
    values["action_log"] = updated_log

    artifacts = {}
    try:
        artifacts = await generate_audit_artifacts(GraphState.model_validate(values))
    except Exception:  # noqa: BLE001 -- the undo itself already succeeded
        logger.exception("Audit report could not be regenerated after undoing %s", action_id)

    return {
        "session_id": session_id,
        "undone_action_id": action_id,
        "actions": updated_log,
        "report_regenerated": bool(artifacts),
    }
