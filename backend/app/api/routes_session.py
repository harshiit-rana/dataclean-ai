"""GET /api/session/{session_id} -- REAL. Reads current graph state from the checkpointer."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query

from app.config import Settings, get_settings
from app.core.errors import AppError, SessionNotFoundError
from app.dependencies import get_graph
from app.graph.checkpointer import thread_config
from app.models.api_schemas import SessionStatusResponse
from app.services import version_diff
from app.services.session_store import dataset_versions, read_session_meta, session_dir

router = APIRouter()


@router.get("/api/session/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(
    session_id: str,
    graph=Depends(get_graph),
    settings: Settings = Depends(get_settings),
) -> SessionStatusResponse:
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
        dataset_versions=dataset_versions(settings.upload_dir, session_id),
    )


class VersionNotFoundError(AppError):
    status_code = 404

    def __init__(self, detail: str):
        super().__init__("No such version", detail, code="version_not_found")


@router.get("/api/session/{session_id}/versions/diff")
async def diff_dataset_versions(
    session_id: str,
    from_version: int = Query(..., alias="from", ge=1),
    to_version: int = Query(..., alias="to", ge=1),
    settings: Settings = Depends(get_settings),
) -> dict:
    """What changed between two versions of the session's data: rows removed and added, columns added and
    removed, changed cells per column, and the fixes applied or undone in between (version_diff.py)."""
    if read_session_meta(settings.upload_dir, session_id) is None:
        raise SessionNotFoundError(session_id)
    versions = dataset_versions(settings.upload_dir, session_id)
    directory = session_dir(settings.upload_dir, session_id)
    try:
        return await asyncio.to_thread(version_diff.diff, directory, versions, from_version, to_version)
    except version_diff.VersionNotFoundError as exc:
        raise VersionNotFoundError(str(exc)) from exc
