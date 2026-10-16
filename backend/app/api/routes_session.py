"""GET /api/session/{session_id} -- REAL. Reads current graph state from the checkpointer."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, field_validator

from app.config import Settings, get_settings
from app.core.errors import AppError, SessionNotFoundError
from app.dependencies import get_graph, get_graph_if_any
from app.graph.checkpointer import delete_thread, persist_thread, thread_config
from app.models.api_schemas import SessionStatusResponse
from app.services import jobs, version_diff
from app.services.session_access import META_KEY, issue_token
from app.services.session_store import (
    KEPT_KEY,
    dataset_versions,
    delete_session_files,
    keep_session,
    read_session_meta,
    replace_token_hash,
    session_dir,
)

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


# --- projects kept until deleted (#120, docs/decisions/0004-platform-architecture.md) -------------------------------

PROJECT_NAME_MAX = 100


class KeepRequest(BaseModel):
    name: str = Field(..., description="What to call the project. Shown in the list of kept projects.")

    @field_validator("name")
    @classmethod
    def _a_readable_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Give the project a name.")
        if len(value) > PROJECT_NAME_MAX:
            raise ValueError(f"Keep the name to {PROJECT_NAME_MAX} characters or fewer.")
        return value


def _project(meta: dict) -> dict:
    return {"kept": bool(meta.get(KEPT_KEY)), "name": meta.get("project_name"), "kept_at": meta.get("kept_at"),
            "file_name": meta.get("file_name")}


def _meta_or_404(settings: Settings, session_id: str) -> dict:
    meta = read_session_meta(settings.upload_dir, session_id)
    if meta is None:
        raise SessionNotFoundError(session_id)
    return meta


@router.get("/api/session/{session_id}/project")
async def get_project(session_id: str, settings: Settings = Depends(get_settings)) -> dict:
    """Whether this session is a kept project, and its name."""
    return _project(_meta_or_404(settings, session_id))


@router.post("/api/session/{session_id}/keep")
async def keep_project(
    session_id: str,
    request: KeepRequest,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_graph_if_any),
) -> dict:
    """Keeps this session as a project until it is deleted; called again, renames it.

    A session from before tokens existed has no stored hash, and a kept project always needs one, so it gets a new
    token here, returned once (`token`). Otherwise the token the browser already holds stays the project's token and
    none is returned. The checkpoint's existing keys lose any TTL they were written with.
    """
    meta = _meta_or_404(settings, session_id)
    token = None
    token_hash = None
    if not meta.get(META_KEY):
        token, token_hash = issue_token()
    updated = keep_session(settings.upload_dir, session_id, request.name, token_hash=token_hash)
    if graph is not None:
        await persist_thread(getattr(graph, "checkpointer", None), session_id)
    return {**_project(updated), "token": token}


@router.post("/api/session/{session_id}/token/rotate")
async def rotate_token(session_id: str, settings: Settings = Depends(get_settings)) -> dict:
    """Replaces the session's token with a new one, returned once; the old one stops working at once. For a token
    that has been shared or exposed. Guarded by the current token like every session route."""
    _meta_or_404(settings, session_id)
    token, token_hash = issue_token()
    replace_token_hash(settings.upload_dir, session_id, token_hash)
    return {"token": token}


@router.delete("/api/session/{session_id}")
async def delete_session(
    session_id: str,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_graph_if_any),
) -> dict:
    """Deletes the session or project for good: its checkpoint in Redis, then its folder with every file in it
    (uploads, cleaned versions, runs and fitted models, reports, the sidecar holding the token hash). Irreversible.
    Files are unlinked, not securely erased; copies in backups remain (see the README)."""
    _meta_or_404(settings, session_id)
    # Unfinished jobs first (#103), so a worker does not pick one up as the folder goes. One already running
    # cannot be stopped mid-fit; it fails when it finds the folder gone.
    jobs.cancel_session_jobs(session_dir(settings.upload_dir, session_id))
    checkpoint_deleted = await delete_thread(getattr(graph, "checkpointer", None), session_id)
    removed = await asyncio.to_thread(delete_session_files, settings.upload_dir, session_id)
    return {"deleted": session_id, "files_removed": removed["files"], "bytes_removed": removed["bytes"],
            "checkpoint_deleted": checkpoint_deleted}
