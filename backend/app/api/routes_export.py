"""GET /api/export/{session_id}/csv -- REAL. Serves the cleaned CSV for download.

The PDF audit report (PRD §8.3, FR-41..44) is still pending audit_agent (§12.4).

Note the download filename is derived from the user's original file name for their
convenience, but the file served is always the session-scoped `cleaned.csv` written by
cleaning_agent -- the uploaded copy is never modified, and the user's local file is never
touched at all.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.config import Settings, get_settings
from app.core.errors import AppError, SessionNotFoundError
from app.services.session_store import read_session_meta, session_dir

router = APIRouter()


class NothingToExportError(AppError):
    status_code = 409

    def __init__(self):
        super().__init__(
            "Nothing to export yet",
            "No cleaned file exists for this session. Approve some fixes first.",
        )


@router.get("/api/export/{session_id}/csv")
async def export_cleaned_csv(
    session_id: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    meta = read_session_meta(settings.upload_dir, session_id)
    if meta is None:
        raise SessionNotFoundError(session_id)

    cleaned_path = session_dir(settings.upload_dir, session_id) / "cleaned.csv"
    if not cleaned_path.exists():
        raise NothingToExportError()

    original_name = meta.get("file_name", "data.csv")
    download_name = f"{Path(original_name).stem}_cleaned.csv"

    return FileResponse(cleaned_path, media_type="text/csv", filename=download_name)


@router.get("/api/export/{session_id}/pdf")
async def export_audit_pdf(session_id: str):
    raise HTTPException(
        status_code=501,
        detail="The PDF audit report is not yet implemented -- see PRD §12.4 (audit_agent).",
    )
