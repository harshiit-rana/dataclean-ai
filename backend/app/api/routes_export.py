"""Export endpoints. PRD §8.3, FR-41..44.

- GET /api/export/{id}/zip  -- cleaned CSV + PDF audit report in one archive (FR-41)
- GET /api/export/{id}/csv  -- cleaned CSV only
- GET /api/export/{id}/pdf  -- audit report only

Download filenames are derived from the user's original file name for convenience, but the
files served are always the session-scoped artifacts written by the graph. The uploaded copy
(`original.csv`) is never modified, and the user's own file is never touched at all.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends
import pandas as pd
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

from app.config import Settings, get_settings
from app.core.errors import AppError, SessionNotFoundError
from app.dependencies import get_graph
from app.graph.checkpointer import thread_config
from app.graph.nodes.audit_agent import generate_audit_artifacts, is_report_stale
from app.models.graph_state import GraphState
from app.services import cleaning_export, model_export, model_store
from app.services.session_store import read_session_meta, session_dir

logger = logging.getLogger(__name__)

router = APIRouter()


class NothingToExportError(AppError):
    status_code = 409

    def __init__(self, detail: str):
        super().__init__("Nothing to export yet", detail)


def _session_files(settings: Settings, session_id: str) -> tuple[dict, Path]:
    meta = read_session_meta(settings.upload_dir, session_id)
    if meta is None:
        raise SessionNotFoundError(session_id)
    return meta, session_dir(settings.upload_dir, session_id)


CLEAN_SCRIPT_NAME = "clean.py"


async def _cleaning_script(session_id: str, directory: Path, graph, file_name: str) -> str | None:
    """clean.py for this session, built from its action log. None when the log can't be read.

    Built on every request rather than written to disk: an undo or a chat instruction changes
    the log after the fact, and a script on disk would go stale the same way a report does --
    except nothing would mark it stale."""
    try:
        state = await graph.aget_state(thread_config(session_id))
        log = list((state.values or {}).get("action_log", [])) if state else []
        original = pd.read_csv(directory / "original.csv")
        return cleaning_export.build_cleaning_script(original, log, file_name=file_name)
    except Exception:  # noqa: BLE001 -- a missing script must not sink the rest of the export
        logger.exception("clean.py could not be built for session %s", session_id)
        return None


@router.get("/api/export/{session_id}/clean-script", response_class=PlainTextResponse)
async def get_cleaning_script(
    session_id: str,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_graph),
) -> PlainTextResponse:
    """The session's cleaning as a runnable script: reproduces the cleaned file from the
    upload, and cleans a newer export the same way."""
    meta, directory = _session_files(settings, session_id)
    script = await _cleaning_script(session_id, directory, graph, meta.get("file_name", "original.csv"))
    if script is None:
        raise NothingToExportError("The cleaning steps for this session could not be read.")
    return PlainTextResponse(
        script, headers={"Content-Disposition": f'attachment; filename="{CLEAN_SCRIPT_NAME}"'}
    )


def _ensure_training_script(directory: Path) -> None:
    """Writes train_model.py if a model exists and the script doesn't.

    It used to be written only as a side effect of generating the model card, so a user who
    trained a model, skipped the write-up and pressed "Export everything" got an archive
    with no script in it -- the one artifact the whole model path exists to produce, missing
    because of an unrelated button they never pressed. The bundle now builds it on demand,
    the same way the audit report is refreshed on demand just above.

    Never raises: the rest of the archive is still worth serving.
    """
    path = directory / model_store.SCRIPT_FILENAME
    if path.exists():
        return
    result = model_store.load_result(directory)
    if result is None:
        return
    try:
        path.write_text(
            model_export.build_training_script(result, file_name="cleaned.csv"), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001 -- see docstring
        logger.exception("Training script could not be written for the export bundle")


async def _ensure_fresh_report(session_id: str, directory: Path, graph) -> None:
    """Rebuilds the audit report if the data changed since it was last written.

    Chat instructions and undos mark it stale instead of regenerating, because each rebuild
    costs an LLM call and a user adjusting their data can do many of those before ever asking
    to see the document. The bill lands here instead -- once, when it is actually wanted.

    Never raises: a report that couldn't be rebuilt is still better served slightly out of
    date than not served at all, and the export is the last step of a session.
    """
    if not is_report_stale(directory):
        return
    try:
        state = await graph.aget_state(thread_config(session_id))
        if not state.values:
            return
        await generate_audit_artifacts(GraphState.model_validate(dict(state.values)))
    except Exception:  # noqa: BLE001
        logger.exception("Audit report could not be refreshed for session %s", session_id)


def _stem(meta: dict) -> str:
    return Path(meta.get("file_name", "data.csv")).stem


@router.get("/api/export/{session_id}/csv")
async def export_cleaned_csv(
    session_id: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    meta, directory = _session_files(settings, session_id)
    cleaned = directory / "cleaned.csv"
    if not cleaned.exists():
        raise NothingToExportError(
            "No cleaned file exists for this session. Approve some fixes first."
        )
    return FileResponse(cleaned, media_type="text/csv", filename=f"{_stem(meta)}_cleaned.csv")


@router.get("/api/export/{session_id}/pdf")
async def export_audit_pdf(
    session_id: str,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_graph),
) -> FileResponse:
    meta, directory = _session_files(settings, session_id)
    await _ensure_fresh_report(session_id, directory, graph)
    pdf = directory / "audit_report.pdf"
    if not pdf.exists():
        raise NothingToExportError(
            "No audit report exists for this session yet. It is written once cleaning finishes."
        )
    return FileResponse(pdf, media_type="application/pdf", filename=f"{_stem(meta)}_audit_report.pdf")


@router.get("/api/export/{session_id}/report")
async def get_report_markdown(
    session_id: str,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_graph),
) -> dict:
    """Report source for the Export page's on-screen preview. The PDF is the artifact meant
    for sharing; this is so the user can read it before downloading."""
    _meta, directory = _session_files(settings, session_id)
    await _ensure_fresh_report(session_id, directory, graph)
    markdown_path = directory / "audit_report.md"
    if not markdown_path.exists():
        raise NothingToExportError(
            "No audit report exists for this session yet. It is written once cleaning finishes."
        )
    return {
        "session_id": session_id,
        "markdown": markdown_path.read_text(encoding="utf-8"),
        "has_pdf": (directory / "audit_report.pdf").exists(),
    }


@router.get("/api/export/{session_id}/zip")
async def export_bundle(
    session_id: str,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_graph),
) -> StreamingResponse:
    """FR-41: cleaned CSV + PDF audit report in one archive, plus the model artifacts if a
    model was trained.

    Built in memory rather than written to disk -- these are small files and it avoids
    leaving a stale archive behind when a session is cleaned up or re-run.
    """
    meta, directory = _session_files(settings, session_id)
    await _ensure_fresh_report(session_id, directory, graph)
    _ensure_training_script(directory)
    stem = _stem(meta)

    # (path on disk, name inside the archive). The markdown report is included as a fallback
    # so the bundle is still useful when WeasyPrint's OS libs are unavailable. Model files
    # keep their own names: train_model.py is meant to be run as-is, and the card refers to
    # model.joblib by name.
    candidates = [
        (directory / "cleaned.csv", f"{stem}_cleaned.csv"),
        (directory / "audit_report.pdf", f"{stem}_audit_report.pdf"),
        (directory / "audit_report.md", f"{stem}_audit_report.md"),
        (directory / model_store.CARD_FILENAME, model_store.CARD_FILENAME),
        (directory / model_store.SCRIPT_FILENAME, model_store.SCRIPT_FILENAME),
        (directory / model_store.PIPELINE_FILENAME, model_store.PIPELINE_FILENAME),
    ]
    present = [(path, name) for path, name in candidates if path.exists()]
    if not present:
        raise NothingToExportError(
            "Nothing has been generated for this session yet. Approve some fixes first."
        )
    # Only once there is something cleaned to reproduce -- a script for an untouched upload
    # would be a file that copies a CSV.
    clean_script = (
        await _cleaning_script(session_id, directory, graph, meta.get("file_name", "original.csv"))
        if (directory / "cleaned.csv").exists()
        else None
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, name in present:
            archive.write(path, arcname=name)
        if clean_script is not None:
            archive.writestr(CLEAN_SCRIPT_NAME, clean_script)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{stem}_dataclean_export.zip"'},
    )
