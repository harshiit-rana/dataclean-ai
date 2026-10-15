"""Replacing the model in use: comparisons between a champion and a challenger, with criteria fixed in advance.

- `POST /api/model/{id}/comparisons`                      start a comparison: two runs and the criteria
- `GET  /api/model/{id}/comparisons`                      every comparison, newest first
- `GET  /api/model/{id}/comparisons/{comparison_id}`      one comparison
- `PUT  /api/model/{id}/comparisons/{comparison_id}/criteria`  change the criteria, until it is evaluated (409 after)
- `POST /api/model/{id}/comparisons/{comparison_id}/evaluate`  score both on the same rows of a labelled file

See services/promotion.py for why the criteria are typed and frozen.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, UploadFile
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.core.errors import AppError, FileTooLargeError, SessionNotFoundError, UnsupportedFileTypeError
from app.dependencies import get_graph_if_any
from app.services import action_log as action_log_service
from app.services import evaluation_ledger, model_store, promotion
from app.services.encoding import read_csv_bytes
from app.services.session_store import current_dataset_version_in, read_session_meta, session_dir

router = APIRouter()


class ComparisonRefusedError(AppError):
    def __init__(self, detail: str, code: str, status: int = 409):
        self.status_code = status
        super().__init__("The comparison cannot do that", detail, code=code)


_NOT_FOUND = {"comparison_not_found", "run_not_found"}


def _refused(exc: promotion.ComparisonError) -> ComparisonRefusedError:
    return ComparisonRefusedError(str(exc), exc.code, 404 if exc.code in _NOT_FOUND else 409)


class StartComparison(BaseModel):
    challenger_run_id: str
    # The model in use when left out: the usual champion.
    champion_run_id: Optional[str] = None
    criteria: promotion.Criteria


class CriteriaUpdate(BaseModel):
    criteria: promotion.Criteria


def _directory(settings: Settings, session_id: str):
    if read_session_meta(settings.upload_dir, session_id) is None:
        raise SessionNotFoundError(session_id)
    return session_dir(settings.upload_dir, session_id)


def _stored_run(directory, run_id: str, role: str) -> dict:
    entry = next((r for r in model_store.load_history(directory) if r.get("run_id") == run_id), None)
    if entry is None or not model_store.succeeded(entry) or model_store.load_run(directory, run_id) is None:
        raise ComparisonRefusedError(
            f"The {role} ({run_id}) is not a stored run with a model in this session.", "run_not_found", 404
        )
    return entry


@router.post("/api/model/{session_id}/comparisons")
async def start_comparison(
    session_id: str, request: StartComparison, settings: Settings = Depends(get_settings)
) -> dict:
    directory = _directory(settings, session_id)
    champion = request.champion_run_id or model_store.current_run_id(directory)
    if not champion:
        raise ComparisonRefusedError("There is no model in use to be the champion; name one.", "no_champion")
    _stored_run(directory, champion, "champion")
    _stored_run(directory, request.challenger_run_id, "challenger")
    try:
        return promotion.create(
            directory, champion_run_id=champion, challenger_run_id=request.challenger_run_id,
            criteria=request.criteria,
        )
    except promotion.ComparisonError as exc:
        raise _refused(exc) from exc


@router.get("/api/model/{session_id}/comparisons")
async def list_comparisons(session_id: str, settings: Settings = Depends(get_settings)) -> dict:
    return {"comparisons": promotion.all_comparisons(_directory(settings, session_id))}


@router.get("/api/model/{session_id}/comparisons/{comparison_id}")
async def get_comparison(session_id: str, comparison_id: str, settings: Settings = Depends(get_settings)) -> dict:
    try:
        return promotion.get(_directory(settings, session_id), comparison_id)
    except promotion.ComparisonError as exc:
        raise _refused(exc) from exc


@router.put("/api/model/{session_id}/comparisons/{comparison_id}/criteria")
async def change_criteria(
    session_id: str, comparison_id: str, request: CriteriaUpdate, settings: Settings = Depends(get_settings)
) -> dict:
    try:
        return promotion.set_criteria(_directory(settings, session_id), comparison_id, request.criteria)
    except promotion.ComparisonError as exc:
        raise _refused(exc) from exc


@router.post("/api/model/{session_id}/comparisons/{comparison_id}/evaluate")
async def evaluate_comparison(
    session_id: str,
    comparison_id: str,
    file: UploadFile,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_graph_if_any),
) -> dict:
    """Scores the champion and the challenger on the same rows of a labelled file, each with its own cleaning and
    inputs, checks every criterion, and freezes the comparison (promotion.evaluate). Recorded in the ledger as an
    evaluation on new data for each model -- never a look at the held-out rows."""
    directory = _directory(settings, session_id)
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise UnsupportedFileTypeError()
    raw = await file.read()
    if len(raw) > settings.max_file_size_bytes:
        raise FileTooLargeError(settings.max_file_size_mb)
    new = read_csv_bytes(raw)
    fingerprint = model_store.full_fingerprint(new)[:16]
    log = await action_log_service.read(graph, session_id)
    try:
        record = await asyncio.to_thread(
            promotion.evaluate, directory, comparison_id, new, file_fingerprint=fingerprint, fallback_actions=log
        )
    except promotion.ComparisonError as exc:
        raise _refused(exc) from exc
    version = current_dataset_version_in(directory)
    for role in ("champion", "challenger"):
        evaluation_ledger.record(
            directory, event=evaluation_ledger.NEW_DATA, split=None, route="compare",
            run_id=record[f"{role}_run_id"], version=version,
            metrics=record["evaluation"]["metrics"][role],
            extra={"comparison_id": comparison_id, "upload_fingerprint": fingerprint, "role": role},
        )
    return record
