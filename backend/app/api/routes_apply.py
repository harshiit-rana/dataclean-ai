"""Using a finished session on a new file.

- `POST /api/apply/{session_id}`                        clean a new CSV the same way, and score it
- `GET  /api/apply/{session_id}/{apply_id}/{kind}`      download `cleaned` or `predictions`

One upload does everything the session can: replays the approved fixes (with the statistics
the session learned, not the new file's own), and -- if a model was trained -- checks the new
rows against the data it learned from, predicts every row that survived cleaning, and, when
the file carries real answers, says how the model did on them.

Every row of the upload is accounted for in the predictions file. A row that a cleaning step
removed comes back with no prediction and the reason, rather than quietly missing: someone
joining predictions back onto their own data by row needs to know which rows have none.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import FileResponse

from app.config import Settings, get_settings
from app.core.errors import (
    AppError,
    FileTooLargeError,
    SessionNotFoundError,
    UnparseableCSVError,
    UnsupportedFileTypeError,
)
from app.dependencies import get_graph
from app.graph.checkpointer import thread_config
from app.services import drift, evaluation_ledger, model_store, replay, scoring
from app.services.encoding import read_csv_bytes
from app.services.session_store import (
    current_dataset_version_in,
    read_session_meta,
    session_dir,
)

logger = logging.getLogger(__name__)

router = APIRouter()

APPLIED_DIR = "applied"
_APPLY_ID = re.compile(r"^[0-9a-f]{12}$")
_FILES = {"cleaned": "cleaned.csv", "predictions": "predictions.csv"}
PREVIEW_ROWS = 20
NOTE_COLUMN = "dataclean_note"


class NothingToApplyError(AppError):
    status_code = 409

    def __init__(self):
        super().__init__(
            "Nothing to apply yet",
            "This session has no approved fixes and no trained model, so there is nothing to "
            "apply to another file. Clean the data or train a model first.",
        )


class ScoringFailedError(AppError):
    status_code = 422

    def __init__(self, detail: str):
        super().__init__("These rows can't be scored", detail)


class AppliedFileNotFoundError(AppError):
    status_code = 404

    def __init__(self):
        super().__init__("Not found", "No such file for this session.")


async def _action_log(graph, session_id: str) -> list[dict]:
    """The session's action log, or [] when it was never cleaned -- a model trained straight
    from the upload is still worth applying."""
    try:
        state = await graph.aget_state(thread_config(session_id))
    except Exception:  # noqa: BLE001 -- no readable log means no fixes, not a failed request
        logger.warning("Could not read the action log for session %s", session_id, exc_info=True)
        return []
    return list((state.values or {}).get("action_log", [])) if state else []


def _run(directory, session_id: str, new: pd.DataFrame, action_log: list[dict], target: str | None) -> dict:
    original = pd.read_csv(directory / "original.csv")
    steps, reproduced = replay.freeze(original, action_log)
    result = model_store.load_result(directory)
    model_path = model_store.pipeline_path(directory)
    if not steps and result is None:
        raise NothingToApplyError()

    notes: list[str] = []
    cleaned_path = directory / "cleaned.csv"
    reference = pd.read_csv(cleaned_path) if cleaned_path.exists() else original
    if cleaned_path.exists() and reproduced.shape != reference.shape:
        # The log is supposed to describe the cleaned file completely. If replaying it does
        # not reproduce the cleaned file, the new file is being cleaned by a description that
        # is known to be incomplete, and the user should hear that before trusting it.
        notes.append(
            f"Replaying this session's fixes on its own file gives {reproduced.shape[0]:,} rows "
            f"and {reproduced.shape[1]} columns, but its cleaned file has "
            f"{reference.shape[0]:,} and {reference.shape[1]}. Some change to the session's data "
            f"isn't in its log, so this file may not be cleaned exactly the same way."
        )

    applied = replay.apply(new, steps, target_column=target)
    apply_id = uuid.uuid4().hex[:12]
    out_dir = directory / APPLIED_DIR / apply_id
    out_dir.mkdir(parents=True, exist_ok=True)
    applied.frame.to_csv(out_dir / _FILES["cleaned"], index=False)

    response: dict = {
        "session_id": session_id,
        "apply_id": apply_id,
        "rows_in": int(len(new)),
        "rows_kept": int(len(applied.frame)),
        "steps": [s.as_dict() for s in applied.steps],
        "dropped": [
            {"row": row + 1, "reason": why}  # 1-based data row, not counting the header
            for row, why in sorted(applied.dropped.items())[:PREVIEW_ROWS]
        ],
        "dropped_count": len(applied.dropped),
        "notes": notes,
        "drift": None,
        "predictions": None,
        "answers": None,
        "downloads": {"cleaned": f"/api/apply/{session_id}/{apply_id}/cleaned"},
    }

    if result is None or model_path is None:
        return response

    try:
        X_new, coerced = scoring.prepare_inputs(applied.frame, result, reference=reference)
        X_ref, _ = scoring.prepare_inputs(reference, result, reference=reference)
    except scoring.ScoringError as exc:
        raise ScoringFailedError(str(exc)) from exc
    notes.extend(coerced)

    numeric, categorical = scoring.input_columns(result)
    response["drift"] = drift.compare(X_ref, X_new, numeric, categorical)

    import joblib

    pipeline = joblib.load(model_path)
    predicted = scoring.predict(pipeline, X_new, result) if len(X_new) else pd.DataFrame(index=X_new.index)

    # Every uploaded row, as uploaded, with its prediction -- or the reason it has none.
    output = new.reset_index(drop=True).join(predicted, how="left")
    output[NOTE_COLUMN] = [
        f"no prediction: {applied.dropped[i]}" if i in applied.dropped else "" for i in output.index
    ]
    output.to_csv(out_dir / _FILES["predictions"], index=False)
    response["downloads"]["predictions"] = f"/api/apply/{session_id}/{apply_id}/predictions"

    columns = list(predicted.columns)
    preview = output.loc[predicted.index[:PREVIEW_ROWS], columns]
    response["predictions"] = {
        "count": int(len(predicted)),
        "columns": columns,
        "preview": [
            {"row": int(i) + 1, **{c: _jsonable(row[c]) for c in columns}}
            for i, row in preview.iterrows()
        ],
        "cutoff": (result.get("operating_point") or {}).get("threshold"),
    }

    target_column = result.get("target_column")
    if target_column and target_column in applied.frame.columns and len(X_new):
        response["answers"] = scoring.check_against_answers(
            pipeline, X_new, applied.frame[target_column], result
        )
        _record_new_data_evaluation(directory, result, apply_id, response["answers"])

    return response


def _record_new_data_evaluation(directory, result: dict, apply_id: str, answers: dict | None) -> None:
    """Writes down that the model was scored on newly uploaded labelled rows.

    A different event from a look, deliberately: these rows are not the session's held-out
    rows, so scoring on them neither uses nor uses up that set (decision 0001, section 1). It
    is recorded so it can be shown, and so a later issue can treat it as a fresh, independent
    evaluation. Only a score that was actually computed is recorded -- a file with too few
    answers to judge reports why and leaves no entry.
    """
    if not answers or answers.get("now") is None:
        return
    evaluation_ledger.record(
        directory,
        event=evaluation_ledger.NEW_DATA,
        split=result.get("split"),
        route="apply_answers",
        run_id=apply_id,
        version=current_dataset_version_in(directory, ensure=True),
        metrics=answers.get("metrics") or {answers.get("metric"): answers.get("now")},
        extra={"rows": answers.get("rows"), "metric": answers.get("metric"), "apply_id": apply_id},
    )


def _jsonable(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


@router.post("/api/apply/{session_id}")
async def apply_to_new_file(
    session_id: str,
    file: UploadFile,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_graph),
) -> dict:
    meta = read_session_meta(settings.upload_dir, session_id)
    if meta is None:
        raise SessionNotFoundError(session_id)
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise UnsupportedFileTypeError()
    raw = await file.read()
    if len(raw) > settings.max_file_size_bytes:
        raise FileTooLargeError(settings.max_file_size_mb)
    new = read_csv_bytes(raw)

    goal = meta.get("goal_context") or {}
    target = (goal.get("target_column") or "").strip() or None
    action_log = await _action_log(graph, session_id)
    directory = session_dir(settings.upload_dir, session_id)

    try:
        # Off the event loop: replay and prediction are CPU-bound pandas and sklearn.
        response = await asyncio.to_thread(_run, directory, session_id, new, action_log, target)
    except (AppError, UnparseableCSVError):
        raise
    except Exception as exc:  # noqa: BLE001 -- the reason goes back, not a bare 500
        logger.exception("Applying session %s to a new file failed", session_id)
        raise ScoringFailedError(f"Stopped unexpectedly: {exc}") from exc
    response["file_name"] = file.filename
    return response


@router.get("/api/apply/{session_id}/{apply_id}/{kind}")
async def download_applied(
    session_id: str,
    apply_id: str,
    kind: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    # Both path parts come from the URL, so both are checked against a closed set before
    # either touches the filesystem -- no `..`, no separators, nothing but what this module
    # wrote.
    if kind not in _FILES or not _APPLY_ID.match(apply_id):
        raise AppliedFileNotFoundError()
    if read_session_meta(settings.upload_dir, session_id) is None:
        raise SessionNotFoundError(session_id)
    path = session_dir(settings.upload_dir, session_id) / APPLIED_DIR / apply_id / _FILES[kind]
    if not path.exists():
        raise AppliedFileNotFoundError()
    return FileResponse(path, media_type="text/csv", filename=f"{kind}_{apply_id}.csv")
