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
    ModelArtifactRefusedError,
    SessionNotFoundError,
    UnparseableCSVError,
    UnsupportedFileTypeError,
)
from app.dependencies import get_graph
from app.graph.nodes.audit_agent import mark_report_stale_if_written
from app.services import action_log, drift, evaluation_ledger, model_store, replay, scoring
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
    return await action_log.read(graph, session_id)


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
    # A derived input (a length, a date part) is never blank itself; it is judged on the blankness
    # of the column it is made from, in each file as it was uploaded.
    blanks_from = {
        output: (reference[recipe["column"]], applied.frame[recipe["column"]])
        for recipe in (result.get("features", {}).get("derived") or [])
        if recipe.get("column") in reference and recipe.get("column") in applied.frame
        for output in recipe.get("outputs") or []
    }
    response["drift"] = drift.compare(X_ref, X_new, numeric, categorical, blanks_from=blanks_from)

    try:
        pipeline, load_notes = model_store.load_pipeline(directory)
    except model_store.ModelArtifactError as exc:
        raise ModelArtifactRefusedError(str(exc), exc.code) from exc
    response.setdefault("notes", []).extend(load_notes)
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
        answers = _check_new_answers(
            pipeline, X_new, applied.frame[target_column], result, new, original, target_column
        )
        response["answers"] = answers
        if answers is not None:
            answers["counted_as_independent"] = _record_new_data_evaluation(
                directory, result, apply_id, answers, model_store.full_fingerprint(new)
            )
            if answers.get("overlap_excluded"):
                notes.append(
                    f"{answers['overlap_excluded']:,} labelled row"
                    f"{'s' if answers['overlap_excluded'] != 1 else ''} in this file were already "
                    f"in the file this model was built from, so they were left out of the check."
                )

    return response


def _rows_seen_at_build(new: pd.DataFrame, original: pd.DataFrame, target_column: str) -> set[int]:
    """Row numbers of the new file whose inputs match a row the model was built from.

    A re-export usually repeats earlier rows, and scoring the model on a row it was trained or
    tested on is not new evidence. Matched on the inputs alone, by their text: a row whose
    label has since been revised is still the same row. Columns are compared as the strings
    pandas read, so a column that is whole numbers in both files matches.
    """
    common = [c for c in new.columns if c in original.columns and c != target_column]
    if not common or new.empty:
        return set()
    seen = set(pd.util.hash_pandas_object(original[common].astype(str), index=False).tolist())
    mine = pd.util.hash_pandas_object(new[common].astype(str), index=False).tolist()
    return {int(row) for row, h in zip(new.index, mine) if h in seen}


def _check_new_answers(pipeline, X_new, y_new, result, new, original, target_column) -> dict | None:
    """How the model did on the file's real answers, leaving out rows it was built from."""
    seen = _rows_seen_at_build(new, original, target_column)
    fresh = [i for i in X_new.index if i not in seen]
    excluded = int(y_new.loc[[i for i in X_new.index if i in seen]].notna().sum())

    answers = scoring.check_against_answers(pipeline, X_new.loc[fresh], y_new.loc[fresh], result)
    if answers is None and excluded:
        key = result.get("primary_metric")
        best = next(
            (r for r in result.get("results", []) if r.get("model") == result.get("best_model")), {}
        )
        answers = {
            "rows": 0,
            "metric": key,
            "reported": (best.get("metrics") or {}).get(key),
            "now": None,
            "headline": f"Every labelled row in this file ({excluded:,}) was already in the file "
            "this model was built from, so none of them can be used to check it.",
        }
    if answers is not None:
        answers["overlap_excluded"] = excluded
        if excluded and answers.get("now") is not None:
            answers["headline"] += (
                f" ({excluded:,} labelled row{'s' if excluded != 1 else ''} the model was built "
                f"from {'were' if excluded != 1 else 'was'} left out.)"
            )
    return answers


def _record_new_data_evaluation(
    directory, result: dict, apply_id: str, answers: dict | None, upload_fingerprint: str
) -> bool:
    """Writes down that the model was scored on newly uploaded labelled rows. Returns whether
    it counted as a new, independent evaluation.

    A different event from a look, deliberately: these rows are not the session's held-out
    rows, so scoring on them neither uses nor uses up that set (decision 0001, section 1). It
    is the cleanest holdout there is -- it did not exist when anything was decided -- and is
    shown beside the original score, never in place of it.

    Only a score that was actually computed is recorded: a file with too few answers to judge
    reports why and leaves no entry. And the same file twice is one piece of evidence, not two.
    """
    if not answers or answers.get("now") is None:
        return False
    if evaluation_ledger.already_evaluated(directory, upload_fingerprint, result.get("run_id")):
        answers["already_counted"] = True
        return False
    mark_report_stale_if_written(directory)
    evaluation_ledger.record(
        directory,
        event=evaluation_ledger.NEW_DATA,
        split=result.get("split"),
        route="apply_answers",
        run_id=apply_id,
        version=current_dataset_version_in(directory, ensure=True),
        metrics=answers.get("metrics") or {answers.get("metric"): answers.get("now")},
        extra={
            "rows": answers.get("rows"),
            "metric": answers.get("metric"),
            "reported": answers.get("reported"),
            "apply_id": apply_id,
            "upload_fingerprint": upload_fingerprint,
            "model_run_id": result.get("run_id"),
            "overlap_excluded": answers.get("overlap_excluded", 0),
        },
    )
    return True


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
    log = await _action_log(graph, session_id)
    directory = session_dir(settings.upload_dir, session_id)

    try:
        # Off the event loop: replay and prediction are CPU-bound pandas and sklearn.
        response = await asyncio.to_thread(_run, directory, session_id, new, log, target)
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
