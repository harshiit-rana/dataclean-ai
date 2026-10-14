"""Persistence for a trained model and its evaluation.

Kept out of the LangGraph checkpoint deliberately. A training result runs to tens of
kilobytes of curve data and the fitted pipeline is a binary blob; putting either through the
checkpointer would bloat every subsequent state read for something no graph node ever looks
at. Both live as files in the session directory instead, alongside cleaned.csv and the audit
report, which also means the retention sweep already cleans them up.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# This module owns the artifact filenames so there is one place to change them and one list
# for `clear` to work from -- a name duplicated in the writer and the cleaner is how stale
# files survive an invalidation.
RESULT_FILENAME = "model_result.json"
PIPELINE_FILENAME = "model.joblib"
CARD_FILENAME = "model_card.md"
SCRIPT_FILENAME = "train_model.py"
HISTORY_FILENAME = "model_history.json"

ALL_ARTIFACTS = (RESULT_FILENAME, PIPELINE_FILENAME, CARD_FILENAME, SCRIPT_FILENAME)

# Runs kept per session. Long enough that a working session of "try it, change something, try
# again" stays entirely visible; short enough that the file remains small and readable.
MAX_HISTORY_ENTRIES = 20


def _jsonable(value: Any) -> Any:
    """numpy scalars and arrays are not JSON-serializable, and they turn up throughout
    sklearn's output. Converted here rather than at each call site so a new diagnostic can't
    quietly reintroduce the problem."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def strip_pipelines(result: dict) -> dict:
    """Removes the fitted estimator objects so what's left can be serialized and sent."""
    cleaned = dict(result)
    cleaned["results"] = [
        {key: value for key, value in entry.items() if key != "pipeline"}
        for entry in result.get("results", [])
    ]
    return _jsonable(cleaned)


def summarise(result: dict, *, fingerprint: str = "") -> dict:
    """One run, small enough to keep twenty of.

    Deliberately not the whole result. A full evaluation runs to tens of kilobytes of curve
    data, and none of it answers the question this record exists for -- "did the thing I
    changed help?" That needs the settings, the headline number and the verdict, and nothing
    else.

    `fingerprint` identifies the data the run was trained on, so a run from before a cleaning
    change can be shown as what it is rather than compared as though the two were like for
    like. Comparing across a cleaning change is often the most interesting comparison there
    is -- it just has to be labelled.
    """
    best = next(
        (r for r in result.get("results", []) if r.get("model") == result.get("best_model")),
        None,
    ) or {}
    split = result.get("split", {}) or {}
    improvement = result.get("improvement") or {}

    return _jsonable(
        {
            "run_id": uuid.uuid4().hex[:12],
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_fingerprint": fingerprint,
            "target_column": result.get("target_column"),
            "task": result.get("task"),
            "best_model": result.get("best_model"),
            "model_label": best.get("label"),
            "models_tried": [r.get("model") for r in result.get("results", [])],
            "primary_metric": result.get("primary_metric"),
            "metrics": best.get("metrics", {}),
            "baseline": result.get("baseline", {}),
            "verdict": (result.get("assessment") or {}).get("verdict"),
            "headline": (result.get("assessment") or {}).get("headline"),
            "split": {
                "strategy": split.get("strategy"),
                "time_column": split.get("time_column"),
                "group_column": split.get("group_column"),
                "train_rows": split.get("train_rows"),
                "test_rows": split.get("test_rows"),
            },
            "feature_count": len(result.get("features", {}).get("numeric", []))
            + len(result.get("features", {}).get("categorical", []))
            + len(result.get("features", {}).get("frequent", [])),
            "excluded_columns": result.get("features", {}).get("excluded", []),
            # Which derived inputs this run used. Two runs on the same columns with different
            # derived inputs are the comparison the fourth round of the loop exists to make.
            "feature_recipes": [
                r.get("id") for r in result.get("features", {}).get("derived", []) or []
            ],
            # Whether the improvement loop produced this, and what it settled on. "Tuning did
            # not help" is a finding, and it is only visible by comparing runs.
            "tuned": bool(improvement),
            "tuned_settings": improvement.get("settings") or {},
            "stopped_because": improvement.get("stopped_because"),
            # Where the cut-off sat and why. Two runs with the same AUC and different recall
            # are the same model at different lines, and the table has to be able to say so.
            "operating_point": (
                {
                    "objective": point.get("objective"),
                    "description": point.get("description"),
                    "threshold": point.get("threshold"),
                    "positive_label": point.get("positive_label"),
                    "observed": point.get("observed"),
                }
                if (point := result.get("operating_point"))
                else None
            ),
        }
    )


def load_history(directory: Path) -> list[dict]:
    path = directory / HISTORY_FILENAME
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
        return entries if isinstance(entries, list) else []
    except (OSError, ValueError):
        logger.warning("Run history at %s could not be read", path, exc_info=True)
        return []


def append_history(directory: Path, entry: dict) -> list[dict]:
    """Newest first, capped. Never allowed to fail a training run.

    A training run that succeeded and then reported an error because its *bookkeeping* failed
    would be the worst of both -- the user loses a result that exists.
    """
    try:
        entries = [entry] + load_history(directory)
        entries = entries[:MAX_HISTORY_ENTRIES]
        (directory / HISTORY_FILENAME).write_text(json.dumps(entries), encoding="utf-8")
        return entries
    except OSError:
        logger.warning("Run history could not be written", exc_info=True)
        return load_history(directory)


def save_result(directory: Path, result: dict, *, fingerprint: str = "") -> dict:
    """Writes the evaluation as JSON and the winning pipeline as a joblib blob.

    Returns the serializable form. A failure to persist the pipeline is logged and swallowed:
    the user can still see their results and retrain, which is a far better outcome than
    losing a completed training run to a disk problem.
    """
    directory.mkdir(parents=True, exist_ok=True)

    # A card and script from a previous run describe a model that no longer exists. Drop
    # them so the next request regenerates rather than serving the old write-up.
    for stale in (CARD_FILENAME, SCRIPT_FILENAME):
        (directory / stale).unlink(missing_ok=True)

    payload = strip_pipelines(result)
    (directory / RESULT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    best = next(
        (r for r in result.get("results", []) if r.get("model") == result.get("best_model")),
        None,
    )
    if best and best.get("pipeline") is not None:
        try:
            import joblib

            joblib.dump(best["pipeline"], directory / PIPELINE_FILENAME)
        except Exception:  # noqa: BLE001
            logger.exception("Fitted pipeline could not be saved; results were still written")

    append_history(directory, summarise(payload, fingerprint=fingerprint))
    return payload


def load_result(directory: Path) -> Optional[dict]:
    path = directory / RESULT_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Stored model result at %s could not be read", path, exc_info=True)
        return None


def pipeline_path(directory: Path) -> Optional[Path]:
    path = directory / PIPELINE_FILENAME
    return path if path.exists() else None


def full_fingerprint(df) -> str:
    """A SHA-256 digest of everything the model would see: every cell, the column names and
    order, the column types and the row order.

    This used to sample the first 50 rows, on the grounds that hashing every cell on every
    training run cost more than it was worth. It cost almost nothing -- pandas hashes a row at
    a time in compiled code -- and the sample missed the case that matters: a fix that only
    touches rows past the first 50 left the fingerprint unchanged, so two runs on different
    data were compared as like for like, and nothing could reliably key an evaluation to the
    data version it was made on.

    Deterministic across processes: `hash_pandas_object` uses a fixed key, not Python's
    per-process randomised `hash`. Types are part of it on purpose -- a column that arrives as
    int64 in one export and float64 in the next changes what the model is given, so it is not
    the same data. The row index is not: a position carries no information the cells do not.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps([str(c) for c in df.columns]).encode("utf-8"))
    digest.update(json.dumps([str(t) for t in df.dtypes]).encode("utf-8"))
    digest.update(str(df.shape).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def fingerprint_frame(df) -> str:
    """The short form of `full_fingerprint`, for display and for tagging run history.

    Sixteen hex characters. Anything that needs to *rely* on the identity of a dataset -- the
    test-set ledger, for one -- should store `full_fingerprint`.

    Never raises: a fingerprint is a nicety, and failing a finished training run over one
    would lose a result that exists. History entries written before the full hash have the
    old twelve-character form; they never match a new fingerprint, which is the right answer,
    because they were taken from a sample and cannot be shown to describe the current data.
    """
    try:
        return full_fingerprint(df)[:16]
    except Exception:  # noqa: BLE001 -- a fingerprint is a nicety; never fail a run for it
        logger.warning("Could not fingerprint the training data", exc_info=True)
        return ""


def clear(directory: Path) -> None:
    """Drops every model artifact. Used when the data changes underneath them -- an
    evaluation, card or script describing rows that no longer exist is worse than none.

    The run history deliberately survives. It is the record of what was tried, and the moment
    it is most useful is exactly this one: the user has just changed the data and wants to
    know whether the change helped. Each entry carries the fingerprint of the data it was
    trained on, so a run from before the change is shown as such rather than compared as
    though nothing had happened.
    """
    for name in ALL_ARTIFACTS:
        try:
            (directory / name).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove %s", name, exc_info=True)
