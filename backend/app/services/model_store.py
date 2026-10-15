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
# The record of every look at the held-out rows (evaluation_ledger.py). Deliberately not in
# ALL_ARTIFACTS, for the reason the run history is not: `clear` fires when the data changes
# underneath a model, and that is exactly when the record matters most -- cleaning after a look
# is the commonest way to select on the test set by hand. A ledger that a cleaning change wiped
# would hand every user a fresh "independent" score after one undo.
LEDGER_FILENAME = "test_set_ledger.json"
# Which rows were sealed as a final holdout, and how many looks had already been taken when they
# were. Kept for the same reason as the ledger: it is evidence about how the evaluation was
# done, and the moment it matters most is after the data has been changed.
SEALED_FILENAME = "sealed_holdout.json"

# What each run said about its held-out rows, one small file per run, so two runs can be compared
# on exactly the same rows. Not in ALL_ARTIFACTS for the reason the history is not: it is part of
# the record, and the moment it is wanted is after the data has changed.
PREDICTIONS_DIR = "run_predictions"

ALL_ARTIFACTS = (RESULT_FILENAME, PIPELINE_FILENAME, CARD_FILENAME, SCRIPT_FILENAME)

# Every trained model in its own folder, `runs/<run_id>/`, holding its result and fitted pipeline, so a new run no
# longer overwrites the last and an earlier one can be compared or brought back. `current.json` names the run in use;
# `load_result` and `pipeline_path` read through it, so no route needs to know. The card and the script stay at the
# top level as caches of the current run's write-up, dropped whenever the pointer moves.
RUNS_DIR = "runs"
CURRENT_FILENAME = "current.json"
RUN_RESULT_FILENAME = "result.json"
# Written into a run's folder when the data changed underneath it: the run is kept, but it describes rows that are
# no longer the data.
SUPERSEDED_FILENAME = "superseded.json"
DEFAULT_MAX_STORED_RUNS = 10

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
    result = {
        **result,
        "results": [
            {k: v for k, v in entry.items() if k != "_held_out"} for entry in result.get("results", [])
        ],
    }
    cleaned = dict(result)
    cleaned["results"] = [
        {key: value for key, value in entry.items() if key != "pipeline"}
        for entry in result.get("results", [])
    ]
    return _jsonable(cleaned)


def _primary_interval(result: dict, best: dict) -> dict | None:
    key = result.get("primary_metric")
    found = ((best.get("metrics_intervals") or {}).get("metrics") or {}).get(key)
    return {"metric": key, **found} if found else None


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
            # Assigned by whoever saves the run when it has to be known elsewhere too -- the
            # test-set ledger names the run a look belongs to -- and made here otherwise.
            "run_id": result.get("run_id") or uuid.uuid4().hex[:12],
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            # "succeeded", or "failed" / "refused" for an attempt that produced no model (see `attempt_entry`).
            # Absent on entries from before attempts were recorded, all of which succeeded.
            "status": "succeeded",
            "data_fingerprint": fingerprint,
            # Which rows were held out, by the rule that held them. Two runs can be compared on
            # the same rows only if this matches as well as the data.
            "split_key": result.get("split_key"),
            # What the app would have recommended for how to divide the rows, and whether this run
            # divided them differently. A score from a split nobody recommended is a different kind
            # of evidence, and the history has to be able to say so.
            "recommended_split": (result.get("split") or {}).get("recommended"),
            "split_overridden": (result.get("split") or {}).get("overridden"),
            "target_column": result.get("target_column"),
            "task": result.get("task"),
            "best_model": result.get("best_model"),
            "model_label": best.get("label"),
            "models_tried": [r.get("model") for r in result.get("results", [])],
            "primary_metric": result.get("primary_metric"),
            "metrics": best.get("metrics", {}),
            # The primary metric's 95% interval, so a later run can be compared with this one
            # without pretending a difference inside the noise is a result.
            "metric_interval": _primary_interval(result, best),
            "baseline": result.get("baseline", {}),
            # How many times the held-out rows had been used when this run was made, and what
            # that made the score. Absent on runs from before the ledger existed, which are
            # shown as "not recorded" -- never as "independent".
            "evaluation": (
                {
                    "look_count": evaluation.get("look_count"),
                    "status": evaluation.get("status"),
                }
                if (evaluation := result.get("evaluation"))
                else None
            ),
            "verdict": (result.get("assessment") or {}).get("verdict"),
            "headline": (result.get("assessment") or {}).get("headline"),
            "split": {
                "strategy": split.get("strategy"),
                "time_column": split.get("time_column"),
                "group_column": split.get("group_column"),
                "train_rows": split.get("train_rows"),
                "test_rows": split.get("test_rows"),
                # What was asked for, not what a date split ended up with. Two runs that held
                # back different fractions are not the same experiment.
                "requested_test_size": split.get("requested_test_size"),
                # The embargo between training and test, in days. Two runs with different gaps hold
                # out different rows.
                "gap_days": split.get("gap_days"),
            },
            "feature_count": len(result.get("features", {}).get("numeric", []))
            + len(result.get("features", {}).get("categorical", []))
            + len(result.get("features", {}).get("frequent", [])),
            "excluded_columns": result.get("features", {}).get("excluded", []),
            # Which inputs had their extreme values clipped. A run that differs only here is
            # the comparison "did taming the outliers help?" needs to be able to show.
            "clipped_columns": result.get("features", {}).get("clipped", []),
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
            # Which search strategy the loop used, so two tuned runs are not compared as if one search.
            "search_strategy": improvement.get("search_strategy"),
            # What the searches actually used, which for "auto" is the part worth showing.
            "search_strategies": improvement.get("search_strategies") or [],
            # Whether the probabilities were calibrated, and by which method. A run that differs
            # only here is the comparison "did calibrating help?" needs to be able to show.
            "calibration": (result.get("calibration") or {}).get("method"),
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
            # What exactly produced it (run_metadata.py): the software and seeds, the request as validated,
            # one hash of everything that decides the result, how long it took, the data version and how the
            # model was chosen. Absent on runs from before they were recorded, which show as "not recorded".
            **{
                key: (result.get("run_metadata") or {}).get(key)
                for key in (
                    "environment", "seeds", "request", "config_hash", "runtime_seconds", "dataset_version",
                    "selection",
                )
            },
        }
    )


# "interrupted": a search whose process died before it finished (search_markers.py).
ATTEMPT_STATUSES = ("failed", "refused", "interrupted")


def succeeded(entry: dict) -> bool:
    """Whether a history entry is a run with a model and a score, rather than an attempt that produced neither."""
    return entry.get("status", "succeeded") == "succeeded"


def attempt_entry(
    *, status: str, reason: str, code: str | None, fingerprint: str = "", target_column: str | None = None,
    task: str | None = None, metadata: dict | None = None,
) -> dict:
    """A history entry for an attempt that produced no model: refused before fitting (the readiness checks said
    no, the target is gone) or failed during it. The record of what was tried omitted exactly the attempts that
    taught something. Same shape as a run's entry, so a reader of the history needs no second schema, with no
    score, no model and the user-facing reason.
    """
    if status not in ATTEMPT_STATUSES:
        raise ValueError(f"not an attempt status: {status}")
    entry = summarise(
        {"target_column": target_column, "task": task, "run_metadata": metadata}, fingerprint=fingerprint
    )
    request = (metadata or {}).get("request") or {}
    entry.update(
        status=status,
        reason=reason,
        code=code,
        excluded_columns=list(request.get("excluded_columns") or []),
        models_tried=list(request.get("model_names") or []),
    )
    return _jsonable(entry)


def restore_entry(restored: dict) -> dict:
    """A history event for bringing back a stored run as the model in use (`restored` is that run's history
    entry). Not a run: no new score was produced and the test rows were not looked at again, so it is left out
    of every comparison, and it names the run it brought back."""
    entry = summarise(
        {"target_column": restored.get("target_column"), "task": restored.get("task"),
         "best_model": restored.get("best_model")},
        fingerprint=restored.get("data_fingerprint") or "",
    )
    entry.update(
        status="restored",
        restored_run_id=restored.get("run_id"),
        model_label=restored.get("model_label"),
        reason=f"Brought back the {restored.get('model_label') or 'model'} trained at {restored.get('at')}.",
    )
    return _jsonable(entry)


def record_attempt(directory: Path, entry: dict) -> None:
    """Appends an attempt to the history. Never raises: the caller is already reporting a failure, and the
    bookkeeping must not replace that failure with one of its own."""
    try:
        append_history(directory, entry)
    except Exception:  # noqa: BLE001
        logger.warning("A failed attempt could not be recorded", exc_info=True)


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
        dropped = entries[MAX_HISTORY_ENTRIES:]
        entries = entries[:MAX_HISTORY_ENTRIES]
        (directory / HISTORY_FILENAME).write_text(json.dumps(entries), encoding="utf-8")
        for old in dropped:
            # A run that has left the history can no longer be compared with, so its rows go too.
            (directory / PREDICTIONS_DIR / f"{old.get('run_id')}.npz").unlink(missing_ok=True)
        return entries
    except OSError:
        logger.warning("Run history could not be written", exc_info=True)
        return load_history(directory)


def run_dir(directory: Path, run_id: str) -> Path:
    return directory / RUNS_DIR / run_id


def current_run_id(directory: Path) -> Optional[str]:
    """The run in use, or None: nothing trained yet, or the data changed underneath the last one."""
    try:
        pointer = json.loads((directory / CURRENT_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    run_id = pointer.get("run_id") if isinstance(pointer, dict) else None
    return run_id if isinstance(run_id, str) and run_id else None


def _drop_write_ups(directory: Path) -> None:
    """A card and script describe one model. When the model in use changes they are regenerated on request."""
    for stale in (CARD_FILENAME, SCRIPT_FILENAME):
        (directory / stale).unlink(missing_ok=True)


def set_current(directory: Path, run_id: Optional[str], **note: Any) -> None:
    """Points the session at a stored run (or at none), and drops the write-ups of the one before."""
    _drop_write_ups(directory)
    (directory / CURRENT_FILENAME).write_text(
        json.dumps({"run_id": run_id, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **note}),
        encoding="utf-8",
    )


def stored_runs(directory: Path) -> list[str]:
    """The run ids with a folder, oldest first by when their result was written."""
    root = directory / RUNS_DIR
    if not root.is_dir():
        return []
    found = [p for p in root.iterdir() if (p / RUN_RESULT_FILENAME).exists()]
    return [p.name for p in sorted(found, key=lambda p: (p / RUN_RESULT_FILENAME).stat().st_mtime)]


def is_superseded(directory: Path, run_id: str) -> bool:
    """Whether the data changed underneath this stored run after it was trained."""
    return (run_dir(directory, run_id) / SUPERSEDED_FILENAME).exists()


def prune_runs(directory: Path, keep: int) -> list[str]:
    """Deletes the oldest stored runs beyond `keep`, never the current one. Returns the ids removed."""
    import shutil

    keep = max(1, int(keep))
    current = current_run_id(directory)
    others = [r for r in stored_runs(directory) if r != current]
    surplus = len(others) + (1 if current else 0) - keep
    removed = others[: max(0, surplus)]
    for run_id in removed:
        shutil.rmtree(run_dir(directory, run_id), ignore_errors=True)
    return removed


def save_result(
    directory: Path, result: dict, *, fingerprint: str = "", keep_runs: int = DEFAULT_MAX_STORED_RUNS
) -> dict:
    """Writes the run to its own folder -- the evaluation as JSON, the winning pipeline as a joblib blob -- and
    makes it the one in use.

    Returns the serializable form. A failure to persist the pipeline is logged and swallowed:
    the user can still see their results and retrain, which is a far better outcome than
    losing a completed training run to a disk problem. Runs beyond `keep_runs` are pruned, oldest
    first, never the one just made.
    """
    directory.mkdir(parents=True, exist_ok=True)
    # One id names the folder, the history entry and the ledger's look.
    result["run_id"] = result.get("run_id") or uuid.uuid4().hex[:12]
    folder = run_dir(directory, result["run_id"])
    folder.mkdir(parents=True, exist_ok=True)

    payload = strip_pipelines(result)
    (folder / RUN_RESULT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    best = next(
        (r for r in result.get("results", []) if r.get("model") == result.get("best_model")),
        None,
    )
    if best and best.get("pipeline") is not None:
        try:
            import joblib

            joblib.dump(best["pipeline"], folder / PIPELINE_FILENAME)
        except Exception:  # noqa: BLE001
            logger.exception("Fitted pipeline could not be saved; results were still written")

    set_current(directory, result["run_id"])
    # The single files of the layout before run folders: superseded by the run just written.
    for legacy in (RESULT_FILENAME, PIPELINE_FILENAME):
        (directory / legacy).unlink(missing_ok=True)
    try:
        prune_runs(directory, keep_runs)
    except OSError:
        logger.warning("Old runs could not be pruned", exc_info=True)

    entry = summarise(payload, fingerprint=fingerprint)
    if best and best.get("_held_out"):
        save_run_predictions(directory, entry["run_id"], best["_held_out"])
    append_history(directory, entry)
    return payload


def save_run_predictions(directory: Path, run_id: str, held_out: dict) -> None:
    """Keeps what a run said about each held-out row. Never fails the run it describes."""
    try:
        folder = directory / PREDICTIONS_DIR
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {
            "row_ids": np.asarray(held_out["row_ids"]),
            "truth": np.asarray(held_out["truth"]).astype(str),
            "called": np.asarray(held_out["called"]).astype(str),
            "positive_label": np.asarray(str(held_out.get("positive_label") or "")),
        }
        for key in ("scores", "groups"):
            if held_out.get(key) is not None:
                arrays[key] = np.asarray(held_out[key])
        np.savez_compressed(folder / f"{run_id}.npz", **arrays)
    except Exception:  # noqa: BLE001
        logger.warning("Could not keep the held-out predictions of run %s", run_id, exc_info=True)


def load_run_predictions(directory: Path, run_id: str) -> Optional[dict]:
    path = directory / PREDICTIONS_DIR / f"{run_id}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except Exception:  # noqa: BLE001
        logger.warning("Could not read the held-out predictions at %s", path, exc_info=True)
        return None


def compare_runs(directory: Path, newer: dict, older: dict) -> dict:
    """Did the change between two consecutive runs help, hurt, or fall inside the noise?

    A difference of 0.003 on a test set that cannot tell 0.78 from 0.71 is not a result, and the
    history used to colour it green or red anyway. Preferred: a *paired* comparison, resampling
    the shared held-out rows once and scoring both runs on every resample, which is far more
    sensitive than comparing two independent intervals. That needs both runs to have been scored
    on the same rows (same data, same split rule) and to have kept their predictions. Otherwise
    the two stored intervals are compared, and "better" or "worse" is said only when they do not
    overlap. With neither, no verdict at all -- never the old fixed threshold.

    Returns `{"method": "paired" | "intervals" | None, "verdict": "better" | "worse" |
    "within_the_noise" | None, ...}`.
    """
    metric = newer.get("primary_metric")
    paired = _paired_comparison(directory, newer, older, metric)
    if paired is not None:
        return paired

    now = newer.get("metric_interval")
    before = older.get("metric_interval")
    if now and before and now.get("metric") == before.get("metric") == metric:
        if now["low"] > before["high"]:
            verdict = "better"
        elif now["high"] < before["low"]:
            verdict = "worse"
        else:
            verdict = "within_the_noise"
        return {"method": "intervals", "verdict": verdict}
    return {"method": None, "verdict": None}


def _paired_comparison(directory: Path, newer: dict, older: dict, metric: str | None) -> dict | None:
    from app.services import uncertainty

    same_rows = (
        metric
        and newer.get("split_key")
        and newer.get("split_key") == older.get("split_key")
        and newer.get("data_fingerprint")
        and newer.get("data_fingerprint") == older.get("data_fingerprint")
    )
    if not same_rows:
        return None
    a = load_run_predictions(directory, newer["run_id"])
    b = load_run_predictions(directory, older["run_id"])
    if a is None or b is None:
        return None
    try:
        # The same rows, possibly in a different order: line the older run up with the newer.
        position = pd.Index(b["row_ids"]).get_indexer(a["row_ids"])
        if len(a["row_ids"]) != len(b["row_ids"]) or (position < 0).any():
            return None
        task = newer.get("task") or "classification"
        label = str(a["positive_label"]) or None
        found = uncertainty.paired_difference(
            a["truth"] if task != "regression" else a["truth"].astype(float),
            {
                "called": a["called"] if task != "regression" else a["called"].astype(float),
                "scores": a.get("scores"),
            },
            {
                "called": b["called"][position] if task != "regression" else b["called"][position].astype(float),
                "scores": None if b.get("scores") is None else b["scores"][position],
            },
            task,
            metric,
            positive_label=label,
            groups=a.get("groups"),
        )
    except Exception:  # noqa: BLE001
        logger.warning("Paired comparison of two runs failed", exc_info=True)
        return None
    if found is None:
        return None
    if found["low"] > 0:
        verdict = "better"
    elif found["high"] < 0:
        verdict = "worse"
    else:
        verdict = "within_the_noise"
    return {"method": "paired", "verdict": verdict, **found}


def load_run(directory: Path, run_id: str) -> Optional[dict]:
    """A stored run's result, whether or not it is the one in use."""
    path = run_dir(directory, run_id) / RUN_RESULT_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Stored run at %s could not be read", path, exc_info=True)
        return None


def load_result(directory: Path) -> Optional[dict]:
    """The result of the run in use. A session from before run folders has a single file at the top level,
    read as it always was."""
    current = current_run_id(directory)
    if current is not None:
        return load_run(directory, current)
    path = directory / RESULT_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Stored model result at %s could not be read", path, exc_info=True)
        return None


def result_path(directory: Path) -> Path:
    """Where the result of the run in use is stored (it may not exist yet)."""
    current = current_run_id(directory)
    return run_dir(directory, current) / RUN_RESULT_FILENAME if current else directory / RESULT_FILENAME


def pipeline_path(directory: Path) -> Optional[Path]:
    """The fitted pipeline of the run in use (or of a session from before run folders)."""
    current = current_run_id(directory)
    path = run_dir(directory, current) / PIPELINE_FILENAME if current else directory / PIPELINE_FILENAME
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
    """Puts no model in use. Used when the data changes underneath them -- an evaluation, card
    or script describing rows that no longer exist is worse than none.

    The stored runs are kept, each marked as trained on data that has since changed, so they
    can still be compared with; the pointer is unset, the write-ups are dropped, and the single
    files of a session from before run folders are removed.

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
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for run_id in stored_runs(directory):
        try:
            marker = run_dir(directory, run_id) / SUPERSEDED_FILENAME
            if not marker.exists():
                marker.write_text(json.dumps({"since": stamp, "reason": "the data changed"}), encoding="utf-8")
        except OSError:
            logger.warning("Could not mark run %s as superseded", run_id, exc_info=True)
    if current_run_id(directory) is not None or (directory / CURRENT_FILENAME).exists():
        try:
            set_current(directory, None, cleared_because="the data changed")
        except OSError:
            logger.warning("Could not unset the current run", exc_info=True)
