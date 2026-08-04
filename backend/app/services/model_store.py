"""Persistence for a trained model and its evaluation.

Kept out of the LangGraph checkpoint deliberately. A training result runs to tens of
kilobytes of curve data and the fitted pipeline is a binary blob; putting either through the
checkpointer would bloat every subsequent state read for something no graph node ever looks
at. Both live as files in the session directory instead, alongside cleaned.csv and the audit
report, which also means the retention sweep already cleans them up.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# This module owns the artifact filenames so there is one place to change them and one list
# for `clear` to work from -- a name duplicated in the writer and the cleaner is how stale
# files survive an invalidation.
RESULT_FILENAME = "model_result.json"
PIPELINE_FILENAME = "model.joblib"
CARD_FILENAME = "model_card.md"
SCRIPT_FILENAME = "train_model.py"

ALL_ARTIFACTS = (RESULT_FILENAME, PIPELINE_FILENAME, CARD_FILENAME, SCRIPT_FILENAME)


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


def save_result(directory: Path, result: dict) -> dict:
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


def clear(directory: Path) -> None:
    """Drops every model artifact. Used when the data changes underneath them -- an
    evaluation, card or script describing rows that no longer exist is worse than none."""
    for name in ALL_ARTIFACTS:
        try:
            (directory / name).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove %s", name, exc_info=True)
