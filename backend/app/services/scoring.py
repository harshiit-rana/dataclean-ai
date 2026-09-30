"""Predictions for new rows, from the model this session trained.

The model could already be downloaded as `model.joblib`, but using it meant writing Python:
rebuilding the derived inputs, getting the columns in the right order, and remembering that
`predict()` answers at 0.50 while the model was reported at the cut-off the user chose. This
does all of that the same way the app did when it evaluated the model, so the number a user
was shown and the answers they get back come from the same place.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.services import features as feature_recipes
from app.services import modeling

logger = logging.getLogger(__name__)


class ScoringError(Exception):
    """The new rows can't be scored. Carries a message written for the user."""


def _recipes(result: dict) -> list[feature_recipes.Recipe]:
    return [
        feature_recipes.Recipe(**r)
        for r in (result.get("features") or {}).get("derived", []) or []
        if isinstance(r, dict)
    ]


def input_columns(result: dict) -> tuple[list[str], list[str]]:
    """(numeric, categorical-like) inputs the model expects, in training order."""
    feats = result.get("features") or {}
    return list(feats.get("numeric", [])), list(feats.get("categorical", [])) + list(
        feats.get("frequent", [])
    )


def prepare_inputs(frame: pd.DataFrame, result: dict, reference: pd.DataFrame | None = None):
    """The model's input matrix for `frame`, plus notes on anything that had to be coerced.

    Returns (X, notes). Raises ScoringError naming every missing input at once -- a user
    fixing their export wants the whole list, not one column per attempt.
    """
    frame = feature_recipes.apply(frame, _recipes(result))
    numeric, categorical = input_columns(result)
    missing = [c for c in numeric + categorical if c not in frame.columns]
    if missing:
        raise ScoringError(
            f"This file is missing {len(missing)} column{'s' if len(missing) != 1 else ''} the "
            f"model needs: {', '.join(missing)}. Every input the model was trained on has to "
            f"be present, with the same name."
        )

    X = frame[numeric + categorical].copy()
    notes: list[str] = []
    for column in numeric:
        before = X[column].notna()
        X[column] = pd.to_numeric(X[column], errors="coerce")
        lost = int((before & X[column].isna()).sum())
        if lost:
            notes.append(
                f"{lost} value{'s' if lost != 1 else ''} in '{column}' weren't numbers and were "
                f"treated as blank, which the model fills the way it did in training."
            )
    for column in categorical:
        # A category column the training file held as text can arrive as numbers (all-digit
        # codes, say). The fitted encoder compares against text categories, and mixing the two
        # fails outright. Match the kind the model saw.
        trained_as_text = reference is None or column not in reference or not pd.api.types.is_numeric_dtype(reference[column])
        if trained_as_text and pd.api.types.is_numeric_dtype(X[column]):
            X[column] = X[column].astype(object).where(X[column].notna(), None).map(
                lambda v: None if v is None else str(v)
            )
    return X, notes


def predict(pipeline, X: pd.DataFrame, result: dict) -> pd.DataFrame:
    """Predicted answer per row, with the probability when there is one.

    A two-class model answers at the cut-off it was reported at -- the operating point the
    user chose -- not at `predict()`'s 0.50. Handing back answers at a different line from
    the one the reported precision and recall were measured at would make those numbers
    describe some other model.
    """
    target = result.get("target_column", "target")
    out = pd.DataFrame(index=X.index)
    if result.get("task") == "regression":
        out[f"predicted_{target}"] = pipeline.predict(X)
        return out

    classes = list(pipeline.classes_)
    point = result.get("operating_point") or {}
    if len(classes) == 2 and hasattr(pipeline, "predict_proba"):
        positive = str(point.get("positive_label") or sorted(str(c) for c in classes)[-1])
        index = [str(c) for c in classes].index(positive)
        other = classes[1 - index]
        threshold = float(point.get("threshold") or 0.5)
        probability = np.asarray(pipeline.predict_proba(X))[:, index]
        out[f"predicted_{target}"] = np.where(probability >= threshold, classes[index], other)
        out[f"probability_{positive}"] = probability.round(4)
        return out

    out[f"predicted_{target}"] = pipeline.predict(X)
    if hasattr(pipeline, "predict_proba"):
        out["probability_of_prediction"] = np.asarray(pipeline.predict_proba(X)).max(axis=1).round(4)
    return out


def check_against_answers(pipeline, X: pd.DataFrame, y: pd.Series, result: dict) -> dict | None:
    """When the new file carries real answers, how the model did on them.

    The natural question a month in -- is it still working? -- asked with the same caution as
    the original verdict: below the size at which a score means anything, it says so instead
    of reporting a number a single row could overturn.
    """
    labelled = y.notna()
    if labelled.sum() == 0:
        return None
    X, y = X[labelled], y[labelled]
    task = result.get("task", "classification")
    class_counts = pd.Series(y).astype(str).value_counts().tolist() if task == "classification" else None

    unjudgeable = modeling._too_small_to_judge(int(len(y)), class_counts)
    best = next((r for r in result.get("results", []) if r.get("model") == result.get("best_model")), {})
    key = result.get("primary_metric")
    reported = (best.get("metrics") or {}).get(key)
    if unjudgeable is not None:
        return {"rows": int(len(y)), "metric": key, "reported": reported, "now": None,
                "headline": unjudgeable["headline"]}

    point = result.get("operating_point") or {}
    try:
        metrics = modeling.evaluate(pipeline, X, y, task, point.get("positive_label"))
    except Exception as exc:  # noqa: BLE001 -- answers in an unexpected form are not a crash
        logger.warning("Could not score against the new answers", exc_info=True)
        return {"rows": int(len(y)), "metric": key, "reported": reported, "now": None,
                "headline": f"The answers in this file could not be compared with the model's: {exc}"}

    now = metrics.get(key)
    if now is None or reported is None:
        headline = f"Scored against {len(y):,} real answers."
    else:
        change = now - reported
        if change < -0.05:
            headline = (
                f"On these {len(y):,} rows the model scores {now:.3f} {key}, down from the "
                f"{reported:.3f} it scored when it was built. Something has changed -- the "
                f"data, or what it takes to predict it. Worth retraining on recent rows."
            )
        else:
            headline = (
                f"On these {len(y):,} rows the model scores {now:.3f} {key}, against "
                f"{reported:.3f} when it was built -- it is holding up."
            )
    return {"rows": int(len(y)), "metric": key, "reported": reported, "now": now,
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            "headline": headline}
