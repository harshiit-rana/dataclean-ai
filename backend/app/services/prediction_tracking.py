"""The distribution of a model's predictions, application by application, against what it predicted at training time.

A sudden change in the share of rows flagged, or in the typical predicted value, is often the earliest visible sign
that something has changed -- in the data, the process that produces it, or the model's fit to it -- and it is
available before any answers arrive. Each application keeps a summary of its predictions (`summarise`); the
reference is the same summary of the model's own held-out predictions when it was trained (`reference`), at the
same cut-off, so the two are comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _quantiles(values) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[~np.isnan(array)]
    if not len(array):
        return {}
    return {str(q): round(float(np.quantile(array, q)), 4) for q in QUANTILES}


def _binary(called, scores, positive: str) -> dict:
    called = np.asarray(called).astype(str)
    return {
        "kind": "binary",
        "positive": positive,
        "rows": int(len(called)),
        "positive_rate": round(float((called == positive).mean()), 4) if len(called) else None,
        "probability_quantiles": _quantiles(scores) if scores is not None else {},
    }


def summarise(predicted: pd.DataFrame, result: dict) -> dict | None:
    """What one application predicted, in a few numbers."""
    if predicted is None or predicted.empty:
        return None
    target = result.get("target_column", "target")
    called = predicted[f"predicted_{target}"]
    if result.get("task") == "regression":
        values = pd.to_numeric(called, errors="coerce")
        return {"kind": "values", "rows": int(len(values)), "mean": round(float(values.mean()), 4),
                "quantiles": _quantiles(values)}
    probability = [c for c in predicted.columns if c.startswith("probability_") and c != "probability_of_prediction"]
    if probability:
        positive = probability[0].removeprefix("probability_")
        return _binary(called, predicted[probability[0]], positive)
    shares = called.astype(str).value_counts(normalize=True)
    return {"kind": "classes", "rows": int(len(called)), "shares": {str(k): round(float(v), 4) for k, v in shares.items()}}


def reference(held_out: dict | None, result: dict) -> dict | None:
    """The same summary for the model's held-out predictions at training time, at the cut-off it answers at."""
    if not held_out:
        return None
    task = result.get("task")
    if task == "regression":
        values = np.asarray(held_out["called"], dtype=float)
        return {"kind": "values", "rows": int(len(values)), "mean": round(float(values.mean()), 4),
                "quantiles": _quantiles(values)}
    scores = held_out.get("scores")
    positive = str(held_out.get("positive_label") or "")
    if scores is not None and positive:
        # Applications answer at the chosen cut-off; so does the reference, or the two rates would differ by design.
        threshold = float((result.get("operating_point") or {}).get("threshold") or 0.5)
        labels = sorted(set(np.asarray(held_out["truth"]).astype(str)))
        other = next((label for label in labels if label != positive), "")
        called = np.where(np.asarray(scores, dtype=float) >= threshold, positive, other)
        return _binary(called, scores, positive)
    called = pd.Series(np.asarray(held_out["called"]).astype(str))
    shares = called.value_counts(normalize=True)
    return {"kind": "classes", "rows": int(len(called)), "shares": {str(k): round(float(v), 4) for k, v in shares.items()}}
