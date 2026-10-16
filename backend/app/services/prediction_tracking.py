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


# How much a distribution of values may move before it counts as changed: its median, by this share of the
# reference's interquartile range.
VALUE_SHIFT_IQR = 0.5
# How different two sets of class shares may be (total variation distance) before they count as changed.
CLASS_SHIFT = 0.1


def compare(reference: dict | None, current: dict | None) -> dict:
    """Whether this file's predictions differ from the training-time reference, by a rule stated per kind: for two
    classes, the 95% interval of the difference in the share flagged excludes zero; for several, the total variation
    distance between the class shares is at least `CLASS_SHIFT`; for values, the median moved by more than
    `VALUE_SHIFT_IQR` of the reference's interquartile range."""
    if not reference or not current or reference.get("kind") != current.get("kind"):
        return {"verdict": "unknown", "headline": "There is no training-time reference to compare the predictions with."}
    kind = current["kind"]
    if kind == "binary":
        p1, n1 = reference.get("positive_rate"), reference.get("rows") or 0
        p2, n2 = current.get("positive_rate"), current.get("rows") or 0
        if p1 is None or p2 is None or not n1 or not n2:
            return {"verdict": "unknown", "headline": "Too few rows to compare the share flagged."}
        gap = p2 - p1
        margin = 1.96 * ((p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) ** 0.5)
        changed = abs(gap) > margin
        return {
            "verdict": "changed" if changed else "no_clear_change",
            "difference": round(gap, 4), "low": round(gap - margin, 4), "high": round(gap + margin, 4),
            "headline": (
                f"{round(p2 * 100)}% of rows flagged as {current.get('positive')}, against {round(p1 * 100)}% on the "
                f"held-out rows at training"
                + (": a change beyond what the number of rows explains." if changed else ", within what the number of rows explains.")
            ),
        }
    if kind == "classes":
        shares1, shares2 = reference.get("shares") or {}, current.get("shares") or {}
        distance = 0.5 * sum(abs(shares1.get(k, 0) - shares2.get(k, 0)) for k in set(shares1) | set(shares2))
        changed = distance >= CLASS_SHIFT
        return {"verdict": "changed" if changed else "no_clear_change", "distance": round(distance, 4),
                "headline": f"The mix of predicted classes moved by {distance:.0%} of rows from the training-time mix"
                            + ("." if changed else ", a small change.")}
    q1, q2 = reference.get("quantiles") or {}, current.get("quantiles") or {}
    if not q1 or not q2:
        return {"verdict": "unknown", "headline": "Too few values to compare."}
    spread = (q1.get("0.75", 0) - q1.get("0.25", 0)) or 1e-9
    shift = q2.get("0.5", 0) - q1.get("0.5", 0)
    changed = abs(shift) > VALUE_SHIFT_IQR * spread
    return {"verdict": "changed" if changed else "no_clear_change", "median_shift": round(shift, 4),
            "headline": f"The median prediction is {q2.get('0.5')} against {q1.get('0.5')} at training"
                        + (", a shift larger than half the usual spread." if changed else ", within the usual spread.")}
