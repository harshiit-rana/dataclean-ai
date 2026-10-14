"""Choosing where a classifier's cut-off sits -- without looking at the test set.

A two-class model produces a probability, and somewhere between 0 and 1 a line is drawn: above
it the row is flagged, below it it is not. Every precision and recall figure in the app is
measured at that line, and by default it sits at 0.50 for no reason other than that it is the
number scikit-learn uses. Where it *should* sit is a business question -- how many false
alarms is catching one more real case worth? -- and the threshold sweep has shown that trade
for a while without letting anyone act on it.

The obvious way to act on it is also the wrong way: look at the sweep, which is computed on
the test rows, pick the cut-off that looks best there, and report the numbers at that cut-off.
That is selecting on the test set. In small it is exactly the mistake the improvement loop is
built to avoid in large, and on a small test set it is enough to manufacture a recall figure
that will not survive contact with new data.

So the user states the trade-off, not the number. The cut-off is then chosen on out-of-fold
predictions over the *training* rows -- each row scored by a model that never saw it, the same
folds the improvement loop searches on -- and only then is the test set opened to report what
that cut-off actually does on rows nothing was chosen against. Both figures are kept, side by
side, because the gap between them is itself information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.services.tuning import cv_splitter

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5

# A fixed grid rather than the unique predicted scores: it makes the chosen number
# reproducible from the exported script's point of view and stops a single row's score from
# becoming the cut-off. Two decimal places is as fine as anyone will ever set by hand.
THRESHOLD_GRID = np.round(np.linspace(0.01, 0.99, 99), 2)

OBJECTIVES = ("default", "cost", "min_recall", "min_precision", "f1")


@dataclass(frozen=True)
class OperatingPoint:
    """What the user said matters. Data, never a number they picked off the test set."""

    objective: str = "default"
    # For "cost": the relative price of the two mistakes. Only the ratio matters, but two
    # numbers read more naturally than one ("a miss costs 5, a false alarm costs 1").
    cost_of_missing: float = 1.0
    cost_of_false_alarm: float = 1.0
    # For "min_recall" / "min_precision": the floor to clear, 0..1.
    target_rate: float | None = None
    # Which answer counts as "a case". Defaults to the last label in sorted order, which is
    # right for yes/no and 0/1 and wrong for churned/retained -- so it is settable.
    positive_label: str | None = None

    @property
    def is_default(self) -> bool:
        return self.objective == "default"

    def describe(self) -> str:
        """The trade-off in the user's terms, for the card, the script and the run history."""
        if self.objective == "cost":
            ratio = self.cost_of_missing / self.cost_of_false_alarm if self.cost_of_false_alarm else float("inf")
            if ratio == 1:
                return "a missed case and a false alarm cost the same"
            if ratio > 1:
                return f"a missed case costs {ratio:g}x a false alarm"
            return f"a false alarm costs {1 / ratio:g}x a missed case"
        if self.objective == "min_recall":
            return f"catch at least {self.target_rate:.0%} of real cases"
        if self.objective == "min_precision":
            return f"be right at least {self.target_rate:.0%} of the time when flagging"
        if self.objective == "f1":
            return "balance catching cases against false alarms"
        return "the default cut-off of 0.50"


def parse(raw: dict | None) -> OperatingPoint:
    """Validates an operating point from the API. Anything malformed becomes the default
    rather than an error, except an unknown objective, which is a client bug worth raising."""
    if not raw:
        return OperatingPoint()
    objective = str(raw.get("objective") or "default")
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown operating-point objective: {objective!r}")

    def _rate(value) -> float | None:
        try:
            rate = float(value)
        except (TypeError, ValueError):
            return None
        return rate if 0.0 < rate <= 1.0 else None

    def _cost(value, default: float = 1.0) -> float:
        try:
            cost = float(value)
        except (TypeError, ValueError):
            return default
        return cost if cost >= 0 else default

    point = OperatingPoint(
        objective=objective,
        cost_of_missing=_cost(raw.get("cost_of_missing")),
        cost_of_false_alarm=_cost(raw.get("cost_of_false_alarm")),
        target_rate=_rate(raw.get("target_rate")),
        positive_label=(str(raw["positive_label"]) if raw.get("positive_label") not in (None, "") else None),
    )
    if point.objective in ("min_recall", "min_precision") and point.target_rate is None:
        raise ValueError(f"{objective} needs a target rate between 0 and 1.")
    if point.objective == "cost" and point.cost_of_missing == 0 and point.cost_of_false_alarm == 0:
        raise ValueError("At least one of the two costs has to be above zero.")
    return point


# --- measuring at a cut-off ----------------------------------------------------------------


def metrics_at(y_binary: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Everything the confusion matrix says at one cut-off, as counts and rates."""
    y = np.asarray(y_binary).astype(int)
    predicted = np.asarray(scores) >= threshold
    tp = int((predicted & (y == 1)).sum())
    fp = int((predicted & (y == 0)).sum())
    fn = int((~predicted & (y == 1)).sum())
    tn = int((~predicted & (y == 0)).sum())
    positives = tp + fn
    flagged = tp + fp
    precision = tp / flagged if flagged else 0.0
    recall = tp / positives if positives else 0.0
    return {
        "threshold": round(float(threshold), 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall) if precision + recall else 0.0, 4),
        "accuracy": round((tp + tn) / len(y), 4) if len(y) else 0.0,
        "flagged": flagged,
        "caught": tp,
        "missed": fn,
        "false_alarms": fp,
        "rows": int(len(y)),
    }


def _cost_of(point: OperatingPoint, at: dict) -> float:
    return point.cost_of_missing * at["missed"] + point.cost_of_false_alarm * at["false_alarms"]


def choose_threshold(y_binary: np.ndarray, scores: np.ndarray, point: OperatingPoint) -> dict:
    """Picks the cut-off on the grid that best serves the stated objective.

    Returns {"threshold", "reachable", "note"}. `reachable` is False when a floor was asked
    for that no cut-off clears -- the app then takes the nearest thing and says so, rather
    than quietly reporting a cut-off that does not do what was asked.
    """
    if point.is_default:
        return {"threshold": DEFAULT_THRESHOLD, "reachable": True, "note": None}

    table = [metrics_at(y_binary, scores, float(t)) for t in THRESHOLD_GRID]

    if point.objective == "cost":
        # Ties broken towards 0.50: when two cut-offs cost the same, the less surprising one.
        best = min(table, key=lambda at: (_cost_of(point, at), abs(at["threshold"] - DEFAULT_THRESHOLD)))
        return {"threshold": best["threshold"], "reachable": True, "note": None}

    if point.objective == "f1":
        best = max(table, key=lambda at: (at["f1"], -abs(at["threshold"] - DEFAULT_THRESHOLD)))
        return {"threshold": best["threshold"], "reachable": True, "note": None}

    floor = float(point.target_rate or 0.0)

    if point.objective == "min_recall":
        # The highest cut-off that still catches enough -- fewest false alarms subject to the
        # floor. Recall only falls as the cut-off rises, so the candidates are a prefix.
        clearing = [at for at in table if at["recall"] >= floor]
        if clearing:
            best = max(clearing, key=lambda at: at["threshold"])
            return {"threshold": best["threshold"], "reachable": True, "note": None}
        best = max(table, key=lambda at: (at["recall"], at["threshold"]))
        return {
            "threshold": best["threshold"],
            "reachable": False,
            "note": (
                f"No cut-off catches {floor:.0%} of cases on the training folds; the most "
                f"this model can catch is about {best['recall']:.0%}, and that is what has "
                f"been set. Catching more would need a better model, not a lower line."
            ),
        }

    # min_precision: the lowest cut-off that is still right often enough -- most caught
    # subject to the floor. Precision is not monotone, so this is a plain search.
    clearing = [at for at in table if at["precision"] >= floor and at["flagged"] > 0]
    if clearing:
        best = max(clearing, key=lambda at: (at["recall"], -at["threshold"]))
        return {"threshold": best["threshold"], "reachable": True, "note": None}
    flagged_any = [at for at in table if at["flagged"] > 0] or table
    best = max(flagged_any, key=lambda at: (at["precision"], -at["threshold"]))
    return {
        "threshold": best["threshold"],
        "reachable": False,
        "note": (
            f"No cut-off is right {floor:.0%} of the time when it flags a row, on the "
            f"training folds; the best this model manages is about {best['precision']:.0%}, "
            f"and that is what has been set. Being right more often would need a better "
            f"model, not a higher line."
        ),
    }


# --- the out-of-fold scores the choice is made on ------------------------------------------


def out_of_fold_scores(
    build, X_train, y_train, positive_label: str, groups=None, ordered: bool = False
) -> tuple[np.ndarray | None, int]:
    """P(positive) for each training row, from a model that did not train on it.

    Returns (scores, folds). Rows no fold validated on are NaN: with forward-in-time folds the
    earliest chunk is only ever trained on, which is also why this is an explicit loop rather
    than `cross_val_predict` -- that refuses folds that don't partition the rows. None when
    the training data can't support two folds, in which case no cut-off can be chosen
    honestly and the caller falls back to the default.
    """
    y = pd.Series(y_train).astype(str).reset_index(drop=True)
    # Folds are taken by position, so the labels do not matter here -- but they are kept, because
    # a calibrated estimator fitted inside each fold finds its rows' groups by label.
    X = X_train
    splitter = cv_splitter("classification", y, groups=groups, ordered=ordered)
    if splitter is None:
        return None, 0

    scores = np.full(len(y), np.nan)
    folds = 0
    group_values = None if groups is None else pd.Series(groups).reset_index(drop=True)
    for train_idx, test_idx in splitter.split(X, y, group_values):
        try:
            model = build().fit(X.iloc[train_idx], y.iloc[train_idx])
        except Exception:  # noqa: BLE001 -- e.g. an early time chunk with one class only
            logger.debug("A fold could not be fitted for the cut-off", exc_info=True)
            continue
        classes = [str(c) for c in model.classes_]
        if str(positive_label) not in classes:
            continue
        scores[test_idx] = np.asarray(model.predict_proba(X.iloc[test_idx]))[:, classes.index(str(positive_label))]
        folds += 1
    if folds == 0:
        return None, 0
    return scores, folds


def apply(
    *,
    build,
    pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    point: OperatingPoint,
    positive_label: str,
    groups=None,
    ordered: bool = False,
    calibrated_with: str | None = None,
) -> dict | None:
    """Chooses the cut-off on the training folds and reports it against the test set.

    `calibrated_with` names the calibration method when `build` and `pipeline` are calibrated
    estimators. The line is then on calibrated probabilities -- chosen on out-of-fold scores
    from a calibrated model, and applied to a calibrated model -- and the result says so.

    Returns None when the model has no probabilities to threshold, or the training data
    cannot be folded. Never raises: an operating point is a refinement of a result that
    already exists, and losing the result to it would be the wrong trade.
    """
    try:
        if not hasattr(pipeline, "predict_proba"):
            return None

        test_binary = (pd.Series(y_test).astype(str) == str(positive_label)).to_numpy().astype(int)
        classes = [str(c) for c in pipeline.classes_]
        test_scores = np.asarray(pipeline.predict_proba(X_test))[:, classes.index(str(positive_label))]

        if point.is_default:
            observed = metrics_at(test_binary, test_scores, DEFAULT_THRESHOLD)
            return {
                **_request_fields(point, positive_label),
                "calibrated_with": calibrated_with,
                "threshold": DEFAULT_THRESHOLD,
                "chosen_on": "the default; nothing was chosen",
                "folds": 0,
                "reachable": True,
                "note": None,
                "expected": None,
                "observed": observed,
                "at_default": observed,
            }

        train_scores, folds = out_of_fold_scores(build, X_train, y_train, positive_label, groups, ordered)
        if train_scores is None:
            return None
        train_binary = (pd.Series(y_train).astype(str) == str(positive_label)).to_numpy().astype(int)
        # Only rows some fold validated on. With forward-in-time folds the earliest rows are
        # never validated, and a score of NaN must not be read as a prediction.
        covered = ~np.isnan(train_scores)
        train_binary, train_scores = train_binary[covered], train_scores[covered]
        if len(train_binary) == 0 or train_binary.min() == train_binary.max():
            return None

        chosen = choose_threshold(train_binary, train_scores, point)
        threshold = chosen["threshold"]

        return {
            **_request_fields(point, positive_label),
            "calibrated_with": calibrated_with,
            "threshold": threshold,
            "chosen_on": (
                f"out-of-fold predictions over {len(train_binary):,} training rows "
                f"({folds} folds{', each validated on rows later than it trained on' if ordered else ''}"
                f"{f', with the probabilities calibrated ({calibrated_with}) inside each fold, so the line is on calibrated probabilities' if calibrated_with else ''}); "
                f"the test set was not consulted"
            ),
            "folds": folds,
            "reachable": chosen["reachable"],
            "note": chosen["note"],
            # What the training folds promised, and what the sealed test set delivered.
            "expected": metrics_at(train_binary, train_scores, threshold),
            "observed": metrics_at(test_binary, test_scores, threshold),
            "at_default": metrics_at(test_binary, test_scores, DEFAULT_THRESHOLD),
        }
    except Exception:  # noqa: BLE001 -- see docstring
        logger.warning("Operating point could not be chosen", exc_info=True)
        return None


def _request_fields(point: OperatingPoint, positive_label: str) -> dict:
    return {
        "objective": point.objective,
        "description": point.describe(),
        "cost_of_missing": point.cost_of_missing,
        "cost_of_false_alarm": point.cost_of_false_alarm,
        "target_rate": point.target_rate,
        "positive_label": str(positive_label),
    }
