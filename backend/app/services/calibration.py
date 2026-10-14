"""Making a model's probabilities honest -- fitted on the training rows, never the test rows.

A model can rank cases well and still say "90% likely" about something that happens 70% of the
time. Tree ensembles do this routinely. The cut-off is chosen on those numbers and the exported
predictions carry them, so a badly calibrated model misleads in two places the ranking metrics
never show. Calibration learns a monotone mapping from the model's score to the rate at which
the answer really happened.

The mapping must be fitted on scores the model did not train on, or it learns the model's
over-confidence on its own training rows and corrects nothing. So `CalibratedClassifierCV` is
given cross-validation folds over the *training* rows: each row is scored by a model that never
saw it, and the mapping is fitted on those scores. The test rows take no part. The folds follow
the same rule as everything else done to the training rows -- whole groups for a grouped split
and copies kept together, forward in time for a date split -- or the mapping would be learned
from the leak the split exists to prevent.

`ensemble=False`: one model fitted on all the training rows, plus one mapping. The default
(`True`) averages one calibrated copy per fold, which changes the ranking and so confuses a
before/after comparison -- AUC moving would be the averaging, not the calibration -- and leaves
an artifact that is k models rather than one. With a single mapping, sigmoid calibration is a
monotone function of the score and AUC is unchanged to the last digit. The exception is a
forward-in-time split: its folds cover only the later rows, so scikit-learn refuses the
single-model form and the averaged form is used, and the result says so.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from app.services import tuning

logger = logging.getLogger(__name__)

METHODS = ("sigmoid", "isotonic")

# Probabilities this far from what happened, on average, are worth offering to fix. Five
# percentage points: "70% likely" that really means anywhere from 65% to 75%. The error is noisy
# on a small test set (see `diagnostics.expected_calibration_error`), so this offers; it does
# not decide.
OFFER_ECE = 0.05

# The error has to clear what chance alone would show, not just the bar above: on a few hundred
# test rows even perfectly honest probabilities score several points, because each bin is a small
# sample. This many times the floor (see `chance_error`) is "clearly more than noise".
CHANCE_MARGIN = 1.5

# Isotonic regression fits a step function with as many steps as the scores allow. On a few
# hundred rows it memorises noise and the "calibrated" probabilities come out worse than the
# originals; scikit-learn's own guidance is about a thousand. The rarer class matters more than
# the total: a thousand rows with twelve positives has twelve points to learn the top of the
# curve from.
ISOTONIC_MIN_TRAIN_ROWS = 1_000
ISOTONIC_MIN_RARE_CLASS = 100
# Sigmoid has two parameters, so it is the method for modest data, but it still needs enough of
# each class to place them.
SIGMOID_MIN_RARE_CLASS = 20


def chance_error(points: list[dict] | None) -> float | None:
    """The calibration error a perfectly calibrated model would still show on this many rows.

    Within a bin the observed rate is a sample, so it differs from the true rate by about
    sqrt(2/pi) * sqrt(p(1-p)/n) on average even when the probabilities are exactly right. Summed
    over the same count-weighted bins as the error itself, that is the floor beneath which a
    reading says nothing. Computed from the chart's own points, so it needs no extra pass.
    """
    rows = [p for p in (points or []) if p.get("count")]
    total = sum(p["count"] for p in rows)
    if not total:
        return None
    return float(
        sum(
            p["count"] / total * math.sqrt(2 / math.pi * p["predicted"] * (1 - p["predicted"]) / p["count"])
            for p in rows
        )
    )


def eligibility(method: str, task: str, y_train, folds: int = tuning.CV_FOLDS) -> str | None:
    """Why this method cannot be used on these training rows, or None if it can.

    Returned as a sentence, because the caller says it to the person who asked.
    """
    if method not in METHODS:
        return f"'{method}' is not a calibration method; the choices are {', '.join(METHODS)}."
    labels = pd.Series(y_train).astype(str)
    counts = labels.value_counts()
    if task != "classification" or len(counts) != 2:
        return (
            "Calibration corrects a probability, so it applies to a target with exactly two "
            "answers."
        )
    rows, rarest = int(len(labels)), int(counts.min())

    if method == "isotonic":
        if rows < ISOTONIC_MIN_TRAIN_ROWS:
            return (
                f"Isotonic calibration needs at least {ISOTONIC_MIN_TRAIN_ROWS:,} training rows "
                f"and this has {rows:,}: it fits a step for every few rows, and on less data it "
                f"memorises noise. Sigmoid needs far less."
            )
        if rarest < ISOTONIC_MIN_RARE_CLASS:
            return (
                f"Isotonic calibration needs at least {ISOTONIC_MIN_RARE_CLASS} training rows of "
                f"the rarer answer and this has {rarest:,}, so the top of the curve would be "
                f"learned from too few cases. Sigmoid needs fewer."
            )
    elif rarest < SIGMOID_MIN_RARE_CLASS:
        return (
            f"Calibration needs at least {SIGMOID_MIN_RARE_CLASS} training rows of the rarer "
            f"answer and this has {rarest:,}."
        )

    if tuning.cv_splitter(task, labels, folds) is None:
        return "The training rows cannot be split into folds, which calibration is fitted on."
    return None


def options(task: str, y_train) -> dict:
    """What can be offered on these training rows: each method, and why not if it cannot."""
    found = {}
    for method in METHODS:
        reason = eligibility(method, task, y_train)
        found[method] = {"available": reason is None, "reason": reason}
    return found


class TrainingFolds:
    """The folds calibration is fitted on, built from whichever rows it is handed.

    scikit-learn calls `split(X, y)` on whatever it was given to fit, which is the training rows
    when the final model is fitted and a subset of them when this is nested inside the cut-off's
    own folds. Groups cannot be passed through `split`, so they are held here and looked up by
    the label of each row of X. That is why X keeps the index it had in the training rows.
    """

    def __init__(self, groups=None, ordered: bool = False, folds: int = tuning.CV_FOLDS):
        self.groups = None if groups is None else pd.Series(groups)
        self.ordered = ordered
        self.folds = folds

    def split(self, X, y=None, groups=None):
        labels = pd.Series(np.asarray(y)).astype(str)
        members = None if self.groups is None else self.groups.loc[X.index].to_numpy()
        splitter = tuning.cv_splitter("classification", labels, self.folds, members, self.ordered)
        if splitter is None:
            raise ValueError("There are too few rows to calibrate on folds.")
        yield from splitter.split(X, labels, members)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.folds


def estimator_factory(build, method: str, *, groups=None, ordered: bool = False):
    """A callable returning a fresh, unfitted calibrated estimator around `build()`.

    The same shape as `build` itself, so the cut-off's out-of-fold loop can use it unchanged and
    choose its line on calibrated probabilities.
    """
    from sklearn.calibration import CalibratedClassifierCV

    def make():
        return CalibratedClassifierCV(
            estimator=build(),
            method=method,
            cv=TrainingFolds(groups, ordered),
            ensemble=ordered,
        )

    return make


def fit(build, method: str, X_train, y_train, *, groups=None, ordered: bool = False):
    """The calibrated model, fitted on the training rows only, and how many folds it used.

    Afterwards the custom fold object is replaced by the plain fold count: it is only read when
    fitting, and leaving it in would make the saved model depend on this application's code to
    be loaded.
    """
    model = estimator_factory(build, method, groups=groups, ordered=ordered)()
    model.fit(X_train, y_train)
    used = sum(1 for _ in TrainingFolds(groups, ordered).split(X_train, y_train))
    model.set_params(cv=used)
    return model, used


# What changes when the probabilities are corrected. The first three are the point; the rest
# move because the default 0.50 line now sits on honest probabilities, and on a rare answer
# that flags fewer rows.
COMPARED = ("brier_score", "expected_calibration_error", "roc_auc", "accuracy", "precision", "recall", "f1")


def before_and_after(before: dict, after: dict) -> dict:
    keep = lambda metrics: {k: metrics[k] for k in COMPARED if isinstance(metrics.get(k), (int, float))}  # noqa: E731
    return {"before": keep(before), "after": keep(after)}


def report(
    *, requested: str, task: str, y_train, uncalibrated_metrics: dict, applied: dict | None = None,
    refused_because: str | None = None, points: list[dict] | None = None
) -> dict | None:
    """What the result says about calibration. None for a target calibration does not apply to.

    Always present for a two-class result, even when nothing was asked for, because it carries
    the offer: whether the probabilities are far enough off to be worth fixing and which
    methods the data supports.
    """
    if task != "classification" or "brier_score" not in uncalibrated_metrics:
        return None
    error = float(uncalibrated_metrics.get("expected_calibration_error", 0.0))
    chance = chance_error(points)
    out = {
        "method": applied["method"] if applied else "none",
        "requested": requested,
        "refused_because": refused_because,
        "ece_threshold": OFFER_ECE,
        # The uncalibrated model's error on the held-out rows, which the offer is made on, and
        # what chance alone would show on this many rows.
        "error": round(error, 4),
        "chance_error": None if chance is None else round(chance, 4),
        "recommended": bool(
            applied is None and error > OFFER_ECE and error > CHANCE_MARGIN * (chance or 0.0)
        ),
        "options": options(task, y_train),
        "folds": applied["folds"] if applied else 0,
        "ensemble": applied["ensemble"] if applied else False,
        "fitted_on": applied["fitted_on"] if applied else None,
        "before": None,
        "after": None,
    }
    if applied:
        out.update(before_and_after(uncalibrated_metrics, applied["metrics"]))
    return out
