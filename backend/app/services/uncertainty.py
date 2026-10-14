"""How much a held-out score could move on a different set of rows.

"AUC 0.78" on 60 rows and on 6,000 look identical on the page. The verdict already asks whether a
score beats what luck could produce on a test set of that size, but the reader never sees the
plausible range. A percentile bootstrap over the test rows gives it: resample the rows with
replacement, recompute every metric, and report the middle 95% of the results.

The resamples are *weights*, not new tables. Each is a vector of how many times each row was drawn,
so every metric is a weighted sum over one set of predictions made once. That is what keeps it
cheap (no refitting, no repeated `predict`) and lets the same function resample entities instead
of rows for a grouped split, where the rows within an entity are not independent and resampling
them one by one would report an interval that is too narrow to be true.

Point estimates computed with all weights equal to one reproduce `modeling.evaluate`, and a test
holds the two together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEVEL = 0.95
DEFAULT_RESAMPLES = 1000
# Below this many resamples a 2.5th/97.5th percentile is too coarse to be worth printing.
MIN_RESAMPLES = 300
# Work is bounded by rows x resamples: past this, fewer resamples are drawn so a 20,000-row test
# set still takes well under a second. 300 is enough for the percentiles; the interval is
# slightly noisier, not wrong.
WORK_BUDGET = 5_000_000
# Matrix cells held in memory at once. The weights are processed in chunks of resamples.
CHUNK_CELLS = 2_000_000


def resample_count(rows: int) -> int:
    return int(min(DEFAULT_RESAMPLES, max(MIN_RESAMPLES, WORK_BUDGET // max(rows, 1))))


def _weights(rng, rows: int, count: int, strata: np.ndarray | None, groups: np.ndarray | None):
    """`count` resamples as a (count, rows) matrix of draw counts.

    Grouped: whole entities are drawn, and every row of a drawn entity comes along with it.
    Otherwise rows are drawn within each stratum (the class, for classification), so every
    resample has exactly the class counts the test set has: a rare class can never vanish from
    a resample and leave recall undefined.
    """
    weights = np.zeros((count, rows), dtype=np.float64)
    if groups is not None:
        codes, uniques = pd.factorize(groups)
        size = len(uniques)
        for b in range(count):
            weights[b] = np.bincount(rng.integers(0, size, size), minlength=size)[codes]
        return weights

    if strata is None:
        strata = np.zeros(rows, dtype=int)
    for stratum in np.unique(strata):
        members = np.flatnonzero(strata == stratum)
        size = len(members)
        for b in range(count):
            weights[b, members] = np.bincount(rng.integers(0, size, size), minlength=size)
    return weights


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """numerator / denominator, with 0 where the denominator is 0 (sklearn's zero_division=0)."""
    out = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _binary_auc(weights: np.ndarray, is_positive: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Weighted ROC AUC, ties counted as half, for a (resamples, rows) weight matrix.

    Rows are sorted by score once. For each group of tied scores, a positive is credited with
    every negative strictly below it plus half the negatives it ties with -- the Mann-Whitney
    form, which is exactly what `roc_auc_score` computes. Resamples that lack a class give NaN.
    """
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    positive = is_positive[order].astype(float)
    negative = 1.0 - positive

    ordered = weights[:, order]
    positives = np.add.reduceat(ordered * positive, starts, axis=1)
    negatives = np.add.reduceat(ordered * negative, starts, axis=1)
    below = np.cumsum(negatives, axis=1) - negatives
    numerator = (positives * (below + 0.5 * negatives)).sum(axis=1)
    denominator = positives.sum(axis=1) * negatives.sum(axis=1)

    out = np.full(len(weights), np.nan)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _classification_metrics(weights, y_true, y_pred, scores, positive_label) -> dict[str, np.ndarray]:
    total = weights.sum(axis=1)
    metrics = {"accuracy": _ratio(weights @ (y_true == y_pred).astype(float), total)}

    labels = sorted(set(y_true) | set(y_pred))
    binary = len(sorted(set(y_true))) == 2

    def prf(label):
        actual, called = y_true == label, y_pred == label
        tp = weights @ (actual & called).astype(float)
        fp = weights @ (~actual & called).astype(float)
        fn = weights @ (actual & ~called).astype(float)
        return _ratio(tp, tp + fp), _ratio(tp, tp + fn), _ratio(2 * tp, 2 * tp + fp + fn)

    if binary:
        precision, recall, f1 = prf(positive_label)
    else:
        per_label = [prf(label) for label in labels]
        precision, recall, f1 = (np.mean([p[i] for p in per_label], axis=0) for i in range(3))
    metrics.update({"precision": precision, "recall": recall, "f1": f1})

    if binary and scores is not None:
        metrics["roc_auc"] = _binary_auc(weights, y_true == positive_label, scores)
    return metrics


def _regression_metrics(weights, y_true, y_pred) -> dict[str, np.ndarray]:
    total = weights.sum(axis=1)
    error = y_pred - y_true
    mean = _ratio(weights @ y_true, total)
    total_ss = weights @ (y_true**2) - total * mean**2
    sse = weights @ (error**2)
    r2 = np.full(len(weights), np.nan)
    np.divide(sse, total_ss, out=r2, where=total_ss > 0)
    return {
        "r2": 1.0 - r2,
        "mae": _ratio(weights @ np.abs(error), total),
        "rmse": np.sqrt(_ratio(sse, total)),
    }


def _metrics(weights, task, y_true, y_pred, scores, positive_label):
    if task == "regression":
        return _regression_metrics(weights, y_true, y_pred)
    return _classification_metrics(weights, y_true, y_pred, scores, positive_label)


def _prepare(task, y_true, y_pred, positive_label):
    if task == "regression":
        return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float), None
    truth = pd.Series(y_true).astype(str).to_numpy()
    called = pd.Series(y_pred).astype(str).to_numpy()
    return truth, called, truth


def _resolve_label(task: str, truth: np.ndarray, requested: str | None) -> str | None:
    """Which of two answers counts as "a case", by the rule `modeling.resolve_positive_label`
    uses: the one asked for if it is present, else the last in sorted order."""
    if task == "regression":
        return None
    classes = sorted(set(truth))
    if len(classes) != 2:
        return None
    return str(requested) if requested is not None and str(requested) in classes else classes[-1]


def point_estimates(y_true, y_pred, scores, task: str, positive_label: str | None = None) -> dict:
    """The same metrics as `modeling.evaluate`, computed with every row drawn once.

    Exists so the bootstrap and `evaluate` can be shown to agree: an interval around a number
    the page does not show would be worse than none.
    """
    truth, called, strata = _prepare(task, y_true, y_pred, positive_label)
    weights = np.ones((1, len(truth)))
    label = _resolve_label(task, truth, positive_label)
    score_array = None if scores is None else np.asarray(scores, dtype=float)
    values = _metrics(weights, task, truth, called, score_array, label)
    return {name: float(v[0]) for name, v in values.items() if not np.isnan(v[0])}


def metric_intervals(
    y_true,
    y_pred,
    scores,
    task: str,
    positive_label: str | None = None,
    groups=None,
    resamples: int | None = None,
    seed: int = 42,
    level: float = LEVEL,
) -> dict:
    """Percentile-bootstrap intervals for every headline metric.

    `scores` is P(positive class) for each row when there are two classes (for ROC AUC), else
    None. `groups`, when given, makes the resampling draw whole groups -- for a grouped split,
    where rows of one customer are not independent observations and drawing them one at a time
    would claim more certainty than the data holds. A classification target is otherwise
    resampled within each class.

    Deterministic for a fixed `seed`. Returns
    `{"level", "resamples", "by", "metrics": {name: {"low", "high"}}}`.
    """
    truth, called, strata = _prepare(task, y_true, y_pred, positive_label)
    rows = len(truth)
    count = resamples or resample_count(rows)
    label = _resolve_label(task, truth, positive_label)
    score_array = None if scores is None else np.asarray(scores, dtype=float)
    group_array = None if groups is None else np.asarray(groups)

    rng = np.random.default_rng(seed)
    chunk = max(1, CHUNK_CELLS // max(rows, 1))
    collected: dict[str, list[np.ndarray]] = {}
    drawn = 0
    while drawn < count:
        size = min(chunk, count - drawn)
        weights = _weights(rng, rows, size, strata, group_array)
        for name, values in _metrics(weights, task, truth, called, score_array, label).items():
            collected.setdefault(name, []).append(values)
        drawn += size

    low_q, high_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    intervals = {}
    for name, parts in collected.items():
        values = np.concatenate(parts)
        usable = values[~np.isnan(values)]
        # A metric that is undefined in most resamples (a group-level draw that lacks a class)
        # has no honest interval.
        if len(usable) < 0.5 * count:
            continue
        low, high = np.percentile(usable, [low_q, high_q])
        intervals[name] = {"low": round(float(low), 4), "high": round(float(high), 4)}

    return {
        "level": level,
        "resamples": count,
        "by": "groups" if groups is not None else "rows",
        "metrics": intervals,
    }
