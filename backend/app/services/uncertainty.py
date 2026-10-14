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
        is_case = y_true == positive_label
        metrics["roc_auc"] = _binary_auc(weights, is_case, scores)
        metrics.update(_calibration_metrics(weights, is_case.astype(float), scores, total))
    return metrics


def _calibration_metrics(weights, y_binary, scores, total) -> dict[str, np.ndarray]:
    """Brier score and expected calibration error, weighted. The error is the count-weighted mean
    of |predicted - observed| per bin, which is |sum of w * (p - y) in the bin| summed over bins and
    divided by the total weight -- so each resample needs one matrix product, not a loop."""
    from app.services import diagnostics

    residual = scores - y_binary
    slots = diagnostics.calibration_slots(scores)
    members = np.zeros((len(scores), diagnostics.CALIBRATION_BINS))
    members[np.arange(len(scores)), slots] = residual
    return {
        "brier_score": _ratio(weights @ (residual**2), total),
        "expected_calibration_error": _ratio(np.abs(weights @ members).sum(axis=1), total),
    }


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


def paired_difference(
    y_true,
    first: dict,
    second: dict,
    task: str,
    metric: str,
    positive_label: str | None = None,
    groups=None,
    resamples: int | None = None,
    seed: int = 42,
    level: float = LEVEL,
) -> dict | None:
    """How much better the first set of predictions is than the second on the *same* rows.

    Two independent intervals ask whether two ranges overlap, and ranges overlap long after the
    two models stopped being the same: most of what makes a test set easy or hard is shared by
    both. Resampling the rows once and scoring both prediction sets on every resample cancels
    that shared part, so the interval is on the *difference* and is far narrower. That is what
    lets "this change helped" be said when it did, and "within the noise" when it did not.

    `first` and `second` are `{"called": predictions, "scores": P(positive) or None}` for the
    same rows in the same order. Returns `{"difference", "low", "high"}`, or None if the metric
    cannot be computed from both.
    """
    truth, called_a, strata = _prepare(task, y_true, first["called"], positive_label)
    _, called_b, _ = _prepare(task, y_true, second["called"], positive_label)
    rows = len(truth)
    label = _resolve_label(task, truth, positive_label)
    scores_a = None if first.get("scores") is None else np.asarray(first["scores"], dtype=float)
    scores_b = None if second.get("scores") is None else np.asarray(second["scores"], dtype=float)
    group_array = None if groups is None else np.asarray(groups)

    ones = np.ones((1, rows))
    a0 = _metrics(ones, task, truth, called_a, scores_a, label).get(metric)
    b0 = _metrics(ones, task, truth, called_b, scores_b, label).get(metric)
    if a0 is None or b0 is None or np.isnan(a0[0]) or np.isnan(b0[0]):
        return None

    count = resamples or resample_count(rows)
    rng = np.random.default_rng(seed)
    chunk = max(1, CHUNK_CELLS // max(rows, 1))
    deltas: list[np.ndarray] = []
    drawn = 0
    while drawn < count:
        size = min(chunk, count - drawn)
        weights = _weights(rng, rows, size, strata, group_array)
        a = _metrics(weights, task, truth, called_a, scores_a, label)[metric]
        b = _metrics(weights, task, truth, called_b, scores_b, label)[metric]
        deltas.append(a - b)
        drawn += size

    values = np.concatenate(deltas)
    usable = values[~np.isnan(values)]
    if len(usable) < 0.5 * count:
        return None
    low_q, high_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    low, high = np.percentile(usable, [low_q, high_q])
    return {
        "difference": round(float(a0[0] - b0[0]), 4),
        "low": round(float(low), 4),
        "high": round(float(high), 4),
    }


def fold_difference(first, second, level: float = LEVEL) -> dict | None:
    """Whether one model beat another across the *same* cross-validation folds.

    `first` and `second` are the per-fold scores of two models scored on identical folds. The
    differences fold by fold are paired, so what makes a fold easy or hard cancels, and the
    interval is on how much the first beat the second.

    Folds are not independent draws: every pair of training sets overlaps, so the plain
    paired t-interval is too narrow and calls differences real far too often (Nadeau and
    Bengio, 2003). The standard correction widens the variance of the mean difference by the
    ratio of test to training rows, 1/(k-1) for k-fold, which is used here for every splitter.
    With only a few folds the t quantile is large, and the interval is wide: that is the
    correct message about how little a handful of folds can establish.

    Returns `{"difference", "low", "high"}` (first minus second), or None when the folds do not
    line up, there are fewer than two, or any score is not a number.
    """
    from scipy import stats

    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        return None
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None

    folds = len(a)
    deltas = a - b
    mean = float(deltas.mean())
    spread = float(deltas.std(ddof=1))
    standard_error = spread * float(np.sqrt(1 / folds + 1 / (folds - 1)))
    half = float(stats.t.ppf((1 + level) / 2, folds - 1)) * standard_error
    return {
        "difference": round(mean, 4),
        "low": round(mean - half, 4),
        "high": round(mean + half, 4),
    }


def _rates_at_threshold(weights, y_binary, flagged) -> dict[str, np.ndarray]:
    """Weighted precision and recall of a fixed cut-off, for a (resamples, rows) weight matrix.

    Undefined is NaN, not 0: a resample in which nothing is flagged has no precision, and calling
    it 0 would pull the interval down for a reason that has nothing to do with the model. (The
    point estimate in `operating_point.metrics_at` follows scikit-learn and says 0.)
    """
    caught = weights @ (flagged * y_binary)
    flags = weights @ flagged
    cases = weights @ y_binary
    precision = np.full(len(weights), np.nan)
    recall = np.full(len(weights), np.nan)
    np.divide(caught, flags, out=precision, where=flags > 0)
    np.divide(caught, cases, out=recall, where=cases > 0)
    return {"precision": precision, "recall": recall}


def threshold_intervals(
    y_binary,
    scores,
    threshold: float,
    groups=None,
    resamples: int | None = None,
    seed: int = 42,
    level: float = LEVEL,
) -> dict:
    """Intervals for the precision and recall a cut-off delivered on the held-out rows.

    The cut-off is fixed -- it was chosen on the training folds, so the test rows had no say in
    it -- and the rows are resampled, within each answer so a rare one cannot vanish, or by whole
    groups for a grouped split. That is the range the delivered figures could plausibly land in
    on a different set of rows, and it is what the promised figure from the training folds has to
    be compared with: a ten-point gap on forty positives is well within it.

    A resample in which a rate is undefined (nothing flagged, no real cases) is dropped; a rate
    undefined in most resamples gets no interval. Returns the same shape as `metric_intervals`.
    """
    y = np.asarray(y_binary, dtype=float)
    flagged = (np.asarray(scores, dtype=float) >= threshold).astype(float)
    rows = len(y)
    count = resamples or resample_count(rows)
    group_array = None if groups is None else np.asarray(groups)
    strata = y.astype(int).astype(str)

    rng = np.random.default_rng(seed)
    chunk = max(1, CHUNK_CELLS // max(rows, 1))
    collected: dict[str, list[np.ndarray]] = {}
    drawn = 0
    while drawn < count:
        size = min(chunk, count - drawn)
        weights = _weights(rng, rows, size, strata, group_array)
        for name, values in _rates_at_threshold(weights, y, flagged).items():
            collected.setdefault(name, []).append(values)
        drawn += size

    low_q, high_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    intervals = {}
    for name, parts in collected.items():
        values = np.concatenate(parts)
        usable = values[~np.isnan(values)]
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
