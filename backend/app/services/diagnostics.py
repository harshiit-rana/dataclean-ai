"""The standard diagnostics you'd expect from any serious model build -- as numbers.

This module deliberately produces **data, not pictures**. Charts are rendered in the browser
where they can be interactive, theme-aware and readable at any size; a PNG baked here would
be none of those, and would drag matplotlib into an API image for no gain. The same numbers
also go into the model card, where tables serve better than figures anyway.

Two tiers, split by cost:

- `compute_diagnostics` runs on predictions the model has already made against the held-out
  test set. It costs one `predict` and is computed for every candidate model.
- `compute_deep_diagnostics` refits the model many times over (cross-validation, learning
  curve, permutation importance). It is run for the chosen model only, and skipped entirely
  on data large enough for the cost to matter.

Everything is downsampled before it leaves: an ROC curve over 50,000 test rows has 50,000
points, of which about a hundred are visually distinguishable.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.services.tuning import scoring_for

logger = logging.getLogger(__name__)

MAX_CURVE_POINTS = 120  # beyond this a line chart gains nothing but payload
MAX_SCATTER_POINTS = 600
CALIBRATION_BINS = 10
RESIDUAL_BINS = 30

# Refitting a model 5-10 times is fine on a spreadsheet and not on a warehouse extract.
MAX_ROWS_FOR_DEEP_DIAGNOSTICS = 20_000
CV_FOLDS = 5
PERMUTATION_REPEATS = 5


def _thin(rows: list, limit: int) -> list:
    """Evenly samples a long series down to `limit` points, always keeping the endpoints --
    the ends of an ROC curve are the part people actually read."""
    if len(rows) <= limit:
        return rows
    indices = np.unique(np.linspace(0, len(rows) - 1, limit).astype(int))
    return [rows[i] for i in indices]


def _positive_scores(pipeline, X, positive_label) -> np.ndarray | None:
    """P(positive class) for each row, or None when the model can't produce probabilities."""
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        classes = [str(c) for c in pipeline.classes_]
        index = classes.index(str(positive_label))
        return np.asarray(pipeline.predict_proba(X))[:, index]
    except Exception:  # noqa: BLE001 -- diagnostics must never break a successful training run
        logger.debug("Probabilities unavailable", exc_info=True)
        return None


# --- classification -------------------------------------------------------------------------


def roc_curve_points(y_binary: np.ndarray, scores: np.ndarray) -> list[dict]:
    """False-positive rate against true-positive rate across every threshold. The diagonal
    is what guessing looks like; distance above it is what the model adds."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_binary, scores)
    points = [
        {
            "fpr": round(float(f), 4),
            "tpr": round(float(t), 4),
            # roc_curve's first threshold is +inf by construction; report it as 1.0 so the
            # value stays JSON-serializable and readable.
            "threshold": round(float(min(th, 1.0)), 4),
        }
        for f, t, th in zip(fpr, tpr, thresholds)
    ]
    return _thin(points, MAX_CURVE_POINTS)


def precision_recall_points(y_binary: np.ndarray, scores: np.ndarray) -> list[dict]:
    """More informative than ROC when the positive class is rare, because it ignores the
    true negatives that a lopsided dataset has in abundance."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_binary, scores)
    # precision_recall_curve returns one more precision/recall than thresholds.
    points = [
        {
            "recall": round(float(r), 4),
            "precision": round(float(p), 4),
            "threshold": round(float(t), 4),
        }
        for p, r, t in zip(precision[:-1], recall[:-1], thresholds)
    ]
    return _thin(points, MAX_CURVE_POINTS)


def calibration_points(y_binary: np.ndarray, scores: np.ndarray, bins: int = CALIBRATION_BINS) -> list[dict]:
    """Does "70% likely" actually happen 70% of the time?

    A model can rank cases perfectly (great AUC) while its probabilities are badly wrong, and
    that matters the moment anyone treats the number as a probability rather than a ranking.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    slots = np.clip(np.digitize(scores, edges[1:-1]), 0, bins - 1)

    points = []
    for b in range(bins):
        mask = slots == b
        count = int(mask.sum())
        if count == 0:
            continue
        points.append(
            {
                "predicted": round(float(scores[mask].mean()), 4),
                "actual": round(float(y_binary[mask].mean()), 4),
                "count": count,
                "bin_start": round(float(edges[b]), 2),
                "bin_end": round(float(edges[b + 1]), 2),
            }
        )
    return points


def threshold_table(y_binary: np.ndarray, scores: np.ndarray, steps: int = 19) -> list[dict]:
    """Precision, recall and volume at each possible cut-off.

    The most directly actionable table here. Models are usually judged at the default 0.5
    cut-off, but that is a business decision, not a modelling one -- how many false alarms is
    catching one more real case worth? This lays the trade out so someone can choose.
    """
    positives = int(y_binary.sum())
    rows = []
    for threshold in np.linspace(0.05, 0.95, steps):
        predicted = scores >= threshold
        flagged = int(predicted.sum())
        true_positives = int((predicted & (y_binary == 1)).sum())
        precision = true_positives / flagged if flagged else 0.0
        recall = true_positives / positives if positives else 0.0
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(
                    2 * precision * recall / (precision + recall) if precision + recall else 0.0, 4
                ),
                "flagged": flagged,
                "caught": true_positives,
                "missed": positives - true_positives,
            }
        )
    return rows


def per_class_report(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> list[dict]:
    """The classic per-class precision/recall/f1/support table. On an imbalanced target this
    is where you see that the rare class is being missed entirely."""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return [
        {
            "label": label,
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    ]


def confusion_cells(matrix: list[list[int]], labels: list[str]) -> list[dict]:
    """Long-form confusion matrix for a heatmap: one row per cell, with the row-normalized
    share alongside the count. Raw counts alone make a lopsided matrix unreadable -- the rare
    class's row is all small numbers regardless of how well it did."""
    cells = []
    for i, actual in enumerate(labels):
        row_total = sum(matrix[i]) or 1
        for j, predicted in enumerate(labels):
            count = int(matrix[i][j])
            cells.append(
                {
                    "actual": actual,
                    "predicted": predicted,
                    "count": count,
                    "share_of_actual": round(count / row_total, 4),
                    "correct": i == j,
                }
            )
    return cells


# --- regression -----------------------------------------------------------------------------


def predicted_vs_actual_points(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """The scatter every regression writeup opens with. Points on the diagonal are exact."""
    rows = [
        {"actual": round(float(a), 4), "predicted": round(float(p), 4)}
        for a, p in zip(y_true, y_pred)
    ]
    return _thin(rows, MAX_SCATTER_POINTS)


def residual_points(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """Residuals against predicted value. Structure here -- a fan, a curve, a step -- means
    the model is systematically wrong somewhere, which a single R² number hides completely."""
    rows = [
        {"predicted": round(float(p), 4), "residual": round(float(a - p), 4)}
        for a, p in zip(y_true, y_pred)
    ]
    return _thin(rows, MAX_SCATTER_POINTS)


def residual_histogram(y_true: np.ndarray, y_pred: np.ndarray, bins: int = RESIDUAL_BINS) -> list[dict]:
    """Residuals should cluster around zero and look roughly symmetric. A shifted centre means
    the model is biased high or low across the board."""
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    if residuals.size == 0:
        return []

    # Effectively one value -- a model wrong by a constant amount, or predicting a single
    # number. Compared relative to magnitude, not against exact zero: `x - (x + 25)` leaves a
    # spread of about 1e-14 rather than 0, which is still far too narrow to split into bins.
    spread = float(residuals.max() - residuals.min())
    scale = max(1.0, float(np.abs(residuals).max()))
    if spread <= scale * 1e-9:
        return [{"centre": round(float(residuals.mean()), 4), "count": int(residuals.size)}]

    counts, edges = np.histogram(residuals, bins=bins)
    return [
        {
            "centre": round(float((edges[i] + edges[i + 1]) / 2), 4),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]


def error_by_actual_range(y_true: np.ndarray, y_pred: np.ndarray, buckets: int = 5) -> list[dict]:
    """Average error within each slice of the target's range. Models are often accurate in
    the middle and badly wrong at the extremes, which is exactly where predictions tend to
    matter most."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.size == 0:
        return []

    try:
        edges = np.unique(np.quantile(actual, np.linspace(0, 1, buckets + 1)))
    except (ValueError, IndexError):
        return []
    if len(edges) < 2:
        return []

    rows = []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        mask = (actual >= low) & (actual <= high if i == len(edges) - 2 else actual < high)
        if not mask.any():
            continue
        rows.append(
            {
                "range": f"{low:,.0f} to {high:,.0f}",
                "mean_absolute_error": round(float(np.abs(actual[mask] - predicted[mask]).mean()), 4),
                "mean_error": round(float((predicted[mask] - actual[mask]).mean()), 4),
                "count": int(mask.sum()),
            }
        )
    return rows


# --- data-level -----------------------------------------------------------------------------


def feature_correlations(df: pd.DataFrame, columns: list[str], limit: int = 12) -> dict:
    """Correlation heatmap over the numeric features.

    Standard EDA, and it earns its place here for a specific reason: two features correlated
    at 0.95 split the credit between them in any importance ranking, so both look unimportant
    and neither gets investigated.
    """
    usable = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])][:limit]
    if len(usable) < 2:
        return {"columns": [], "cells": [], "strong_pairs": []}

    matrix = df[usable].corr(numeric_only=True)
    cells = []
    strong_pairs = []
    for i, row in enumerate(usable):
        for j, column in enumerate(usable):
            value = matrix.iloc[i, j]
            value = 0.0 if pd.isna(value) else float(value)
            cells.append({"row": row, "column": column, "correlation": round(value, 4)})
            if i < j and abs(value) >= 0.9:
                strong_pairs.append(
                    {"a": row, "b": column, "correlation": round(value, 4)}
                )

    return {"columns": usable, "cells": cells, "strong_pairs": strong_pairs}


# --- entry points ---------------------------------------------------------------------------


def compute_diagnostics(pipeline, X_test, y_test, task: str, metrics: dict) -> dict:
    """Everything derivable from one pass of predictions over the test set. Cheap enough to
    run for every candidate model. Never raises -- a missing chart must not fail a training
    run that otherwise succeeded."""
    try:
        predictions = pipeline.predict(X_test)

        if task == "regression":
            actual = np.asarray(y_test, dtype=float)
            predicted = np.asarray(predictions, dtype=float)
            return {
                "predicted_vs_actual": predicted_vs_actual_points(actual, predicted),
                "residuals": residual_points(actual, predicted),
                "residual_histogram": residual_histogram(actual, predicted),
                "error_by_range": error_by_actual_range(actual, predicted),
            }

        labels = metrics.get("labels") or sorted(pd.Series(y_test).astype(str).unique())
        y_true = pd.Series(y_test).astype(str)
        y_pred = pd.Series(predictions).astype(str)

        result: dict = {
            "confusion_cells": confusion_cells(metrics.get("confusion_matrix", []), labels),
            "per_class": per_class_report(y_true, y_pred, labels),
            "class_distribution": [
                {"label": label, "count": int((y_true == label).sum())} for label in labels
            ],
        }

        positive_label = metrics.get("positive_label")
        if positive_label is None:
            return result  # ROC/PR/calibration are two-class notions

        scores = _positive_scores(pipeline, X_test, positive_label)
        if scores is None:
            return result

        y_binary = (y_true == str(positive_label)).to_numpy().astype(int)
        result.update(
            {
                "roc_curve": roc_curve_points(y_binary, scores),
                "precision_recall_curve": precision_recall_points(y_binary, scores),
                "calibration": calibration_points(y_binary, scores),
                "thresholds": threshold_table(y_binary, scores),
            }
        )
        return result
    except Exception:  # noqa: BLE001
        logger.warning("Diagnostics could not be computed", exc_info=True)
        return {}


def cross_validate_pipeline(
    build, X, y, task: str, folds: int = CV_FOLDS, groups=None, ordered: bool = False
) -> dict:
    """Scores across k folds instead of one split.

    A single held-out set gives one number with no sense of how much it would move on a
    different split. Spread across folds is the honest answer to "how much should I trust
    this?", and a wide spread on small data is itself the finding.

    `groups` makes the folds entity-aware. Without it, a run that carefully grouped its
    train/test split would still report a cross-validated score built by scattering each
    entity across folds -- an inflated number sitting directly beside the honest one, which is
    worse than not showing it, because the two disagree and nothing says why.
    """
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_score

    scoring = scoring_for(task, y)

    if ordered:
        # Date-based split: each fold validated on rows later than it trained on. X arrives
        # in time order (the training rows come out of `split_data` that way).
        from sklearn.model_selection import TimeSeriesSplit

        usable_folds = int(min(folds, len(y) - 1))
        if usable_folds < 2:
            return {}
        y_input = pd.Series(y).astype(str) if task == "classification" else y
        scores = cross_val_score(
            build(), X, y_input, cv=TimeSeriesSplit(n_splits=usable_folds), scoring=scoring, n_jobs=1
        )
        values = [round(float(s), 4) for s in scores]
        return {
            "metric": scoring,
            "folds": values,
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
            "spread": round(float(max(values) - min(values)), 4),
            "ordered": True,
        }

    if groups is not None:
        distinct = int(pd.Series(groups).nunique())
        usable_folds = int(min(folds, distinct))
        if usable_folds < 2:
            return {}
        # Not stratified: keeping an entity whole and balancing the classes are competing
        # demands, and splitting the entity is the one that inflates the score.
        splitter = GroupKFold(n_splits=usable_folds)
        y_input = pd.Series(y).astype(str) if task == "classification" else y
        scores = cross_val_score(
            build(), X, y_input, cv=splitter, scoring=scoring, n_jobs=1, groups=groups
        )
        values = [round(float(s), 4) for s in scores]
        return {
            "metric": scoring,
            "folds": values,
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
            "spread": round(float(max(values) - min(values)), 4),
            "grouped": True,
        }

    if task == "classification":
        counts = pd.Series(y).value_counts()
        usable_folds = int(min(folds, counts.min()))
        if usable_folds < 2:
            return {}
        splitter = StratifiedKFold(n_splits=usable_folds, shuffle=True, random_state=42)
        y_input = pd.Series(y).astype(str)
    else:
        usable_folds = folds
        splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
        y_input = y

    scores = cross_val_score(build(), X, y_input, cv=splitter, scoring=scoring, n_jobs=1)
    values = [round(float(s), 4) for s in scores]
    return {
        "metric": scoring,
        "folds": values,
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values)), 4),
        "spread": round(float(max(values) - min(values)), 4),
    }


def learning_curve_points(
    build, X, y, task: str, ordered: bool = False, groups=None
) -> list[dict]:
    """Score against training-set size, for both the training and validation folds.

    Reads the two failure modes directly off the shape: the curves converging at a poor score
    means the model is too simple for the problem; a wide persistent gap means it is
    memorizing, and more data would help more than more tuning.

    `groups` keeps an entity whole inside each fold, for the same reason `cross_validate_pipeline`
    takes it. The curve used to be the one fold-producing call that never received it, so a
    grouped run drew a curve from folds that scattered each customer across both sides.
    """
    from sklearn.model_selection import learning_curve

    # The metric the cross-validation spread beside it uses. This chart used to score with
    # accuracy while the spread used AUC, so on a lopsided target the curve could sit flat while
    # the AUC next to it improved, and the two read as disagreeing about the same model.
    scoring = scoring_for(task, y)
    y_input = pd.Series(y).astype(str) if task == "classification" else y

    if ordered:
        # Validated forwards in time, and not shuffled -- see cross_validate_pipeline.
        from sklearn.model_selection import TimeSeriesSplit

        cv, shuffle = TimeSeriesSplit(n_splits=3), False
    elif groups is not None:
        from sklearn.model_selection import GroupKFold

        usable_folds = int(min(3, pd.Series(groups).nunique()))
        if usable_folds < 2:
            return []
        cv, shuffle = GroupKFold(n_splits=usable_folds), True
    else:
        cv, shuffle = 3, True
    sizes, train_scores, validation_scores = learning_curve(
        build(),
        X,
        y_input,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        shuffle=shuffle,
        random_state=42 if shuffle else None,
        groups=groups,
    )
    return [
        {
            "training_rows": int(size),
            "training_score": round(float(train.mean()), 4),
            "validation_score": round(float(validation.mean()), 4),
            "metric": scoring,
        }
        for size, train, validation in zip(sizes, train_scores, validation_scores)
    ]


def permutation_importance_scores(
    pipeline, X_test, y_test, task: str, top_n: int = 15, scoring: str | None = None
) -> list[dict]:
    """Importance measured by breaking each column in turn and seeing what the score loses.

    Preferred over the model's built-in importances, which are biased toward high-cardinality
    numeric columns regardless of whether they help. This measures the thing people assume
    importance means: how much worse would predictions get without this column?
    """
    from sklearn.inspection import permutation_importance

    # The primary metric, like every other chart beside it. `scoring` lets the caller fix it
    # from the training rows, so a test set that happens to hold one class does not change which
    # metric the chart speaks.
    scoring = scoring or scoring_for(task, y_test)
    y_input = pd.Series(y_test).astype(str) if task == "classification" else y_test

    result = permutation_importance(
        pipeline,
        X_test,
        y_input,
        n_repeats=PERMUTATION_REPEATS,
        random_state=42,
        scoring=scoring,
        n_jobs=1,
    )
    rows = [
        {
            "column": str(column),
            "importance": round(float(mean), 4),
            "std": round(float(std), 4),
            "metric": scoring,
        }
        for column, mean, std in zip(X_test.columns, result.importances_mean, result.importances_std)
    ]
    rows.sort(key=lambda item: item["importance"], reverse=True)
    return rows[:top_n]


def compute_deep_diagnostics(
    build, pipeline, X_train, X_test, y_train, y_test, task: str, groups=None, ordered: bool = False
) -> dict:
    """The expensive tier: refits the model many times. Run for the chosen model only, and
    skipped on data large enough for the cost to be felt.

    Cross-validation and the learning curve use the **training rows only**. They used to be
    handed the training and test rows joined together, which put the test rows inside the
    folds of the number shown as the independent check on the test score. The test set's
    one job is to be the single evaluation nothing else touched; permutation importance still
    reads it, because that explains the model already evaluated and chooses nothing.

    `groups` is aligned with `X_train`, not with train plus test.

    Each part is guarded separately -- cross-validation failing is no reason to lose the
    learning curve.
    """
    rows = len(X_train) + len(X_test)
    if rows > MAX_ROWS_FOR_DEEP_DIAGNOSTICS:
        return {
            "skipped": (
                f"Cross-validation, the learning curve and permutation importance were "
                f"skipped: at {rows:,} rows they would each mean refitting the model several "
                f"times over. The single held-out test set above still applies."
            )
        }

    result: dict = {"skipped": None}
    for key, work in (
        (
            "cross_validation",
            lambda: cross_validate_pipeline(
                build, X_train, y_train, task, groups=groups, ordered=ordered
            ),
        ),
        (
            "learning_curve",
            lambda: learning_curve_points(
                build, X_train, y_train, task, ordered=ordered, groups=groups
            ),
        ),
        (
            "permutation_importance",
            lambda: permutation_importance_scores(
                pipeline, X_test, y_test, task, scoring=scoring_for(task, y_train)
            ),
        ),
    ):
        try:
            result[key] = work()
        except Exception:  # noqa: BLE001 -- one diagnostic failing must not lose the others
            logger.warning("Deep diagnostic %s failed", key, exc_info=True)
            result[key] = None

    return result
