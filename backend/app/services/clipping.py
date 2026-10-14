"""Clipping extreme values inside the model pipeline, so the limits come from training rows.

Capping outliers while cleaning reads its limits from every row of the file, test rows
included, before any split -- see `ml_readiness.DATASET_STATISTIC_OPERATIONS`. This is the same
job done where it cannot leak: `fit` learns each column's limits from whatever it is handed,
which inside the pipeline is the training fold only, and `transform` applies them to
everything else.

The class is deliberately self-contained (numpy, pandas and scikit-learn, nothing from this
app) because `model_export` copies its source verbatim into `train_model.py`. The script has
to run with only those three installed, and a second hand-written copy would be one more thing
to drift from the first. A test runs the script and compares its numbers with the app's.
"""

from __future__ import annotations

import inspect

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# The same 1.5 x IQR rule the profiler uses to flag an outlier (`profiling._detect_outliers`),
# so "this column has outliers" and "these are the values that get clipped" mean one thing.
WHISKER = 1.5


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clips each column to [Q1 - 1.5 x IQR, Q3 + 1.5 x IQR], learned in `fit`.

    A value beyond a limit is pulled back to the limit rather than removed, so no row is lost
    and the model still sees that the value was large. Blanks stay blank, for the imputer after
    this step. A column with no spread (constant, or too few values) is passed through
    unchanged: its limits would be a single point, and clipping to it would erase the column.
    """

    def __init__(self, whisker=1.5):
        self.whisker = whisker

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        lower = np.full(values.shape[1], -np.inf)
        upper = np.full(values.shape[1], np.inf)
        for index in range(values.shape[1]):
            column = values[:, index]
            column = column[~np.isnan(column)]
            if column.size < 4:
                continue
            q1, q3 = np.percentile(column, [25, 75])
            spread = q3 - q1
            if spread > 0:
                lower[index] = q1 - self.whisker * spread
                upper[index] = q3 + self.whisker * spread
        self.lower_, self.upper_ = lower, upper
        self.n_features_in_ = values.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return np.clip(values, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return getattr(
            self,
            "feature_names_in_",
            np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object),
        )


def source() -> str:
    """The class as text, for `train_model.py`. Read from the class itself rather than kept
    as a second copy."""
    return inspect.getsource(IQRClipper)


# Far enough out that ordinary data almost never gets there. A bell-shaped column puts a few
# values past 1.5 x IQR by chance alone (about one in a hundred), so offering clipping whenever
# there is one would offer it on nearly every numeric column -- and a list padded with offers
# that catch nothing is how people learn to approve without reading. Past 3 x IQR is where a
# bell curve runs out, and where a skewed or contaminated column does not.
FAR_OUT_WHISKER = 3.0


def outliers_in(series, whisker: float = WHISKER) -> dict | None:
    """What clipping would touch in one numeric column, or None when it is not worth offering.

    Offered only when some value lies far out (past `FAR_OUT_WHISKER` x IQR); the counts are
    of everything beyond the clipping limits, since that is what clipping would pull back.
    Read off the column alone -- never the answers -- so offering it cannot leak. Counts are
    over every row because the offer is about the file; the limits the model uses are learned
    on the training rows when it is fitted.
    """
    values = series.dropna().astype(float)
    if len(values) < 4:
        return None
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    spread = q3 - q1
    if spread <= 0:
        return None
    lower, upper = q1 - whisker * spread, q3 + whisker * spread
    low, high = values[values < lower], values[values > upper]
    far = int(
        ((values < q1 - FAR_OUT_WHISKER * spread) | (values > q3 + FAR_OUT_WHISKER * spread)).sum()
    )
    if far == 0:
        return None
    inside = values[(values >= lower) & (values <= upper)]
    return {
        "far_out": far,
        "count": int(len(low) + len(high)),
        "share": round(float((len(low) + len(high)) / len(values)), 4),
        "low": int(len(low)),
        "high": int(len(high)),
        "lowest": float(low.min()) if len(low) else None,
        "highest": float(high.max()) if len(high) else None,
        "typical_min": float(inside.min()),
        "typical_max": float(inside.max()),
    }
