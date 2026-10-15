"""Forward-in-time folds that never split a timestamp and can leave a gap.

A date-based split trains on older rows and tests on newer ones, and every fold inside the training
rows follows the same rule: each chunk is validated only on a model trained on rows before it.
`TimeSeriesSplit` does that, but it cuts at row positions, so rows that share a timestamp can straddle
a fold boundary exactly as they did at the train/test cut (daily data, or records loaded in batches,
cut mid-day: the model trains on part of a day and is validated on the rest of it). It also cannot leave
a gap, which a target defined over a future window needs: "churned within 30 days" for a row just before
the cut is partly decided by what happens in the test period.

This splitter is given the timestamps explicitly, keyed by row label, and never infers order from row
position. It is self-contained (numpy and pandas only) because `model_export` copies its source into
`train_model.py`, so the script cuts the same folds the app did.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class ForwardInTimeFolds:
    """`n_splits` folds, each validated on a later chunk than the rows it trains on.

    The chunks are those `TimeSeriesSplit` would make -- the last `n_splits` of `n_splits + 1` equal
    parts -- but every boundary is moved to the nearest point where the timestamp changes, so no
    timestamp is on both sides of any fold. With `gap_days` the training rows of a fold are further
    cut back to those at least that many days before the first validation row.

    `timestamps` is a Series keyed by the label of each row of the `X` that `split` is given (so the
    same object serves the full training rows and any subset of them, as inside a cut-off's own
    folds). Unreadable dates sort last and count as one timestamp, as at the train/test cut.
    """

    def __init__(self, timestamps, n_splits: int = 5, gap_days: int = 0):
        self.timestamps = pd.Series(timestamps)
        self.n_splits = n_splits
        self.gap_days = gap_days

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        times = pd.to_datetime(self.timestamps.loc[X.index], errors="coerce", format="mixed")
        order = np.argsort(times.to_numpy(), kind="stable")  # row positions, earliest first
        ranked = times.iloc[order].reset_index(drop=True)
        rows = len(ranked)
        # Where the timestamp changes: the only places a boundary may fall.
        starts = [i for i, repeat in enumerate(ranked.duplicated().tolist()) if not repeat]
        chunk = rows // (self.n_splits + 1)
        if chunk < 1:
            raise ValueError(
                f"Cannot have {self.n_splits} forward-in-time folds from {rows} rows."
            )

        def snap(wanted: int) -> int:
            allowed = [s for s in starts if 0 < s < rows]
            return min(allowed, key=lambda s: (abs(s - wanted), -s)) if allowed else rows

        edges = []
        for k in range(self.n_splits):
            edge = snap(rows - (self.n_splits - k) * chunk)
            if not edges or edge > edges[-1]:
                edges.append(edge)
        edges.append(rows)

        gap = pd.Timedelta(days=self.gap_days) if self.gap_days else None
        for begin, end in zip(edges, edges[1:]):
            if end <= begin:
                continue
            train = order[:begin]
            if gap is not None and not pd.isna(ranked.iloc[begin]):
                train = train[(ranked.iloc[:begin] <= ranked.iloc[begin] - gap).to_numpy()]
            if len(train) == 0:
                continue
            yield train, order[begin:end]


@dataclass(frozen=True)
class TimeFoldRule:
    """What every fold producer needs to cut forward-in-time folds: the timestamps and the gap.

    Passed wherever the flag `ordered=True` used to go. It is truthy, so code that only asks
    "is this a date split?" is unchanged, and `tuning.cv_splitter` -- the one place that makes
    folds -- turns it into a `ForwardInTimeFolds` instead of a position-based `TimeSeriesSplit`.
    """

    timestamps: pd.Series
    gap_days: int = 0

    def splitter(self, n_splits: int) -> ForwardInTimeFolds:
        return ForwardInTimeFolds(self.timestamps, n_splits, self.gap_days)


def time_rule(df: pd.DataFrame, column: str, gap_days: int = 0) -> TimeFoldRule:
    """The rule for a date split on `column` of `df`, keyed by the frame's row labels."""
    return TimeFoldRule(pd.to_datetime(df[column], errors="coerce", format="mixed"), int(gap_days or 0))
