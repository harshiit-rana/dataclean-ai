"""Answers that arrive late: never present a partial set of answers as current performance.

Outcomes often arrive weeks after the prediction -- churn within 90 days, a default within a year. A file whose recent
rows have no answers yet, or answers only for the cases that resolved quickly, gives a performance figure about the
past and a biased one at that. When the file has a date column, `coverage` says which period the answered rows cover
against the unanswered ones, warns when the answered rows are a small share or bunched in the older part of the
file, and, given an outcome window, marks the rows too recent for their answer to have matured, which the
performance check then leaves out and counts.
"""

from __future__ import annotations

import pandas as pd

# Fewer answered rows than this share of the file is worth a warning on its own.
SMALL_SHARE = 0.5
# When the answered rows' median date is this far into the file's span before the unanswered rows', the answered
# rows are bunched in the older period.
OLDER_BY_SHARE_OF_SPAN = 0.25


def date_column(frame: pd.DataFrame, result: dict) -> str | None:
    """The column that dates the rows: the model's own time column when its split used one, else the first column
    that reads as dates."""
    from app.services.features import parse_dates

    preferred = ((result.get("split") or {}).get("time_column"))
    if preferred and preferred in frame.columns:
        return preferred
    target = result.get("target_column")
    for column in frame.columns:
        if column != target and parse_dates(frame[column]) is not None:
            return column
    return None


def _span(dates: pd.Series) -> dict | None:
    dates = dates.dropna()
    if dates.empty:
        return None
    return {"rows": int(len(dates)), "from": dates.min().date().isoformat(), "to": dates.max().date().isoformat()}


def coverage(frame: pd.DataFrame, target: str, column: str, window_days: int | None = None) -> tuple[dict, set]:
    """Which period the answers cover, the warnings that follow, and the rows an outcome window leaves out."""
    dates = pd.to_datetime(frame[column], errors="coerce", format="mixed")
    answered = frame[target].notna()
    excluded: set = set()
    if window_days:
        cutoff = dates.max() - pd.Timedelta(days=int(window_days))
        excluded = set(frame.index[(dates > cutoff).to_numpy()])
    report: dict = {
        "date_column": column,
        "answered": _span(dates[answered]),
        "unanswered": _span(dates[~answered]),
        "share_answered": round(float(answered.mean()), 4) if len(frame) else None,
        "window_days": window_days,
        "excluded_by_window": int(len(excluded & set(frame.index[answered.to_numpy()]))),
        "warnings": [],
    }
    if report["share_answered"] is not None and report["share_answered"] < SMALL_SHARE:
        report["warnings"].append(
            f"Only {report['share_answered']:.0%} of the rows have an answer, so the performance figure describes "
            f"a minority of the file."
        )
    if answered.any() and (~answered).any():
        span = (dates.max() - dates.min()) / pd.Timedelta(days=1) or 1
        lag = (dates[~answered].median() - dates[answered].median()) / pd.Timedelta(days=1)
        if pd.notna(lag) and lag > OLDER_BY_SHARE_OF_SPAN * span:
            report["warnings"].append(
                f"The rows with answers are mostly older ({report['answered']['from']} to {report['answered']['to']}) "
                f"than those without ({report['unanswered']['from']} to {report['unanswered']['to']}): the answers "
                f"may not have arrived yet for recent rows, and the ones that have may be the quick ones. Treat the "
                f"figure as being about the older period, or set an outcome window."
            )
    return report, excluded
