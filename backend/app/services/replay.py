"""Cleaning a new file the way this session's file was cleaned.

A session used to be single-use: next month's export meant uploading it and approving every
fix again, and a model could only be used by writing Python against the downloaded joblib.
But the action log is already a complete description of the cleaning -- undo depends on that,
re-deriving the data from `original.csv` through every applied step -- so the same log can be
run against a different file.

Two things make that more than a loop over the log:

**Learned values are frozen.** `fill_median` on the new file must fill with the *session's*
median, not the new file's, or the new rows are cleaned differently from the rows a model was
trained on. `freeze` replays the log over the original once, reading each data-dependent
step's statistic off the data as it stood at that point (see `operations.learn`), and the new
file is cleaned with those.

**Nothing disappears silently.** Several steps drop rows. When the point is to get a
prediction for every row in a file, a row that quietly vanishes is worse than one that fails
loudly -- so the input's row numbers are carried through every step and each dropped row is
reported with the step that dropped it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from app.services import operations

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Step:
    """One applied fix, ready to run on another file."""

    action_id: str
    operation: str
    params: dict
    column: str | None
    learned: dict | None = None
    source: str = "suggestion_card"


@dataclass
class StepReport:
    action_id: str
    operation: str
    column: str | None
    status: str  # "applied" | "skipped" | "failed"
    note: str | None = None
    rows_before: int = 0
    rows_after: int = 0
    dropped_rows: list[int] = field(default_factory=list)
    learned: dict | None = None

    def as_dict(self) -> dict:
        return {
            "action_id": self.action_id,
            "operation": self.operation,
            "column": self.column,
            "status": self.status,
            "note": self.note,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "dropped_count": len(self.dropped_rows),
            "learned": self.learned,
        }


def _params_for(entry: dict) -> dict:
    params = dict(entry.get("operation_params") or {})
    operation = operations.OPERATIONS.get(entry.get("operation", ""))
    if operation is not None and operation.needs_column:
        params.setdefault("column", entry.get("column_name"))
    return params


def freeze(original: pd.DataFrame, action_log: list[dict]) -> tuple[list[Step], pd.DataFrame]:
    """The applied steps in order, with every learned value read off the session's own data.

    Returns the steps and the frame the replay produced -- which is the session's cleaned
    data, the same invariant undo relies on. A step that fails here was recorded as applied
    but cannot be reproduced; it is left out rather than guessed at, and logged.
    """
    df = original
    steps: list[Step] = []
    for entry in action_log:
        if entry.get("status") != "applied":
            continue
        name = entry.get("operation", "")
        params = _params_for(entry)
        try:
            learned = operations.learn(df, name, params)
            df, _before, _after = operations.apply_operation(df, name, params, learned=learned)
        except (operations.OperationFailedError, operations.UnknownOperationError) as exc:
            logger.warning("Step %s (%s) could not be reproduced: %s", entry.get("action_id"), name, exc)
            continue
        steps.append(
            Step(
                action_id=str(entry.get("action_id")),
                operation=name,
                params=params,
                column=params.get("column"),
                learned=learned,
                source=entry.get("source", "suggestion_card"),
            )
        )
    return steps, df


# Row-dropping steps whose reason is worth stating in words, so a user reading the output file
# knows why a row came back without a prediction.
_WHY_DROPPED = {
    "drop_rows_with_nulls": "{column} was empty",
    "drop_duplicate_rows": "it duplicated an earlier row",
    "drop_outlier_rows": "{column} was outside the range the original data allowed",
    "filter_rows": "it matched a filter on {column}",
}


def reason_for(step: Step) -> str:
    template = _WHY_DROPPED.get(step.operation, "the step '{operation}' removed it")
    return template.format(column=step.column or "a column", operation=step.operation)


@dataclass
class Applied:
    frame: pd.DataFrame  # indexed by input row number, 0-based
    steps: list[StepReport]
    # input row number -> why it was dropped, for every row not in `frame`
    dropped: dict[int, str]


def apply(new: pd.DataFrame, steps: list[Step], *, target_column: str | None = None) -> Applied:
    """Runs the frozen steps against a new file.

    A step whose column is missing is skipped with the reason, never failed silently and never
    allowed to stop the rest: the target column being absent is *expected* when the point is
    to predict it, and a column the original had but this file lacks is a finding the user
    needs to see, alongside everything that did apply.
    """
    df = new.reset_index(drop=True)
    reports: list[StepReport] = []
    dropped: dict[int, str] = {}

    for step in steps:
        rows_before = len(df)
        report = StepReport(
            action_id=step.action_id,
            operation=step.operation,
            column=step.column,
            status="applied",
            rows_before=rows_before,
            learned=step.learned,
        )
        needs_column = operations.OPERATIONS[step.operation].needs_column
        if needs_column and step.column and step.column not in df.columns:
            report.status = "skipped"
            report.rows_after = rows_before
            if step.column == target_column:
                report.note = (
                    f"This file has no '{step.column}' column -- expected when these are the "
                    f"rows you want predictions for."
                )
            elif step.operation == "drop_column":
                report.note = f"'{step.column}' is already absent from this file."
            else:
                report.note = (
                    f"This file has no '{step.column}' column, which the original did. The fix "
                    f"was skipped; if a model uses this column it will say so below."
                )
            reports.append(report)
            continue

        try:
            result, _before, _after = operations.apply_operation(
                df, step.operation, step.params, learned=step.learned
            )
        except operations.OperationFailedError as exc:
            report.status = "failed"
            report.note = str(exc)
            report.rows_after = rows_before
            reports.append(report)
            continue

        lost = sorted(set(df.index) - set(result.index))
        if lost:
            why = reason_for(step)
            for row in lost:
                dropped[int(row)] = why
        report.dropped_rows = [int(r) for r in lost]
        report.rows_after = len(result)
        df = result
        reports.append(report)

    return Applied(frame=df, steps=reports, dropped=dropped)
