"""A new file checked against the schema its model was trained to (input_schema.py), column by column and row by row.

Applying a model used to refuse a missing column by name and coerce everything else quietly, so a user whose export
had a renamed category or a column of text where numbers belong learned nothing about which rows were affected.
`validate` reports, for the cleaned new rows:

- **missing columns** -- a column the model needs is not in the file. *Blocking*: nothing can be scored without it.
- **unexpected columns** -- in the file, not in the schema. Ignored, and listed.
- **type problems** -- a numeric column with values that are not numbers, or a date column with values that are not
  dates. *Warning*: those cells are treated as blank, which the model fills the way it was trained to.
- **unseen categories** -- values a categorical column never had in training. *Warning*: the model treats an unseen
  value as none of the ones it knows.
- **outside the training range** -- numbers below the smallest or above the largest training value. *Warning*: the
  model is extrapolating there.
- **blank where training never was** -- a column with no blanks in training has some now. *Warning*.

Each finding has a count, up to `EXAMPLE_ROWS` row numbers (1-based, header not counted) and example values, and each
row gets a one-line summary of its findings for the predictions file. Vectorised per column: a hundred thousand rows
of twenty columns check in well under a second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXAMPLE_ROWS = 5
EXAMPLE_VALUES = 5
BLOCKING = "blocking"
WARNING = "warning"


def _examples(mask: pd.Series, frame_index: pd.Index) -> list[int]:
    return [int(i) + 1 for i in frame_index[mask.to_numpy()][:EXAMPLE_ROWS]]


def validate(schema: dict, frame: pd.DataFrame, target_column: str | None = None) -> tuple[dict, pd.Series]:
    """The findings for `frame` against `schema`, and a per-row summary (empty string for a clean row)."""
    columns = schema.get("columns") or []
    expected = [c["name"] for c in columns]
    target = target_column or (schema.get("target") or {}).get("name")
    missing = [c for c in expected if c not in frame.columns]
    unexpected = [c for c in frame.columns if c not in expected and c != target]
    notes = np.full(len(frame), "", dtype=object)
    findings: list[dict] = []

    def add(column: str, kind: str, mask: pd.Series, message: str, values=None) -> None:
        flagged = mask.to_numpy(dtype=bool)
        count = int(flagged.sum())
        if not count:
            return
        entry = {"column": column, "finding": kind, "severity": WARNING, "count": count,
                 "rows": _examples(mask, frame.index), "message": message}
        if values is not None:
            entry["values"] = [str(v) for v in pd.unique(values)[:EXAMPLE_VALUES]]
        findings.append(entry)
        # Only the flagged rows are touched: most rows of most columns are fine.
        text = f"{column}: {message}"
        current = notes[flagged]
        notes[flagged] = np.where(current == "", text, current + ("; " + text))

    for column in columns:
        name = column["name"]
        if name not in frame.columns:
            continue
        series = frame[name]
        is_text = series.dtype == object or pd.api.types.is_string_dtype(series)
        blank = series.isna() | (series == "") if is_text else series.isna()
        present = ~blank
        if column.get("nullable") is False:
            add(name, "blank_new", blank, "blank, which it never was in training")
        if column["kind"] == "numeric":
            numbers = series if pd.api.types.is_numeric_dtype(series) else pd.to_numeric(series, errors="coerce")
            not_number = present & numbers.isna()
            add(name, "type", not_number, "not a number (treated as blank)", series[not_number])
            span = column.get("range") or {}
            low, high = span.get("min"), span.get("max")
            if low is not None and high is not None:
                outside = numbers.notna() & ((numbers < low) | (numbers > high))
                add(name, "out_of_range", outside, f"outside the training range {low:g} to {high:g}", series[outside])
        elif column["kind"] == "date":
            parsed = pd.to_datetime(series.where(present), errors="coerce", format="mixed")
            not_date = present & parsed.isna()
            add(name, "type", not_date, "not a date (treated as blank)", series[not_date])
        elif column["kind"] == "categorical" and not column.get("categories_truncated"):
            known = set(column.get("categories") or {})
            text = series if is_text else series.astype(str)
            unseen = present & ~text.isin(known)
            add(name, "unseen_category", unseen, "a value not seen in training", series[unseen])

    report = {
        "schema_version": schema.get("schema_version"),
        "blocking": bool(missing),
        "missing_columns": missing,
        "unexpected_columns": unexpected,
        "findings": findings,
        "rows_with_findings": int((notes != "").sum()),
        "rows_checked": int(len(frame)),
    }
    return report, pd.Series(notes, index=frame.index, dtype=object)


def describe_blocking(report: dict) -> str:
    """The refusal, in words, when a file cannot be scored at all."""
    return (
        "This file is missing columns the model needs, so nothing could be scored: "
        + ", ".join(report["missing_columns"])
        + "."
    )
