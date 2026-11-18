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
  model is extrapolating there. **Far outside** (beyond the range by more than half its width) is reported apart,
  with the row's prediction called not reliable: usually a unit or decimal slip, and the predictions are still
  written, never refused or clipped, because the value may be real (#133).
- **a category apart from spaces or capitals** -- reported as matched, and scored as the training value (#134).
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
FAR_OUTSIDE = 0.5  # beyond the training range by more than this share of its width
BLOCKING = "blocking"
WARNING = "warning"


def known_spellings(column: dict) -> dict[str, str] | None:
    """For a categorical schema column: each training category's key (spaces trimmed, case ignored) mapped to the
    category, for the keys only one category has. None when the schema lists only some of the categories: a value
    missing from a partial list may be a real category, so nothing can be matched safely (#134)."""
    if column.get("kind") != "categorical" or column.get("categories_truncated"):
        return None
    by_key: dict[str, list[str]] = {}
    for category in column.get("categories") or {}:
        by_key.setdefault(str(category).strip().casefold(), []).append(str(category))
    return {key: found[0] for key, found in by_key.items() if len(found) == 1}


def respelled(values: pd.Series, known: set, spellings: dict[str, str]) -> pd.Series:
    """For each value that is not a training category but is one apart from spaces or capitals, that category; NaN
    elsewhere. `values` are the column's present values as text."""
    out = pd.Series(None, index=values.index, dtype=object)
    if not spellings:
        return out
    # Worked out once per distinct value, not per row: this runs on every categorical column of every file checked,
    # and a pass over every row in Python made the check of 100,000 rows twice as slow.
    lookup = {}
    for v in pd.unique(values):  # a blank is not a str and is skipped below
        if isinstance(v, str) and v not in known:
            match = spellings.get(v.strip().casefold())
            if match is not None:
                lookup[v] = match
    if lookup:
        hit = values.isin(list(lookup))
        out[hit] = values[hit].map(lookup)
    return out


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
                # Far outside: beyond the range by more than half its width (#133). A value there is usually a unit
                # or decimal-point slip, and a linear model extrapolates it into an absurd answer (abalone's RMSE
                # went from 2.3 to about 2,600). Measured on six OpenML tasks, this caught 87% of values multiplied
                # by 10-1000 and 2 of 47,466 real ones.
                margin = FAR_OUTSIDE * ((high - low) or abs(high) or 1.0)
                far = outside & ((numbers < low - margin) | (numbers > high + margin))
                add(name, "out_of_range", outside & ~far, f"outside the training range {low:g} to {high:g}",
                    series[outside & ~far])
                add(name, "far_out_of_range", far,
                    f"far outside the training range {low:g} to {high:g}; this row's prediction is not reliable",
                    series[far])
        elif column["kind"] == "date":
            parsed = pd.to_datetime(series.where(present), errors="coerce", format="mixed")
            not_date = present & parsed.isna()
            add(name, "type", not_date, "not a date (treated as blank)", series[not_date])
        elif column["kind"] == "categorical" and not column.get("categories_truncated"):
            known = set(column.get("categories") or {})
            text = series if is_text else series.astype(str)
            matched = present & respelled(text, known, known_spellings(column) or {}).notna()
            unseen = present & ~text.isin(known) & ~matched
            add(name, "respelled_category", matched,
                "a training value apart from spaces or capitals; scored as that value", series[matched])
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
