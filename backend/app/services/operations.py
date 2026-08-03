"""Whitelisted cleaning operations. PRD §12.3.

The PRD specifies the Cleaning Agent generates pandas code and runs it in a sandbox. This
module takes a deliberately different approach: a fixed registry of parameterized
operations that the LLM *selects* rather than writes.

Rationale:
- The profiler detects a closed set of issue types (see ColumnIssue.issue_type), so the set
  of possible fixes is bounded -- arbitrary code generation buys no coverage here.
- No arbitrary code execution means no sandbox to harden, no escape surface, and no
  isolated-compute cost in production. A restricted-globals exec() is NOT a real sandbox.
- Deterministic and unit-testable: the same card always produces the same transformation.
- The operation is chosen during the Explanation Agent's existing per-issue call, so
  executing approved fixes costs zero additional LLM tokens.

Free-form conversational instructions (PRD FR-37..40) are the case that genuinely needs
code generation; that path will need the sandbox, and is not implemented yet.

Every operation is pure: it returns a new DataFrame and never mutates the input.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

_OUTLIER_IQR_MULTIPLIER = 1.5  # matches profiling.py's detector


class UnknownOperationError(Exception):
    def __init__(self, name: str):
        super().__init__(f"Unknown cleaning operation: {name!r}")
        self.name = name


class OperationFailedError(Exception):
    """Raised when a whitelisted operation can't run against this data (e.g. median of a
    non-numeric column). Callers log it against the action and continue -- one bad fix must
    not abort the rest of the run."""


@dataclass(frozen=True)
class Operation:
    name: str
    description: str
    needs_column: bool
    apply: Callable[[pd.DataFrame, dict], pd.DataFrame]


# --- stats (PRD §12.3: null_count, unique_value_count, row_count, min/max/mean/std) -------


def compute_stats(df: pd.DataFrame, column: str | None) -> dict:
    stats: dict = {"row_count": int(len(df))}
    if not column or column not in df.columns:
        return stats

    series = df[column]
    stats["null_count"] = int(series.isna().sum())
    stats["unique_value_count"] = int(series.nunique(dropna=True))

    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 0:
            stats.update(
                {
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                    "mean": float(non_null.mean()),
                    "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                }
            )
    return stats


def _require_column(df: pd.DataFrame, params: dict) -> str:
    column = params.get("column")
    if not column:
        raise OperationFailedError("This operation requires a column, but none was given.")
    if column not in df.columns:
        raise OperationFailedError(f"Column {column!r} is not in the data.")
    return column


def _require_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    series = pd.to_numeric(df[column], errors="coerce")
    if series.notna().sum() == 0:
        raise OperationFailedError(f"Column {column!r} has no numeric values to work with.")
    return series


# --- missing values -----------------------------------------------------------------------


def _fill_median(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = _require_column(df, params)
    numeric = _require_numeric(df, column)
    out = df.copy()
    out[column] = numeric.fillna(numeric.median())
    return out


def _fill_mean(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = _require_column(df, params)
    numeric = _require_numeric(df, column)
    out = df.copy()
    out[column] = numeric.fillna(numeric.mean())
    return out


def _fill_mode(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = _require_column(df, params)
    modes = df[column].mode(dropna=True)
    if modes.empty:
        raise OperationFailedError(f"Column {column!r} has no values to derive a mode from.")
    out = df.copy()
    out[column] = out[column].fillna(modes.iloc[0])
    return out


def _fill_constant(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = _require_column(df, params)
    if "value" not in params:
        raise OperationFailedError("fill_constant requires a 'value' parameter.")
    out = df.copy()
    out[column] = out[column].fillna(params["value"])
    return out


def _drop_rows_with_nulls(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = _require_column(df, params)
    return df[df[column].notna()].copy()


# --- duplicates ---------------------------------------------------------------------------


def _drop_duplicate_rows(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    subset = params.get("columns") or None
    if subset:
        missing = [c for c in subset if c not in df.columns]
        if missing:
            raise OperationFailedError(f"Columns not in the data: {', '.join(missing)}")
    return df.drop_duplicates(subset=subset, keep="first").copy()


# --- type mismatches ----------------------------------------------------------------------


def _cast_to_numeric(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = _require_column(df, params)
    out = df.copy()
    out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _cast_to_datetime(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = _require_column(df, params)
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce", format="mixed")
    return out


# --- formatting ---------------------------------------------------------------------------


def _standardize_case(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = _require_column(df, params)
    case = params.get("case", "lower")
    if case not in {"lower", "upper", "title"}:
        raise OperationFailedError(f"Unsupported case {case!r}; expected lower, upper, or title.")
    out = df.copy()
    text = out[column].astype("string")
    out[column] = getattr(text.str, case)()
    return out


def _trim_whitespace(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = _require_column(df, params)
    out = df.copy()
    out[column] = out[column].astype("string").str.strip()
    return out


def _standardize_dates(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Normalizes mixed date formats to one consistent output format, keeping the column as
    text so the rendered values stay exactly as specified (unlike cast_to_datetime, which
    hands formatting off to pandas' dtype repr)."""
    column = _require_column(df, params)
    fmt = params.get("format", "%Y-%m-%d")
    parsed = pd.to_datetime(df[column], errors="coerce", format="mixed")
    if parsed.notna().sum() == 0:
        raise OperationFailedError(f"No values in {column!r} could be read as dates.")
    out = df.copy()
    out[column] = parsed.dt.strftime(fmt).where(parsed.notna(), None)
    return out


# --- outliers -----------------------------------------------------------------------------


def _iqr_bounds(series: pd.Series) -> tuple[float, float]:
    non_null = series.dropna()
    if len(non_null) < 4:
        raise OperationFailedError("Not enough values to compute an outlier range.")
    q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        raise OperationFailedError("All values are too similar to identify outliers.")
    return q1 - _OUTLIER_IQR_MULTIPLIER * iqr, q3 + _OUTLIER_IQR_MULTIPLIER * iqr


def _cap_outliers(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = _require_column(df, params)
    numeric = _require_numeric(df, column)
    lower, upper = _iqr_bounds(numeric)
    out = df.copy()
    out[column] = numeric.clip(lower=lower, upper=upper)
    return out


def _drop_outlier_rows(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = _require_column(df, params)
    numeric = _require_numeric(df, column)
    lower, upper = _iqr_bounds(numeric)
    keep = numeric.isna() | ((numeric >= lower) & (numeric <= upper))
    return df[keep].copy()


# --- structural ---------------------------------------------------------------------------


def _drop_column(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = _require_column(df, params)
    return df.drop(columns=[column]).copy()


OPERATIONS: dict[str, Operation] = {
    op.name: op
    for op in [
        Operation("fill_median", "Fill missing values with the column median", True, _fill_median),
        Operation("fill_mean", "Fill missing values with the column mean", True, _fill_mean),
        Operation("fill_mode", "Fill missing values with the most common value", True, _fill_mode),
        Operation("fill_constant", "Fill missing values with a fixed value", True, _fill_constant),
        Operation("drop_rows_with_nulls", "Drop rows where this column is empty", True, _drop_rows_with_nulls),
        Operation("drop_duplicate_rows", "Remove duplicate rows, keeping the first", False, _drop_duplicate_rows),
        Operation("cast_to_numeric", "Convert this column to numbers", True, _cast_to_numeric),
        Operation("cast_to_datetime", "Convert this column to real dates", True, _cast_to_datetime),
        Operation("standardize_case", "Make capitalization consistent", True, _standardize_case),
        Operation("trim_whitespace", "Remove leading/trailing spaces", True, _trim_whitespace),
        Operation("standardize_dates", "Rewrite all dates in one consistent format", True, _standardize_dates),
        Operation("cap_outliers", "Clamp extreme values to the normal range", True, _cap_outliers),
        Operation("drop_outlier_rows", "Drop rows with extreme values", True, _drop_outlier_rows),
        Operation("drop_column", "Remove this column entirely", True, _drop_column),
    ]
}


def operation_catalog() -> str:
    """Human-readable operation list, injected into the Explanation Agent's prompt so the
    model can only ever pick something that actually exists."""
    return "\n".join(f"- {op.name}: {op.description}" for op in OPERATIONS.values())


_CALL_PATTERN = re.compile(r"^\s*([a-z_]+)\s*(?:\((.*)\))?\s*$", re.DOTALL)


def parse_operation(text: str) -> tuple[str, dict] | None:
    """Parses the Explanation Agent's OPERATION line, e.g.
    `fill_median(column="Salary")` -> ("fill_median", {"column": "Salary"}).

    Returns None if the line is absent, malformed, or names an operation not in the registry
    -- the caller treats that as "this card has no executable fix" rather than crashing."""
    if not text:
        return None
    match = _CALL_PATTERN.match(text.strip())
    if not match:
        return None

    name, raw_args = match.group(1), match.group(2) or ""
    if name not in OPERATIONS:
        logger.warning("LLM proposed unknown operation %r; ignoring", name)
        return None

    params: dict = {}
    for part in re.findall(r'(\w+)\s*=\s*("[^"]*"|\'[^\']*\'|[^,]+)', raw_args):
        key, value = part[0], part[1].strip().strip("\"'")
        params[key] = value
    return name, params


def apply_operation(df: pd.DataFrame, name: str, params: dict) -> tuple[pd.DataFrame, dict, dict]:
    """Runs one whitelisted operation. Returns (new_df, before_stats, after_stats)."""
    operation = OPERATIONS.get(name)
    if operation is None:
        raise UnknownOperationError(name)

    column = params.get("column")
    before = compute_stats(df, column)
    result = operation.apply(df, params)
    after = compute_stats(result, column)
    return result, before, after
