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

Free-form conversational instructions (PRD FR-37..40) were expected to be the case that
needed code generation and a sandbox. They didn't: routes_chat.py maps each instruction onto
this same registry (filter_rows, replace_values and friends cover what people actually ask
for) and declines what can't be expressed, so nothing in the app executes generated code.

Every operation is pure: it returns a new DataFrame and never mutates the input.
"""

from __future__ import annotations

import ast

import logging
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


def _fill_median(df: pd.DataFrame, params: dict, learned: dict | None = None) -> pd.DataFrame:
    column = _require_column(df, params)
    numeric = _require_numeric(df, column)
    value = learned["fill_value"] if learned else numeric.median()
    out = df.copy()
    out[column] = numeric.fillna(value)
    return out


def _fill_mean(df: pd.DataFrame, params: dict, learned: dict | None = None) -> pd.DataFrame:
    column = _require_column(df, params)
    numeric = _require_numeric(df, column)
    value = learned["fill_value"] if learned else numeric.mean()
    out = df.copy()
    out[column] = numeric.fillna(value)
    return out


def _mode_of(df: pd.DataFrame, column: str):
    modes = df[column].mode(dropna=True)
    if modes.empty:
        raise OperationFailedError(f"Column {column!r} has no values to derive a mode from.")
    return modes.iloc[0]


def _fill_mode(df: pd.DataFrame, params: dict, learned: dict | None = None) -> pd.DataFrame:
    column = _require_column(df, params)
    value = learned["fill_value"] if learned else _mode_of(df, column)
    out = df.copy()
    out[column] = out[column].fillna(value)
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


def _cap_outliers(df: pd.DataFrame, params: dict, learned: dict | None = None) -> pd.DataFrame:
    column = _require_column(df, params)
    numeric = _require_numeric(df, column)
    lower, upper = (learned["lower"], learned["upper"]) if learned else _iqr_bounds(numeric)
    out = df.copy()
    out[column] = numeric.clip(lower=lower, upper=upper)
    return out


def _drop_outlier_rows(df: pd.DataFrame, params: dict, learned: dict | None = None) -> pd.DataFrame:
    column = _require_column(df, params)
    numeric = _require_numeric(df, column)
    lower, upper = (learned["lower"], learned["upper"]) if learned else _iqr_bounds(numeric)
    keep = numeric.isna() | ((numeric >= lower) & (numeric <= upper))
    return df[keep].copy()


# --- structural ---------------------------------------------------------------------------


def _drop_column(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = _require_column(df, params)
    return df.drop(columns=[column]).copy()


def _rename_column(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = _require_column(df, params)
    new_name = str(params.get("new_name", "")).strip()
    if not new_name:
        raise OperationFailedError("rename_column requires a 'new_name' parameter.")
    if new_name in df.columns:
        raise OperationFailedError(f"A column named {new_name!r} already exists.")
    return df.rename(columns={column: new_name}).copy()


def _round_numeric(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = _require_column(df, params)
    numeric = _require_numeric(df, column)
    try:
        decimals = int(params.get("decimals", 0))
    except (TypeError, ValueError) as exc:
        raise OperationFailedError("'decimals' must be a whole number.") from exc
    out = df.copy()
    out[column] = numeric.round(decimals)
    return out


def _replace_values(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = _require_column(df, params)
    if "find" not in params:
        raise OperationFailedError("replace_values requires a 'find' parameter.")
    out = df.copy()
    out[column] = out[column].replace(params["find"], params.get("replace"))
    return out


# --- row filtering (the main thing free-form instructions ask for) -------------------------

_COMPARISONS = {
    "eq": lambda s, v: s == v,
    "ne": lambda s, v: s != v,
    "gt": lambda s, v: s > v,
    "gte": lambda s, v: s >= v,
    "lt": lambda s, v: s < v,
    "lte": lambda s, v: s <= v,
    "contains": lambda s, v: s.astype("string").str.contains(str(v), case=False, na=False),
    "not_contains": lambda s, v: ~s.astype("string").str.contains(str(v), case=False, na=False),
    "is_null": lambda s, _v: s.isna(),
    "not_null": lambda s, _v: s.notna(),
}


def _coerce_for_comparison(series: pd.Series, value):
    """Comparison values arrive as text. Compare numerically when the column is numeric so
    "age < 0" behaves like a number rather than a string comparison."""
    if pd.api.types.is_numeric_dtype(series):
        coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(coerced):
            raise OperationFailedError(
                f"{value!r} is not a number, but that column holds numbers."
            )
        return coerced
    return value


def _filter_rows(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Keeps or drops rows matching a comparison. `mode` is "keep" (default) or "drop"."""
    column = _require_column(df, params)
    comparison = str(params.get("comparison", "")).lower()
    if comparison not in _COMPARISONS:
        raise OperationFailedError(
            f"Unsupported comparison {comparison!r}; expected one of "
            f"{', '.join(sorted(_COMPARISONS))}."
        )

    series = df[column]
    value = params.get("value")
    if comparison not in {"is_null", "not_null"}:
        value = _coerce_for_comparison(series, value)

    mask = _COMPARISONS[comparison](series, value)
    mode = str(params.get("mode", "keep")).lower()
    if mode not in {"keep", "drop"}:
        raise OperationFailedError(f"Unsupported mode {mode!r}; expected 'keep' or 'drop'.")
    return df[mask if mode == "keep" else ~mask].copy()


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
        Operation("rename_column", "Rename this column", True, _rename_column),
        Operation("round_numeric", "Round numbers to a set number of decimals", True, _round_numeric),
        Operation("replace_values", "Replace one value with another", True, _replace_values),
        Operation("filter_rows", "Keep or drop rows matching a condition", True, _filter_rows),
    ]
}

# Operations that only make sense for free-form instructions -- the Explanation Agent picks
# fixes for *detected issues*, and none of these correspond to a detectable issue type.
# Kept out of that agent's catalog so it can't propose them for a profiling finding.
CONVERSATIONAL_ONLY = {"rename_column", "round_numeric", "replace_values", "filter_rows"}


def operation_catalog(include_conversational: bool = False) -> str:
    """Human-readable operation list, injected into a prompt so the model can only ever pick
    something that actually exists. The Explanation Agent gets the issue-fixing subset; the
    conversational path gets everything."""
    return "\n".join(
        f"- {op.name}: {op.description}"
        for op in OPERATIONS.values()
        if include_conversational or op.name not in CONVERSATIONAL_ONLY
    )


# Every parameter name an operation reads. Anything else in a proposed line is dropped (#114): a model's OPERATION line
# is written from uploaded data, and unknown keys used to ride along into the action log and the exported script.
KNOWN_PARAMETERS = frozenset(
    {"column", "columns", "value", "case", "format", "new_name", "decimals", "find", "replace", "comparison", "mode"}
)


def _literal(node: ast.expr):
    """A parameter's value: a literal, or a bare word taken as text (models sometimes leave strings unquoted).
    Anything else -- a call, an attribute, an expression -- is not a value and the line is refused."""
    if isinstance(node, ast.Name):
        return node.id
    value = ast.literal_eval(node)
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    # Text, as the parser has always returned it: the operations coerce what they need.
    return value if isinstance(value, str) else str(value)


def parse_operation(text: str) -> tuple[str, dict] | None:
    """Parses the Explanation Agent's OPERATION line, e.g.
    `fill_median(column="Salary")` -> ("fill_median", {"column": "Salary"}).

    Returns None if the line is absent, malformed, or names an operation not in the registry
    -- the caller treats that as "this card has no executable fix" rather than crashing.

    Parsed as a Python expression with `ast`, which reads it without running anything, rather than with a regular
    expression (#114): the regex cut a value at an escaped quote, so a column whose name holds a `"` got a truncated
    name, and a file can choose its column names. Only keyword arguments with literal values are accepted, and only
    parameter names an operation reads (`KNOWN_PARAMETERS`)."""
    if not text or not text.strip():
        return None
    line = text.strip().splitlines()[0].strip()
    try:
        node = ast.parse(line, mode="eval").body
    except SyntaxError:
        return None

    if isinstance(node, ast.Name):
        name, keywords = node.id, []
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.args:
        name, keywords = node.func.id, node.keywords
    else:
        return None
    if name not in OPERATIONS:
        logger.warning("LLM proposed unknown operation %r; ignoring", name)
        return None

    params: dict = {}
    for keyword in keywords:
        if keyword.arg is None:  # **something
            return None
        try:
            value = _literal(keyword.value)
        except (ValueError, TypeError, SyntaxError):
            return None
        if keyword.arg not in KNOWN_PARAMETERS:
            logger.warning("Dropped unknown parameter %r from a proposed %s", keyword.arg, name)
            continue
        params[keyword.arg] = value
    return name, params


# Operations whose effect depends on a statistic read from the data they are applied to.
LEARNED_OPERATIONS = frozenset(
    {"fill_median", "fill_mean", "fill_mode", "cap_outliers", "drop_outlier_rows"}
)


def _plain(value):
    """numpy scalars -> Python ones, so a learned value survives JSON and repr()."""
    return value.item() if hasattr(value, "item") else value


def learn(df: pd.DataFrame, name: str, params: dict) -> dict | None:
    """What a data-dependent operation reads from the data it is applied to, or None.

    `fill_median` on January's file fills with January's median. Replayed on February's file
    it must fill with January's median too -- the model learned from rows filled that way,
    and filling February's gaps with February's own median quietly cleans the new rows
    differently from the rows the model was trained on. Same for the mode and for the
    outlier limits. So replay reads these once, from the session's own data at the point the
    step ran, and hands them back through `apply_operation(..., learned=)`.

    A separate argument rather than a parameter on purpose: parameters arrive from the LLM
    and from chat instructions, and neither should be able to supply a "learned" value.
    """
    if name not in LEARNED_OPERATIONS:
        return None
    column = _require_column(df, params)
    if name in ("fill_median", "fill_mean"):
        numeric = _require_numeric(df, column)
        value = numeric.median() if name == "fill_median" else numeric.mean()
        return {"fill_value": _plain(value)}
    if name == "fill_mode":
        return {"fill_value": _plain(_mode_of(df, column))}
    lower, upper = _iqr_bounds(_require_numeric(df, column))
    return {"lower": _plain(lower), "upper": _plain(upper)}


def apply_operation(
    df: pd.DataFrame, name: str, params: dict, learned: dict | None = None
) -> tuple[pd.DataFrame, dict, dict]:
    """Runs one whitelisted operation. Returns (new_df, before_stats, after_stats).

    `learned` replaces the statistic a data-dependent operation would otherwise read from
    `df` -- see `learn`. Ignored by every other operation."""
    operation = OPERATIONS.get(name)
    if operation is None:
        raise UnknownOperationError(name)

    column = params.get("column")
    before = compute_stats(df, column)
    if learned and name in LEARNED_OPERATIONS:
        result = operation.apply(df, params, learned=learned)
    else:
        result = operation.apply(df, params)
    after = compute_stats(result, column)
    return result, before, after
