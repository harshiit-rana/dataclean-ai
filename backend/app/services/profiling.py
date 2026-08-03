"""Statistical profiling: ydata-profiling wrapper + custom issue detectors -> list[ColumnIssue].

PRD FR-11..19. Per §9.3, ColumnIssue.description is "Plain English description of what
was found. No LLM involved." -- everything in this module is deterministic pandas/numpy;
semantic interpretation and goal-aware priority adjustment happens later, in the LLM-backed
Profiling Agent node (app/graph/nodes/profiling_agent.py, PRD §12.1).

ydata-profiling itself is wrapped defensively and kept independent of issue detection: it is
a heavyweight, version-sensitive dependency (PRD §17.1 flags it as a medium schedule risk),
and a profiling failure must not crash the session. Its JSON output is stored verbatim as
GraphState.raw_profile ("source of truth for statistics" per §9.1) but the ColumnIssue list
below is derived independently so the core detection logic never depends on ydata-profiling
succeeding.
"""

from __future__ import annotations

import json
import logging
import re
import uuid

import pandas as pd

from app.models.graph_state import ColumnIssue

logger = logging.getLogger(__name__)

PROFILING_TIMEOUT_SECONDS = 60  # PRD NFR-02

_NUMERIC_MISMATCH_THRESHOLD = 0.9  # share of non-null values that must parse as numeric
_DATE_MISMATCH_THRESHOLD = 0.9  # share of non-null values that must parse as a date
_OUTLIER_IQR_MULTIPLIER = 1.5


def generate_issue_id() -> str:
    return f"issue_{uuid.uuid4().hex[:8]}"


def run_ydata_profile(df: pd.DataFrame) -> dict:
    """Runs ydata-profiling in minimal mode (PRD §17.1 mitigation) and returns the raw report
    as a dict. Never raises -- returns {} on any failure so the pipeline can continue."""
    try:
        from ydata_profiling import ProfileReport

        report = ProfileReport(df, minimal=True, progress_bar=False, explorative=False)
        return json.loads(report.to_json())
    except Exception:
        logger.exception("ydata-profiling failed; continuing without raw_profile")
        return {}


def _severity_and_score(rate: float, high_cut: float, medium_cut: float) -> tuple[str, float]:
    """Baseline severity/priority before goal-context adjustment (FR-19 happens in the
    Profiling Agent node, not here)."""
    if rate >= high_cut:
        return "high", min(1.0, 0.75 + rate * 0.2)
    if rate >= medium_cut:
        return "medium", min(1.0, 0.45 + rate * 0.2)
    return "low", min(1.0, 0.15 + rate * 0.2)


def _detect_missing_values(df: pd.DataFrame) -> list[ColumnIssue]:
    issues = []
    n = len(df)
    if n == 0:
        return issues
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count == 0:
            continue
        rate = null_count / n
        severity, score = _severity_and_score(rate, high_cut=0.3, medium_cut=0.05)
        issues.append(
            ColumnIssue(
                issue_id=generate_issue_id(),
                column_name=col,
                issue_type="missing_values",
                severity=severity,
                description=f"Column '{col}' has {null_count} missing values ({rate:.1%} of rows).",
                stats={"null_count": null_count, "null_rate": round(rate, 4), "row_count": n},
                priority_score=score,
            )
        )
    return issues


def _detect_duplicates(df: pd.DataFrame) -> list[ColumnIssue]:
    n = len(df)
    if n == 0:
        return []
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        return []
    rate = dup_count / n
    severity, score = _severity_and_score(rate, high_cut=0.1, medium_cut=0.02)
    return [
        ColumnIssue(
            issue_id=generate_issue_id(),
            column_name="DATASET",
            issue_type="duplicates",
            severity=severity,
            description=f"Found {dup_count} duplicate rows ({rate:.1%} of the dataset).",
            stats={"duplicate_count": dup_count, "duplicate_rate": round(rate, 4), "row_count": n},
            priority_score=score,
        )
    ]


def _detect_type_mismatches(df: pd.DataFrame) -> list[ColumnIssue]:
    """FR-13 (numeric-as-string) and FR-14 (date-as-string)."""
    issues = []
    for col in df.columns:
        series = df[col]
        if series.dtype != object:
            continue
        non_null = series.dropna()
        if len(non_null) == 0:
            continue

        numeric = pd.to_numeric(non_null, errors="coerce")
        numeric_rate = numeric.notna().sum() / len(non_null)
        if numeric_rate >= _NUMERIC_MISMATCH_THRESHOLD:
            issues.append(
                ColumnIssue(
                    issue_id=generate_issue_id(),
                    column_name=col,
                    issue_type="type_mismatch",
                    severity="medium",
                    description=f"Column '{col}' looks numeric but is stored as text ({numeric_rate:.0%} of values parse as numbers).",
                    stats={"parseable_rate": round(float(numeric_rate), 4), "detected_as": "numeric"},
                    priority_score=0.55,
                )
            )
            continue  # a column is either numeric-as-string or date-as-string, not both

        dates = pd.to_datetime(non_null, errors="coerce", format="mixed")
        date_rate = dates.notna().sum() / len(non_null)
        if date_rate >= _DATE_MISMATCH_THRESHOLD:
            issues.append(
                ColumnIssue(
                    issue_id=generate_issue_id(),
                    column_name=col,
                    issue_type="type_mismatch",
                    severity="medium",
                    description=f"Column '{col}' looks like dates but is stored as text ({date_rate:.0%} of values parse as dates).",
                    stats={"parseable_rate": round(float(date_rate), 4), "detected_as": "date"},
                    priority_score=0.55,
                )
            )
    return issues


def _date_shape(value: str) -> str:
    """Normalizes a date-like string to a shape signature, e.g. '2024-01-15' -> 'DDDD-DD-DD',
    '01/15/2024' -> 'DD/DD/DDDD'. Two different shapes in the same column indicate
    inconsistent date formatting even when every value individually parses as a valid date."""
    return re.sub(r"\d", "D", str(value).strip())


def _detect_inconsistent_date_formats(df: pd.DataFrame) -> list[ColumnIssue]:
    """FR-16-adjacent: bad_dates.csv fixture requires an inconsistent_format issue on a date
    column even when every individual value is a parseable date (§15.1)."""
    issues = []
    for col in df.columns:
        series = df[col]
        if series.dtype != object:
            continue
        non_null = series.dropna().astype(str)
        if len(non_null) < 2:
            continue
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
        date_rate = parsed.notna().sum() / len(non_null)
        if date_rate < _DATE_MISMATCH_THRESHOLD:
            continue  # not a date column
        shapes = non_null[parsed.notna()].map(_date_shape)
        distinct_shapes = shapes.value_counts()
        if len(distinct_shapes) <= 1:
            continue
        issues.append(
            ColumnIssue(
                issue_id=generate_issue_id(),
                column_name=col,
                issue_type="inconsistent_format",
                severity="medium",
                description=f"Column '{col}' contains dates in {len(distinct_shapes)} different formats.",
                stats={"formats_found": int(len(distinct_shapes)), "format_counts": distinct_shapes.to_dict()},
                priority_score=0.5,
            )
        )
    return issues


def _detect_inconsistent_casing(df: pd.DataFrame) -> list[ColumnIssue]:
    """FR-16: inconsistent casing in categorical columns."""
    issues = []
    for col in df.columns:
        series = df[col]
        if series.dtype != object:
            continue
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            continue
        distinct_exact = non_null.nunique()
        distinct_lower = non_null.str.lower().nunique()
        if distinct_lower < distinct_exact:
            collapsed = distinct_exact - distinct_lower
            issues.append(
                ColumnIssue(
                    issue_id=generate_issue_id(),
                    column_name=col,
                    issue_type="inconsistent_format",
                    severity="low",
                    description=f"Column '{col}' has {collapsed} value(s) that only differ by letter case (e.g. 'USA' vs 'usa').",
                    stats={"distinct_values": int(distinct_exact), "distinct_case_insensitive": int(distinct_lower)},
                    priority_score=0.3,
                )
            )
    return issues


def _detect_outliers(df: pd.DataFrame) -> list[ColumnIssue]:
    """FR-15: IQR method."""
    issues = []
    numeric_df = df.select_dtypes(include="number")
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - _OUTLIER_IQR_MULTIPLIER * iqr
        upper = q3 + _OUTLIER_IQR_MULTIPLIER * iqr
        outliers = series[(series < lower) | (series > upper)]
        if len(outliers) == 0:
            continue
        rate = len(outliers) / len(series)
        severity, score = _severity_and_score(rate, high_cut=0.1, medium_cut=0.02)
        issues.append(
            ColumnIssue(
                issue_id=generate_issue_id(),
                column_name=col,
                issue_type="outliers",
                severity=severity,
                description=f"Column '{col}' has {len(outliers)} outlier values outside the expected range ({lower:.2f} to {upper:.2f}).",
                stats={
                    "outlier_count": int(len(outliers)),
                    "outlier_rate": round(float(rate), 4),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "q1": float(q1),
                    "q3": float(q3),
                },
                priority_score=score,
            )
        )
    return issues


def _detect_high_cardinality(df: pd.DataFrame) -> list[ColumnIssue]:
    """Not required by an FR, but a valid issue_type per §9.3 -- included so the category
    isn't left entirely unimplemented in the scaffold."""
    issues = []
    n = len(df)
    if n < 20:
        return issues
    for col in df.select_dtypes(include="object").columns:
        unique_ratio = df[col].nunique(dropna=True) / n
        if unique_ratio > 0.95:
            issues.append(
                ColumnIssue(
                    issue_id=generate_issue_id(),
                    column_name=col,
                    issue_type="high_cardinality",
                    severity="low",
                    description=f"Column '{col}' has {df[col].nunique()} distinct values across {n} rows -- likely an identifier, not a category.",
                    stats={"unique_count": int(df[col].nunique()), "unique_ratio": round(float(unique_ratio), 4)},
                    priority_score=0.2,
                )
            )
    return issues


def detect_issues(df: pd.DataFrame) -> list[ColumnIssue]:
    """Runs all detectors (FR-11..16) and returns issues sorted by priority_score descending
    (FR-18). This baseline ordering is later overwritten by the goal-aware Profiling Agent."""
    issues: list[ColumnIssue] = []
    issues += _detect_missing_values(df)
    issues += _detect_duplicates(df)
    issues += _detect_type_mismatches(df)
    issues += _detect_inconsistent_date_formats(df)
    issues += _detect_inconsistent_casing(df)
    issues += _detect_outliers(df)
    issues += _detect_high_cardinality(df)
    issues.sort(key=lambda i: i.priority_score, reverse=True)
    return issues
