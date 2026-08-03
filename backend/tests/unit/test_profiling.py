"""Profiler unit tests against the fixtures in tests/fixtures/. PRD §15.1."""

from __future__ import annotations

import pandas as pd

from app.services.profiling import detect_issues


def _load(fixtures_dir, name: str) -> pd.DataFrame:
    return pd.read_csv(fixtures_dir / name)


def test_missing_values_fixture(fixtures_dir):
    df = _load(fixtures_dir, "missing_values.csv")
    issues = detect_issues(df)
    missing = [i for i in issues if i.issue_type == "missing_values"]
    assert len(missing) >= 3
    assert any(i.severity == "high" for i in missing)


def test_duplicates_fixture(fixtures_dir):
    df = _load(fixtures_dir, "duplicates.csv")
    issues = detect_issues(df)
    dup = [i for i in issues if i.issue_type == "duplicates"]
    assert len(dup) >= 1
    assert dup[0].stats["duplicate_count"] == 2


def test_mixed_types_fixture(fixtures_dir):
    df = _load(fixtures_dir, "mixed_types.csv")
    issues = detect_issues(df)
    mismatches = [i for i in issues if i.issue_type == "type_mismatch"]
    assert len(mismatches) >= 2


def test_bad_dates_fixture(fixtures_dir):
    df = _load(fixtures_dir, "bad_dates.csv")
    issues = detect_issues(df)
    inconsistent = [
        i
        for i in issues
        if i.issue_type == "inconsistent_format" and i.column_name == "order_date"
    ]
    assert len(inconsistent) >= 1


def test_outliers_fixture(fixtures_dir):
    df = _load(fixtures_dir, "outliers.csv")
    issues = detect_issues(df)
    outliers = [i for i in issues if i.issue_type == "outliers"]
    assert len(outliers) >= 3
    columns_with_outliers = {i.column_name for i in outliers}
    assert {"salary", "performance_score", "hours_worked"} <= columns_with_outliers


def test_issues_sorted_by_priority_descending(fixtures_dir):
    df = _load(fixtures_dir, "missing_values.csv")
    issues = detect_issues(df)
    scores = [i.priority_score for i in issues]
    assert scores == sorted(scores, reverse=True)
