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


# --- The two card-text defects found live (STATUS.md "What's next" §3) -------------------
#
# Both were the profiler's fault, not the model's: the LLM quoted the issue description it
# was handed, and the description was wrong.


def test_casing_issue_quotes_real_values_from_the_column():
    # Seen live: a card about a `plan` column read "'USA' and 'usa' are considered different
    # values" because that pair was hard-coded into every casing description.
    df = pd.DataFrame({"plan": ["basic", "Basic", "premium", "basic", "standard", "BASIC"]})
    issues = detect_issues(df)
    casing = [i for i in issues if i.issue_type == "inconsistent_format"]
    assert len(casing) == 1
    issue = casing[0]
    assert "USA" not in issue.description
    assert "'basic' vs 'Basic' vs 'BASIC'" in issue.description
    assert issue.stats["example_variants"] == ["basic", "Basic", "BASIC"]
    assert issue.stats["distinct_values"] == 5
    assert issue.stats["distinct_case_insensitive"] == 3


def test_casing_issue_example_is_the_most_common_collision():
    # Two colliding groups; the description should quote the one the user will recognise.
    df = pd.DataFrame({"city": ["paris"] * 5 + ["Paris"] * 4 + ["rome", "Rome"]})
    issue = [i for i in detect_issues(df) if i.issue_type == "inconsistent_format"][0]
    assert issue.stats["example_variants"] == ["paris", "Paris"]


def test_outlier_description_never_quotes_a_fence_below_the_data():
    # Seen live: "These extreme values are between -19.32 and 63.15" for monthly spend --
    # a negative lower bound on a strictly positive column.
    spend = [10.0, 12.0, 11.0, 13.0, 12.5, 11.5, 10.5, 12.0, 250.0]
    df = pd.DataFrame({"monthly_spend": spend})
    issue = [i for i in detect_issues(df) if i.issue_type == "outliers"][0]
    assert "-" not in issue.description.replace("Column 'monthly_spend'", "")
    assert issue.description == (
        "Column 'monthly_spend' has 1 unusually high value (the largest is 250.00), "
        "while the other 8 values sit between 10.00 and 13.00."
    )
    assert issue.stats["high_outlier_count"] == 1
    assert issue.stats["low_outlier_count"] == 0
    assert issue.stats["highest_outlier"] == 250.0
    assert issue.stats["lowest_outlier"] is None
    assert issue.stats["typical_min"] == 10.0
    assert issue.stats["typical_max"] == 13.0
    assert "lower_bound" not in issue.stats
    assert "upper_bound" not in issue.stats


def test_outlier_description_covers_both_tails():
    values = [48.0, 49.0, 50.0, 51.0, 52.0, 48.5, 49.5, 50.5, 51.5, 50.0, -5.0, 400.0, 500.0]
    df = pd.DataFrame({"x": values})
    issue = [i for i in detect_issues(df) if i.issue_type == "outliers"][0]
    assert issue.description == (
        "Column 'x' has 2 unusually high values (the largest is 500.00) and "
        "1 unusually low value (the smallest is -5.00), "
        "while the other 10 values sit between 48.00 and 52.00."
    )
    assert issue.stats["outlier_count"] == 3
    assert issue.stats["lowest_outlier"] == -5.0



# --- Stray whitespace (found on the churn sample) -----------------------------------------


def _whitespace_issues(df):
    return [i for i in detect_issues(df) if "spaces before or after" in i.description]


def test_padded_values_that_split_a_category_are_found():
    """`' basic '` beside `basic`: to the eye one word, to a model two unrelated values."""
    df = pd.DataFrame({"plan": [" basic ", "basic", "Basic", " pro", "pro", "pro"]})
    issues = _whitespace_issues(df)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.severity == "medium"
    assert issue.stats["padded_values"] == 2
    assert issue.stats["distinct_values"] == 5
    assert issue.stats["distinct_after_trim"] == 3
    assert issue.stats["distinct_after_trim_and_case"] == 2
    assert "merges 5 different spellings into 3" in issue.description
    assert "leaves 2" in issue.description


def test_padding_that_merges_nothing_is_low_priority():
    df = pd.DataFrame({"note": [" alpha", "beta ", "gamma"]})
    issue = _whitespace_issues(df)[0]
    assert issue.severity == "low"
    assert "merges" not in issue.description


def test_the_example_is_a_real_padded_value_from_the_column():
    df = pd.DataFrame({"plan": [" basic ", " basic ", "basic", "pro"]})
    assert _whitespace_issues(df)[0].stats["example"] == " basic "


def test_clean_text_and_numbers_are_left_alone():
    df = pd.DataFrame({"plan": ["basic", "pro"], "spend": [1.5, 2.5]})
    assert _whitespace_issues(df) == []
