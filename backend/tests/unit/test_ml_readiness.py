"""Target-aware detection and the guards around the prediction target.

The datasets here are built so the *reason* a check fires is unambiguous -- the leakage
fixtures contain a column that is a deterministic function of the target and nothing else
that comes close, so a passing test means the detector found the planted leak rather than
tripping over noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.models.graph_state import GoalContext
from app.services import ml_readiness


def goal(**overrides) -> GoalContext:
    base = {
        "use_case": "ml_training",
        "expected_output": "one row per employee",
        "target_column": "attrition",
    }
    return GoalContext(**{**base, **overrides})


@pytest.fixture
def employees() -> pd.DataFrame:
    """120 rows. `termination_date` is filled in for exactly the leavers and empty for
    everyone else -- the textbook leak, and one that carries no signal in its *values*."""
    rng = np.random.default_rng(7)
    n = 120
    attrition = np.where(rng.random(n) < 0.3, "yes", "no")
    return pd.DataFrame(
        {
            "employee_id": [f"E{i:04d}" for i in range(n)],
            "salary": rng.normal(60000, 12000, n).round(2),
            "department": rng.choice(["sales", "eng", "support"], n),
            "termination_date": [
                f"2024-0{rng.integers(1, 9)}-15" if a == "yes" else None for a in attrition
            ],
            "attrition": attrition,
        }
    )


class TestTaskInference:
    def test_text_column_is_classification(self):
        assert ml_readiness.infer_task_type(pd.Series(["yes", "no", "yes"])) == "classification"

    def test_continuous_numeric_column_is_regression(self):
        assert ml_readiness.infer_task_type(pd.Series(range(500))) == "regression"

    def test_few_valued_numeric_column_is_classification(self):
        assert ml_readiness.infer_task_type(pd.Series([1, 2, 3, 1, 2] * 20)) == "classification"

    def test_declared_type_overrides_inference(self):
        """A 1-5 rating is legitimately either, so the user's choice has to win."""
        ratings = pd.Series([1, 2, 3, 4, 5] * 20)
        assert ml_readiness.infer_task_type(ratings, "regression") == "regression"

    def test_empty_column_has_no_task_type(self):
        assert ml_readiness.infer_task_type(pd.Series([None, None], dtype=object)) is None


class TestLeakageDetection:
    def test_finds_missingness_leak(self, employees):
        issues = ml_readiness.detect_leakage(employees, "attrition", "classification")
        leaks = [i for i in issues if i.column_name == "termination_date"]
        assert leaks, "the planted missingness leak was not detected"
        assert any(i.stats["via_missingness"] for i in leaks)
        assert all(i.issue_type == "target_leakage" for i in leaks)

    def test_ignores_unrelated_columns(self, employees):
        issues = ml_readiness.detect_leakage(employees, "attrition", "classification")
        assert not [i for i in issues if i.column_name in {"salary", "department"}]

    def test_ignores_row_unique_identifiers(self, employees):
        """A unique ID maps one-to-one onto every target by construction. Flagging it would
        be technically true and completely useless; high_cardinality already covers it."""
        issues = ml_readiness.detect_leakage(employees, "attrition", "classification")
        assert not [i for i in issues if i.column_name == "employee_id"]

    def test_finds_value_leak(self):
        n = 80
        outcome = ["pass" if i % 3 else "fail" for i in range(n)]
        df = pd.DataFrame(
            {
                "grade": ["B" if o == "pass" else "F" for o in outcome],  # perfect predictor
                "hours_studied": list(range(n)),
                "outcome": outcome,
            }
        )
        issues = ml_readiness.detect_leakage(df, "outcome", "classification")
        assert [i for i in issues if i.column_name == "grade" and i.severity == "high"]

    def test_finds_near_perfect_correlation_for_regression(self):
        n = 60
        price = np.linspace(100_000, 900_000, n)
        df = pd.DataFrame(
            {
                "price_in_thousands": price / 1000,  # the same number, different units
                "bedrooms": np.tile([2, 3, 4, 5], n // 4),
                "price": price,
            }
        )
        issues = ml_readiness.detect_leakage(df, "price", "regression")
        flagged = {i.column_name for i in issues}
        assert "price_in_thousands" in flagged
        assert "bedrooms" not in flagged

    def test_no_leakage_reported_without_enough_labelled_rows(self):
        df = pd.DataFrame({"a": [1, 2, 3], "target": ["x", "y", "x"]})
        assert ml_readiness.detect_leakage(df, "target", "classification") == []


class TestTargetIssues:
    def test_missing_target_values_are_reported_as_target_missing(self, employees):
        employees.loc[:9, "attrition"] = None
        issues = ml_readiness.detect_target_issues(employees, "attrition", "classification")
        missing = [i for i in issues if i.issue_type == "target_missing"]
        assert len(missing) == 1
        assert missing[0].severity == "high"
        assert missing[0].stats["null_count"] == 10

    def test_single_class_target_is_flagged(self):
        df = pd.DataFrame({"a": range(60), "target": ["yes"] * 60})
        issues = ml_readiness.detect_target_issues(df, "target", "classification")
        assert [i for i in issues if i.issue_type == "target_imbalance" and i.severity == "high"]

    def test_severe_imbalance_is_high_severity(self):
        df = pd.DataFrame({"a": range(200), "target": ["no"] * 197 + ["yes"] * 3})
        issues = ml_readiness.detect_target_issues(df, "target", "classification")
        imbalance = [i for i in issues if i.issue_type == "target_imbalance"]
        assert imbalance and imbalance[0].severity == "high"
        assert imbalance[0].stats["minority_count"] == 3

    def test_balanced_target_produces_no_issue(self):
        df = pd.DataFrame({"a": range(100), "target": ["yes", "no"] * 50})
        assert ml_readiness.detect_target_issues(df, "target", "classification") == []

    def test_constant_regression_target_is_flagged(self):
        df = pd.DataFrame({"a": range(60), "price": [10.0] * 60})
        issues = ml_readiness.detect_target_issues(df, "price", "regression")
        assert [i for i in issues if i.issue_type == "target_imbalance"]


class TestAugmentIssues:
    def test_no_target_leaves_issues_untouched(self, employees):
        from app.services import profiling

        issues = profiling.detect_issues(employees)
        unchanged = ml_readiness.augment_issues(
            employees, issues, GoalContext(use_case="analysis", expected_output="x")
        )
        assert unchanged == issues

    def test_generic_missing_values_issue_on_the_target_is_replaced(self, employees):
        from app.services import profiling

        employees.loc[:9, "attrition"] = None
        issues = profiling.detect_issues(employees)
        assert [i for i in issues if i.column_name == "attrition" and i.issue_type == "missing_values"]

        augmented = ml_readiness.augment_issues(employees, issues, goal())
        on_target = [i for i in augmented if i.column_name == "attrition"]
        assert not [i for i in on_target if i.issue_type == "missing_values"]
        assert [i for i in on_target if i.issue_type == "target_missing"]

    def test_unknown_target_column_is_survivable(self, employees):
        """The target can vanish -- a rename during cleaning, or a typo. Degrade, never crash."""
        issues = ml_readiness.augment_issues(employees, [], goal(target_column="not_a_column"))
        assert issues == []


class TestTargetSafety:
    def test_imputing_the_target_is_rewritten_to_dropping_rows(self):
        operation, params, warning = ml_readiness.enforce_target_safety(
            "fill_mode", {"column": "attrition"}, goal()
        )
        assert operation == "drop_rows_with_nulls"
        assert params["column"] == "attrition"
        assert warning and "predict" in warning

    def test_dropping_the_target_column_is_blocked(self):
        operation, _params, warning = ml_readiness.enforce_target_safety(
            "drop_column", {"column": "attrition"}, goal()
        )
        assert operation is None
        assert warning

    def test_imputing_a_feature_is_allowed_but_warned_about(self):
        """Legal, and often right -- but it bakes a whole-file statistic into rows that will
        later be split into train and test, so the user should approve it knowingly."""
        operation, _params, warning = ml_readiness.enforce_target_safety(
            "fill_mean", {"column": "salary"}, goal()
        )
        assert operation == "fill_mean"
        assert warning and "training" in warning

    def test_ordinary_fix_on_a_feature_is_untouched(self):
        params = {"column": "department", "case": "lower"}
        assert ml_readiness.enforce_target_safety("standardize_case", params, goal()) == (
            "standardize_case",
            params,
            None,
        )

    def test_no_target_means_no_interference(self):
        """A reporting session must behave exactly as it did before this feature existed."""
        params = {"column": "salary"}
        no_target = GoalContext(use_case="reporting", expected_output="x")
        assert ml_readiness.enforce_target_safety("fill_mean", params, no_target) == (
            "fill_mean",
            params,
            None,
        )


class TestReadinessVerdict:
    def test_clean_dataset_is_ready(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame(
            {
                "feature": rng.normal(0, 1, 200),
                "target": rng.choice(["yes", "no"], 200),
            }
        )
        report = ml_readiness.assess_readiness(df, goal(target_column="target"))
        assert report["verdict"] == "ready"
        assert report["blockers"] == []

    def test_missing_target_values_block(self, employees):
        employees.loc[:9, "attrition"] = None
        report = ml_readiness.assess_readiness(employees, goal())
        assert report["verdict"] == "blocked"
        assert any("no value for 'attrition'" in b for b in report["blockers"])

    def test_too_few_rows_blocks(self):
        df = pd.DataFrame({"a": range(10), "target": ["yes", "no"] * 5})
        report = ml_readiness.assess_readiness(df, goal(target_column="target"))
        assert report["verdict"] == "blocked"
        assert any("too few" in b for b in report["blockers"])

    def test_target_removed_during_cleaning_blocks(self, employees):
        report = ml_readiness.assess_readiness(employees.drop(columns=["attrition"]), goal())
        assert report["verdict"] == "blocked"
        assert any("no longer in the data" in b for b in report["blockers"])

    def test_remaining_leakage_is_a_warning_not_a_blocker(self, employees):
        report = ml_readiness.assess_readiness(employees, goal())
        assert report["verdict"] == "warnings"
        assert report["blockers"] == []
        assert any("termination_date" in w for w in report["warnings"])

    def test_session_without_a_target_is_not_applicable(self, employees):
        report = ml_readiness.assess_readiness(
            employees, GoalContext(use_case="reporting", expected_output="x")
        )
        assert report["verdict"] == "not_applicable"
        assert report["target_column"] is None
