"""Decision provenance: why the app believes what it believes about the data.

The verdict is a claim of *absence* -- "nothing found that would stop you" -- and a reader
cannot verify that by looking at the data. These tests pin down the three things that make
the claim worth anything: that every check reports itself even when it passes, that a check
which could not run says so instead of staying silent, and that the flat summary the rest of
the app consumes is derived from the same record rather than written separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.models.graph_state import GoalContext
from app.services import ml_readiness, readiness_checks


def goal(**overrides) -> GoalContext:
    return GoalContext(
        use_case="ml_training",
        expected_output="predict the outcome",
        **overrides,
    )


@pytest.fixture
def healthy() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 300
    return pd.DataFrame(
        {
            "signal": rng.normal(0, 1, n).round(3),
            "region": rng.choice(["north", "south", "east"], n),
            "outcome": rng.choice(["yes", "no"], n, p=[0.45, 0.55]),
        }
    )


def find(report: dict, name: str) -> dict:
    match = [c for c in report["checks"] if c["name"] == name]
    assert match, f"no check named {name!r} in {[c['name'] for c in report['checks']]}"
    return match[0]


class TestEveryCheckReportsItself:
    def test_a_clean_dataset_still_produces_a_full_transcript(self, healthy):
        """Passing silently is the behaviour being replaced. "We looked for duplicate rows
        with contradictory answers and there are none" is information; silence isn't."""
        report = ml_readiness.assess_readiness(healthy, goal(target_column="outcome"))
        assert report["verdict"] == "ready"
        assert report["check_counts"]["passed"] >= 8
        assert report["check_counts"]["failed"] == 0
        for check in report["checks"]:
            assert check["finding"], f"{check['name']} reported no finding"

    def test_a_passing_check_carries_its_evidence(self, healthy):
        report = ml_readiness.assess_readiness(healthy, goal(target_column="outcome"))
        assert find(report, "leakage")["evidence"]
        assert find(report, "contradictory_labels")["evidence"]

    def test_a_failing_check_says_what_would_change_it(self):
        df = pd.DataFrame({"a": range(10), "outcome": ["yes", "no"] * 5})
        report = ml_readiness.assess_readiness(df, goal(target_column="outcome"))
        rows = find(report, "enough_rows")
        assert rows["status"] == "failed"
        assert rows["remedy"], "a finding the user cannot act on is only half-reported"


class TestEnoughRowsCountsRowsItCanLearnFrom:
    """Found live on the RSNA knee dataset. This check reported "4,407 rows, comfortably
    above the 50 needed" while the check two lines below it reported 4,349 rows with no
    answer. Both read the same frame; only one described what would be trained on."""

    @staticmethod
    def frame(rows: int, labelled: int) -> pd.DataFrame:
        rng = np.random.default_rng(11)
        answers = list(rng.choice(["yes", "no"], labelled)) + [None] * (rows - labelled)
        return pd.DataFrame({"signal": rng.normal(0, 1, rows).round(3), "outcome": answers})

    def test_a_mostly_unlabelled_file_does_not_pass_on_its_size(self):
        report = ml_readiness.assess_readiness(
            self.frame(4407, 58), goal(target_column="outcome")
        )
        rows = find(report, "enough_rows")
        assert rows["status"] != "passed"
        assert "58" in rows["finding"]

    def test_below_the_minimum_it_blocks_and_names_both_numbers(self):
        report = ml_readiness.assess_readiness(
            self.frame(4407, 20), goal(target_column="outcome")
        )
        rows = find(report, "enough_rows")
        assert rows["status"] == "failed"
        assert rows["severity"] == "blocker"
        assert "20" in rows["finding"] and "4,407" in rows["finding"]

    def test_it_refuses_to_recommend_filling_the_gap_in(self):
        """The remedy for a missing answer is finding the answer. Imputing it would train the
        model on invented labels, which is worse than having fewer rows."""
        report = ml_readiness.assess_readiness(
            self.frame(4407, 20), goal(target_column="outcome")
        )
        assert "guess" in find(report, "enough_rows")["remedy"]

    def test_enough_labelled_rows_still_warns_about_the_gap(self):
        report = ml_readiness.assess_readiness(
            self.frame(4407, 900), goal(target_column="outcome")
        )
        rows = find(report, "enough_rows")
        assert rows["status"] == "warning"
        assert "900" in rows["finding"]

    def test_a_fully_labelled_file_passes_without_a_caveat(self, healthy):
        report = ml_readiness.assess_readiness(healthy, goal(target_column="outcome"))
        assert find(report, "enough_rows")["status"] == "passed"


class TestChecksThatCouldNotRun:
    def test_no_target_leaves_a_transcript_of_what_was_skipped(self, healthy):
        """The case that motivated the refactor. The old code returned early, so a report
        with no target was indistinguishable from one where every check passed."""
        report = ml_readiness.assess_readiness(healthy, goal())
        assert report["verdict"] == "not_applicable"
        assert report["check_counts"]["not_run"] >= 8
        for check in report["checks"]:
            if check["status"] == "not_run" and check["name"] != "target_named":
                assert check["blocked_by"] == "target_named"

    def test_a_removed_target_does_not_silently_skip_the_rest(self, healthy):
        report = ml_readiness.assess_readiness(healthy, goal(target_column="gone"))
        assert report["verdict"] == "blocked"
        assert find(report, "target_present")["status"] == "failed"
        # Everything downstream is accounted for rather than absent.
        leakage = find(report, "leakage")
        assert leakage["status"] == "not_run"
        assert leakage["blocked_by"] == "target_present"

    def test_not_run_is_distinguishable_from_passed(self, healthy):
        skipped = ml_readiness.assess_readiness(healthy, goal())
        clean = ml_readiness.assess_readiness(healthy, goal(target_column="outcome"))
        assert find(skipped, "leakage")["status"] == "not_run"
        assert find(clean, "leakage")["status"] == "passed"


class TestTheSummaryIsDerived:
    def test_blockers_come_from_the_checks(self):
        df = pd.DataFrame({"a": range(10), "outcome": ["yes", "no"] * 5})
        report = ml_readiness.assess_readiness(df, goal(target_column="outcome"))
        failed = [c for c in report["checks"] if c["status"] == "failed" and c["severity"] == "blocker"]
        assert len(report["blockers"]) == len(failed)

    def test_the_summary_keeps_the_evidence(self):
        """Dropping it would make the one-line summary vaguer than the version it replaced."""
        df = pd.DataFrame({"a": range(10), "outcome": ["yes", "no"] * 5})
        report = ml_readiness.assess_readiness(df, goal(target_column="outcome"))
        assert any("too few" in b for b in report["blockers"])


class TestContradictoryLabels:
    def test_identical_rows_with_different_answers_are_caught(self):
        """No statistical check finds this, and no model can survive it: two rows agreeing on
        every input and disagreeing on the outcome are a hard ceiling on accuracy."""
        rows = [{"a": 1, "b": "x", "outcome": "yes"}, {"a": 1, "b": "x", "outcome": "no"}] * 40
        df = pd.DataFrame(rows)
        check = readiness_checks.check_contradictory_labels(df, "outcome", ["a", "b"])
        assert check.status == "failed"
        assert check.severity == "blocker"
        assert "hard ceiling" in check.evidence

    def test_duplicate_rows_that_agree_are_not_a_contradiction(self):
        df = pd.DataFrame([{"a": 1, "b": "x", "outcome": "yes"}] * 50)
        check = readiness_checks.check_contradictory_labels(df, "outcome", ["a", "b"])
        assert check.status == "passed"
        assert "agree" in check.finding or "same answer" in check.finding

    def test_a_handful_of_contradictions_warns_rather_than_blocks(self):
        """A few are normal in real data. Blocking on them is the kind of false alarm that
        teaches people to ignore the panel."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "a": rng.normal(0, 1, 400).round(4),
                "b": rng.choice(["x", "y"], 400),
                "outcome": rng.choice(["yes", "no"], 400),
            }
        )
        df = pd.concat(
            [df, pd.DataFrame([{"a": 99.0, "b": "z", "outcome": o} for o in ("yes", "no")])],
            ignore_index=True,
        )
        check = readiness_checks.check_contradictory_labels(df, "outcome", ["a", "b"])
        assert check.status == "failed"
        assert check.severity == "warning"

    def test_no_repeated_inputs_at_all_passes(self, healthy):
        check = readiness_checks.check_contradictory_labels(
            healthy, "outcome", ["signal", "region"]
        )
        assert check.status == "passed"

    def test_it_declines_rather_than_crawling_on_a_huge_file(self):
        df = pd.DataFrame({"a": [1], "outcome": ["yes"]})
        check = readiness_checks.check_contradictory_labels(df, "outcome", ["a"])
        assert check.status == "passed"  # sanity: the small path works

        big = readiness_checks.MAX_ROWS_FOR_CONTRADICTION_CHECK + 1
        frame = pd.DataFrame({"a": np.zeros(big), "outcome": ["yes"] * big})
        check = readiness_checks.check_contradictory_labels(frame, "outcome", ["a"])
        assert check.status == "not_run"


class TestRepeatedEntities:
    def test_a_repeating_identifier_is_flagged(self):
        """The same customer in train and test lets the model recognise them rather than
        learn the pattern, and every score comes out better than it should."""
        customers = [f"CUST-{i:04d}" for i in range(60)]
        df = pd.DataFrame(
            {
                "customer_id": customers * 4,
                "amount": range(240),
                "outcome": ["yes", "no"] * 120,
            }
        )
        check = readiness_checks.check_repeated_entities(df, "outcome")
        assert check.status == "warning"
        assert "customer_id" in check.finding

    def test_a_two_value_flag_is_not_an_entity(self, healthy):
        check = readiness_checks.check_repeated_entities(healthy, "outcome")
        assert check.status == "passed"

    def test_a_numeric_measurement_is_not_a_repeated_entity(self):
        """Found live. `last_login_days_ago` holds a few hundred distinct integers each
        repeating a dozen times -- exactly the shape this check looks for, and nothing to do
        with entities. An identifier is a label whose values carry no order; a numeric column
        with moderate cardinality is a measurement."""
        rng = np.random.default_rng(8)
        n = 1200
        df = pd.DataFrame(
            {
                "days_since_login": rng.integers(0, 120, n),
                "tickets": rng.integers(0, 40, n),
                "outcome": rng.choice(["yes", "no"], n),
            }
        )
        check = readiness_checks.check_repeated_entities(df, "outcome")
        assert check.status == "passed", check.finding

    def test_a_row_unique_column_is_not_a_repeated_entity(self):
        df = pd.DataFrame(
            {"row_id": [f"R{i}" for i in range(200)], "outcome": ["yes", "no"] * 100}
        )
        check = readiness_checks.check_repeated_entities(df, "outcome")
        assert check.status == "passed"
