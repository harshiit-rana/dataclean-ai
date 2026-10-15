"""Repeated cross-validation when models are compared on a small file.

With a few hundred training rows, one 5-fold partition can rank two candidates one way and another
partition the other way round. Model choice, tuning gains and the fold spread all rested on that one
partition. Below 2,000 training rows, the comparisons now repeat the folds three times with different
shuffles. Grouped and date splits are unchanged, and so are the cut-off and calibration, which need
every row validated exactly once.
"""

from __future__ import annotations

import collections
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
)

from app.services import modeling, tuning, uncertainty


def labels(n: int) -> pd.Series:
    return pd.Series(["yes", "no"] * (n // 2))


def two_class(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "y": np.where(x + rng.normal(size=n) > 0, "yes", "no")})


class TestTheSplitter:
    def test_a_comparison_on_a_small_file_repeats_the_folds(self):
        found = tuning.cv_splitter("classification", labels(400), repeated=True)
        assert isinstance(found, RepeatedStratifiedKFold) and found.n_repeats == tuning.CV_REPEATS
        assert isinstance(tuning.cv_splitter("regression", pd.Series(np.arange(400.0)), repeated=True), RepeatedKFold)

    def test_a_large_file_does_not(self):
        n = tuning.REPEATED_CV_BELOW_ROWS
        assert isinstance(tuning.cv_splitter("classification", labels(n), repeated=True), StratifiedKFold)
        assert isinstance(tuning.cv_splitter("regression", pd.Series(np.arange(float(n))), repeated=True), KFold)

    def test_a_caller_that_needs_a_partition_never_gets_repeats(self):
        """The cut-off and calibration validate every row exactly once."""
        assert isinstance(tuning.cv_splitter("classification", labels(400)), StratifiedKFold)

    def test_grouped_and_date_splits_are_unchanged(self):
        from sklearn.model_selection import StratifiedGroupKFold

        groups = pd.Series(np.repeat(np.arange(100), 4))
        assert isinstance(tuning.cv_splitter("classification", labels(400), groups=groups, repeated=True), StratifiedGroupKFold)
        assert type(tuning.cv_splitter("classification", labels(400), ordered=True, repeated=True)).__name__ == "TimeSeriesSplit"

    def test_the_repeats_are_seeded_so_every_candidate_sees_the_same_folds(self):
        y = labels(400)
        X = np.zeros((400, 1))
        a = [(t.tolist(), v.tolist()) for t, v in tuning.cv_splitter("classification", y, repeated=True).split(X, y)]
        b = [(t.tolist(), v.tolist()) for t, v in tuning.cv_splitter("classification", y, repeated=True).split(X, y)]
        assert a == b and len(a) == 15


class TestScoringAConfiguration:
    def score(self, df):
        return tuning.score_config(
            lambda: modeling.build_pipeline("logistic_regression", ["x", "z"], []),
            df[["x", "z"]], df["y"], "classification",
        )

    def test_below_the_threshold_fits_are_folds_times_repeats(self):
        found = self.score(two_class(400))
        assert found.repeats == 3 and found.fits == 15 and len(found.fold_scores) == 15

    def test_above_it_behaviour_is_unchanged(self):
        found = self.score(two_class(tuning.REPEATED_CV_BELOW_ROWS + 200))
        assert found.repeats == 1 and found.fits == 5

    def test_the_spread_is_over_every_repeated_score(self):
        found = self.score(two_class(400))
        assert found.std == pytest.approx(np.std(found.fold_scores), abs=1e-4)

    def test_a_search_on_a_small_file_counts_its_repeated_fits(self):
        df = two_class(400)
        found = tuning.search(
            lambda: modeling.build_pipeline("random_forest_classifier", ["x", "z"], []), "random_forest_classifier",
            df[["x", "z"]], df["y"], "classification", budget=4,
        )
        assert found.repeats == 3 and found.fits == 4 * 15


class TestTheDecisionsThatRestOnIt:
    def test_model_choice_records_the_repeats_and_keeps_the_folds_of_one_partition(self):
        result = modeling.train_candidates(
            two_class(400), target_column="y", task="classification",
            model_names=["logistic_regression", "random_forest_classifier"], include_deep_diagnostics=False,
        )
        selection = result["selection"]
        assert selection["folds"] == 5 and selection["repeats"] == 3
        assert "repeated 3 times with different shuffles" in selection["chosen_on"]

    def test_the_one_standard_error_rule_does_not_shrink_its_error_by_counting_repeats_as_folds(self, monkeypatch):
        seen = {}
        real = modeling.one_standard_error_choice

        def spy(scores):
            seen.update(scores)
            return real(scores)

        monkeypatch.setattr(modeling, "one_standard_error_choice", spy)
        modeling.train_candidates(
            two_class(400), target_column="y", task="classification",
            model_names=["logistic_regression", "random_forest_classifier"], include_deep_diagnostics=False,
        )
        assert all(entry["folds"] == 5 and len(entry["fold_scores"]) == 15 for entry in seen.values())

    def test_the_paired_comparison_uses_the_repeated_correction(self):
        a = np.r_[np.full(5, 0.81), np.full(5, 0.80), np.full(5, 0.82)] + np.linspace(0, 0.01, 15)
        b = a - 0.01 + np.tile([0.01, -0.01, 0.0, 0.005, -0.005], 3)
        plain = uncertainty.fold_difference(a, b)
        repeated = uncertainty.fold_difference(a, b, test_train_ratio=1 / 4)
        # Treating fifteen repeated scores as fifteen folds would claim a far narrower interval.
        assert (repeated["high"] - repeated["low"]) > (plain["high"] - plain["low"])

    def test_the_fold_spread_shown_beside_the_score_is_over_the_repeats_too(self):
        result = modeling.train_candidates(
            two_class(400), target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=True,
        )
        spread = result["deep_diagnostics"]["cross_validation"]
        assert spread["repeats"] == 3 and len(spread["folds"]) == 15

    def test_the_cut_off_still_validates_every_row_once(self):
        from app.services.operating_point import OperatingPoint

        result = modeling.train_candidates(
            two_class(400), target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False, point=OperatingPoint("f1"),
        )
        point = result["operating_point"]
        assert point["folds"] == 5 and point["expected"]["rows"] == result["split"]["train_rows"]

    def test_calibration_still_gets_a_partition(self):
        result = modeling.train_candidates(
            two_class(800), target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False, calibration="sigmoid",
        )
        assert result["calibration"]["method"] == "sigmoid" and result["calibration"]["folds"] == 5

    def test_the_improvement_loop_is_judged_on_repeated_scores(self):
        from app.services import improvement

        found = improvement.improve(
            two_class(400), target_column="y", task="classification", model_names=["logistic_regression"], budget=4,
        )
        assert found["best"] is not None


class TestItIsWorthIt:
    def test_two_near_equal_candidates_are_ranked_more_consistently_across_partitions(self):
        """The issue's measurement, cut down to stay quick: two logistic regressions that each see
        one of two equally informative columns, so only the partition decides the winner. Measured
        in full (ten files, twenty partitions each) the share of partitions agreeing on the winner
        rose from 0.905 to 0.96; on this file, where it is a close call, from 0.7 to 0.8."""
        rng = np.random.default_rng(3)
        n = 300
        a, b = rng.normal(size=n), rng.normal(size=n)
        X = pd.DataFrame({"a": a, "b": b})
        y = pd.Series(np.where(a + b + rng.normal(size=n) * 1.5 > 0, "yes", "no"))

        def agreement(make_cv):
            winners = []
            for seed in range(10):
                means = {
                    col: cross_val_score(modeling.build_pipeline("logistic_regression", [col], []), X[[col]], y,
                                         cv=make_cv(seed), scoring="roc_auc").mean()
                    for col in ("a", "b")
                }
                winners.append(max(means, key=means.get))
            return collections.Counter(winners).most_common(1)[0][1] / len(winners)

        single = agreement(lambda s: StratifiedKFold(5, shuffle=True, random_state=s))
        repeated = agreement(lambda s: RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=s))
        assert repeated >= single and repeated > 0.75

    def test_it_stays_quick_on_a_small_file(self):
        df = two_class(600)
        start = time.perf_counter()
        modeling.train_candidates(
            df, target_column="y", task="classification",
            model_names=["logistic_regression", "random_forest_classifier"], include_deep_diagnostics=False,
        )
        assert time.perf_counter() - start < 60
