"""Choosing where the cut-off sits.

The property that matters here is in `TestChosenOnTheTrainingFolds`: the threshold has to be
picked without reference to the test set, or the recall figure it produces is the best of a
hundred draws on those particular rows and will not survive new data. The chooser is given
out-of-fold predictions over the training rows and nothing else; that is asserted by handing
`apply` a test set whose answers would pull the threshold somewhere else entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import modeling, operating_point
from app.services.operating_point import OperatingPoint, choose_threshold, metrics_at, parse


def _scores_and_truth(n: int = 400, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A well-calibrated-ish score: cases cluster high, non-cases low, with overlap."""
    rng = np.random.default_rng(seed)
    truth = (rng.random(n) < 0.3).astype(int)
    scores = np.clip(rng.normal(0.65, 0.18, n) * truth + rng.normal(0.35, 0.18, n) * (1 - truth), 0, 1)
    return truth, scores


class TestMetricsAtACutoff:
    def test_counts_add_up(self):
        truth, scores = _scores_and_truth()
        at = metrics_at(truth, scores, 0.5)
        assert at["caught"] + at["missed"] == int(truth.sum())
        assert at["caught"] + at["false_alarms"] == at["flagged"]
        assert at["rows"] == len(truth)

    def test_a_lower_cutoff_catches_more_and_is_wrong_more(self):
        truth, scores = _scores_and_truth()
        low, high = metrics_at(truth, scores, 0.3), metrics_at(truth, scores, 0.7)
        assert low["recall"] > high["recall"]
        assert low["false_alarms"] > high["false_alarms"]


class TestChoosingAThreshold:
    def test_default_is_half_and_nothing_is_computed(self):
        truth, scores = _scores_and_truth()
        chosen = choose_threshold(truth, scores, OperatingPoint())
        assert chosen == {"threshold": 0.5, "reachable": True, "note": None}

    def test_an_expensive_miss_moves_the_line_down(self):
        """A missed case costing far more than a false alarm should flag more rows, not
        fewer -- the whole point of stating the trade-off."""
        truth, scores = _scores_and_truth()
        even = choose_threshold(truth, scores, OperatingPoint("cost", 1, 1))["threshold"]
        misses_hurt = choose_threshold(truth, scores, OperatingPoint("cost", 10, 1))["threshold"]
        alarms_hurt = choose_threshold(truth, scores, OperatingPoint("cost", 1, 10))["threshold"]
        assert misses_hurt < even < alarms_hurt

    def test_cost_minimises_the_stated_cost(self):
        truth, scores = _scores_and_truth()
        point = OperatingPoint("cost", 4, 1)
        chosen = choose_threshold(truth, scores, point)["threshold"]
        cost_at = lambda t: 4 * metrics_at(truth, scores, t)["missed"] + metrics_at(truth, scores, t)["false_alarms"]
        assert cost_at(chosen) == min(cost_at(t) for t in operating_point.THRESHOLD_GRID)

    def test_a_recall_floor_takes_the_highest_line_that_clears_it(self):
        """Fewest false alarms subject to catching enough. Any lower line would also clear
        the floor while flagging more rows for nothing."""
        truth, scores = _scores_and_truth()
        chosen = choose_threshold(truth, scores, OperatingPoint("min_recall", target_rate=0.9))
        assert chosen["reachable"]
        t = chosen["threshold"]
        assert metrics_at(truth, scores, t)["recall"] >= 0.9
        assert metrics_at(truth, scores, round(t + 0.01, 2))["recall"] < 0.9

    def test_a_precision_floor_takes_the_line_that_catches_most(self):
        truth, scores = _scores_and_truth()
        chosen = choose_threshold(truth, scores, OperatingPoint("min_precision", target_rate=0.7))
        assert chosen["reachable"]
        t = chosen["threshold"]
        at = metrics_at(truth, scores, t)
        assert at["precision"] >= 0.7
        clearing = [
            metrics_at(truth, scores, float(g))
            for g in operating_point.THRESHOLD_GRID
            if metrics_at(truth, scores, float(g))["precision"] >= 0.7
        ]
        assert at["recall"] == max(row["recall"] for row in clearing)

    def test_an_unreachable_floor_is_said_not_hidden(self):
        """Asking to be right 99% of the time when the scores overlap heavily has no answer.
        The app sets the nearest thing and says so -- it must not report a cut-off as
        satisfying a floor it does not."""
        rng = np.random.default_rng(1)
        truth = (rng.random(300) < 0.5).astype(int)
        scores = rng.random(300)  # no signal at all
        chosen = choose_threshold(truth, scores, OperatingPoint("min_precision", target_rate=0.99))
        assert not chosen["reachable"]
        assert "99%" in chosen["note"]
        assert "better model" in chosen["note"]

    def test_f1_picks_the_peak(self):
        truth, scores = _scores_and_truth()
        t = choose_threshold(truth, scores, OperatingPoint("f1"))["threshold"]
        best = max(metrics_at(truth, scores, float(g))["f1"] for g in operating_point.THRESHOLD_GRID)
        assert metrics_at(truth, scores, t)["f1"] == best


class TestParsing:
    def test_nothing_means_default(self):
        assert parse(None).is_default
        assert parse({}).is_default

    def test_unknown_objective_is_refused(self):
        with pytest.raises(ValueError, match="objective"):
            parse({"objective": "vibes"})

    def test_a_floor_needs_a_rate(self):
        with pytest.raises(ValueError, match="target rate"):
            parse({"objective": "min_recall"})
        with pytest.raises(ValueError, match="target rate"):
            parse({"objective": "min_recall", "target_rate": 1.5})

    def test_both_costs_zero_is_refused(self):
        with pytest.raises(ValueError, match="above zero"):
            parse({"objective": "cost", "cost_of_missing": 0, "cost_of_false_alarm": 0})

    def test_describes_the_trade_off_in_the_users_terms(self):
        assert parse({"objective": "cost", "cost_of_missing": 5}).describe() == "a missed case costs 5x a false alarm"
        assert parse({"objective": "cost", "cost_of_false_alarm": 3}).describe() == "a false alarm costs 3x a missed case"
        assert parse({"objective": "min_recall", "target_rate": 0.8}).describe() == "catch at least 80% of real cases"
        assert "0.50" in parse(None).describe()


@pytest.fixture
def learnable() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 400
    signal = rng.normal(0, 1, n)
    probability = 1 / (1 + np.exp(-(1.8 * signal)))
    return pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(0, 1, n),
            "outcome": np.where(rng.random(n) < probability, "yes", "no"),
        }
    )


class TestChosenOnTheTrainingFolds:
    def test_the_test_set_has_no_say_in_the_threshold(self, learnable):
        """Two runs with the same training rows and wildly different test rows must choose
        the same line. If the test set could move it, the chooser is looking at it."""
        X = learnable[["signal", "noise"]]
        y = learnable["outcome"]
        X_train, y_train = X.iloc[:300], y.iloc[:300]
        X_test = X.iloc[300:]
        build = lambda: modeling.build_pipeline("logistic_regression", ["signal", "noise"], [])
        pipeline = build().fit(X_train, y_train)
        point = OperatingPoint("cost", 5, 1)

        honest = operating_point.apply(
            build=build, pipeline=pipeline, X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y.iloc[300:], point=point, positive_label="yes",
        )
        # Every test answer flipped: a chooser peeking at the test set would move the line.
        flipped = y.iloc[300:].map({"yes": "no", "no": "yes"})
        rigged = operating_point.apply(
            build=build, pipeline=pipeline, X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=flipped, point=point, positive_label="yes",
        )
        assert honest["threshold"] == rigged["threshold"]
        assert honest["observed"] != rigged["observed"]  # the test set is still *reported*

    def test_expected_and_observed_are_both_kept(self, learnable):
        result = modeling.train_candidates(
            learnable, target_column="outcome", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False,
            point=OperatingPoint("min_recall", target_rate=0.9),
        )
        point = result["operating_point"]
        assert point["objective"] == "min_recall"
        assert point["threshold"] != 0.5
        assert point["expected"]["recall"] >= 0.9  # the promise, on the training folds
        assert "test set was not consulted" in point["chosen_on"]
        assert point["folds"] >= 2
        for key in ("expected", "observed", "at_default"):
            assert {"precision", "recall", "flagged", "missed", "false_alarms"} <= set(point[key])

    def test_the_default_records_that_nothing_was_chosen(self, learnable):
        result = modeling.train_candidates(
            learnable, target_column="outcome", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False,
        )
        point = result["operating_point"]
        assert point["objective"] == "default"
        assert point["threshold"] == 0.5
        assert point["expected"] is None
        assert point["observed"] == point["at_default"]

    def test_the_positive_label_can_be_named(self, learnable):
        """yes/no sorts 'yes' last, so the default is right here -- name 'no' instead and
        precision and recall have to be about 'no'."""
        result = modeling.train_candidates(
            learnable, target_column="outcome", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False,
            point=OperatingPoint("cost", 3, 1, positive_label="no"),
        )
        assert result["operating_point"]["positive_label"] == "no"
        assert result["results"][0]["metrics"]["positive_label"] == "no"

    def test_a_label_not_in_the_data_falls_back_rather_than_failing(self, learnable):
        result = modeling.train_candidates(
            learnable, target_column="outcome", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False,
            point=OperatingPoint("cost", 3, 1, positive_label="maybe"),
        )
        assert result["operating_point"]["positive_label"] == "yes"

    def test_regression_has_no_operating_point(self):
        rng = np.random.default_rng(9)
        size = rng.uniform(50, 250, 300)
        df = pd.DataFrame({"size": size, "price": size * 1000 + rng.normal(0, 5000, 300)})
        result = modeling.train_candidates(
            df, target_column="price", task="regression",
            model_names=["linear_regression"], include_deep_diagnostics=False,
            point=OperatingPoint("cost", 3, 1),
        )
        assert result["operating_point"] is None

    def test_a_grouped_run_chooses_on_grouped_folds(self, learnable):
        """Same rule as everywhere else: a grouped split has to be grouped all the way down,
        or the out-of-fold scores the line is chosen on are inflated by the leak."""
        learnable["customer"] = [f"c{i % 40}" for i in range(len(learnable))]
        result = modeling.train_candidates(
            learnable, target_column="outcome", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False,
            split_strategy="group", group_column="customer",
            point=OperatingPoint("f1"),
        )
        assert result["operating_point"] is not None
        assert result["operating_point"]["folds"] >= 2
