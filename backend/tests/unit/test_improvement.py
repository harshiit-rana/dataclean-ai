"""The improvement loop.

The tests that matter here are not "does it find a better model" -- it usually will. They are
the ones that pin down the things that make the answer trustworthy: that the test set is never
what a round competes on, that a gain smaller than the fold-to-fold noise is refused, and that
the loop is willing to stop and say the goal is out of reach rather than search until
something clears the bar by luck.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import improvement, modeling, tuning


@pytest.fixture
def learnable() -> pd.DataFrame:
    """600 rows where `signal` genuinely predicts the outcome, with enough noise that no
    configuration gets near perfect."""
    rng = np.random.default_rng(11)
    n = 600
    signal = rng.normal(0, 1, n)
    extra = rng.normal(0, 1, n)
    probability = 1 / (1 + np.exp(-(1.6 * signal + 0.7 * extra)))
    return pd.DataFrame(
        {
            "signal": signal,
            "extra": extra,
            "noise": rng.normal(0, 1, n),
            "region": rng.choice(["north", "south", "east"], n),
            "outcome": np.where(rng.random(n) < probability, "yes", "no"),
        }
    )


@pytest.fixture
def pure_noise() -> pd.DataFrame:
    """Nothing predicts the target. No amount of searching should change that."""
    rng = np.random.default_rng(4)
    n = 400
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.choice(["x", "y"], n),
            "outcome": rng.choice(["yes", "no"], n),
        }
    )


class TestGainRule:
    def test_a_gain_smaller_than_the_noise_is_not_a_gain(self):
        """The rule the whole loop rests on. Folds that disagree by 0.04 cannot demonstrate
        an improvement of 0.01, however many configurations produced it."""
        assert improvement._is_real_gain(0.81, 0.80, std=0.04) is False

    def test_a_gain_larger_than_the_noise_counts(self):
        assert improvement._is_real_gain(0.87, 0.80, std=0.02) is True

    def test_a_tiny_gain_is_refused_even_when_the_folds_all_agree(self):
        """Perfectly agreeing folds drive the spread to zero, and without a floor the rule
        would accept a gain of 0.0001 and the loop would 'improve' forever."""
        assert improvement._is_real_gain(0.8001, 0.80, std=0.0) is False

    def test_going_backwards_is_never_a_gain(self):
        assert improvement._is_real_gain(0.75, 0.80, std=0.0) is False


class TestSealedTestSet:
    def test_the_reported_score_is_not_the_score_anything_was_chosen_on(self, learnable):
        """The point of the whole design: rounds compete on cross-validated training folds,
        and the test set is opened once at the end. If the two were the same number, the
        search would be selecting on the rows it then reports."""
        result = improvement.improve(
            learnable, target_column="outcome", task="classification", budget=8
        )
        assert result["best"]["cv_score"] is not None
        assert result["test_metrics"]["roc_auc"] is not None
        # Different measurements of different rows -- they should not be identical.
        assert result["best"]["cv_score"] != result["test_metrics"]["roc_auc"]

    def test_every_round_scores_on_the_same_metric_as_the_goal(self, learnable):
        result = improvement.improve(
            learnable, target_column="outcome", task="classification", budget=8
        )
        assert result["metric"] == "roc_auc"
        for entry in result["rounds"]:
            assert entry["score"] is None or 0.0 <= entry["score"] <= 1.0


class TestStopping:
    def test_it_stops_once_the_goal_is_reached(self, learnable):
        result = improvement.improve(
            learnable, target_column="outcome", task="classification", goal=0.55, budget=8
        )
        assert result["stopped_because"] == "goal_reached"
        # And it didn't keep burning fits after that.
        assert len(result["rounds"]) == 1

    def test_it_admits_when_a_goal_is_out_of_reach(self, pure_noise):
        """The answer that matters. On data with no signal the loop must stop and say so --
        searching until something clears 0.90 is exactly how you manufacture a useless model
        that looks fine."""
        result = improvement.improve(
            pure_noise, target_column="outcome", task="classification", goal=0.90, budget=8
        )
        assert result["stopped_because"] == "goal_out_of_reach"
        assert result["best"]["cv_score"] < 0.90

    def test_searching_noise_does_not_manufacture_a_score(self, pure_noise):
        """Dozens of configurations against data with nothing in it. The honest test-set AUC
        has to stay near a coin flip."""
        result = improvement.improve(
            pure_noise, target_column="outcome", task="classification", budget=20
        )
        assert result["test_metrics"]["roc_auc"] < 0.62

    def test_with_no_goal_it_runs_out_of_things_to_try(self, learnable):
        result = improvement.improve(
            learnable, target_column="outcome", task="classification", budget=8
        )
        assert result["stopped_because"] == "no_further_gains"


class TestTheTrail:
    def test_it_reports_what_it_tried_including_what_did_not_work(self, learnable):
        """A rejected round is a finding: 'tuning did not help' says something about the data
        that a silent success would not."""
        result = improvement.improve(
            learnable, target_column="outcome", task="classification", budget=8
        )
        names = [r["name"] for r in result["rounds"]]
        assert names[0] == "baseline"
        for entry in result["rounds"]:
            assert entry["what_it_tried"]
            assert entry["outcome"]

    def test_progress_is_reported_as_it_goes(self, learnable):
        seen: list[str] = []
        improvement.improve(
            learnable,
            target_column="outcome",
            task="classification",
            budget=8,
            on_progress=lambda r: seen.append(r.name),
        )
        assert seen[0] == "baseline"
        assert len(seen) > 1

    def test_the_chosen_settings_are_reported_without_plumbing(self, learnable):
        result = improvement.improve(
            learnable, target_column="outcome", task="classification", budget=12
        )
        for key in result["best"]["params"]:
            assert not key.startswith("model__")


class TestRefusals:
    def test_it_refuses_when_the_smallest_class_cannot_be_folded(self):
        rng = np.random.default_rng(2)
        n = 120
        df = pd.DataFrame(
            {
                "a": rng.normal(0, 1, n),
                "outcome": ["yes"] + ["no"] * (n - 1),  # one row in a class
            }
        )
        with pytest.raises(modeling.ModelingError, match="cross-validate"):
            improvement.improve(df, target_column="outcome", task="classification")

    def test_it_refuses_when_there_is_nothing_to_predict_from(self, learnable):
        with pytest.raises(modeling.ModelingError, match="nothing to improve"):
            improvement.improve(
                learnable,
                target_column="outcome",
                task="classification",
                excluded_columns=["signal", "extra", "noise", "region"],
            )


class TestRegression:
    def test_it_works_on_a_continuous_target(self):
        rng = np.random.default_rng(7)
        n = 500
        size = rng.uniform(50, 250, n)
        df = pd.DataFrame(
            {
                "size": size,
                "noise": rng.normal(0, 1, n),
                "kind": rng.choice(["flat", "house"], n),
                "price": size * 1000 + rng.normal(0, 20000, n),
            }
        )
        result = improvement.improve(df, target_column="price", task="regression", budget=8)
        assert result["metric"] == "r2"
        assert result["test_metrics"]["r2"] > 0.5


class TestSearchSpaces:
    def test_every_model_has_a_space_worth_searching(self):
        """A model with no space silently skips its tuning round, which would look like
        'tuning did not help' when nothing was tried."""
        for name in modeling.MODELS:
            assert name in tuning.SEARCH_SPACES, f"{name} has no search space"
            assert tuning.SEARCH_SPACES[name]

    def test_search_parameters_address_the_model_step(self):
        """They are set on the Pipeline, so anything without the prefix silently does
        nothing -- sklearn raises, the search fails, and the round reports no gain."""
        for space in tuning.SEARCH_SPACES.values():
            for key in space:
                assert key.startswith("model__"), key


class TestTheFourthRound:
    """Tuning is the small lever. On the churn sample the settings search found nothing
    worth keeping, which is the evidence that the ceiling is the columns -- so the loop also
    tries the columns."""

    @pytest.fixture
    def derivable(self) -> pd.DataFrame:
        """The outcome depends on the weekday of a date column and nothing else, so no
        amount of tuning can reach it and one derived input reaches it completely."""
        n = 600
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        rng = np.random.default_rng(7)
        flip = rng.random(n) < 0.1  # a little noise, so nothing scores 1.0
        return pd.DataFrame(
            {
                "noise": rng.normal(0, 1, n),
                "when": dates.astype(str),
                "outcome": np.where((dates.weekday >= 5) ^ flip, "yes", "no"),
            }
        )

    def test_it_tries_the_derived_inputs_when_the_run_started_without_them(self, derivable):
        found = improvement.improve(
            derivable, target_column="outcome", task="classification",
            model_names=["random_forest_classifier"], budget=4,
        )
        round_4 = next(r for r in found["rounds"] if r["name"] == "derived_features")
        assert round_4["accepted"]
        assert "date_parts:when" in round_4["detail"]["recipes"]
        assert found["best"]["feature_recipes"] == ["date_parts:when"]
        # The signal was unreachable without them, so this is a large gain, not a nudge.
        assert round_4["gain"] > 0.2

    def test_it_checks_that_derived_inputs_are_earning_their_place(self, learnable):
        """Started with them on, so the round tries removing them. On data where they carry
        nothing, removing them must not be reported as an improvement either."""
        learnable["joined"] = pd.date_range("2021-01-01", periods=len(learnable), freq="D").astype(str)
        found = improvement.improve(
            learnable, target_column="outcome", task="classification",
            model_names=["logistic_regression"], budget=4,
            feature_recipes_wanted=["date_parts:joined"],
        )
        round_4 = next(r for r in found["rounds"] if r["name"] == "derived_features")
        assert "Removed" in round_4["what_it_tried"]
        assert not round_4["accepted"]  # a random date adds nothing, and removing it adds nothing

    def test_it_says_so_when_there_is_nothing_to_derive(self, pure_noise):
        found = improvement.improve(
            pure_noise, target_column="outcome", task="classification",
            model_names=["logistic_regression"], budget=4,
        )
        round_4 = next(r for r in found["rounds"] if r["name"] == "derived_features")
        assert not round_4["accepted"]
        assert "can be derived" in round_4["outcome"]

    def test_a_derived_round_is_still_judged_on_the_folds(self, derivable):
        """Same rule as every other round: the test set is scored once, at the end, and the
        reported number is not the number anything was selected against."""
        found = improvement.improve(
            derivable, target_column="outcome", task="classification",
            model_names=["random_forest_classifier"], budget=4,
        )
        round_4 = next(r for r in found["rounds"] if r["name"] == "derived_features")
        assert "training folds" in round_4["what_it_tried"]
        assert found["test_metrics"]["roc_auc"] != round_4["score"]

    def test_the_winning_recipes_ride_along_for_the_rebuild(self, derivable):
        """The route rebuilds the result through train_candidates using best["feature_recipes"].
        If the loop's choice did not survive that handoff, the reported model would be the one
        the loop rejected."""
        found = improvement.improve(
            derivable, target_column="outcome", task="classification",
            model_names=["random_forest_classifier"], budget=4,
        )
        rebuilt = modeling.train_candidates(
            derivable, target_column="outcome", task="classification",
            model_names=[found["best"]["model"]], include_deep_diagnostics=False,
            feature_recipes_wanted=found["best"]["feature_recipes"],
        )
        bare = modeling.train_candidates(
            derivable, target_column="outcome", task="classification",
            model_names=[found["best"]["model"]], include_deep_diagnostics=False,
        )
        assert "when__weekday" in rebuilt["features"]["numeric"]
        # Compared against the same model on the same rows without them, rather than against
        # an absolute bar: one row in ten carries a flipped label here, which caps the
        # reachable AUC well below 1 no matter what the model does.
        assert rebuilt["results"][0]["metrics"]["roc_auc"] > bare["results"][0]["metrics"]["roc_auc"] + 0.25
        assert rebuilt["assessment"]["verdict"] == "useful"
        assert bare["assessment"]["verdict"] == "no_better_than_guessing"
