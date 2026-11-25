"""With more than two classes, models are chosen and tuned by log loss (#132).

Chosen by accuracy, a model that was right as often but far more over-confident won: on the eucalyptus benchmark task
the app picked gradient boosting (test log loss 1.06) where the forest had 0.75. The score is compared as sklearn's
`neg_log_loss`, so higher is still better everywhere, and shown to people as the positive log loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import improvement, modeling, tuning


def three_class(n: int = 450) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {"x": x, "z": rng.normal(0, 1, n), "y": np.digitize(x + rng.normal(0, 1, n), [-0.5, 0.5]).astype(str)}
    )


class TestTheMetric:
    def test_more_than_two_classes_are_chosen_by_log_loss(self):
        assert tuning.scoring_for("classification", ["a", "b", "c", "a"]) == "neg_log_loss"

    def test_two_classes_and_numbers_are_unchanged(self):
        assert tuning.scoring_for("classification", ["a", "b", "a"]) == "roc_auc"
        assert tuning.scoring_for("regression", [1.0, 2.0, 3.0]) == "r2"

    def test_the_charts_keep_accuracy(self):
        """The learning curve and importance charts stay in accuracy: a chart of negative log loss reads badly."""
        assert tuning.chart_metric("classification", ["a", "b", "c"]) == "accuracy"

    def test_log_loss_is_shown_as_itself(self):
        assert tuning.shown(-0.75, "neg_log_loss") == 0.75
        assert tuning.shown(0.9, "roc_auc") == 0.9


class TestChanceAndGoals:
    def test_chance_is_the_log_loss_of_predicting_the_class_shares(self):
        y = ["a"] * 50 + ["b"] * 30 + ["c"] * 20
        shares = np.array([0.5, 0.3, 0.2])
        assert improvement.chance_level("neg_log_loss", y) == pytest.approx(float((shares * np.log(shares)).sum()))

    def test_a_score_no_better_than_the_class_shares_is_not_clear_of_chance(self):
        y = ["a"] * 50 + ["b"] * 30 + ["c"] * 20
        chance = improvement.chance_level("neg_log_loss", y)
        assert not improvement.clear_of_chance(chance + 0.01, 0.05, "neg_log_loss", y)
        assert improvement.clear_of_chance(-0.3, 0.05, "neg_log_loss", y)

    def test_a_log_loss_goal_is_met_at_or_below_it(self):
        assert improvement._goal_met(-0.40, 0.5, "neg_log_loss")
        assert not improvement._goal_met(-0.60, 0.5, "neg_log_loss")
        assert improvement._goal_met(0.9, 0.8, "roc_auc")


class TestTheTrainAndTheLoop:
    def test_the_plain_train_chooses_by_log_loss(self):
        df = three_class()
        result = modeling.train_candidates(
            df, target_column="y", task="classification",
            model_names=["logistic_regression", "random_forest_classifier"], include_deep_diagnostics=False,
        )
        assert result["selection"]["metric"] == "neg_log_loss"
        scores = [r["cv_score"] for r in result["results"] if r.get("cv_score") is not None]
        assert scores and all(s < 0 for s in scores)

    def test_the_loop_says_log_loss_as_a_positive_number(self):
        df = three_class()
        out = improvement.improve(
            df, target_column="y", task="classification", model_names=["logistic_regression"], budget=2,
        )
        rounds = out["rounds"]
        baseline = rounds[0]
        assert baseline["metric"] == "neg_log_loss"
        assert "log loss" in baseline["outcome"]
        assert f"{-baseline['score']:.3f}" in baseline["outcome"]
        assert "-0." not in baseline["outcome"]


class TestNoSignal:
    def test_no_round_keeps_a_gain_that_is_not_clear_of_chance(self, monkeypatch):
        """On cmc with its labels shuffled, dropping inputs pulled a below-chance log loss toward the class shares
        and was kept: the input rounds did not ask whether the model was clear of chance, as the others do. Here
        every gain counts as real and nothing is clear of chance, so no round may be kept."""
        monkeypatch.setattr(improvement, "_is_real_gain", lambda *a, **k: True)
        monkeypatch.setattr(improvement, "clear_of_chance", lambda *a, **k: False)
        rng = np.random.default_rng(11)
        n = 400
        df = pd.DataFrame({f"x{i}": rng.normal(0, 1, n) for i in range(6)})
        df["when"] = pd.date_range("2024-01-01", periods=n, freq="D").astype(str)
        df["y"] = rng.choice(["a", "b", "c"], n, p=[0.45, 0.35, 0.2])
        out = improvement.improve(
            df, target_column="y", task="classification", model_names=["logistic_regression"], budget=2,
        )
        kept = [r["name"] for r in out["rounds"] if r["accepted"] and r["name"] != "baseline"]
        assert kept == []


class TestNoModelBeatsChance:
    """On MIC every candidate's fold log loss was worse than predicting the class shares, and the selection did
    not say so (#140). The selection now says so, for any metric with a chance level."""

    @staticmethod
    def noise(classes, n=400, seed=5):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({f"x{i}": rng.normal(0, 1, n) for i in range(5)})
        df["y"] = rng.choice(classes, n)
        return df

    def train(self, df):
        return modeling.train_candidates(
            df, target_column="y", task="classification",
            model_names=["logistic_regression", "random_forest_classifier"], include_deep_diagnostics=False,
        )["selection"]

    def test_labels_with_no_signal_are_said_to_be_no_better_than_guessing(self):
        selection = self.train(self.noise(["a", "b", "c"]))
        assert selection["clear_of_chance"] is False
        assert "class shares" in selection["chance_note"]
        assert "do not rely on the probabilities" in selection["chance_note"]

    def test_two_classes_with_no_signal_are_said_so_too(self):
        selection = self.train(self.noise(["yes", "no"]))
        assert selection["clear_of_chance"] is False
        assert "guessing" in selection["chance_note"]

    def test_a_target_with_signal_gets_no_such_note(self):
        selection = self.train(three_class())
        assert selection["clear_of_chance"] is True
        assert selection["chance_note"] is None
