"""Reading the learning curve against how much its folds disagree.

The card read the curve with fixed sizes -- a train/unseen gap over 0.15 is "memorizing", a rise over
0.05 is "still climbing" -- whatever the folds' own disagreement. On small data those statements can be
produced by noise. Each point now carries the folds' spread, a statement is made only when the
difference beats it, and otherwise the curve is called inconclusive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services import diagnostics, modeling
from app.services.model_card import build_model_card


def point(rows, train, valid, train_std=0.01, valid_std=0.01):
    return {"training_rows": rows, "training_score": train, "validation_score": valid,
            "training_std": train_std, "validation_std": valid_std, "metric": "roc_auc"}


class TestTheReading:
    def test_a_big_gap_that_beats_the_noise_is_memorizing(self):
        found = diagnostics.read_learning_curve([point(100, 1.0, 0.70), point(400, 0.99, 0.72)])
        assert found["kind"] == "memorizing" and "partly memorizing" in found["text"]

    def test_the_same_gap_inside_the_noise_is_inconclusive(self):
        found = diagnostics.read_learning_curve([point(100, 1.0, 0.70), point(400, 0.99, 0.80, 0.08, 0.10)])
        assert found["kind"] == "inconclusive"
        assert "looks like memorizing" in found["text"] and "folds disagree by about" in found["text"]

    def test_a_rise_that_beats_the_noise_is_still_climbing(self):
        found = diagnostics.read_learning_curve([point(100, 0.85, 0.70), point(400, 0.86, 0.80)])
        assert found["kind"] == "climbing"

    def test_the_same_rise_inside_the_noise_is_inconclusive(self):
        found = diagnostics.read_learning_curve([point(100, 0.85, 0.70, 0.05, 0.06), point(400, 0.86, 0.78, 0.05, 0.06)])
        assert found["kind"] == "inconclusive" and "still climbing is inconclusive" in found["text"]

    def test_a_flat_curve_with_tight_folds_is_flat(self):
        found = diagnostics.read_learning_curve([point(100, 0.84, 0.80, 0.005, 0.005), point(400, 0.84, 0.81, 0.005, 0.005)])
        assert found["kind"] == "flat" and "flattened out" in found["text"]

    def test_a_flat_curve_with_noisy_folds_cannot_rule_out_a_real_rise(self):
        found = diagnostics.read_learning_curve([point(100, 0.84, 0.80, 0.04, 0.04), point(400, 0.84, 0.81, 0.04, 0.04)])
        assert found["kind"] == "inconclusive" and "enough to hide a real rise" in found["text"]

    def test_the_inconclusive_wording_says_three_folds_give_a_rough_spread(self):
        found = diagnostics.read_learning_curve([point(100, 0.84, 0.80, 0.04, 0.04), point(400, 0.84, 0.81, 0.04, 0.04)])
        assert "With three folds that spread is itself only roughly known" in found["text"]

    def test_a_curve_from_before_the_spread_was_kept_is_read_by_the_fixed_sizes(self):
        old = [{"training_rows": 100, "training_score": 1.0, "validation_score": 0.7},
               {"training_rows": 400, "training_score": 0.99, "validation_score": 0.72}]
        assert diagnostics.read_learning_curve(old)["kind"] == "memorizing"
        flat = [{"training_rows": 100, "training_score": 0.84, "validation_score": 0.80},
                {"training_rows": 400, "training_score": 0.84, "validation_score": 0.81}]
        assert diagnostics.read_learning_curve(flat)["kind"] == "flat"

    def test_no_curve_has_no_reading(self):
        assert diagnostics.read_learning_curve([]) is None


def frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "y": np.where(x + rng.normal(size=n) > 0, "yes", "no")})


class TestWhereItComesFrom:
    def test_each_point_carries_the_folds_spread(self):
        points = diagnostics.learning_curve_points(
            lambda: modeling.build_pipeline("logistic_regression", ["x", "z"], []),
            frame()[["x", "z"]], frame()["y"], "classification",
        )
        assert points and all("validation_std" in p and "training_std" in p for p in points)
        assert all(p["validation_std"] >= 0 for p in points)

    def test_deep_diagnostics_carry_the_reading_and_the_card_prints_the_same_words(self):
        result = modeling.train_candidates(
            frame(), target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=True,
        )
        reading = result["deep_diagnostics"]["learning_curve_reading"]
        assert reading["kind"] in {"memorizing", "climbing", "flat", "inconclusive"}
        card = build_model_card(session_id="s", file_name="f.csv", result=result, summary="A.", interpretation="B.")
        assert reading["text"] in card

    def test_a_stored_result_without_a_reading_is_read_when_the_card_is_written(self):
        result = modeling.train_candidates(
            frame(), target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=True,
        )
        expected = result["deep_diagnostics"].pop("learning_curve_reading")["text"]
        card = build_model_card(session_id="s", file_name="f.csv", result=result, summary="A.", interpretation="B.")
        assert expected in card
