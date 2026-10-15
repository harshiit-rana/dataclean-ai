"""What the chosen model adds over its best single input column.

A model that scores well may be doing little more than one strong column. The simplest registry
model is cross-validated on each input alone, on the training rows and the folds the candidates were
compared on, and the best of those is set beside the chosen model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services import modeling
from app.services.model_card import build_model_card


def one_column_explains_it(n=400, seed=0):
    rng = np.random.default_rng(seed)
    key = rng.normal(size=n)
    return pd.DataFrame({
        "key": key,
        "noise_a": rng.normal(size=n),
        "noise_b": rng.normal(size=n),
        "y": np.where(key > 0, "yes", "no"),
    })


def two_columns_needed(n=600, seed=1):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": b, "y": np.where(a * b > 0, "yes", "no")})


def train(df, models=("logistic_regression", "random_forest_classifier"), **kwargs):
    return modeling.train_candidates(
        df, target_column="y", task="classification", model_names=list(models),
        include_deep_diagnostics=False, **kwargs,
    )


class TestWhenOneColumnExplainsTheTarget:
    def test_it_is_named_and_the_chosen_model_adds_nothing_over_it(self):
        single = train(one_column_explains_it())["baseline"]["best_single_column"]
        assert single["applicable"] is True
        assert single["column"] == "key"
        assert single["score"] > 0.99
        assert single["adds_over_it"] is False
        assert "'key'" in single["note"] and "not clearly better" in single["note"]

    def test_it_is_scored_with_the_simplest_model_for_the_task(self):
        assert train(one_column_explains_it())["baseline"]["best_single_column"]["model"] == "Logistic regression"


class TestWhenTheColumnsTogetherMatter:
    def test_the_chosen_model_is_clearly_ahead_of_any_one_input(self):
        result = train(two_columns_needed())
        single = result["baseline"]["best_single_column"]
        assert single["applicable"] is True
        assert result["best_model"] == "random_forest_classifier"
        assert single["adds_over_it"] is True and single["lead"] > 0.2
        assert single["paired_difference"]["low"] > 0
        assert "more than noise" in single["note"]


class TestRegression:
    def test_a_numeric_target_is_compared_on_r_squared(self):
        rng = np.random.default_rng(2)
        key = rng.normal(size=300)
        df = pd.DataFrame({"key": key, "other": rng.normal(size=300), "y": 3 * key + rng.normal(0, 0.1, 300)})
        result = modeling.train_candidates(
            df, target_column="y", task="regression",
            model_names=["linear_regression", "random_forest_regressor"], include_deep_diagnostics=False,
        )
        single = result["baseline"]["best_single_column"]
        assert single["column"] == "key" and single["metric"] == "r2" and single["score"] > 0.95


class TestOnlyOneModel:
    def test_the_chosen_model_is_scored_for_the_comparison_when_selection_kept_no_scores(self):
        result = train(two_columns_needed(), models=("random_forest_classifier",))
        assert result["selection"]["method"] == "only_candidate"
        assert result["baseline"]["best_single_column"]["applicable"] is True


class TestWhenItIsSkipped:
    def test_one_input_column_has_nothing_to_compare_with(self):
        df = one_column_explains_it()[["key", "y"]]
        single = train(df)["baseline"]["best_single_column"]
        assert single["applicable"] is False and "only one input column" in single["reason"]

    def test_too_many_columns_is_skipped_with_the_reason(self, monkeypatch):
        monkeypatch.setattr(modeling, "BASELINE_MAX_COLUMNS", 2)
        single = train(one_column_explains_it())["baseline"]["best_single_column"]
        assert single["applicable"] is False and "3 input columns" in single["reason"]

    def test_too_many_rows_is_skipped_with_the_reason(self, monkeypatch):
        monkeypatch.setattr(modeling, "BASELINE_MAX_ROWS", 100)
        single = train(one_column_explains_it())["baseline"]["best_single_column"]
        assert single["applicable"] is False and "training rows" in single["reason"]


class TestCategoricalAndCardAndBounds:
    def test_a_category_column_can_be_the_best_single_input(self):
        rng = np.random.default_rng(3)
        n = 400
        colour = rng.choice(["red", "green", "blue"], n)
        df = pd.DataFrame({"colour": colour, "noise": rng.normal(size=n),
                           "y": np.where(colour == "red", "yes", "no")})
        assert train(df)["baseline"]["best_single_column"]["column"] == "colour"

    def test_the_card_prints_the_same_sentence(self):
        result = train(one_column_explains_it())
        card = build_model_card(session_id="s", file_name="f.csv", result=result, summary="A.", interpretation="B.")
        assert result["baseline"]["best_single_column"]["note"] in card

    def test_the_card_prints_the_reason_when_it_was_skipped(self, monkeypatch):
        monkeypatch.setattr(modeling, "BASELINE_MAX_ROWS", 100)
        result = train(one_column_explains_it())
        card = build_model_card(session_id="s", file_name="f.csv", result=result, summary="A.", interpretation="B.")
        assert "Best single input not compared" in card

    def test_a_result_from_before_it_existed_still_builds_a_card(self):
        result = train(one_column_explains_it())
        result["baseline"].pop("best_single_column")
        assert build_model_card(session_id="s", file_name="f.csv", result=result, summary="A.", interpretation="B.")
