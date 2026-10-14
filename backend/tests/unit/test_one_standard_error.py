"""The one-standard-error rule: among candidates the folds cannot separate, keep the simplest.

Taking the plain maximum treats a lead of 0.002 as a lead even when the folds disagree with each
other by 0.03. A forest "beats" logistic regression, though the evidence does not distinguish them
and the simpler model is easier to explain, to export and to trust.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.services import improvement, modeling, tuning

ALL = ["logistic_regression", "random_forest_classifier", "gradient_boosting_classifier"]


@pytest.fixture
def learnable() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    n = 400
    a, b = rng.normal(0, 1, n), rng.normal(0, 1, n)
    probability = 1 / (1 + np.exp(-(2.0 * a * b + 0.8 * a)))
    return pd.DataFrame(
        {"a": a, "b": b, "noise": rng.normal(0, 1, n),
         "outcome": np.where(rng.random(n) < probability, "yes", "no")}
    )


def folds_scoring(table: dict[str, tuple[float, float]], folds: int = 5):
    """A stand-in for tuning.score_config that gives each model a chosen (score, std)."""

    def scoring(build, X_train, y_train, task, *, folds=folds, **_kwargs):
        name = type(build().named_steps["model"]).__name__
        score, std = table[name]
        return tuning.SearchResult(params={}, score=score, std=std, metric="roc_auc", fits=folds)

    return scoring


def train(df):
    return modeling.train_candidates(
        df, target_column="outcome", task="classification", model_names=ALL,
        include_deep_diagnostics=False,
    )


def scores(**named):
    return {name: {"score": s, "std": sd, "folds": 5} for name, (s, sd) in named.items()}


class TestTheRule:
    def test_the_simpler_candidate_within_one_standard_error_wins(self):
        pick = modeling.one_standard_error_choice(
            scores(logistic_regression=(0.840, 0.03), random_forest_classifier=(0.842, 0.03))
        )
        assert pick["chosen"] == "logistic_regression"
        assert pick["highest_scorer"] == "random_forest_classifier"
        assert pick["rule"] == "one_standard_error"

    def test_the_standard_error_is_the_spread_over_root_folds(self):
        pick = modeling.one_standard_error_choice(
            scores(logistic_regression=(0.80, 0.03), random_forest_classifier=(0.85, 0.05))
        )
        assert pick["standard_error"] == pytest.approx(0.05 / math.sqrt(5), abs=1e-4)

    def test_a_clear_lead_is_respected(self):
        pick = modeling.one_standard_error_choice(
            scores(logistic_regression=(0.70, 0.02), random_forest_classifier=(0.85, 0.02))
        )
        assert pick["chosen"] == "random_forest_classifier"
        assert pick["rule"] == "highest_score"

    def test_the_boundary_is_inclusive(self):
        se = 0.03 / math.sqrt(5)
        pick = modeling.one_standard_error_choice(
            scores(logistic_regression=(0.85 - se, 0.03), random_forest_classifier=(0.85, 0.03))
        )
        assert pick["chosen"] == "logistic_regression"

    def test_just_outside_the_boundary_is_not_enough(self):
        se = 0.03 / math.sqrt(5)
        pick = modeling.one_standard_error_choice(
            scores(logistic_regression=(0.85 - se - 0.001, 0.03), random_forest_classifier=(0.85, 0.03))
        )
        assert pick["chosen"] == "random_forest_classifier"

    def test_the_simplest_of_three_wins_not_merely_the_simpler_of_two(self):
        pick = modeling.one_standard_error_choice(
            scores(
                logistic_regression=(0.838, 0.03),
                random_forest_classifier=(0.841, 0.03),
                gradient_boosting_classifier=(0.845, 0.03),
            )
        )
        assert pick["chosen"] == "logistic_regression"
        assert pick["within_one_standard_error"] == ALL

    def test_among_equally_complex_candidates_the_higher_score_wins(self, monkeypatch):
        twin = modeling.ModelSpec(
            "twin_forest", "Twin forest", "", "classification", "sklearn.ensemble",
            "RandomForestClassifier", {}, complexity_rank=2,
        )
        monkeypatch.setitem(modeling.MODELS, "twin_forest", twin)
        pick = modeling.one_standard_error_choice(
            scores(random_forest_classifier=(0.84, 0.03), twin_forest=(0.85, 0.03))
        )
        assert pick["chosen"] == "twin_forest"

    def test_a_lone_candidate_is_just_itself(self):
        pick = modeling.one_standard_error_choice(scores(logistic_regression=(0.7, 0.05)))
        assert pick["chosen"] == "logistic_regression" and pick["rule"] == "highest_score"

    def test_zero_spread_means_the_best_wins_outright(self):
        pick = modeling.one_standard_error_choice(
            scores(logistic_regression=(0.84, 0.0), random_forest_classifier=(0.85, 0.0))
        )
        assert pick["chosen"] == "random_forest_classifier"

    def test_the_complexity_order_is_written_down(self):
        ranks = {name: spec.complexity_rank for name, spec in modeling.MODELS.items()}
        assert ranks["logistic_regression"] < ranks["random_forest_classifier"] < ranks["gradient_boosting_classifier"]
        assert ranks["linear_regression"] < ranks["random_forest_regressor"] < ranks["gradient_boosting_regressor"]


class TestInTraining:
    def test_a_simpler_model_within_the_noise_is_chosen_and_the_rule_is_recorded(self, learnable, monkeypatch):
        monkeypatch.setattr(
            tuning, "score_config",
            folds_scoring({"LogisticRegression": (0.840, 0.03), "RandomForestClassifier": (0.842, 0.03),
                           "HistGradientBoostingClassifier": (0.830, 0.03)}),
        )
        result = train(learnable)
        selection = result["selection"]
        assert result["best_model"] == "logistic_regression"
        assert selection["rule"] == "one_standard_error"
        assert selection["highest_scorer"] == "random_forest_classifier"
        assert "within one standard error" in selection["rule_note"]
        assert "Random forest" in selection["rule_note"] and "Logistic regression" in selection["rule_note"]

    def test_a_clear_winner_is_chosen_as_before(self, learnable, monkeypatch):
        monkeypatch.setattr(
            tuning, "score_config",
            folds_scoring({"LogisticRegression": (0.70, 0.01), "RandomForestClassifier": (0.90, 0.01),
                           "HistGradientBoostingClassifier": (0.80, 0.01)}),
        )
        result = train(learnable)
        assert result["best_model"] == "random_forest_classifier"
        assert result["selection"]["rule"] == "highest_score"
        assert result["selection"]["rule_note"] is None

    def test_the_fold_scores_are_still_the_raw_ones(self, learnable, monkeypatch):
        monkeypatch.setattr(
            tuning, "score_config",
            folds_scoring({"LogisticRegression": (0.840, 0.03), "RandomForestClassifier": (0.842, 0.03),
                           "HistGradientBoostingClassifier": (0.830, 0.03)}),
        )
        by_model = {r["model"]: r["cv_score"] for r in train(learnable)["results"]}
        assert by_model == {"logistic_regression": 0.840, "random_forest_classifier": 0.842,
                            "gradient_boosting_classifier": 0.830}

    def test_the_choice_still_ignores_the_test_set(self, learnable, monkeypatch):
        """The rule reads fold scores only. Flip every test answer: same model, same note."""
        monkeypatch.setattr(
            tuning, "score_config",
            folds_scoring({"LogisticRegression": (0.840, 0.03), "RandomForestClassifier": (0.842, 0.03),
                           "HistGradientBoostingClassifier": (0.830, 0.03)}),
        )
        honest = train(learnable)
        real_split = modeling.split_data

        def flipped(*args, **kwargs):
            X_train, X_test, y_train, y_test = real_split(*args, **kwargs)
            return X_train, X_test, y_train, y_test.map({"yes": "no", "no": "yes"})

        monkeypatch.setattr(modeling, "split_data", flipped)
        rigged = train(learnable)
        assert rigged["best_model"] == honest["best_model"]
        assert rigged["selection"] == honest["selection"]

    def test_the_real_folds_pick_something_sensible_end_to_end(self, learnable):
        result = train(learnable)
        assert result["selection"]["rule"] in {"highest_score", "one_standard_error"}
        assert result["selection"]["highest_scorer"] in ALL

    def test_the_rule_does_not_apply_to_a_lone_candidate(self, learnable):
        result = modeling.train_candidates(
            learnable, target_column="outcome", task="classification",
            model_names=["random_forest_classifier"], include_deep_diagnostics=False,
        )
        assert result["selection"]["method"] == "only_candidate"
        assert "rule" not in result["selection"]


class TestTheImprovementLoopStartsFromTheSamePlace:
    def test_the_starting_family_follows_the_same_rule(self, learnable, monkeypatch):
        monkeypatch.setattr(
            tuning, "score_config",
            folds_scoring({"LogisticRegression": (0.840, 0.03), "RandomForestClassifier": (0.842, 0.03),
                           "HistGradientBoostingClassifier": (0.830, 0.03)}),
        )
        monkeypatch.setattr(tuning, "search", lambda *a, **k: None)
        found = improvement.improve(
            learnable, target_column="outcome", task="classification", model_names=ALL, budget=4,
        )
        start = found["rounds"][0]
        assert start["detail"]["chosen"] == "logistic_regression"
        assert start["detail"]["highest_scorer"] == "random_forest_classifier"
        assert start["detail"]["rule"] == "one_standard_error"
