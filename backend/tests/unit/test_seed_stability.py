"""Whether the chosen model's lead survives a different random seed.

Random forests and boosting are fitted with seed 42 everywhere. A result that only holds for that seed
is not a result, so after the winner is chosen it is cross-validated again on the same training folds
with four more seeds, and the spread of its fold score is set against the gap to the runner-up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import modeling, tuning
from app.services.model_card import build_model_card

FOREST, BOOST, LOGISTIC = "random_forest_classifier", "gradient_boosting_classifier", "logistic_regression"


def two_class(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "y": np.where(x + rng.normal(size=n) > 0, "yes", "no")})


def train(df, models, **kwargs):
    return modeling.train_candidates(
        df, target_column="y", task="classification", model_names=list(models),
        include_deep_diagnostics=False, **kwargs,
    )


def controlled(by_model: dict[str, float], seed_scores: dict[int, float]):
    """A score_config whose answer is fixed per model, and for the chosen forest per seed -- so the
    decision is tested on numbers chosen for the case."""

    def fake(build, X, y, task, **kwargs):
        model = build().named_steps["model"]
        name = FOREST if "Forest" in type(model).__name__ else BOOST
        score = seed_scores.get(model.random_state, by_model[name]) if name == FOREST else by_model[name]
        values = (score,) * 5
        return tuning.SearchResult(params={}, score=score, std=0.01, metric="roc_auc", fits=5, fold_scores=values)

    return fake


class TestWhichModelsAreChecked:
    def test_forests_and_boosting_are_seeded_and_the_linear_models_are_not(self):
        seeded = {name for name, spec in modeling.MODELS.items() if spec.seeded}
        assert seeded == {
            "random_forest_classifier", "extra_trees_classifier", "gradient_boosting_classifier",
            "random_forest_regressor", "extra_trees_regressor", "gradient_boosting_regressor",
        }

    def test_a_deterministic_winner_reports_not_applicable_with_the_reason(self):
        selection = {"method": "cross_validation", "chosen": LOGISTIC, "scores": {LOGISTIC: {"score": 0.8}}}
        found = modeling.seed_stability(selection, lambda seed: None, two_class()[["x"]], two_class()["y"], "classification")
        assert found == {"applicable": False, "reason": "Logistic regression's fit has no randomness, so the seed cannot change it."}

    def test_a_single_candidate_was_not_chosen_on_folds_so_there_is_nothing_to_check(self):
        assert train(two_class(), [FOREST])["selection"].get("seed_stability") is None

    def test_a_large_file_is_skipped_with_the_reason(self, monkeypatch):
        monkeypatch.setattr(modeling, "SEED_CHECK_MAX_ROWS", 100)
        found = train(two_class(), [FOREST, BOOST])["selection"]["seed_stability"]
        assert found["applicable"] is False and "Skipped on" in found["reason"]


class TestWhatItMeasures:
    def test_five_seeds_and_the_first_is_the_score_already_measured(self):
        selection = train(two_class(), [FOREST, BOOST])["selection"]
        found = selection["seed_stability"]
        assert found["applicable"] is True and found["seeds"] == list(modeling.SEEDS_TO_CHECK)
        assert len(found["scores"]) == 5
        assert found["spread"] == pytest.approx(max(found["scores"]) - min(found["scores"]), abs=1e-4)

    def test_each_seed_really_reaches_the_estimator(self, monkeypatch):
        seen = []
        real = modeling.build_pipeline

        def spy(name, numeric, categorical, model_params=None, *args, **kwargs):
            if model_params and "random_state" in model_params:
                seen.append(model_params["random_state"])
            return real(name, numeric, categorical, model_params, *args, **kwargs)

        monkeypatch.setattr(modeling, "build_pipeline", spy)
        train(two_class(), [FOREST, BOOST])
        assert set(modeling.SEEDS_TO_CHECK[1:]) <= set(seen)

    def test_the_gap_is_to_the_runner_up(self):
        selection = train(two_class(), [FOREST, BOOST])["selection"]
        found = selection["seed_stability"]
        assert found["runner_up"] == selection["runner_up"] and found["gap"] is not None


class TestTheMessage:
    def selection(self, monkeypatch, seed_scores, boost=0.78):
        monkeypatch.setattr(tuning, "score_config", controlled({FOREST: 0.80, BOOST: boost}, seed_scores))
        return train(two_class(), [FOREST, BOOST])["selection"]

    def test_it_appears_when_the_seed_moves_the_score_at_least_as_much_as_the_lead(self, monkeypatch):
        found = self.selection(monkeypatch, {1: 0.79, 2: 0.81, 3: 0.775, 4: 0.80})["seed_stability"]
        assert found["spread"] == pytest.approx(0.035) and found["gap"] == pytest.approx(0.02)
        assert found["seed_dependent"] is True
        assert "depends on the seed, not on the data" in found["note"]

    def test_it_does_not_when_the_lead_is_bigger_than_the_seed_spread(self, monkeypatch):
        found = self.selection(monkeypatch, {1: 0.801, 2: 0.799, 3: 0.80, 4: 0.802})["seed_stability"]
        assert found["seed_dependent"] is False and found["note"] is None

    def test_a_spread_exactly_equal_to_the_gap_counts_as_dependent(self, monkeypatch):
        found = self.selection(monkeypatch, {1: 0.78, 2: 0.80, 3: 0.80, 4: 0.80})["seed_stability"]
        assert found["spread"] == found["gap"] and found["seed_dependent"] is True


class TestWhereItIsSaid:
    def card(self, result):
        return build_model_card(session_id="s", file_name="f.csv", result=result, summary="A.", interpretation="B.")

    def test_the_card_carries_the_warning_when_the_choice_is_the_seeds(self, monkeypatch):
        monkeypatch.setattr(tuning, "score_config", controlled({FOREST: 0.80, BOOST: 0.795}, {1: 0.79, 2: 0.81, 3: 0.80, 4: 0.80}))
        assert "depends on the seed, not on the data" in self.card(train(two_class(), [FOREST, BOOST]))

    def test_the_card_says_it_was_checked_when_it_is_not(self, monkeypatch):
        monkeypatch.setattr(tuning, "score_config", controlled({FOREST: 0.80, BOOST: 0.77}, {1: 0.801, 2: 0.799, 3: 0.80, 4: 0.80}))
        card = self.card(train(two_class(), [FOREST, BOOST]))
        assert "Refitted with 5 random seeds, the chosen model's fold score moved by 0.002" in card
        assert "so the choice is not the seed's doing" in card
