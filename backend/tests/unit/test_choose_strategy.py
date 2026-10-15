"""Choosing the search strategy automatically, and letting the user override it.

The rule is a pure function of the model, the number of training rows and the budget, so it is tested at each
boundary. The choice and its reason are in the trail and the run history; an explicit choice is respected and
recorded as the user's.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import improvement, model_store, tuning

FOREST = "random_forest_classifier"  # 288 settings
LOGISTIC = "logistic_regression"  # 14 settings
RIDGE = "linear_regression"  # 5 settings


class TestTheRule:
    def test_many_settings_a_real_budget_and_many_rows_get_halving(self):
        name, reason = tuning.choose_strategy(FOREST, tuning.HALVING_MIN_ROWS, tuning.HALVING_MIN_BUDGET)
        assert name == "halving"
        assert f"{tuning.HALVING_MIN_ROWS:,} training rows" in reason and "288 possible settings" in reason

    @pytest.mark.parametrize("rows", [tuning.HALVING_MIN_ROWS - 1, 100, 0])
    def test_just_below_the_row_threshold_is_random(self, rows):
        name, reason = tuning.choose_strategy(FOREST, rows, tuning.HALVING_MIN_BUDGET)
        assert name == "random" and "training rows is below" in reason

    def test_just_below_the_budget_threshold_is_random(self):
        name, reason = tuning.choose_strategy(FOREST, 50_000, tuning.HALVING_MIN_BUDGET - 1)
        assert name == "random" and f"a budget of {tuning.HALVING_MIN_BUDGET - 1}" in reason

    def test_a_small_space_is_random_whatever_else_is_true(self):
        for model, size in ((RIDGE, 5),):
            name, reason = tuning.choose_strategy(model, 1_000_000, 120)
            assert name == "random" and f"only {size} settings" in reason

    def test_the_space_boundary_is_the_documented_one(self):
        # 14 settings (logistic) clears the 12 needed; 5 (ridge) does not.
        assert tuning.choose_strategy(LOGISTIC, 50_000, 24)[0] == "halving"
        assert tuning.choose_strategy(RIDGE, 50_000, 24)[0] == "random"

    def test_a_model_with_no_space_is_random(self):
        name, reason = tuning.choose_strategy("no_such_model", 50_000, 24)
        assert name == "random" and "only 0 settings" in reason

    def test_it_is_pure(self):
        assert tuning.choose_strategy(FOREST, 9_000, 24) == tuning.choose_strategy(FOREST, 9_000, 24)

    def test_the_thresholds_are_what_the_docs_say(self):
        assert (tuning.HALVING_MIN_ROWS, tuning.HALVING_MIN_SETTINGS, tuning.HALVING_MIN_BUDGET) == (5_000, 12, 12)


def frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "outcome": np.where(x + rng.normal(size=n) > 0, "yes", "no")})


def settings_round(found: dict) -> dict:
    return next(r for r in found["rounds"] if r["name"] == "hyperparameters")


class TestInTheLoop:
    def test_auto_on_a_small_file_picks_random_and_says_why(self):
        found = improvement.improve(
            frame(240), target_column="outcome", task="classification",
            model_names=[LOGISTIC], budget=12,
        )
        detail = settings_round(found)["detail"]
        assert detail["strategy"] == "random" and "training rows is below" in detail["strategy_reason"]
        assert found["search_strategy"] == "auto" and found["search_strategies"] == ["random"]

    def test_auto_picks_halving_when_the_rule_says_so(self, monkeypatch):
        monkeypatch.setattr(tuning, "HALVING_MIN_ROWS", 100)
        found = improvement.improve(
            frame(240), target_column="outcome", task="classification",
            model_names=[LOGISTIC], budget=12,
        )
        detail = settings_round(found)["detail"]
        assert detail["strategy"] == "halving" and "cheaper to rank them on a few folds" in detail["strategy_reason"]
        assert found["search_strategies"] == ["halving"] and detail["rungs"]

    def test_an_explicit_choice_beats_the_rule_and_is_recorded_as_the_users(self, monkeypatch):
        monkeypatch.setattr(tuning, "HALVING_MIN_ROWS", 10_000_000)  # the rule would say random
        found = improvement.improve(
            frame(240), target_column="outcome", task="classification",
            model_names=[LOGISTIC], budget=12, search_strategy="halving",
        )
        detail = settings_round(found)["detail"]
        assert detail["strategy"] == "halving" and detail["strategy_reason"] == "you chose halving search"
        assert found["search_strategy"] == "halving"

    def test_an_explicit_random_is_respected_even_where_the_rule_would_halve(self, monkeypatch):
        monkeypatch.setattr(tuning, "HALVING_MIN_ROWS", 100)
        found = improvement.improve(
            frame(240), target_column="outcome", task="classification",
            model_names=[LOGISTIC], budget=12, search_strategy="random",
        )
        assert settings_round(found)["detail"]["strategy"] == "random"
        assert found["search_strategies"] == ["random"]

    def test_a_halving_search_that_fell_back_gives_its_own_reason_not_the_rules(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=240)
        regression = pd.DataFrame({"x": x, "z": rng.normal(size=240), "outcome": 2 * x + rng.normal(size=240)})
        found = improvement.improve(
            regression, target_column="outcome", task="regression",
            model_names=[RIDGE], budget=12, search_strategy="halving",
        )
        detail = settings_round(found)["detail"]
        # Ridge has five settings: nothing to halve over, so it was searched at random and says so.
        assert detail["strategy"] == "random"
        assert detail["strategy_reason"].startswith("Searched at random: only 5 settings to try")
        assert found["search_strategies"] == ["random"]


class TestTheHistory:
    def test_the_run_history_keeps_the_request_and_what_was_used(self):
        result = {"results": [], "split": {}, "improvement": {
            "search_strategy": "auto", "search_strategies": ["halving", "random"], "stopped_because": "no_further_gains",
            "settings": {}}}
        summary = model_store.summarise(result)
        assert summary["search_strategy"] == "auto" and summary["search_strategies"] == ["halving", "random"]

    def test_an_untuned_run_has_none(self):
        summary = model_store.summarise({"results": [], "split": {}})
        assert summary["search_strategy"] is None and summary["search_strategies"] == []


class TestOverHttp:
    def session(self):
        client = TestClient(app)
        buffer = io.StringIO()
        frame(240).to_csv(buffer, index=False)
        sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
        goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
        assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
        return client, sid

    def test_by_default_the_strategy_is_chosen_and_the_trail_says_so(self):
        client, sid = self.session()
        body = client.post(
            f"/api/model/{sid}/improve",
            json={"model_names": [LOGISTIC], "include_deep_diagnostics": False, "budget": 12},
        ).json()
        assert body["improvement"]["search_strategy"] == "auto"
        assert body["improvement"]["search_strategies"] == ["random"]

    def test_the_override_is_respected_and_recorded(self):
        client, sid = self.session()
        body = client.post(
            f"/api/model/{sid}/improve",
            json={"model_names": [LOGISTIC], "include_deep_diagnostics": False, "budget": 12, "search_strategy": "halving"},
        ).json()
        assert body["improvement"]["search_strategy"] == "halving"
        assert body["improvement"]["search_strategies"] == ["halving"]
        history = client.get(f"/api/model/{sid}/history").json()["runs"]
        assert history[0]["search_strategy"] == "halving" and history[0]["search_strategies"] == ["halving"]

    def test_an_unknown_strategy_is_refused(self):
        client, sid = self.session()
        assert client.post(f"/api/model/{sid}/improve", json={"search_strategy": "magic"}).status_code == 422
