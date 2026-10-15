"""A processing limit on one training request.

Deep diagnostics were already skipped above 20,000 rows, but plain training, model choice and the loop had no limit
on rows x candidates x folds. Over the limit a request is now reduced -- model choice on a sample, then the dearest
families skipped, then refused -- and says what it reduced; within it, nothing changes.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.routes_model import TrainingFailedError  # noqa: F401  (the code list lives in its docstring)
from app.config import Settings, get_settings
from app.main import app
from app.services import improvement, model_card, modeling

LR, RF, GB = "logistic_regression", "random_forest_classifier", "gradient_boosting_classifier"
ALL = [LR, RF, GB]


class TestTheEstimate:
    def test_a_forest_and_boosting_are_dearer_per_row_than_a_linear_model(self):
        assert modeling.MODELS[LR].cost_weight == 1
        assert modeling.MODELS[RF].cost_weight > modeling.MODELS[GB].cost_weight > modeling.MODELS[LR].cost_weight
        assert modeling.MODELS["linear_regression"].cost_weight == 1

    def test_it_counts_one_fit_on_every_row_plus_the_folds_of_the_comparison(self):
        # 1,000 rows: five folds, no repeats above 2,000... below 2,000 the folds repeat three times.
        folds, repeats = modeling._selection_folds(1_000)
        assert (folds, repeats) == (5, 3)
        assert modeling.estimate_cost(1_000, [LR]) == 1 * (1_000 + 1_000 * 5 * 3)

    def test_more_than_20000_rows_uses_fewer_folds_and_no_repeats(self):
        assert modeling._selection_folds(20_001) == (3, 1)
        assert modeling._selection_folds(5_000) == (5, 1)

    def test_a_sample_for_the_comparison_costs_less_and_the_final_fit_does_not(self):
        full = modeling.estimate_cost(50_000, [RF])
        sampled = modeling.estimate_cost(50_000, [RF], selection_rows=10_000)
        assert sampled < full
        folds, repeats = modeling._selection_folds(50_000)
        assert (folds, repeats) == (3, 1)  # above 20,000 rows
        assert sampled == 40 * (50_000 + 10_000 * 3 * 1)

    def test_unknown_names_cost_nothing(self):
        assert modeling.estimate_cost(1_000, ["no_such_model"]) == 0


class TestThePlan:
    def test_within_the_limit_nothing_changes(self):
        plan = modeling.plan_within_limit(ALL, 20_000, limit=30_000_000, can_sample=True)
        assert plan["models"] == ALL and plan["selection_rows"] is None and plan["reductions"] == []
        assert modeling.resource_report(plan) is None

    @pytest.mark.parametrize("limit", [None, 0])
    def test_no_limit_means_no_reduction_however_big(self, limit):
        plan = modeling.plan_within_limit(ALL, 50_000_000, limit=limit, can_sample=True)
        assert plan["models"] == ALL and plan["reductions"] == []

    def test_exactly_at_the_limit_is_unchanged_and_one_more_row_is_not(self):
        rows = 100_000
        at = modeling.estimate_cost(rows, ALL)
        assert modeling.plan_within_limit(ALL, rows, limit=at, can_sample=True)["reductions"] == []
        assert modeling.plan_within_limit(ALL, rows, limit=at - 1, can_sample=True)["reductions"]

    def test_the_comparison_is_sampled_first_and_every_family_is_kept(self):
        plan = modeling.plan_within_limit(ALL, 200_000, limit=30_000_000, can_sample=True)
        assert plan["models"] == ALL
        assert 0 < plan["selection_rows"] < 200_000
        assert plan["estimated"] <= 30_000_000 < plan["estimated_before"]
        first = plan["reductions"][0]
        assert first["kind"] == "selection_rows" and first["from"] == 200_000 and first["to"] == plan["selection_rows"]
        assert "the test rows took no part" in first["reason"]

    def test_the_sample_is_never_below_the_minimum_unless_the_file_is(self):
        plan = modeling.plan_within_limit([LR, RF], 400_000, limit=40_000_000, can_sample=True)
        assert plan["selection_rows"] is None or plan["selection_rows"] >= modeling.MIN_SELECTION_ROWS

    def test_where_it_cannot_sample_the_dearest_family_goes_first(self):
        plan = modeling.plan_within_limit(ALL, 200_000, limit=30_000_000, can_sample=False)
        assert plan["selection_rows"] is None
        assert RF not in plan["models"] and LR in plan["models"]
        skipped = [r for r in plan["reductions"] if r["kind"] == "skipped_model"]
        assert skipped[0]["model"] == RF and "Random forest" in skipped[0]["reason"]
        assert plan["estimated"] <= 30_000_000

    def test_a_file_so_big_that_only_the_cheapest_model_fits_keeps_just_that(self):
        plan = modeling.plan_within_limit(ALL, 5_000_000, limit=30_000_000, can_sample=True)
        assert plan["models"] == [LR]
        assert [r["kind"] for r in plan["reductions"]].count("skipped_model") == 2

    def test_a_file_too_big_for_even_the_cheapest_is_refused_with_what_would_fit(self):
        with pytest.raises(modeling.ModelingError) as caught:
            modeling.plan_within_limit(ALL, 50_000_000, limit=30_000_000, can_sample=True)
        assert caught.value.code == "too_large"
        assert "50,000,000 training rows" in str(caught.value) and "rows would fit" in str(caught.value)
        assert "TRAINING_COST_LIMIT" in str(caught.value)

    def test_a_single_cheap_model_is_never_skipped(self):
        plan = modeling.plan_within_limit([LR], 5_000_000, limit=30_000_000, can_sample=True)
        assert plan["models"] == [LR]

    def test_it_is_deterministic(self):
        a = modeling.plan_within_limit(ALL, 150_000, limit=30_000_000, can_sample=True)
        assert a == modeling.plan_within_limit(ALL, 150_000, limit=30_000_000, can_sample=True)


class TestTheSample:
    def classification(self, n=6000, seed=0):
        rng = np.random.default_rng(seed)
        y = pd.Series(rng.choice(["a", "b", "rare"], n, p=[0.6, 0.38, 0.02]), index=np.arange(n) + 10)
        return pd.DataFrame({"x": rng.normal(size=n)}, index=y.index), y

    def test_a_file_under_the_cap_comes_back_whole(self):
        X, y = self.classification(1000)
        out = modeling.selection_sample(X, y, None, "classification", 5000)
        assert out[0] is X and out[3] == 1000

    def test_a_classification_sample_keeps_the_answers_in_proportion_and_enough_of_the_rare_one(self):
        X, y = self.classification()
        Xs, ys, _, rows = modeling.selection_sample(X, y, None, "classification", 1500)
        assert rows == len(Xs) == len(ys) and 1300 < rows < 1800
        shares = ys.value_counts(normalize=True)
        assert abs(shares["a"] - 0.6) < 0.05 and abs(shares["b"] - 0.38) < 0.05
        assert ys.value_counts()["rare"] >= 30  # kept enough to fold, whatever the proportion says
        assert set(Xs.index) == set(ys.index) and set(Xs.index) <= set(X.index)

    def test_a_grouped_sample_keeps_whole_groups(self):
        rng = np.random.default_rng(1)
        n = 6000
        groups = pd.Series(np.repeat(np.arange(600), 10), index=np.arange(n))
        X = pd.DataFrame({"x": rng.normal(size=n)}, index=groups.index)
        y = pd.Series(rng.choice(["a", "b"], n), index=groups.index)
        Xs, ys, gs, rows = modeling.selection_sample(X, y, groups, "classification", 1500)
        assert rows >= 1500
        for group in gs.unique():
            assert (gs == group).sum() == 10  # not one group cut in half
        assert len(Xs) == len(ys) == len(gs)

    def test_a_regression_sample_is_a_plain_random_sample(self):
        rng = np.random.default_rng(2)
        X = pd.DataFrame({"x": rng.normal(size=4000)})
        y = pd.Series(rng.normal(size=4000))
        Xs, ys, _, rows = modeling.selection_sample(X, y, None, "regression", 1000)
        assert rows == 1000 == len(Xs) == len(ys)

    def test_it_is_deterministic(self):
        X, y = self.classification()
        a = modeling.selection_sample(X, y, None, "classification", 1500)
        b = modeling.selection_sample(X, y, None, "classification", 1500)
        assert list(a[0].index) == list(b[0].index)


def frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "outcome": np.where(x + rng.normal(0, 1, n) > 0, "yes", "no")})


def train(df, limit, models=(LR, RF), **kwargs):
    return modeling.train_candidates(
        df, target_column="outcome", task="classification", model_names=list(models),
        include_deep_diagnostics=False, cost_limit=limit, **kwargs,
    )


class TestInATrain:
    def test_without_a_limit_there_is_no_report_and_the_choice_is_on_every_row(self):
        result = train(frame(1500), None)
        assert result["resource_limits"] is None
        assert result["selection"]["rows_used"] == result["selection"]["rows_available"]
        assert "a sample of" not in result["selection"]["chosen_on"]

    def test_within_a_generous_limit_the_result_is_exactly_the_unlimited_one(self):
        free, within = train(frame(1500), None), train(frame(1500), 10**12)
        assert within["resource_limits"] is None
        assert within["best_model"] == free["best_model"]
        assert [(r["model"], r["cv_score"]) for r in within["results"]] == [(r["model"], r["cv_score"]) for r in free["results"]]

    def test_over_the_limit_the_choice_is_made_on_a_sample_and_the_result_says_so(self, monkeypatch):
        monkeypatch.setattr(modeling, "MIN_SELECTION_ROWS", 300)
        result = train(frame(3000), 400_000)
        report = result["resource_limits"]
        assert report["reductions"][0]["kind"] == "selection_rows" and report["limit"] == 400_000
        used, available = result["selection"]["rows_used"], result["selection"]["rows_available"]
        assert used < available == result["split"]["train_rows"]
        assert f"a sample of {used:,} of the {available:,} training rows" in result["selection"]["chosen_on"]
        # Every candidate was still trained on every training row and scored on the held-out rows.
        assert {r["model"] for r in result["results"]} == {LR, RF}
        assert all(r["metrics"] for r in result["results"])

    def test_a_family_is_skipped_where_sampling_is_not_possible(self):
        df = frame(600).assign(when=pd.date_range("2022-01-01", periods=600, freq="D").astype(str))
        result = train(df, 120_000, split_strategy="time", time_column="when")
        skipped = [r for r in result["resource_limits"]["reductions"] if r["kind"] == "skipped_model"]
        assert skipped and skipped[0]["model"] == RF
        assert [r["model"] for r in result["results"]] == [LR]

    def test_a_file_too_big_for_the_limit_is_refused_with_a_code(self):
        with pytest.raises(modeling.ModelingError) as caught:
            train(frame(600), 100)
        assert caught.value.code == "too_large"

    def test_the_reductions_reach_the_cautions_and_the_card(self, monkeypatch):
        monkeypatch.setattr(modeling, "MIN_SELECTION_ROWS", 300)
        result = train(frame(3000), 400_000)
        best = next(r for r in result["results"] if r["model"] == result["best_model"])
        texts = [c["text"] for c in model_card.caution_items(result, best)]
        assert any("Models were compared on a sample of" in t for t in texts)
        card = model_card.build_model_card(session_id="s", file_name="f.csv", result=result, summary="A.", interpretation="B.")
        assert "Models were compared on a sample of" in card


class TestInTheLoop:
    def test_dearer_families_are_not_tried_and_the_search_gets_a_smaller_budget(self):
        found = improvement.improve(
            frame(1500), target_column="outcome", task="classification",
            model_names=[LR, RF], budget=24, cost_limit=300_000,
        )
        limits = found["resource_limits"]
        kinds = [r["kind"] for r in limits["reductions"]]
        assert "skipped_model" in kinds and "search_budget" in kinds
        assert found["best"]["model"] == LR
        assert limits["limit"] == 300_000

    def test_within_the_limit_the_loop_is_unchanged(self):
        found = improvement.improve(
            frame(300), target_column="outcome", task="classification",
            model_names=[LR], budget=4, cost_limit=10**12,
        )
        assert found["resource_limits"] is None

    def test_a_file_too_big_even_for_the_simplest_model_is_refused(self):
        with pytest.raises(modeling.ModelingError) as caught:
            improvement.improve(
                frame(300), target_column="outcome", task="classification", model_names=[LR], budget=4, cost_limit=10,
            )
        assert caught.value.code == "too_large"


class TestSettingsAndRoutes:
    def session(self, rows: int):
        client = TestClient(app)
        buffer = io.StringIO()
        frame(rows).to_csv(buffer, index=False)
        sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
        goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
        assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
        return client, sid

    def limited(self, limit: int):
        app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(update={"training_cost_limit": limit})

    def teardown_method(self):
        app.dependency_overrides.pop(get_settings, None)

    def test_the_default_limit_is_what_the_docs_say(self):
        assert Settings().training_cost_limit == 30_000_000

    def test_a_train_over_the_limit_is_reduced_and_says_so_in_its_cautions(self, monkeypatch):
        monkeypatch.setattr(modeling, "MIN_SELECTION_ROWS", 300)
        client, sid = self.session(3000)
        self.limited(400_000)
        body = client.post(
            f"/api/model/{sid}/train",
            json={"model_names": [LR, RF], "include_deep_diagnostics": False},
        ).json()
        assert body["resource_limits"]["reductions"][0]["kind"] == "selection_rows"
        assert any(c["id"].startswith("resource_limit_selection_rows") for c in body["cautions"])

    def test_a_train_that_cannot_fit_is_refused_with_the_code_the_page_acts_on(self):
        client, sid = self.session(600)
        self.limited(100)
        response = client.post(f"/api/model/{sid}/train", json={"include_deep_diagnostics": False})
        assert response.status_code == 422 and response.json()["code"] == "too_large"

    def test_with_the_limit_off_a_train_is_not_reduced(self):
        client, sid = self.session(600)
        self.limited(0)
        body = client.post(f"/api/model/{sid}/train", json={"model_names": [LR], "include_deep_diagnostics": False}).json()
        assert body["resource_limits"] is None
