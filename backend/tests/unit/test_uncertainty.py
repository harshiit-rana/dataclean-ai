"""Confidence intervals for the headline held-out metrics.

"AUC 0.78" on 60 rows and on 6,000 look identical on the page. A percentile bootstrap over the
test rows gives the plausible range. The tests that matter: the point estimates the bootstrap
works from are exactly the ones `modeling.evaluate` reports (an interval around a number the page
does not show is worse than none); a 95% interval really does contain the truth about 95% of the
time on a problem where the truth is known; a grouped split resamples entities, because resampling
the rows of one customer one by one would claim certainty the data does not hold.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.stats import norm

from app.config import get_settings
from app.main import app
from app.services import modeling, uncertainty
from app.services.session_store import session_dir

client = TestClient(app)


def fitted_binary(n: int = 500, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    y = pd.Series(np.where(X.a + rng.normal(0, 1, n) > 0, "yes", "no"))
    pipeline = modeling.build_pipeline("logistic_regression", ["a", "b"], []).fit(X, y)
    scores = pipeline.predict_proba(X)[:, list(pipeline.classes_).index("yes")]
    return X, y, pipeline, scores


class TestThePointEstimatesAreWhatTheAppReports:
    def test_two_class_metrics_match_evaluate_exactly(self):
        X, y, pipeline, scores = fitted_binary()
        reported = modeling.evaluate(pipeline, X, y, "classification", "yes")
        ours = uncertainty.point_estimates(y, pipeline.predict(X), scores, "classification", "yes")
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert round(ours[key], 4) == reported[key], key

    def test_the_other_class_can_be_the_positive_one(self):
        X, y, pipeline, _ = fitted_binary()
        scores = pipeline.predict_proba(X)[:, list(pipeline.classes_).index("no")]
        reported = modeling.evaluate(pipeline, X, y, "classification", "no")
        ours = uncertainty.point_estimates(y, pipeline.predict(X), scores, "classification", "no")
        for key in ("precision", "recall", "f1", "roc_auc"):
            assert round(ours[key], 4) == reported[key], key

    def test_ties_in_the_scores_are_counted_as_half(self):
        """A tree model gives many tied probabilities; AUC with ties is where hand-rolled versions go wrong."""
        rng = np.random.default_rng(3)
        y = pd.Series(np.where(rng.random(400) < 0.4, "yes", "no"))
        scores = np.round(rng.random(400) * 0.5 + (y == "yes") * 0.2, 1)  # heavy ties
        from sklearn.metrics import roc_auc_score

        ours = uncertainty.point_estimates(y, np.where(scores > 0.3, "yes", "no"), scores, "classification", "yes")
        assert ours["roc_auc"] == pytest.approx(roc_auc_score(y == "yes", scores), abs=1e-9)

    def test_multi_class_metrics_match_evaluate(self):
        rng = np.random.default_rng(4)
        X = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500)})
        y = pd.Series(np.digitize(X.a + rng.normal(0, 1, 500), [-0.5, 0.5]).astype(str))
        pipeline = modeling.build_pipeline("logistic_regression", ["a", "b"], []).fit(X, y)
        reported = modeling.evaluate(pipeline, X, y, "classification")
        ours = uncertainty.point_estimates(y, pipeline.predict(X), None, "classification")
        for key in ("accuracy", "precision", "recall", "f1"):
            assert round(ours[key], 4) == reported[key], key
        assert "roc_auc" not in ours  # no AUC for three classes

    def test_regression_metrics_match_evaluate(self):
        rng = np.random.default_rng(5)
        X = pd.DataFrame({"a": rng.normal(0, 1, 400), "b": rng.normal(0, 1, 400)})
        y = X.a * 3 + rng.normal(0, 1, 400)
        pipeline = modeling.build_pipeline("linear_regression", ["a", "b"], []).fit(X, y)
        reported = modeling.evaluate(pipeline, X, y, "regression")
        ours = uncertainty.point_estimates(y, pipeline.predict(X), None, "regression")
        for key in ("r2", "mae", "rmse"):
            assert round(ours[key], 4) == reported[key], key


class TestTheIntervalsAreHonest:
    def test_a_95_percent_interval_contains_the_true_auc_about_95_percent_of_the_time(self):
        """Scores are N(1, 1) for cases and N(0, 1) for non-cases, so the true AUC is
        Phi(1 / sqrt(2)). Draw many test sets of 300 rows and count how often the interval
        contains it. A percentile bootstrap undercovers a little on small samples; the bar is
        'close to nominal', not 'exactly'."""
        true_auc = float(norm.cdf(1 / np.sqrt(2)))
        covered, trials = 0, 120
        for seed in range(trials):
            rng = np.random.default_rng(1000 + seed)
            y = np.where(rng.random(300) < 0.5, "yes", "no")
            scores = rng.normal(0, 1, 300) + (y == "yes") * 1.0
            found = uncertainty.metric_intervals(
                y, np.where(scores > 0.5, "yes", "no"), scores, "classification", "yes",
                resamples=300, seed=seed,
            )["metrics"]["roc_auc"]
            covered += found["low"] <= true_auc <= found["high"]
        assert 0.88 <= covered / trials <= 1.0, f"coverage was {covered / trials:.2f}"

    def test_the_interval_brackets_the_point_estimate(self):
        X, y, pipeline, scores = fitted_binary()
        estimates = uncertainty.point_estimates(y, pipeline.predict(X), scores, "classification", "yes")
        found = uncertainty.metric_intervals(y, pipeline.predict(X), scores, "classification", "yes")
        for key, interval in found["metrics"].items():
            assert interval["low"] <= estimates[key] <= interval["high"], key

    def test_fewer_rows_give_a_wider_interval(self):
        """The reason to print it at all: 60 rows and 6,000 must not look alike."""
        rng = np.random.default_rng(2)

        def width(rows):
            y = np.where(rng.random(rows) < 0.5, "yes", "no")
            scores = rng.normal(0, 1, rows) + (y == "yes") * 1.0
            found = uncertainty.metric_intervals(
                y, np.where(scores > 0.5, "yes", "no"), scores, "classification", "yes", resamples=400
            )["metrics"]["roc_auc"]
            return found["high"] - found["low"]

        assert width(60) > 2 * width(6000)

    def test_a_regression_interval_for_r2_is_wide_on_a_small_set(self):
        rng = np.random.default_rng(6)
        truth = rng.normal(0, 1, 40)
        predicted = truth * 0.3 + rng.normal(0, 1, 40)
        found = uncertainty.metric_intervals(truth, predicted, None, "regression")["metrics"]["r2"]
        assert found["high"] - found["low"] > 0.3

    def test_it_is_deterministic_for_a_fixed_seed(self):
        X, y, pipeline, scores = fitted_binary()
        args = (y, pipeline.predict(X), scores, "classification", "yes")
        assert uncertainty.metric_intervals(*args, seed=7) == uncertainty.metric_intervals(*args, seed=7)

    def test_a_different_seed_gives_a_slightly_different_one(self):
        X, y, pipeline, scores = fitted_binary()
        args = (y, pipeline.predict(X), scores, "classification", "yes")
        assert uncertainty.metric_intervals(*args, seed=1) != uncertainty.metric_intervals(*args, seed=2)


class TestWholeGroupsAreResampled:
    @staticmethod
    def _clustered(groups: int = 40, per: int = 10, seed: int = 8):
        """Every row of a group shares one answer and nearly one score: forty independent facts,
        not four hundred."""
        rng = np.random.default_rng(seed)
        label = np.repeat(rng.random(groups) < 0.5, per)
        effect = np.repeat(rng.normal(0, 1, groups), per)
        scores = effect + label * 0.8 + rng.normal(0, 0.05, groups * per)
        y = np.where(label, "yes", "no")
        called = np.where(scores > 0.4, "yes", "no")
        return y, called, scores, np.repeat(np.arange(groups), per)

    def test_a_grouped_interval_is_wider_than_a_row_level_one_on_the_same_data(self):
        y, called, scores, groups = self._clustered()
        by_rows = uncertainty.metric_intervals(y, called, scores, "classification", "yes", resamples=500)
        by_groups = uncertainty.metric_intervals(
            y, called, scores, "classification", "yes", groups=groups, resamples=500
        )
        width = lambda found: found["metrics"]["accuracy"]["high"] - found["metrics"]["accuracy"]["low"]  # noqa: E731
        assert by_rows["by"] == "rows" and by_groups["by"] == "groups"
        assert width(by_groups) > 1.5 * width(by_rows)

    def test_every_row_of_a_drawn_group_comes_along(self):
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(30), 7)
        weights = uncertainty._weights(rng, len(groups), 20, None, groups)
        for row in weights:
            per_group = [set(row[groups == g]) for g in range(30)]
            assert all(len(counts) == 1 for counts in per_group)  # one draw count per group


class TestARareClassCannotVanish:
    def test_five_cases_in_two_hundred_rows_still_give_a_recall_and_an_auc(self):
        rng = np.random.default_rng(11)
        y = np.array(["no"] * 195 + ["yes"] * 5)
        scores = rng.random(200) * 0.5 + (y == "yes") * 0.4
        called = np.where(scores > 0.5, "yes", "no")
        found = uncertainty.metric_intervals(y, called, scores, "classification", "yes", resamples=300)
        assert {"recall", "roc_auc", "precision", "f1"} <= set(found["metrics"])
        assert not any(np.isnan(v["low"]) or np.isnan(v["high"]) for v in found["metrics"].values())

    def test_each_resample_keeps_exactly_the_classes_the_test_set_has(self):
        rng = np.random.default_rng(12)
        strata = np.array([0] * 90 + [1] * 10)
        weights = uncertainty._weights(rng, 100, 50, strata, None)
        assert (weights[:, :90].sum(axis=1) == 90).all()
        assert (weights[:, 90:].sum(axis=1) == 10).all()


class TestCost:
    def test_a_large_test_set_takes_well_under_the_budget(self):
        rng = np.random.default_rng(13)
        rows = 20_000
        y = np.where(rng.random(rows) < 0.3, "yes", "no")
        scores = rng.normal(0, 1, rows) + (y == "yes") * 0.8
        started = time.perf_counter()
        found = uncertainty.metric_intervals(y, np.where(scores > 0.4, "yes", "no"), scores, "classification", "yes")
        elapsed = time.perf_counter() - started
        assert elapsed < 3, f"20,000 rows took {elapsed:.1f}s"  # the target is 1s; the bound is loose for slow runners
        assert found["resamples"] >= uncertainty.MIN_RESAMPLES

    def test_the_resample_count_shrinks_as_the_set_grows_but_never_below_the_floor(self):
        assert uncertainty.resample_count(100) == 1000
        assert uncertainty.resample_count(20_000) == uncertainty.MIN_RESAMPLES
        assert uncertainty.resample_count(10**7) == uncertainty.MIN_RESAMPLES


# ------------------------------------------------------------------ in training


def learnable(n: int = 600, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    df = pd.DataFrame({"x": x.round(4), "z": rng.normal(0, 1, n).round(4)})
    df["y"] = np.where(x + rng.normal(0, 1, n) > 0, "yes", "no")
    return df


def train(df, **kwargs):
    return modeling.train_candidates(
        df, target_column="y", task="classification",
        model_names=["logistic_regression", "random_forest_classifier"],
        include_deep_diagnostics=False, **kwargs,
    )


class TestInTraining:
    def test_the_chosen_model_has_an_interval_for_every_headline_metric(self):
        result = train(learnable())
        chosen = next(r for r in result["results"] if r["model"] == result["best_model"])
        intervals = chosen["metrics_intervals"]
        assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= set(intervals["metrics"])
        assert intervals["level"] == 0.95 and intervals["by"] == "rows"
        for key, found in intervals["metrics"].items():
            assert found["low"] <= chosen["metrics"][key] <= found["high"], key

    def test_other_candidates_keep_point_estimates_with_their_fold_spread(self):
        result = train(learnable())
        others = [r for r in result["results"] if r["model"] != result["best_model"]]
        assert others and all(not r.get("metrics_intervals") for r in others)
        assert all(r["cv_std"] is not None for r in others)

    def test_a_regression_result_has_r2_mae_and_rmse_intervals(self):
        rng = np.random.default_rng(9)
        df = pd.DataFrame({"x": rng.normal(0, 1, 400)})
        df["price"] = df["x"] * 3 + rng.normal(0, 1, 400)
        result = modeling.train_candidates(
            df, target_column="price", task="regression", model_names=["linear_regression"],
            include_deep_diagnostics=False,
        )
        assert set(result["results"][0]["metrics_intervals"]["metrics"]) == {"r2", "mae", "rmse"}

    def test_a_test_set_too_small_to_judge_gets_no_interval(self):
        """The verdict already says the test set is too small; a wide interval beside it would
        look like more of an answer than there is."""
        small = learnable(120)
        result = train(small)
        assert result["results"][0]["metrics_intervals"] is None
        assert result["assessment"]["verdict"] == "not_enough_test_data"

    def test_a_grouped_split_resamples_whole_groups(self):
        rng = np.random.default_rng(5)
        customers = 120
        base = rng.normal(0, 1, customers)
        df = pd.DataFrame(
            {
                "customer": np.repeat([f"c{i:03d}" for i in range(customers)], 8),
                "x": (np.repeat(base, 8) + rng.normal(0, 0.4, customers * 8)).round(4),
            }
        )
        df["y"] = np.repeat(np.where(base + rng.normal(0, 0.8, customers) > 0, "yes", "no"), 8)
        result = train(df, split_strategy="group", group_column="customer")
        chosen = next(r for r in result["results"] if r["model"] == result["best_model"])
        assert chosen["metrics_intervals"]["by"] == "groups"

    def test_the_run_history_records_the_primary_metrics_interval(self):
        from app.services import model_store

        result = train(learnable())
        entry = model_store.summarise(model_store.strip_pipelines(result))
        chosen = next(r for r in result["results"] if r["model"] == result["best_model"])
        expected = chosen["metrics_intervals"]["metrics"][result["primary_metric"]]
        assert entry["metric_interval"] == {"metric": result["primary_metric"], **expected}

    def test_a_pipeline_that_cannot_predict_does_not_sink_the_run(self):
        class Broken:
            def predict(self, _X):
                raise RuntimeError("boom")

        assert modeling.interval_report(Broken(), learnable(), learnable()["y"], "classification") is None


def _session(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    session_id = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "y", "task_type": "auto"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


class TestOverTheApi:
    FAST = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}

    def test_the_result_carries_the_intervals_and_the_card_prints_them(self):
        session_id = _session(learnable())
        body = client.post(f"/api/model/{session_id}/train", json=self.FAST).json()
        chosen = body["results"][0]
        assert chosen["metrics_intervals"]["resamples"] >= 300

        card = client.get(f"/api/model/{session_id}/card").json()["markdown"]
        from app.services.model_card import _fmt

        bounds = chosen["metrics_intervals"]["metrics"]["roc_auc"]
        assert f"({_fmt(bounds['low'])}–{_fmt(bounds['high'])})" in card
        assert "Figures in brackets are the middle 95%" in card

    def test_the_history_entry_has_the_interval(self):
        session_id = _session(learnable())
        client.post(f"/api/model/{session_id}/train", json=self.FAST)
        run = client.get(f"/api/model/{session_id}/history").json()["runs"][0]
        assert run["metric_interval"]["metric"] == "roc_auc"
        assert run["metric_interval"]["low"] < run["metric_interval"]["high"]

    def test_the_sealed_evaluation_has_its_own_interval(self):
        session_id = _session(learnable(800))
        client.post(f"/api/model/{session_id}/train", json={**self.FAST, "sealed_holdout": True})
        opened = client.post(f"/api/model/{session_id}/finalise").json()
        assert opened["metrics_intervals"]["metrics"]["roc_auc"]["low"] < opened["metrics"]["roc_auc"]
        again = client.post(f"/api/model/{session_id}/finalise").json()
        assert again["metrics_intervals"] == opened["metrics_intervals"]
