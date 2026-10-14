"""A median baseline and a relative error for regression.

The regression baseline always predicted the training average. On a skewed target -- spend, price --
a few huge values drag the average away from where most rows sit, so the median is the stronger naive
guess for the typical miss, and beating only the average flatters a model. An absolute MAE with no
scale is also hard to read: is 1,995 good? It is as a fraction of what just guessing the middle
would miss by.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import modeling
from app.services.model_card import build_model_card

client = TestClient(app)


def skewed(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Spend-like: lognormal, so the average sits well above the typical value."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "z": rng.normal(0, 1, n), "y": np.exp(1.0 + 0.8 * x + rng.normal(0, 0.5, n)) * 1000})


def symmetric(n: int = 800, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "z": rng.normal(0, 1, n), "y": 50 + 10 * x + rng.normal(0, 3, n)})


def train(df, **kwargs):
    return modeling.train_candidates(
        df, target_column="y", task="regression", model_names=["linear_regression"],
        include_deep_diagnostics=False, **kwargs,
    )


class TestTwoBaselines:
    def test_both_the_average_and_the_median_have_all_three_scores(self):
        base = modeling.baseline_score(skewed()["y"], skewed(seed=2)["y"], "regression")
        median = base["others"][0]
        for row in (base, median):
            assert {"r2", "mae", "rmse", "mae_vs_baseline"} <= set(row)
        assert "average" in base["strategy"] and "median" in median["strategy"]

    def test_on_a_skewed_target_the_median_beats_the_average_on_mae(self):
        base = modeling.baseline_score(skewed()["y"], skewed(seed=2)["y"], "regression")
        assert base["others"][0]["mae"] < base["mae"]

    def test_on_a_symmetric_target_they_are_nearly_the_same(self):
        base = modeling.baseline_score(symmetric()["y"], symmetric(seed=3)["y"], "regression")
        assert base["mae"] == pytest.approx(base["others"][0]["mae"], rel=0.03)

    def test_the_median_guess_is_the_scale_so_it_is_one_by_definition(self):
        base = modeling.baseline_score(skewed()["y"], skewed(seed=2)["y"], "regression")
        assert base["others"][0]["mae_vs_baseline"] == 1.0
        assert base["mae_vs_baseline"] == pytest.approx(base["mae"] / base["others"][0]["mae"], abs=1e-4)
        assert base["mae_vs_baseline"] > 1.0  # the average misses by more than the median does

    def test_the_scores_are_those_of_a_constant_guess_from_the_training_rows(self):
        train_y = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])  # mean 22, median 3
        test_y = pd.Series([2.0, 4.0, 6.0, 8.0])
        base = modeling.baseline_score(train_y, test_y, "regression")
        assert base["mae"] == pytest.approx(np.mean(np.abs(test_y - 22.0)), abs=1e-3)
        assert base["others"][0]["mae"] == pytest.approx(np.mean(np.abs(test_y - 3.0)), abs=1e-3)
        assert base["others"][0]["rmse"] == pytest.approx(np.sqrt(np.mean((test_y - 3.0) ** 2)), abs=1e-3)

    def test_the_test_rows_never_choose_the_guess(self):
        """Fitted on the training rows only: swap in test answers from a different distribution and
        the guess, and so the strategy text, is unchanged."""
        train_y = skewed()["y"]
        one = modeling.baseline_score(train_y, skewed(seed=2)["y"], "regression")
        other = modeling.baseline_score(train_y, skewed(seed=3)["y"] * 5, "regression")
        assert one["strategy"] == other["strategy"] and one["others"][0]["strategy"] == other["others"][0]["strategy"]

    def test_a_target_with_no_spread_around_its_median_has_no_scale(self):
        constant = pd.Series([5.0] * 50)
        base = modeling.baseline_score(constant, constant, "regression")
        assert base["others"][0]["mae"] == 0.0
        assert "mae_vs_baseline" not in base and "mae_vs_baseline" not in base["others"][0]

    def test_classification_is_untouched(self):
        labels = pd.Series(["a", "b"] * 30)
        assert "mae_vs_baseline" not in modeling.baseline_score(labels, labels, "classification")


class TestEachModelsRelativeError:
    def test_it_is_the_models_mae_over_the_median_guesses(self):
        result = train(skewed())
        metrics = result["results"][0]["metrics"]
        median_mae = result["baseline"]["others"][0]["mae"]
        assert metrics["mae_vs_baseline"] == pytest.approx(metrics["mae"] / median_mae, abs=1e-4)

    def test_a_model_that_learned_something_is_below_one(self):
        assert train(skewed())["results"][0]["metrics"]["mae_vs_baseline"] < 0.9

    def test_a_model_that_learned_nothing_is_about_one(self):
        rng = np.random.default_rng(4)
        noise = pd.DataFrame({"x": rng.normal(0, 1, 800), "y": np.exp(rng.normal(1, 0.5, 800)) * 1000})
        assert train(noise)["results"][0]["metrics"]["mae_vs_baseline"] == pytest.approx(1.0, abs=0.15)

    def test_every_candidate_gets_one(self):
        result = modeling.train_candidates(
            skewed(), target_column="y", task="regression",
            model_names=["linear_regression", "random_forest_regressor"], include_deep_diagnostics=False,
        )
        assert all("mae_vs_baseline" in r["metrics"] for r in result["results"])

    def test_the_verdict_is_unchanged_by_it(self):
        result = train(skewed())
        again = modeling.assess_result(
            result["results"][0]["metrics"], {}, "regression", test_rows=result["split"]["test_rows"],
            intervals=result["results"][0]["metrics_intervals"],
        )
        assert result["assessment"] == again

    def test_a_classification_model_does_not_get_one(self):
        rng = np.random.default_rng(5)
        x = rng.normal(0, 1, 400)
        df = pd.DataFrame({"x": x, "y": np.where(x + rng.normal(0, 1, 400) > 0, "yes", "no")})
        result = modeling.train_candidates(
            df, target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False,
        )
        assert "mae_vs_baseline" not in result["results"][0]["metrics"]


class TestInTheCardAndTheApi:
    def card(self, result):
        return build_model_card(session_id="s", file_name="f.csv", result=result, summary="A.", interpretation="B.")

    def test_the_card_has_the_column_both_baselines_and_the_explanation(self):
        card = self.card(train(skewed()))
        header = next(line for line in card.splitlines() if line.startswith("| Model |") and "median guess" in line)
        assert "Error vs. the median guess" in header
        rows = [line for line in card.splitlines() if line.startswith("| _No model at all")]
        assert len(rows) == 2 and "average" in rows[0] and "median" in rows[1]
        for row in rows:
            assert "—" not in row, row
        assert "**Error vs. the median guess** — Its typical miss as a fraction" in card

    def test_a_classification_card_has_no_such_column(self):
        rng = np.random.default_rng(6)
        x = rng.normal(0, 1, 400)
        df = pd.DataFrame({"x": x, "y": np.where(x + rng.normal(0, 1, 400) > 0, "yes", "no")})
        result = modeling.train_candidates(
            df, target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False,
        )
        assert "median guess" not in self.card(result)

    def _session(self, df):
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        session_id = client.post(
            "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
        ).json()["session_id"]
        goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "y", "task_type": "regression"}
        assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
        return session_id

    def test_the_train_response_carries_both_baselines_and_the_ratio(self):
        session_id = self._session(skewed())
        body = client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["linear_regression"], "include_deep_diagnostics": False},
        ).json()
        assert "mae_vs_baseline" in body["results"][0]["metrics"]
        assert len(body["baseline"]["others"]) == 1 and "rmse" in body["baseline"]

    def test_the_sealed_final_evaluation_reports_it_too(self):
        session_id = self._session(skewed(1200))
        client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["linear_regression"], "include_deep_diagnostics": False, "sealed_holdout": True},
        )
        opened = client.post(f"/api/model/{session_id}/finalise").json()
        assert opened["metrics"]["mae_vs_baseline"] == pytest.approx(
            opened["metrics"]["mae"] / opened["baseline"]["others"][0]["mae"], abs=1e-4
        )
