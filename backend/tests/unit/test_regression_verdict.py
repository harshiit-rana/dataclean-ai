"""A regression verdict that depends on how many rows it was judged on.

The classification verdict compares AUC with what luck could produce on a test set of that size.
The regression verdict used fixed R² cut-offs -- under 0.05 is guessing, under 0.3 is marginal --
for 35 test rows and for 35,000 alike. It now stands on the lower end of the R² interval from the
bootstrap over the test rows: the interval is wide on a small set and tight on a large one, so the
same R² can mean different things, which is the point.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import modeling, uncertainty

client = TestClient(app)


def interval(low: float, high: float) -> dict:
    return {"level": 0.95, "resamples": 1000, "by": "rows", "metrics": {"r2": {"low": low, "high": high}}}


def verdict(r2: float, low: float, high: float, rows: int = 200) -> dict:
    return modeling.assess_result(
        {"r2": r2, "mae": 1.0, "rmse": 1.2}, {"r2": 0.0}, "regression",
        test_rows=rows, intervals=interval(low, high),
    )


class TestTheVerdictStandsOnTheLowerEndOfTheInterval:
    def test_a_range_that_includes_zero_is_no_better_than_guessing(self):
        found = verdict(0.18, -0.10, 0.45, rows=40)
        assert found["verdict"] == "no_better_than_guessing"

    def test_even_a_respectable_point_estimate_is_not_enough_on_a_small_set(self):
        """R² of 0.35 would have been 'useful' under the fixed cut-offs."""
        assert verdict(0.35, -0.05, 0.65, rows=40)["verdict"] == "no_better_than_guessing"

    def test_a_range_wholly_below_zero_says_it_does_worse_not_that_it_includes_zero(self):
        found = verdict(-0.4, -0.7, -0.1)
        assert found["verdict"] == "no_better_than_guessing"
        assert "below 0" in found["headline"] and "includes 0" not in found["headline"]

    def test_a_range_above_zero_but_reaching_under_the_bar_is_marginal(self):
        assert verdict(0.45, 0.18, 0.66, rows=60)["verdict"] == "marginal"

    def test_a_small_but_real_signal_on_a_large_set_is_marginal_not_guessing(self):
        """R² 0.04 on thousands of rows with a tight range above zero: real, and weak. The old
        cut-off called it no better than guessing."""
        assert verdict(0.04, 0.02, 0.06, rows=5000)["verdict"] == "marginal"

    def test_a_range_that_clears_the_bar_throughout_is_useful(self):
        assert verdict(0.62, 0.54, 0.69, rows=400)["verdict"] == "useful"

    def test_the_edges_belong_to_the_more_cautious_verdict(self):
        assert verdict(0.1, 0.0, 0.2)["verdict"] == "no_better_than_guessing"
        assert verdict(0.5, modeling.R2_USEFUL_FLOOR, 0.7)["verdict"] == "useful"
        assert verdict(0.5, modeling.R2_USEFUL_FLOOR - 0.001, 0.7)["verdict"] == "marginal"

    @pytest.mark.parametrize("seed", range(6))
    def test_the_same_quality_of_prediction_is_judged_by_how_much_evidence_there_is(self, seed):
        """A prediction with R² around 0.4, bootstrapped for real. On 60 rows the range is far too
        wide to call it useful; on 4,000 it is not."""
        rng = np.random.default_rng(seed)

        def judged(rows: int) -> str:
            truth = rng.normal(0, 1, rows)
            predicted = truth + rng.normal(0, 0.77, rows)
            found = uncertainty.metric_intervals(truth, predicted, None, "regression")
            r2 = float(1 - ((truth - predicted) ** 2).sum() / ((truth - truth.mean()) ** 2).sum())
            return modeling.assess_result(
                {"r2": r2}, {}, "regression", test_rows=rows, intervals=found
            )["verdict"]

        assert judged(60) in {"marginal", "no_better_than_guessing"}
        assert judged(4000) == "useful"


class TestTheHeadlineStatesTheInterval:
    def test_each_verdict_prints_its_range(self):
        guessing = verdict(0.18, -0.10, 0.45, rows=40)["headline"]
        marginal = verdict(0.45, 0.18, 0.66)["headline"]
        useful = verdict(0.62, 0.54, 0.69)["headline"]
        assert "-0.10 to 0.45" in guessing and "95% range" in guessing and "40 test rows" in guessing
        assert "18% to 66%" in marginal and "95% range" in marginal
        assert "54% to 69%" in useful and "95% range" in useful

    def test_it_says_what_the_bar_for_a_strong_signal_is(self):
        assert "30%" in verdict(0.45, 0.18, 0.66)["headline"]

    def test_it_does_not_blame_the_model_when_the_data_cannot_show_it(self):
        assert "more or different data" in verdict(0.18, -0.10, 0.45, rows=40)["headline"]


class TestWithoutAnInterval:
    """A result saved before intervals existed, or one whose interval could not be computed."""

    @pytest.mark.parametrize(
        "r2,expected",
        [(0.02, "no_better_than_guessing"), (-0.3, "no_better_than_guessing"), (0.2, "marginal"), (0.5, "useful")],
    )
    def test_the_fixed_cutoffs_still_apply(self, r2, expected):
        assert modeling.assess_result({"r2": r2}, {}, "regression", test_rows=200)["verdict"] == expected

    def test_an_interval_for_other_metrics_only_is_not_used(self):
        only_mae = {"level": 0.95, "metrics": {"mae": {"low": 0.1, "high": 0.2}}}
        found = modeling.assess_result({"r2": 0.5}, {}, "regression", test_rows=200, intervals=only_mae)
        assert found["verdict"] == "useful"

    def test_an_empty_interval_report_falls_back(self):
        found = modeling.assess_result({"r2": 0.5}, {}, "regression", test_rows=200, intervals={"metrics": {}})
        assert found["verdict"] == "useful"


class TestTheOtherChecksStillComeFirst:
    def test_too_few_test_rows_is_still_unjudgeable_whatever_the_interval_says(self):
        found = verdict(0.9, 0.8, 0.95, rows=12)
        assert found["verdict"] == "not_enough_test_data"

    def test_classification_is_untouched(self):
        found = modeling.assess_result(
            {"accuracy": 0.7, "roc_auc": 0.65}, {"accuracy": 0.7}, "classification",
            test_rows=2000, class_counts=[1400, 600], intervals=interval(0.0, 0.1),
        )
        assert found["verdict"] == "marginal"


def noise(seed: int, rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, rows),
            "b": rng.normal(0, 1, rows),
            "c": rng.normal(0, 1, rows),
            "target": rng.normal(0, 1, rows),
        }
    )


class TestNoiseIsCalledNoise:
    def test_noise_with_forty_test_rows_is_no_better_than_guessing_in_nearly_every_seed(self):
        """Mirrors the classification benchmark: 25 datasets where the target is independent of
        every column, 200 rows so that 40 are held out. At least 23 must be called guessing, and
        none may be called useful."""
        verdicts = []
        for seed in range(25):
            result = modeling.train_candidates(
                noise(seed), target_column="target", task="regression",
                model_names=["linear_regression"], include_deep_diagnostics=False,
            )
            assert result["split"]["test_rows"] == 40
            verdicts.append(result["assessment"]["verdict"])
        guessing = verdicts.count("no_better_than_guessing")
        assert guessing >= 23, f"noise slipped through as signal too often: {verdicts}"
        assert "useful" not in verdicts

    def test_a_clearly_learnable_problem_is_still_useful(self):
        rng = np.random.default_rng(5)
        x = rng.normal(0, 1, 600)
        df = pd.DataFrame({"x": x, "z": rng.normal(0, 1, 600), "target": 3 * x + rng.normal(0, 1, 600)})
        result = modeling.train_candidates(
            df, target_column="target", task="regression",
            model_names=["linear_regression"], include_deep_diagnostics=False,
        )
        assert result["assessment"]["verdict"] == "useful"
        assert "95% range" in result["assessment"]["headline"]

    def test_the_verdict_comes_from_the_interval_the_result_reports(self):
        result = modeling.train_candidates(
            noise(3), target_column="target", task="regression",
            model_names=["linear_regression"], include_deep_diagnostics=False,
        )
        reported = result["results"][0]["metrics_intervals"]["metrics"]["r2"]
        assert f"{reported['low']:.2f}" in result["assessment"]["headline"]


def _upload(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    session_id = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "target", "task_type": "regression"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


class TestOverTheApi:
    def test_the_train_response_carries_the_range_in_its_headline(self):
        session_id = _upload(noise(1, rows=400))
        response = client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["linear_regression"], "include_deep_diagnostics": False},
        )
        assert response.status_code == 200, response.text
        assessment = response.json()["assessment"]
        assert assessment["verdict"] == "no_better_than_guessing"
        assert "95% range" in assessment["headline"]

    def test_the_sealed_final_evaluation_is_judged_the_same_way(self):
        rng = np.random.default_rng(8)
        x = rng.normal(0, 1, 900)
        df = pd.DataFrame({"x": x.round(4), "target": (2 * x + rng.normal(0, 1, 900)).round(4)})
        session_id = _upload(df)
        trained = client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["linear_regression"], "include_deep_diagnostics": False, "sealed_holdout": True},
        )
        assert trained.status_code == 200, trained.text
        opened = client.post(f"/api/model/{session_id}/finalise")
        assert opened.status_code == 200, opened.text
        body = opened.json()
        assert body["assessment"]["verdict"] == "useful"
        assert "95% range" in body["assessment"]["headline"]
        assert body["metrics_intervals"]["metrics"]["r2"]["low"] > modeling.R2_USEFUL_FLOOR
