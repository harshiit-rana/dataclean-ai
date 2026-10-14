"""Brier score and expected calibration error beside the calibration chart.

The app drew a calibration curve but gave no number for how far the probabilities are from what
happened, so two runs could not be compared on it. The cut-off is chosen on these probabilities
and the exported predictions carry them, so they matter even when the AUC is good: AUC is a
ranking and says nothing about whether "70% likely" happens 70% of the time.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import brier_score_loss

from app.main import app
from app.services import diagnostics, modeling, uncertainty
from app.services.model_card import build_model_card
from app.services import model_store

client = TestClient(app)


class TestTheNumbers:
    def test_a_hand_checked_example(self):
        """Two bins of two rows. Low bin: said 0.1, happened 1 time in 2 -> off by 0.4. High bin:
        said 0.9, happened every time -> off by 0.1. Weighted by rows: (2*0.4 + 2*0.1) / 4."""
        y = np.array([0, 1, 1, 1])
        p = np.array([0.1, 0.1, 0.9, 0.9])
        assert diagnostics.expected_calibration_error(y, p) == pytest.approx(0.25)
        found = diagnostics.calibration_summary(y, p)
        assert found["expected_calibration_error"] == 0.25
        assert found["brier_score"] == pytest.approx((0.01 + 0.81 + 0.01 + 0.01) / 4)

    def test_the_brier_score_is_sklearns(self):
        rng = np.random.default_rng(1)
        p = rng.random(500)
        y = (rng.random(500) < p).astype(int)
        assert diagnostics.calibration_summary(y, p)["brier_score"] == pytest.approx(
            brier_score_loss(y, p), abs=1e-4
        )

    def test_it_uses_the_same_bins_as_the_chart_and_weights_them_by_count(self):
        rng = np.random.default_rng(2)
        p = np.clip(rng.beta(2, 5, 700), 0, 1)
        y = (rng.random(700) < np.clip(p * 1.4, 0, 1)).astype(int)
        points = diagnostics.calibration_points(y, p)
        from_the_chart = sum(pt["count"] * abs(pt["predicted"] - pt["actual"]) for pt in points) / len(y)
        assert diagnostics.expected_calibration_error(y, p) == pytest.approx(from_the_chart, abs=1e-3)

    def test_scores_of_exactly_zero_and_one_land_in_the_end_bins(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([0.0, 1.0, 1.0, 0.0])
        assert diagnostics.expected_calibration_error(y, p) == 0.0
        assert diagnostics.calibration_summary(y, p)["brier_score"] == 0.0

    def test_probabilities_that_are_exactly_right_score_near_zero(self):
        rng = np.random.default_rng(3)
        p = rng.random(50_000)
        y = (rng.random(50_000) < p).astype(int)
        found = diagnostics.calibration_summary(y, p)
        assert found["expected_calibration_error"] < 0.02
        assert found["brier_score"] == pytest.approx(1 / 6, abs=0.01)  # E[p(1-p)] for uniform p

    def test_overconfident_probabilities_score_badly(self):
        """The truth is 70% likely; the model says 97% every time."""
        rng = np.random.default_rng(4)
        y = (rng.random(5000) < 0.7).astype(int)
        p = np.full(5000, 0.97)
        calibrated = diagnostics.calibration_summary(y, np.full(5000, 0.7))
        overconfident = diagnostics.calibration_summary(y, p)
        assert overconfident["expected_calibration_error"] == pytest.approx(0.27, abs=0.02)
        assert overconfident["expected_calibration_error"] > 10 * calibrated["expected_calibration_error"]
        assert overconfident["brier_score"] > calibrated["brier_score"]

    def test_ranking_perfectly_does_not_make_the_probabilities_honest(self):
        """The reason this exists: a perfect AUC with probabilities that are wrong."""
        y = np.array([0] * 50 + [1] * 50)
        p = np.array([0.40] * 50 + [0.60] * 50)  # ranks perfectly, but says 40% where it never happens
        found = diagnostics.calibration_summary(y, p)
        assert found["expected_calibration_error"] == pytest.approx(0.4)


def two_class(n: int = 600, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {"x": x, "z": rng.normal(0, 1, n), "y": np.where(x + rng.normal(0, 1, n) > 0, "yes", "no")}
    )


def three_class(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(6)
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {"x": x, "z": rng.normal(0, 1, n), "y": np.digitize(x + rng.normal(0, 1, n), [-0.5, 0.5]).astype(str)}
    )


def numeric(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "z": rng.normal(0, 1, n), "y": 2 * x + rng.normal(0, 1, n)})


def train(df, task="classification", model="logistic_regression"):
    return modeling.train_candidates(
        df, target_column="y", task=task, model_names=[model], include_deep_diagnostics=False
    )


CALIBRATION_KEYS = {"brier_score", "expected_calibration_error", "brier_baseline"}


class TestInTheResult:
    def test_a_two_class_result_carries_all_three(self):
        result = train(two_class())
        metrics = result["results"][0]["metrics"]
        assert CALIBRATION_KEYS <= set(metrics)
        assert 0 <= metrics["brier_score"] <= 1 and 0 <= metrics["expected_calibration_error"] <= 1

    def test_the_baseline_is_the_brier_score_of_always_predicting_the_training_rate(self):
        df = two_class()
        result = train(df)
        _, X_test, y_train, y_test = modeling.split_data(df, "y", "classification")
        positive = result["results"][0]["metrics"]["positive_label"]
        rate = float((y_train.astype(str) == positive).mean())
        expected = brier_score_loss((y_test.astype(str) == positive).astype(int), np.full(len(y_test), rate))
        assert result["baseline"]["brier_score"] == pytest.approx(expected, abs=1e-4)
        assert result["results"][0]["metrics"]["brier_baseline"] == result["baseline"]["brier_score"]

    def test_a_useful_model_beats_the_baseline(self):
        metrics = train(two_class(1200))["results"][0]["metrics"]
        assert metrics["brier_score"] < metrics["brier_baseline"]

    def test_the_baseline_follows_the_chosen_positive_label(self):
        df = two_class()
        yes = modeling.baseline_score(df["y"], df["y"], "classification", "yes")["brier_score"]
        no = modeling.baseline_score(df["y"], df["y"], "classification", "no")["brier_score"]
        assert yes == pytest.approx(no, abs=1e-4)  # symmetric: the rate and its complement
        skewed = np.where(np.arange(100) < 20, "yes", "no")
        rare = modeling.baseline_score(pd.Series(skewed), pd.Series(skewed), "classification", "yes")
        assert rare["brier_score"] == pytest.approx(0.2 * 0.8, abs=1e-4)

    def test_more_than_two_classes_are_unaffected(self):
        result = train(three_class())
        assert not (CALIBRATION_KEYS & set(result["results"][0]["metrics"]))
        assert "brier_score" not in result["baseline"]

    def test_a_regression_is_unaffected(self):
        result = train(numeric(), task="regression", model="linear_regression")
        assert not (CALIBRATION_KEYS & set(result["results"][0]["metrics"]))
        assert set(result["baseline"]) == {"strategy", "r2", "mae"}

    def test_a_model_without_probabilities_does_not_lose_its_other_scores(self):
        class NoProbabilities:
            classes_ = np.array(["no", "yes"])

            def predict(self, X):
                return np.where(X["x"] > 0, "yes", "no")

        df = two_class()
        scores = modeling.evaluate(NoProbabilities(), df[["x"]], df["y"], "classification")
        assert "accuracy" in scores and "brier_score" not in scores

    def test_every_candidate_gets_its_own_brier_score_but_the_same_baseline(self):
        result = modeling.train_candidates(
            two_class(), target_column="y", task="classification",
            model_names=["logistic_regression", "random_forest_classifier"], include_deep_diagnostics=False,
        )
        scores = [r["metrics"]["brier_score"] for r in result["results"]]
        assert scores[0] != scores[1]
        assert {r["metrics"]["brier_baseline"] for r in result["results"]} == {result["baseline"]["brier_score"]}


class TestTheirIntervals:
    """ECE is noisy on a small test set, so it is never shown without its range."""

    def test_the_bootstrap_agrees_with_evaluate_to_the_last_digit(self):
        df = two_class(400)
        X_train, X_test, y_train, y_test = modeling.split_data(df, "y", "classification")
        from app.services.modeling import build_pipeline

        pipeline = build_pipeline("logistic_regression", ["x", "z"], [])
        pipeline.fit(X_train, y_train)
        scores = modeling.evaluate(pipeline, X_test, y_test, "classification")
        called, probabilities = modeling.held_out_predictions(pipeline, X_test, y_test, "classification")
        found = uncertainty.point_estimates(y_test, called, probabilities, "classification", scores["positive_label"])
        assert found["brier_score"] == pytest.approx(scores["brier_score"], abs=5e-5)
        assert found["expected_calibration_error"] == pytest.approx(scores["expected_calibration_error"], abs=5e-5)

    def test_the_chosen_model_has_an_interval_for_each(self):
        result = train(two_class())
        intervals = result["results"][0]["metrics_intervals"]["metrics"]
        for key in ("brier_score", "expected_calibration_error"):
            assert intervals[key]["low"] <= intervals[key]["high"]
        assert intervals["brier_score"]["low"] <= result["results"][0]["metrics"]["brier_score"] <= intervals["brier_score"]["high"]

    def test_a_target_with_more_than_two_classes_has_none(self):
        intervals = train(three_class())["results"][0]["metrics_intervals"]["metrics"]
        assert "brier_score" not in intervals and "expected_calibration_error" not in intervals

    def test_a_small_test_set_gives_a_wide_range(self):
        small = train(two_class(150))["results"][0]["metrics_intervals"]["metrics"]["expected_calibration_error"]
        large = train(two_class(3000))["results"][0]["metrics_intervals"]["metrics"]["expected_calibration_error"]
        assert (small["high"] - small["low"]) > 2 * (large["high"] - large["low"])


class TestTheCard:
    def card(self, result):
        return build_model_card(session_id="s", file_name="f.csv", result=result, summary="A model.", interpretation="Fine.")

    def test_the_results_table_has_both_columns_and_a_baseline_brier(self):
        card = self.card(train(two_class()))
        assert "Brier score" in card and "Calibration error" in card
        header = next(line for line in card.splitlines() if line.startswith("| Model |") and "Brier" in line)
        columns = [cell.strip() for cell in header.strip("|").split("|")]
        assert columns.index("Brier score") < columns.index("Calibration error")
        baseline_row = next(line for line in card.splitlines() if line.startswith("| _No model at all"))
        cells = [cell.strip() for cell in baseline_row.strip("|").split("|")]
        assert cells[columns.index("Brier score")] not in {"", "—"}  # a Brier baseline in its column
        assert cells[columns.index("Calibration error")] == "—"  # no baseline for the error

    def test_the_glossary_explains_them_in_plain_english(self):
        card = self.card(train(two_class()))
        assert "**Brier score** — How far its probabilities are from what happened" in card
        assert "**Calibration error** — On average, how far 'X% likely' is from how often it really happens" in card

    def test_the_range_is_printed_beside_the_error(self):
        card = self.card(train(two_class()))
        row = next(line for line in card.splitlines() if "**Logistic regression**" in line and "(chosen)" in line)
        assert row.count("(") >= 3  # at least a few metrics carry their interval

    def test_a_multiclass_card_does_not_mention_them(self):
        card = self.card(train(three_class()))
        assert "Brier" not in card and "Calibration error" not in card

    def test_a_regression_card_does_not_mention_them(self):
        card = self.card(train(numeric(), task="regression", model="linear_regression"))
        assert "Brier" not in card


class TestInTheHistory:
    def test_a_run_records_them_so_two_runs_can_be_compared(self):
        summary = model_store.summarise({**train(two_class()), "run_id": "r1"})
        assert CALIBRATION_KEYS <= set(summary["metrics"])


def _session(df: pd.DataFrame, task: str = "auto") -> str:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    session_id = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "y", "task_type": task}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


class TestOverTheApi:
    def test_the_train_response_carries_them_for_two_classes(self):
        session_id = _session(two_class())
        response = client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert CALIBRATION_KEYS <= set(body["results"][0]["metrics"])
        assert "brier_score" in body["baseline"]

    def test_the_sealed_final_evaluation_reports_them_with_a_baseline(self):
        session_id = _session(two_class(900))
        trained = client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False, "sealed_holdout": True},
        )
        assert trained.status_code == 200, trained.text
        opened = client.post(f"/api/model/{session_id}/finalise")
        assert opened.status_code == 200, opened.text
        body = opened.json()
        assert CALIBRATION_KEYS <= set(body["metrics"])
        assert body["baseline"]["brier_score"] == body["metrics"]["brier_baseline"]
        assert "brier_score" in body["metrics_intervals"]["metrics"]
