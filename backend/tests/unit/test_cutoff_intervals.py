"""A range around the precision and recall a cut-off delivered.

The cut-off card compared what the training folds promised with what the test rows delivered and
called any gap over ten points an over-promise. On forty positives a ten-point gap is well inside
what a different set of rows would give, so the message was noise as often as it was news. The
delivered figures now carry the range they could land in, and the message appears only when the
promise lies outside it.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import modeling, operating_point, uncertainty
from app.services.model_card import build_model_card
from app.services.operating_point import OperatingPoint

client = TestClient(app)


def scored(n: int = 400, seed: int = 0, strength: float = 0.6):
    rng = np.random.default_rng(seed)
    truth = (rng.random(n) < 0.35).astype(int)
    scores = np.clip(0.5 + strength * (truth - 0.35) * 0.4 + rng.normal(0, 0.2, n), 0, 1)
    return truth, scores


class TestTheRates:
    def test_with_every_row_drawn_once_they_are_the_point_estimates(self):
        truth, scores = scored()
        flagged = (scores >= 0.5).astype(float)
        found = uncertainty._rates_at_threshold(np.ones((1, len(truth))), truth.astype(float), flagged)
        at = operating_point.metrics_at(truth, scores, 0.5)
        assert found["precision"][0] == pytest.approx(at["precision"], abs=1e-4)
        assert found["recall"][0] == pytest.approx(at["recall"], abs=1e-4)

    def test_nothing_flagged_is_undefined_not_zero(self):
        found = uncertainty._rates_at_threshold(np.ones((1, 4)), np.array([1.0, 0, 1, 0]), np.zeros(4))
        assert np.isnan(found["precision"][0]) and found["recall"][0] == 0.0

    def test_no_real_cases_is_undefined_recall(self):
        found = uncertainty._rates_at_threshold(np.ones((1, 4)), np.zeros(4), np.array([1.0, 0, 1, 0]))
        assert np.isnan(found["recall"][0]) and found["precision"][0] == 0.0


class TestTheIntervals:
    def test_each_range_has_the_delivered_figure_inside_it(self):
        truth, scores = scored()
        found = uncertainty.threshold_intervals(truth, scores, 0.5)["metrics"]
        at = operating_point.metrics_at(truth, scores, 0.5)
        for key in ("precision", "recall"):
            assert found[key]["low"] <= at[key] <= found[key]["high"], key

    def test_it_is_deterministic(self):
        truth, scores = scored()
        assert uncertainty.threshold_intervals(truth, scores, 0.5) == uncertainty.threshold_intervals(truth, scores, 0.5)

    def test_more_rows_give_a_narrower_range(self):
        small = uncertainty.threshold_intervals(*scored(80), 0.5)["metrics"]["recall"]
        large = uncertainty.threshold_intervals(*scored(4000), 0.5)["metrics"]["recall"]
        assert (small["high"] - small["low"]) > 2.5 * (large["high"] - large["low"])

    def test_a_ten_point_gap_on_forty_cases_is_inside_the_range(self):
        """The reason the fixed rule was wrong."""
        rng = np.random.default_rng(1)
        truth = np.r_[np.ones(40), np.zeros(160)].astype(int)
        scores = np.where(truth == 1, rng.random(200) * 0.6 + 0.4, rng.random(200) * 0.7)
        recall = uncertainty.threshold_intervals(truth, scores, 0.6)["metrics"]["recall"]
        at = operating_point.metrics_at(truth, scores, 0.6)["recall"]
        assert recall["high"] - at > 0.05 and at - recall["low"] > 0.05

    def test_a_rare_answer_never_vanishes_from_a_resample(self):
        truth = np.r_[np.ones(3), np.zeros(97)].astype(int)
        scores = np.r_[[0.9, 0.8, 0.2], np.linspace(0.0, 0.7, 97)]
        found = uncertainty.threshold_intervals(truth, scores, 0.5)["metrics"]
        assert "recall" in found  # every resample has its three cases, so recall is always defined

    def test_a_cutoff_that_flags_nothing_has_no_precision_range_but_a_recall_range(self):
        truth, scores = scored()
        found = uncertainty.threshold_intervals(truth, scores, 1.01)["metrics"]
        assert "precision" not in found and found["recall"] == {"low": 0.0, "high": 0.0}

    def test_resamples_without_a_precision_are_dropped_not_counted_as_zero(self):
        """One flagged row, and it is a real case: wherever precision is defined it is 1. Counting
        the resamples that miss the row as 0 would drag the lower end to 0 for a reason that has
        nothing to do with the model."""
        truth = np.r_[np.ones(30), np.zeros(170)].astype(int)
        scores = np.zeros(200)
        scores[0] = 0.9
        assert uncertainty.threshold_intervals(truth, scores, 0.5)["metrics"]["precision"] == {"low": 1.0, "high": 1.0}

    def test_a_grouped_test_set_is_resampled_by_whole_groups(self):
        rng = np.random.default_rng(2)
        groups = np.repeat(np.arange(40), 10)
        by_group = (rng.random(40) < 0.4).astype(int)
        truth = np.repeat(by_group, 10)
        # A customer's rows share a score, so whether they are flagged is a property of the customer.
        group_score = np.clip(by_group * 0.35 + 0.3 + rng.normal(0, 0.2, 40), 0, 1)
        scores = np.clip(np.repeat(group_score, 10) + rng.normal(0, 0.03, 400), 0, 1)
        by_rows = uncertainty.threshold_intervals(truth, scores, 0.5)
        by_groups = uncertainty.threshold_intervals(truth, scores, 0.5, groups=groups)
        assert by_groups["by"] == "groups" and by_rows["by"] == "rows"
        width = lambda r: r["metrics"]["recall"]["high"] - r["metrics"]["recall"]["low"]  # noqa: E731
        assert width(by_groups) > width(by_rows)  # ten rows of one customer are not ten observations


def frame(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "z": rng.normal(0, 1, n), "y": np.where(x + rng.normal(0, 1, n) > 0.2, "yes", "no")})


def train(df, objective=("min_recall", 1, 1, 0.8), **kwargs):
    return modeling.train_candidates(
        df, target_column="y", task="classification", model_names=["logistic_regression"],
        include_deep_diagnostics=False, point=OperatingPoint(*objective), **kwargs,
    )


class TestInTheResult:
    def test_the_delivered_precision_and_recall_carry_ranges(self):
        point = train(frame())["operating_point"]
        ranges = point["observed_intervals"]
        assert ranges["level"] == 0.95 and ranges["by"] == "rows"
        for key in ("precision", "recall"):
            assert ranges["metrics"][key]["low"] <= point["observed"][key] <= ranges["metrics"][key]["high"]

    def test_they_are_the_ranges_of_the_figures_the_table_reports(self):
        df = frame()
        result = train(df)
        _, X_test, _, y_test = modeling.split_data(df, "y", "classification")
        model = result["results"][0]["pipeline"]
        again = uncertainty.threshold_intervals(
            (y_test == "yes").to_numpy().astype(int), model.predict_proba(X_test)[:, 1],
            result["operating_point"]["threshold"],
        )
        assert result["operating_point"]["observed_intervals"] == again

    def test_the_default_line_has_them_too(self):
        assert train(frame(), objective=("default",))["operating_point"]["observed_intervals"] is not None

    def test_a_test_set_too_small_to_judge_gets_none(self):
        small = train(frame(120))
        assert small["split"]["test_rows"] < modeling.MIN_TEST_ROWS_TO_JUDGE
        assert small["operating_point"]["observed_intervals"] is None

    def test_a_grouped_split_resamples_whole_groups(self):
        rng = np.random.default_rng(3)
        base = np.repeat(rng.normal(0, 1, 200), 6)
        df = pd.DataFrame(
            {"customer": np.repeat(np.arange(200), 6), "x": base + rng.normal(0, 0.5, 1200),
             "y": np.where(base + rng.normal(0, 0.5, 1200) > 0.2, "yes", "no")}
        )
        result = train(df, split_strategy="group", group_column="customer")
        assert result["operating_point"]["observed_intervals"]["by"] == "groups"

    def test_a_calibrated_line_has_them_on_the_calibrated_probabilities(self):
        result = train(frame(1200), calibration="sigmoid")
        ranges = result["operating_point"]["observed_intervals"]["metrics"]
        point = result["operating_point"]
        assert ranges["recall"]["low"] <= point["observed"]["recall"] <= ranges["recall"]["high"]


class TestTheCardAndTheApi:
    def test_the_card_prints_the_range_in_the_delivered_column(self):
        result = train(frame())
        card = build_model_card(session_id="s", file_name="f.csv", result=result, summary="A.", interpretation="B.")
        row = next(line for line in card.splitlines() if line.startswith("| Share of real cases caught"))
        assert len(re.findall(r"\(\d", row)) == 1 and "–" in row  # one range, in the held-out column

    def test_the_card_for_a_small_test_set_prints_none(self):
        card = build_model_card(session_id="s", file_name="f.csv", result=train(frame(120)), summary="A.", interpretation="B.")
        row = next(line for line in card.splitlines() if line.startswith("| Share of real cases caught"))
        assert "–" not in row

    def test_the_api_returns_them(self):
        buffer = io.StringIO()
        frame().to_csv(buffer, index=False)
        session_id = client.post(
            "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
        ).json()["session_id"]
        goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "y", "task_type": "auto"}
        client.post(f"/api/goal/{session_id}", json=goal)
        body = client.post(
            f"/api/model/{session_id}/train",
            json={
                "model_names": ["logistic_regression"], "include_deep_diagnostics": False,
                "operating_point": {"objective": "f1"},
            },
        ).json()
        assert set(body["operating_point"]["observed_intervals"]["metrics"]) == {"precision", "recall"}
