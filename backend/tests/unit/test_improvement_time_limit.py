"""A wall-clock limit on the improvement loop, and what the trail says when it bites.

A fake clock stands in for time: every cross-validated score "takes" ten seconds, so a limit of 25 seconds
is reached after a known number of them and nothing here waits for real.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
import sklearn.model_selection as sklearn_selection
from fastapi.testclient import TestClient

from app.main import app
from app.services import improvement, tuning

SECONDS_PER_SCORE = 10.0


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(improvement, "time", fake)
    monkeypatch.setattr(tuning, "time", fake)
    real = sklearn_selection.cross_val_score

    def slow(*args, **kwargs):
        fake.now += SECONDS_PER_SCORE
        return real(*args, **kwargs)

    monkeypatch.setattr(sklearn_selection, "cross_val_score", slow)
    return fake


def frame(n: int = 220, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({
        "x": x, "z": rng.normal(size=n),
        "outcome": np.where(x + rng.normal(0, 1, n) > 0, "yes", "no"),
    })


def improve(**kwargs) -> dict:
    return improvement.improve(
        frame(), target_column="outcome", task="classification",
        model_names=["logistic_regression", "random_forest_classifier"], budget=6, **kwargs,
    )


def names(found: dict) -> list[str]:
    return [r["name"] for r in found["rounds"]]


class TestWhenTheLimitBites:
    def test_a_search_cut_short_says_how_far_it_got_and_the_later_rounds_are_skipped(self, clock):
        found = improve(time_limit_seconds=25)
        assert found["stopped_because"] == "time_limit" and found["time_limit_seconds"] == 25
        assert names(found) == ["baseline", "hyperparameters", "other_models", "derived_features", "interactions", "ensemble"]
        settings = next(r for r in found["rounds"] if r["name"] == "hyperparameters")
        assert settings["detail"]["stopped_for_time"] is True and settings["detail"]["tried"] < settings["detail"]["planned"]
        assert "The time limit stopped it after" in settings["what_it_tried"]
        for skipped in found["rounds"][2:]:
            assert skipped["detail"] == {"skipped": "time_limit"} and skipped["accepted"] is False
            assert skipped["outcome"] == "Not run: the 25-second time limit was reached first."
        assert found["skipped_for_time"] == ["other_models", "derived_features", "interactions", "ensemble"]

    def test_a_limit_gone_before_the_second_round_skips_every_search_round(self, clock):
        found = improve(time_limit_seconds=1)
        assert found["stopped_because"] == "time_limit"
        assert found["skipped_for_time"] == ["hyperparameters", "other_models", "derived_features", "interactions"]
        baseline = found["rounds"][0]
        # One family is always scored; the rest were left for want of time and the round says so.
        assert len(baseline["detail"]["scores"]) == 1 and baseline["detail"]["unscored_for_time"]
        assert "The time limit left 1 more unscored." in baseline["what_it_tried"]

    def test_the_result_is_still_a_result_with_its_one_look_at_the_test_set(self, clock):
        found = improve(time_limit_seconds=1)
        assert found["pipeline"] is not None and "roc_auc" in found["test_metrics"]
        assert found["best"]["model"]

    def test_each_skipped_round_is_announced_to_the_stream(self, clock):
        stages = []
        improve(time_limit_seconds=1, on_stage=lambda name, description: stages.append((name, description)))
        skipped = [name for name, description in stages if description.startswith("Skipped:")]
        assert skipped == ["hyperparameters", "other_models", "derived_features", "interactions"]

    def test_runner_ups_are_skipped_one_by_one_when_time_runs_out_between_them(self, clock):
        found = improvement.improve(
            frame(), target_column="outcome", task="classification",
            model_names=["logistic_regression", "random_forest_classifier", "gradient_boosting_classifier"],
            budget=4, time_limit_seconds=55,
        )
        assert found["stopped_because"] == "time_limit"
        others = next((r for r in found["rounds"] if r["name"] == "other_models"), None)
        assert others is not None
        text = others["outcome"] + others["what_it_tried"]
        assert "time limit" in text or others["detail"].get("skipped") == "time_limit"


class TestWhenItDoesNot:
    def test_no_limit_means_no_skipped_rounds_and_nothing_new_in_the_trail(self, clock):
        found = improve()
        assert found["time_limit_seconds"] is None and found["skipped_for_time"] == []
        assert found["stopped_because"] != "time_limit"
        assert all("skipped" not in r["detail"] for r in found["rounds"])

    def test_a_limit_that_is_never_reached_changes_nothing(self, clock):
        without = improve()
        clock.now = 1000.0
        within = improve(time_limit_seconds=100_000)
        assert [r["score"] for r in within["rounds"]] == [r["score"] for r in without["rounds"]]
        assert within["stopped_because"] == without["stopped_because"]
        assert within["best"] == without["best"]

    def test_a_goal_that_is_met_wins_over_the_clock(self, clock):
        found = improve(goal=0.0, time_limit_seconds=1)
        assert found["stopped_because"] == "goal_reached"

    def test_no_limit_never_reads_the_clock_for_a_deadline(self, monkeypatch):
        class Boom:
            def monotonic(self):
                raise AssertionError("read")

        monkeypatch.setattr(improvement, "time", Boom())
        monkeypatch.setattr(tuning, "time", Boom())
        assert improve()["time_limit_seconds"] is None


class TestOverHttp:
    def session(self):
        client = TestClient(app)
        buffer = io.StringIO()
        frame().to_csv(buffer, index=False)
        sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
        goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
        assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
        return client, sid

    def test_the_trail_in_the_response_carries_the_limit_and_what_was_skipped(self, clock):
        client, sid = self.session()
        response = client.post(
            f"/api/model/{sid}/improve",
            json={"model_names": ["logistic_regression", "random_forest_classifier"],
                  "include_deep_diagnostics": False, "budget": 4, "time_limit_seconds": 1},
        )
        assert response.status_code == 200, response.text
        trail = response.json()["improvement"]
        assert trail["stopped_because"] == "time_limit" and trail["time_limit_seconds"] == 1
        assert trail["skipped_for_time"] == ["hyperparameters", "other_models", "derived_features", "interactions"]

    @pytest.mark.parametrize("bad", [0, 0.5, -3, 3601])
    def test_a_limit_out_of_range_is_refused(self, bad):
        client, sid = self.session()
        response = client.post(f"/api/model/{sid}/improve", json={"time_limit_seconds": bad})
        assert response.status_code == 422
