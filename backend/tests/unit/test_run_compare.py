"""Any two runs compared: every setting that differs, and the score difference with its uncertainty."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import run_compare
from app.services.session_store import session_dir

client = TestClient(app)
LR = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}


def entry(**over) -> dict:
    base = {
        "run_id": "r", "data_fingerprint": "f", "split_key": "k", "primary_metric": "roc_auc", "target_column": "y",
        "excluded_columns": [], "feature_count": 3, "feature_recipes": [], "clipped_columns": [],
        "split": {"strategy": "random", "requested_test_size": 0.2}, "models_tried": ["logistic_regression"],
        "best_model": "logistic_regression", "tuned": False, "tuned_settings": {}, "calibration": None,
        "operating_point": None, "metrics": {"roc_auc": 0.8, "accuracy": 0.7},
    }
    return {**base, **over}


class TestTheDiff:
    def test_identical_settings_differ_in_nothing(self):
        assert run_compare.settings_diff(entry(), entry()) == []

    def test_every_differing_field_is_listed_with_both_values(self):
        a = entry(excluded_columns=["z"], feature_count=2, split={"strategy": "time", "time_column": "day"},
                  calibration="sigmoid", operating_point={"objective": "cost", "threshold": 0.3})
        fields = {d["field"]: (d["a"], d["b"]) for d in run_compare.settings_diff(a, entry())}
        assert fields["excluded_columns"] == (["z"], [])
        assert fields["feature_count"] == (2, 3)
        assert fields["split.strategy"] == ("time", "random")
        assert fields["split.time_column"] == ("day", None)
        assert fields["calibration"] == ("sigmoid", None)
        assert fields["operating_point.objective"] == ("cost", "default")
        assert fields["operating_point.threshold"] == (0.3, None)

    def test_order_of_lists_is_not_a_difference(self):
        assert run_compare.settings_diff(entry(excluded_columns=["a", "b"]), entry(excluded_columns=["b", "a"])) == []


class TestComparability:
    def test_different_data_cannot_be_compared(self):
        able = run_compare.comparability(entry(data_fingerprint="g"), entry())
        assert able == {"comparable": False, "reasons": ["they were trained on different data"], "same_rows": False}

    def test_different_metrics_cannot_be_compared(self):
        able = run_compare.comparability(entry(primary_metric="accuracy"), entry())
        assert not able["comparable"] and "different metrics (roc_auc and accuracy)" in able["reasons"][0]

    def test_the_same_split_means_the_same_rows(self):
        assert run_compare.comparability(entry(), entry())["same_rows"] is True
        assert run_compare.comparability(entry(split_key="other"), entry())["same_rows"] is False


class TestTheSummary:
    def test_leaving_out_a_column_inside_the_noise(self):
        diff = run_compare.settings_diff(entry(excluded_columns=["region"]), entry())
        text = run_compare.summary(diff, {"comparable": True, "reasons": []}, {"verdict": "within_the_noise"}, "roc_auc")
        assert text == "Leaving out 'region' did not change roc_auc beyond the noise."

    def test_a_paired_gain_gives_its_range(self):
        diff = run_compare.settings_diff(entry(calibration="sigmoid"), entry())
        primary = {"verdict": "better", "method": "paired", "difference": 0.04, "low": 0.01, "high": 0.07}
        text = run_compare.summary(diff, {"comparable": True, "reasons": []}, primary, "roc_auc")
        assert text == "Changing calibration raised roc_auc beyond the noise (by +0.040; 95% range +0.010 to +0.070)."

    def test_nothing_differs(self):
        text = run_compare.summary([], {"comparable": True, "reasons": []}, {"verdict": "within_the_noise"}, "roc_auc")
        assert text == "Nothing in the settings differs, and the two scores agree within the noise."

    def test_incomparable_runs_say_why(self):
        able = run_compare.comparability(entry(data_fingerprint="g"), entry())
        assert run_compare.summary([], able, None, "roc_auc") == (
            "These two runs cannot be compared: they were trained on different data."
        )


def frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "w": rng.normal(size=n),
                         "outcome": np.where(x + rng.normal(0, 1, n) > 0, "yes", "no")})


@pytest.fixture
def sid():
    buffer = io.StringIO()
    frame().to_csv(buffer, index=False)
    session_id = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


class TestOverHttp:
    def test_two_runs_on_the_same_rows_get_a_paired_difference(self, sid):
        older = client.post(f"/api/model/{sid}/train", json=LR).json()["run_id"]
        newer = client.post(f"/api/model/{sid}/train", json={**LR, "excluded_columns": ["z"]}).json()["run_id"]
        body = client.get(f"/api/model/{sid}/runs/compare", params={"a": newer, "b": older}).json()
        assert body["comparable"] is True and body["same_rows"] is True
        assert body["primary"]["method"] == "paired" and body["primary"]["metric"] == "roc_auc"
        assert body["primary"]["low"] <= body["primary"]["difference"] <= body["primary"]["high"]
        assert [d["field"] for d in body["settings"]] == ["excluded_columns", "feature_count"]
        assert body["summary"].startswith("Leaving out 'z' ")  # the input count follows from it
        assert {m["metric"] for m in body["metrics"]} >= {"roc_auc", "accuracy"}

    def test_runs_on_different_data_get_the_reason_and_no_paired_numbers(self, sid):
        older = client.post(f"/api/model/{sid}/train", json=LR).json()["run_id"]
        directory = session_dir(get_settings().upload_dir, sid)
        changed = frame()
        changed.loc[0, "x"] = 99.0
        changed.to_csv(directory / "cleaned.csv", index=False)
        newer = client.post(f"/api/model/{sid}/train", json=LR).json()["run_id"]
        body = client.get(f"/api/model/{sid}/runs/compare", params={"a": newer, "b": older}).json()
        assert body["comparable"] is False and body["primary"] is None
        assert body["reasons"] == ["they were trained on different data"]
        assert body["summary"].startswith("These two runs cannot be compared")

    def test_an_unknown_run_or_an_attempt_is_not_found(self, sid):
        run_id = client.post(f"/api/model/{sid}/train", json=LR).json()["run_id"]
        missing = client.get(f"/api/model/{sid}/runs/compare", params={"a": run_id, "b": "nope"})
        assert missing.status_code == 404 and missing.json()["code"] == "run_not_found"
        client.post(f"/api/model/{sid}/train", json={**LR, "split_strategy": "time", "time_column": "z"})
        attempt = client.get(f"/api/model/{sid}/history").json()["runs"][0]["run_id"]
        refused = client.get(f"/api/model/{sid}/runs/compare", params={"a": run_id, "b": attempt})
        assert refused.status_code == 404 and "produced no model" in refused.json()["detail"]
