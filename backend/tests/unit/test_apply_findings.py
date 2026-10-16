"""The apply report as three findings -- inputs, predictions, performance -- each with its own meaning."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.routes_apply import _three_findings
from app.main import app
from app.services import prediction_tracking

client = TestClient(app)


class TestThePredictionRule:
    REF = {"kind": "binary", "positive": "yes", "rows": 400, "positive_rate": 0.3, "probability_quantiles": {}}

    def test_a_share_flagged_well_outside_the_noise_has_changed(self):
        found = prediction_tracking.compare(self.REF, {**self.REF, "rows": 400, "positive_rate": 0.6})
        assert found["verdict"] == "changed" and found["low"] > 0 and "beyond what the number of rows" in found["headline"]

    def test_a_small_move_on_few_rows_is_not_a_change(self):
        found = prediction_tracking.compare(self.REF, {**self.REF, "rows": 20, "positive_rate": 0.4})
        assert found["verdict"] == "no_clear_change" and found["low"] < 0 < found["high"]

    def test_values_and_classes_have_their_own_rule(self):
        ref = {"kind": "values", "rows": 100, "mean": 0, "quantiles": {"0.25": -1, "0.5": 0, "0.75": 1}}
        assert prediction_tracking.compare(ref, {**ref, "quantiles": {"0.25": 0, "0.5": 1.5, "0.75": 2}})["verdict"] == "changed"
        assert prediction_tracking.compare(ref, {**ref, "quantiles": {"0.25": -1, "0.5": 0.5, "0.75": 1}})["verdict"] == "no_clear_change"
        mix = {"kind": "classes", "rows": 100, "shares": {"a": 0.5, "b": 0.5}}
        assert prediction_tracking.compare(mix, {**mix, "shares": {"a": 0.8, "b": 0.2}})["verdict"] == "changed"

    def test_without_a_reference_it_says_so(self):
        assert prediction_tracking.compare(None, self.REF)["verdict"] == "unknown"


class TestTheThreeFindings:
    @pytest.mark.parametrize("inputs,predictions,expect", [
        ("shifted", "changed", "both moved"),
        ("similar", "changed", "predictions moved although the inputs look similar"),
        ("different", "no_clear_change", "inputs moved but the predictions look much as they did"),
        ("similar", "no_clear_change", "not proof the model still works"),
    ])
    def test_the_summary_follows_what_moved_when_there_are_no_answers(self, inputs, predictions, expect):
        found = _three_findings({"verdict": inputs, "headline": "x"}, {"verdict": predictions, "headline": "y"}, None)
        assert found["performance"] == {"verdict": "unknown", "headline": "Unknown -- this file has no answers yet."}
        assert expect in found["summary"]

    def test_performance_is_stated_only_when_there_are_answers(self):
        found = _three_findings({"verdict": "similar"}, {"verdict": "no_clear_change"},
                                {"now": 0.81, "headline": "AUC 0.81 on 200 new rows."})
        assert found["performance"] == {"verdict": "measured", "headline": "AUC 0.81 on 200 new rows."}


async def _state(_config):
    return SimpleNamespace(values={"action_log": []}, next=())


def test_the_apply_response_carries_the_three_findings(monkeypatch):
    monkeypatch.setattr(app.state, "graph", SimpleNamespace(aget_state=_state), raising=False)
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    df = pd.DataFrame({"x": x.round(4), "z": rng.normal(size=300).round(4),
                       "churned": np.where(x + rng.normal(0, 1, 300) > 0, "yes", "no")})
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    sid = client.post("/api/upload", files={"file": ("t.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "churned", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    client.post(f"/api/model/{sid}/train", json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False})
    new = pd.DataFrame({"x": rng.normal(2, 1, 150).round(4), "z": rng.normal(size=150).round(4)})
    buffer = io.StringIO()
    new.to_csv(buffer, index=False)
    body = client.post(f"/api/apply/{sid}", files={"file": ("n.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()
    findings = body["findings"]
    assert set(findings) == {"inputs", "predictions", "performance", "summary"}
    assert findings["predictions"]["verdict"] == "changed"
    assert findings["performance"]["verdict"] == "unknown"
