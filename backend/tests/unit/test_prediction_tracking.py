"""What the model predicted, file by file, against what it predicted on its held-out rows at training time."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services import prediction_tracking

client = TestClient(app)
RESULT = {"task": "classification", "target_column": "churned", "operating_point": {"threshold": 0.3}}


class TestTheSummaries:
    def test_a_two_class_application(self):
        predicted = pd.DataFrame({"predicted_churned": ["yes", "no", "no", "yes"],
                                  "probability_yes": [0.9, 0.1, 0.2, 0.6]})
        summary = prediction_tracking.summarise(predicted, RESULT)
        assert summary["kind"] == "binary" and summary["positive"] == "yes" and summary["positive_rate"] == 0.5
        assert summary["probability_quantiles"]["0.5"] == 0.4

    def test_the_reference_answers_at_the_model_s_cut_off(self):
        held_out = {"called": np.array(["no", "no", "no", "no"]), "truth": np.array(["yes", "no", "no", "yes"]),
                    "scores": np.array([0.35, 0.1, 0.2, 0.31]), "positive_label": np.array("yes")}
        summary = prediction_tracking.reference(held_out, RESULT)
        # predict() at 0.5 called nothing positive; at the 0.3 cut-off the model in use flags half.
        assert summary["positive_rate"] == 0.5

    def test_a_regression(self):
        predicted = pd.DataFrame({"predicted_price": [1.0, 2.0, 3.0]})
        summary = prediction_tracking.summarise(predicted, {"task": "regression", "target_column": "price"})
        assert summary == {"kind": "values", "rows": 3, "mean": 2.0, "quantiles": summary["quantiles"]}

    def test_several_classes(self):
        predicted = pd.DataFrame({"predicted_plan": ["a", "b", "b", "c"], "probability_of_prediction": [0.5] * 4})
        summary = prediction_tracking.summarise(predicted, {"task": "classification", "target_column": "plan"})
        assert summary["kind"] == "classes" and summary["shares"] == {"b": 0.5, "a": 0.25, "c": 0.25}


async def _state(_config):
    return SimpleNamespace(values={"action_log": []}, next=())


def csv(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def rows(n: int, seed: int, shift: float = 0.0, target: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(shift, 1, n)
    df = pd.DataFrame({"x": x.round(4), "z": rng.normal(size=n).round(4)})
    if target:
        df["churned"] = np.where(x + rng.normal(0, 1, n) > 0, "yes", "no")
    return df


def test_the_history_shows_the_reference_and_each_file(monkeypatch):
    monkeypatch.setattr(app.state, "graph", SimpleNamespace(aget_state=_state), raising=False)
    sid = client.post("/api/upload", files={"file": ("t.csv", io.BytesIO(csv(rows(300, 1))), "text/csv")}).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "churned", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    client.post(f"/api/model/{sid}/train", json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False})
    for seed, shift in ((7, 0.0), (8, 2.0)):  # the second file has drifted towards "yes"
        new = rows(80, seed, shift, target=False)
        assert client.post(f"/api/apply/{sid}", files={"file": ("n.csv", io.BytesIO(csv(new)), "text/csv")}).status_code == 200
    body = client.get(f"/api/model/{sid}/prediction-history").json()
    assert body["reference"]["kind"] == "binary" and 0.2 < body["reference"]["positive_rate"] < 0.8
    rates = [a["prediction_summary"]["positive_rate"] for a in body["applications"]]
    assert len(rates) == 2 and rates[1] > rates[0] + 0.2
