"""The export is checked against the app's own model for this session, not a fixture."""

from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import export_acceptance, model_export, model_store
from app.services.session_store import session_dir

client = TestClient(app)

LOG = [{"action_id": "a1", "operation": "fill_median", "column_name": "spend", "operation_params": {},
        "status": "applied"}]


class FakeGraph:
    async def aget_state(self, _config):
        return SimpleNamespace(values={"action_log": LOG}, next=())


def frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    signal = rng.normal(0, 1, n)
    return pd.DataFrame({
        "signal": signal.round(4),
        "spend": np.where(rng.random(n) < 0.1, np.nan, rng.normal(50, 10, n).round(2)),
        "plan": rng.choice(["basic", "pro"], n),
        "churned": np.where(rng.random(n) < 1 / (1 + np.exp(-2.5 * signal)), "yes", "no"),
    })


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(app.state, "graph", FakeGraph(), raising=False)
    original = frame()
    buffer = io.StringIO()
    original.to_csv(buffer, index=False)
    sid = client.post("/api/upload", files={"file": ("t.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
    directory = session_dir(get_settings().upload_dir, sid)
    original.assign(spend=original.spend.fillna(original.spend.median())).to_csv(directory / "cleaned.csv", index=False)
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "churned", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    assert client.post(f"/api/model/{sid}/train", json={"model_names": ["logistic_regression"],
                                                        "include_deep_diagnostics": False}).status_code == 200
    return sid, directory


class TestTheCheck:
    def test_a_real_session_s_export_reproduces_its_model(self, session):
        sid, directory = session
        body = client.post(f"/api/export/{sid}/verify").json()
        assert body["status"] == "passed", body
        assert body["rows_compared"] == model_store.load_result(directory)["split"]["test_rows"]
        assert body["max_probability_difference"] <= export_acceptance.TOLERANCE
        assert client.get(f"/api/export/{sid}/verify").json()["status"] == "passed"

    def test_the_result_goes_in_the_bundle(self, session):
        sid, _ = session
        client.post(f"/api/export/{sid}/verify")
        with zipfile.ZipFile(io.BytesIO(client.get(f"/api/export/{sid}/zip").content)) as archive:
            assert json.loads(archive.read(export_acceptance.FILENAME))["status"] == "passed"

    def test_a_broken_template_fails_with_the_first_differing_rows(self, session, monkeypatch):
        sid, _ = session
        real = model_export.build_training_script

        def broken(result, **kwargs):
            # A template that has drifted from the app: a different regularisation strength.
            return real(result, **kwargs).replace("LogisticRegression(", "LogisticRegression(C=0.0005, ", 1)

        monkeypatch.setattr(model_export, "build_training_script", broken)
        body = client.post(f"/api/export/{sid}/verify").json()
        assert body["status"] == "failed" and body["rows_differing"] > 0
        assert body["first_differences"] and body["first_differences"][0]["row"] >= 1

    def test_a_check_that_runs_out_of_time_is_reported_not_raised(self, session):
        _, directory = session
        result = model_store.load_result(directory)
        outcome = export_acceptance.verify(directory, result, LOG, timeout=0.001)
        assert outcome["status"] == "error" and "longer than" in outcome["message"]

    def test_the_download_never_waits_on_or_fails_for_the_check(self, session, monkeypatch):
        sid, _ = session

        def explode(*args, **kwargs):
            raise RuntimeError("no subprocesses today")

        monkeypatch.setattr(export_acceptance.subprocess, "run", explode)
        assert client.post(f"/api/export/{sid}/verify").json()["status"] == "error"
        assert client.get(f"/api/export/{sid}/zip").status_code == 200


class TestTheProbes:
    SCHEMA = {"columns": [
        {"name": "amount", "kind": "numeric", "quantiles": {"0.5": 12.5}, "range": {"min": 0, "max": 50}},
        {"name": "plan", "kind": "categorical", "categories": {"basic": 9, "pro": 3}},
        {"name": "joined", "kind": "date", "range": {"min": "2022-01-01T00:00:00"}},
    ]}

    def test_every_input_column_gets_its_probe_rows(self):
        probes, labels = export_acceptance.probe_frame(self.SCHEMA)
        assert labels[0] == "a typical row" and probes.iloc[0].to_dict() == {
            "amount": 12.5, "plan": "basic", "joined": "2022-01-01T00:00:00"}
        assert labels[1:] == [
            "amount blank", "amount written as text", "amount not a number",
            "plan blank", "plan a value never seen in training",
            "joined blank", "joined not a date",
        ]
        assert probes.loc[labels.index("plan a value never seen in training"), "plan"] == export_acceptance.NEVER_SEEN
        assert probes.loc[labels.index("amount not a number"), "amount"] == "n/a"

    def test_the_app_and_predict_py_agree_on_every_probe_row(self, session):
        sid, _ = session
        body = client.post(f"/api/export/{sid}/verify").json()
        assert body["status"] == "passed", body
        assert body["probes"]["rows"] > 5 and body["probes"]["agreed"] == body["probes"]["rows"]
        assert "predict.py and the app agree" in body["message"]

    def test_a_disagreement_on_a_probe_row_fails_the_check_and_is_listed(self, session, monkeypatch):
        from app.services import predict_export

        sid, _ = session
        real = predict_export.build_predict_script

        def drifted(*args, **kwargs):
            return real(*args, **kwargs).replace(
                "    output.to_csv(destination, index=False)",
                "    output.iloc[1, -1] = 'drifted'\n    output.to_csv(destination, index=False)",
            )

        monkeypatch.setattr(predict_export, "build_predict_script", drifted)
        body = client.post(f"/api/export/{sid}/verify").json()
        assert body["status"] == "failed"
        assert body["probes"]["disagreements"][0]["probe"] == "signal blank"
