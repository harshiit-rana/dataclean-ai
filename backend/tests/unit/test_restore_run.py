"""Bringing back an earlier run as the model in use.

With each run's model kept (#66), the commonest reason for wanting a history -- going back to a better earlier
model -- is one request. It is allowed only for a run trained on the data as it is now; the result, download, script,
card and bundle follow it; the held-out rows are not looked at again; and the history records the restore as an
event, never as a run.
"""

from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import evaluation_ledger as ledger
from app.services import model_store
from app.services.session_store import session_dir

client = TestClient(app)
LR = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}
RF = {"model_names": ["random_forest_classifier"], "include_deep_diagnostics": False}


def frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "outcome": np.where(x + rng.normal(0, 1, n) > 0, "yes", "no")})


@pytest.fixture
def two_runs():
    buffer = io.StringIO()
    frame().to_csv(buffer, index=False)
    sid = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    first = client.post(f"/api/model/{sid}/train", json=LR).json()["run_id"]
    second = client.post(f"/api/model/{sid}/train", json=RF).json()["run_id"]
    return sid, session_dir(get_settings().upload_dir, sid), first, second


def restore(sid: str, run_id: str):
    return client.post(f"/api/model/{sid}/runs/{run_id}/restore")


class TestRestoring:
    def test_the_result_download_and_script_follow_the_restored_run(self, two_runs):
        sid, directory, first, second = two_runs
        assert client.get(f"/api/model/{sid}/result").json()["best_model"] == "random_forest_classifier"
        client.get(f"/api/model/{sid}/script")  # cache the newer run's script

        response = restore(sid, first)
        assert response.status_code == 200 and response.json()["run_id"] == first

        assert client.get(f"/api/model/{sid}/result").json()["run_id"] == first
        expected = (model_store.run_dir(directory, first) / model_store.PIPELINE_FILENAME).read_bytes()
        assert client.get(f"/api/model/{sid}/download").content == expected
        script = client.get(f"/api/model/{sid}/script").text
        assert "LogisticRegression(" in script and "RandomForestClassifier(" not in script

    def test_the_bundle_follows_the_restored_run(self, two_runs, monkeypatch):
        class FakeGraph:
            async def aget_state(self, _config):
                return SimpleNamespace(values={"action_log": []}, next=())

        monkeypatch.setattr(app.state, "graph", FakeGraph(), raising=False)
        sid, directory, first, _ = two_runs
        restore(sid, first)
        bundle = client.get(f"/api/export/{sid}/zip")
        with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
            assert archive.read(model_store.PIPELINE_FILENAME) == model_store.pipeline_path(directory).read_bytes()
            assert b"LogisticRegression(" in archive.read(model_store.SCRIPT_FILENAME)

    def test_the_history_records_the_restore_as_an_event_not_a_run(self, two_runs):
        sid, _, first, second = two_runs
        restore(sid, first)
        body = client.get(f"/api/model/{sid}/history").json()
        event = body["runs"][0]
        assert event["status"] == "restored" and event["restored_run_id"] == first
        assert event["metrics"] == {} and "Brought back" in event["reason"]
        assert body["current_run_id"] == first
        assert set(body["stored_run_ids"]) == {first, second}
        assert event["run_id"] not in body["comparisons"] and second in body["comparisons"]

    def test_a_restore_is_not_a_new_look_at_the_test_rows(self, two_runs):
        sid, directory, first, _ = two_runs
        before = len(ledger.looks(directory))
        restore(sid, first)
        assert len(ledger.looks(directory)) == before

    def test_restoring_the_model_already_in_use_changes_nothing(self, two_runs):
        sid, _, _, second = two_runs
        before = client.get(f"/api/model/{sid}/history").json()["runs"]
        assert restore(sid, second).status_code == 200
        assert client.get(f"/api/model/{sid}/history").json()["runs"] == before


class TestRefusing:
    def test_a_run_from_before_the_data_changed_is_refused_with_the_reason(self, two_runs):
        sid, directory, first, second = two_runs
        changed = frame()
        changed.loc[0, "x"] = 99.0
        changed.to_csv(directory / "cleaned.csv", index=False)
        response = restore(sid, first)
        assert response.status_code == 409 and response.json()["code"] == "data_changed"
        assert "changed since this run was trained" in response.json()["detail"]
        assert model_store.current_run_id(directory) == second

    def test_undoing_back_to_the_data_a_run_was_trained_on_makes_it_restorable_again(self, two_runs):
        sid, directory, first, _ = two_runs
        changed = frame()
        changed.loc[0, "x"] = 99.0
        changed.to_csv(directory / "cleaned.csv", index=False)
        model_store.clear(directory)
        assert restore(sid, first).status_code == 409
        (directory / "cleaned.csv").unlink()
        assert restore(sid, first).status_code == 200
        assert model_store.current_run_id(directory) == first

    def test_a_run_with_no_stored_model_is_not_found(self, two_runs):
        sid, *_ = two_runs
        response = restore(sid, "nosuchrun")
        assert response.status_code == 404 and response.json()["code"] == "run_not_stored"

    def test_an_unknown_session_is_not_found(self):
        assert restore("does-not-exist", "x").status_code == 404
