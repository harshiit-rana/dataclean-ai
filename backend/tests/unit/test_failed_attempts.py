"""Refused and failed training attempts are part of the run history.

The record of "what was tried" used to hold only the runs that produced a model, so it left out exactly the attempts
that taught something: a blocked dataset, a split that could not be made, a crash. Each now leaves an entry with its
status, the reason the user was given, the request and the data version. Score comparisons skip them, and recording
one never replaces the error the user needs to see.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services import model_store, modeling

client = TestClient(app)
TRAIN = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}


def learnable(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "outcome": np.where(x + rng.normal(0, 1, n) > 0, "yes", "no")})


def blocked(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    outcome = list(rng.choice(["yes", "no"], 40)) + [None] * (n - 40)
    return pd.DataFrame({"signal": rng.normal(size=n).round(4), "outcome": outcome})


def session(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    sid = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    return sid


def history(sid: str) -> dict:
    return client.get(f"/api/model/{sid}/history").json()


class TestRefused:
    def test_a_blocked_train_leaves_a_refused_entry_with_the_blocker_text(self):
        sid = session(blocked())
        response = client.post(f"/api/model/{sid}/train", json=TRAIN)
        assert response.status_code == 422
        entry = history(sid)["runs"][0]
        assert entry["status"] == "refused" and entry["code"] == "data_blocked"
        assert entry["reason"] == response.json()["detail"]
        assert entry["best_model"] is None and entry["metrics"] == {}
        assert entry["request"]["model_names"] == ["logistic_regression"]
        assert entry["dataset_version"] is not None and entry["config_hash"]

    def test_a_blocked_stream_is_refused_and_recorded_before_it_opens(self):
        sid = session(blocked())
        response = client.post(f"/api/model/{sid}/improve/stream", json={})
        assert response.status_code == 422
        assert history(sid)["runs"][0]["status"] == "refused"

    def test_a_blocked_improve_is_recorded(self):
        sid = session(blocked())
        assert client.post(f"/api/model/{sid}/improve", json={}).status_code == 422
        entry = history(sid)["runs"][0]
        assert entry["status"] == "refused" and entry["request"]["budget"]


class TestFailed:
    def test_a_modelling_error_leaves_a_failed_entry_with_its_message(self):
        sid = session(learnable())
        response = client.post(f"/api/model/{sid}/train", json={**TRAIN, "split_strategy": "time", "time_column": "z"})
        assert response.status_code == 422 and response.json()["code"] == "split_unusable"
        entry = history(sid)["runs"][0]
        assert entry["status"] == "failed" and entry["code"] == "split_unusable"
        assert entry["reason"] == response.json()["detail"]
        assert entry["task"] == "classification" and entry["runtime_seconds"] is not None

    def test_an_unexpected_error_is_recorded_as_a_failure(self, monkeypatch):
        def crash(*args, **kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(modeling, "train_candidates", crash)
        sid = session(learnable())
        response = client.post(f"/api/model/{sid}/train", json=TRAIN)
        assert response.status_code == 422
        entry = history(sid)["runs"][0]
        assert entry["status"] == "failed" and entry["code"] == "training_error"
        assert "disk on fire" in entry["reason"]

    def test_a_failed_improve_is_recorded(self, monkeypatch):
        from app.services import improvement

        def crash(*args, **kwargs):
            raise modeling.ModelingError("No usable columns.", code="no_usable_columns")

        monkeypatch.setattr(improvement, "improve", crash)
        sid = session(learnable())
        assert client.post(f"/api/model/{sid}/improve", json={"budget": 4}).status_code == 422
        entry = history(sid)["runs"][0]
        assert entry["status"] == "failed" and entry["reason"] == "No usable columns."

    def test_a_failed_stream_is_recorded(self, monkeypatch):
        from app.services import improvement

        def crash(*args, **kwargs):
            raise modeling.ModelingError("No usable columns.", code="no_usable_columns")

        monkeypatch.setattr(improvement, "improve", crash)
        sid = session(learnable())
        response = client.post(f"/api/model/{sid}/improve/stream", json={"budget": 4})
        assert '"type": "error"' in response.text
        entry = history(sid)["runs"][0]
        assert entry["status"] == "failed" and entry["code"] == "no_usable_columns"

    def test_recording_never_replaces_the_error_the_user_needs(self, monkeypatch):
        def broken(**kwargs):
            raise OSError("history is read-only")

        monkeypatch.setattr(model_store, "attempt_entry", broken)
        sid = session(blocked())
        response = client.post(f"/api/model/{sid}/train", json=TRAIN)
        assert response.status_code == 422 and response.json()["code"] == "data_blocked"


class TestInTheHistory:
    def test_a_successful_run_says_so(self):
        sid = session(learnable())
        client.post(f"/api/model/{sid}/train", json=TRAIN)
        assert history(sid)["runs"][0]["status"] == "succeeded"

    def test_comparisons_skip_the_attempts_between_runs(self):
        sid = session(learnable())
        client.post(f"/api/model/{sid}/train", json=TRAIN)
        client.post(f"/api/model/{sid}/train", json={**TRAIN, "split_strategy": "time", "time_column": "z"})
        client.post(f"/api/model/{sid}/train", json={**TRAIN, "model_names": ["random_forest_classifier"]})
        body = history(sid)
        newest, failed, oldest = body["runs"][:3]
        assert failed["status"] == "failed"
        assert newest["run_id"] in body["comparisons"]
        assert failed["run_id"] not in body["comparisons"]

    def test_an_attempt_entry_has_the_same_shape_as_a_run(self):
        run = model_store.summarise({"results": [], "best_model": None})
        attempt = model_store.attempt_entry(status="failed", reason="x", code=None)
        assert set(run) <= set(attempt)
        assert {"reason", "code"} <= set(attempt)

    def test_only_the_two_attempt_statuses_are_accepted(self):
        import pytest

        with pytest.raises(ValueError):
            model_store.attempt_entry(status="succeeded", reason="x", code=None)
        assert model_store.succeeded({}) and model_store.succeeded({"status": "succeeded"})
        assert not model_store.succeeded({"status": "refused"})
