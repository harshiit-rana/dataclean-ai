"""An unexpected failure in a long-running route answers with the documented shape, a stable message and the
request id -- never the exception's own text -- and the server log carries the same id."""

from __future__ import annotations

import io
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.request_context import RequestIdFilter
from app.main import app
from app.services import improvement, modeling, scoring

client = TestClient(app)
SECRET = "secret internals at /srv/app/private.py line 42"
TRAIN = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}


def explode(*args, **kwargs):
    raise RuntimeError(SECRET)


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(app.state, "graph", SimpleNamespace(aget_state=_state), raising=False)
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    df = pd.DataFrame({"x": x, "z": rng.normal(size=300), "outcome": np.where(x + rng.normal(0, 1, 300) > 0, "yes", "no")})
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    return sid, df


async def _state(_config):
    return SimpleNamespace(values={"action_log": []}, next=())


def assert_safe(body: dict, request_id: str) -> None:
    assert set(body) >= {"error", "detail"} and body.get("request_id") == request_id
    assert SECRET not in body["detail"] and "stopped unexpectedly" in body["detail"]
    assert request_id in body["detail"]


@pytest.fixture
def logged(caplog):
    caplog.handler.addFilter(RequestIdFilter())
    caplog.set_level(logging.ERROR)
    return caplog


def logged_with(caplog, request_id: str) -> bool:
    return any(r.request_id == request_id and r.levelno >= logging.ERROR for r in caplog.records)


class TestEachRoute:
    def test_train(self, session, monkeypatch, logged):
        sid, _ = session
        monkeypatch.setattr(modeling, "train_candidates", explode)
        response = client.post(f"/api/model/{sid}/train", json=TRAIN, headers={"X-Request-Id": "train-1"})
        assert response.status_code == 422
        assert_safe(response.json(), "train-1")
        assert logged_with(logged, "train-1")

    def test_improve(self, session, monkeypatch, logged):
        sid, _ = session
        monkeypatch.setattr(improvement, "improve", explode)
        response = client.post(f"/api/model/{sid}/improve", json={**TRAIN, "budget": 4},
                               headers={"X-Request-Id": "improve-1"})
        assert_safe(response.json(), "improve-1")
        assert logged_with(logged, "improve-1")

    def test_the_streamed_search(self, session, monkeypatch, logged):
        sid, _ = session
        monkeypatch.setattr(improvement, "improve", explode)
        response = client.post(f"/api/model/{sid}/improve/stream", json={**TRAIN, "budget": 4},
                               headers={"X-Request-Id": "stream-1"})
        assert SECRET not in response.text and "stopped unexpectedly" in response.text and "stream-1" in response.text
        assert logged_with(logged, "stream-1")

    def test_apply(self, session, monkeypatch, logged):
        sid, df = session
        assert client.post(f"/api/model/{sid}/train", json=TRAIN).status_code == 200
        monkeypatch.setattr(scoring, "predict", explode)
        buffer = io.StringIO()
        df.drop(columns=["outcome"]).head(20).to_csv(buffer, index=False)
        response = client.post(f"/api/apply/{sid}", files={"file": ("n.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")},
                               headers={"X-Request-Id": "apply-1"})
        assert_safe(response.json(), "apply-1")
        assert logged_with(logged, "apply-1")

    def test_the_export_check(self, session, monkeypatch):
        from app.services import export_acceptance

        sid, _ = session
        assert client.post(f"/api/model/{sid}/train", json=TRAIN).status_code == 200
        monkeypatch.setattr(export_acceptance.subprocess, "run", explode)
        body = client.post(f"/api/export/{sid}/verify", headers={"X-Request-Id": "verify-1"}).json()
        assert body["status"] == "error" and SECRET not in body["message"] and "verify-1" in body["message"]

    def test_an_unhandled_error_anywhere_has_the_same_shape(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        from app.config import get_settings

        get_settings.cache_clear()
        def unhandled():
            raise RuntimeError(SECRET)

        app.router.add_api_route("/api/_test_unhandled", unhandled)
        body = TestClient(app, raise_server_exceptions=False).get(
            "/api/_test_unhandled", headers={"X-Request-Id": "any-1"}).json()
        assert body["request_id"] == "any-1" and SECRET not in body["detail"]

    def test_a_malformed_request_has_the_same_shape_with_readable_text(self, session):
        sid, _ = session
        response = client.post(f"/api/model/{sid}/improve", json={"budget": 2}, headers={"X-Request-Id": "bad-1"})
        body = response.json()
        assert response.status_code == 422 and body["code"] == "invalid_request" and body["request_id"] == "bad-1"
        assert isinstance(body["detail"], str) and body["detail"].startswith("budget: ")
