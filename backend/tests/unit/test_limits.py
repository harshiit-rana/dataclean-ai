"""Optional limits for hosted deployments (#123): off by default, readable when they refuse."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import limits
from app.config import get_settings
from app.main import app
from app.services import jobs

client = TestClient(app)
CSV = b"x,y,outcome\n" + b"".join(f"{i},{i % 7},{'yes' if i % 3 else 'no'}\n".encode() for i in range(120))
TRAIN = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}


@pytest.fixture(autouse=True)
def _fresh():
    limits.reset()
    yield
    limits.reset()


def _set(monkeypatch, **values):
    for key, value in values.items():
        monkeypatch.setenv(key, str(value))
    get_settings.cache_clear()


def _upload(headers=None):
    return client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(CSV), "text/csv")}, headers=headers or {})


def _session() -> str:
    session_id = _upload().json()["session_id"]
    client.post(f"/api/goal/{session_id}", json={"use_case": "ml_training", "expected_output": "p",
                                                 "target_column": "outcome", "task_type": "auto"})
    return session_id


def test_everything_is_off_by_default():
    for _ in range(12):
        assert _upload().status_code == 200


class TestTheRateLimit:
    def test_past_the_limit_a_client_is_told_to_wait(self, monkeypatch):
        _set(monkeypatch, RATE_LIMIT_PER_MINUTE=2)
        assert _upload().status_code == 200
        assert _upload().status_code == 200
        refused = _upload()
        assert refused.status_code == 429
        assert refused.json()["code"] == "rate_limited"
        assert "Try again in about" in refused.json()["detail"]

    def test_clients_are_counted_apart_by_their_token(self, monkeypatch):
        _set(monkeypatch, RATE_LIMIT_PER_MINUTE=1)
        assert _upload({"X-Session-Token": "a"}).status_code == 200
        assert _upload({"X-Session-Token": "b"}).status_code == 200
        assert _upload({"X-Session-Token": "a"}).status_code == 429

    def test_the_window_moves_on(self):
        settings = get_settings().model_copy(update={"rate_limit_per_minute": 1})

        class Request:
            headers = {"x-session-token": "t"}
            client = None

        limits._check_rate(settings, Request(), now=0.0)
        with pytest.raises(limits.TooManyRequestsError):
            limits._check_rate(settings, Request(), now=30.0)
        limits._check_rate(settings, Request(), now=61.0)


class TestTheQuota:
    def test_a_project_over_its_quota_cannot_start_more_work_and_loses_nothing(self, monkeypatch):
        session_id = _session()
        directory = Path(get_settings().upload_dir) / session_id
        (directory / "big.bin").write_bytes(b"x" * (2 * 1024 * 1024))
        _set(monkeypatch, PROJECT_QUOTA_MB=1)
        refused = client.post(f"/api/model/{session_id}/train", json=TRAIN)
        assert refused.status_code == 413
        assert refused.json()["code"] == "quota_exceeded"
        assert (directory / "big.bin").exists() and (directory / "original.csv").exists()

    def test_under_the_quota_work_goes_ahead(self, monkeypatch):
        session_id = _session()
        _set(monkeypatch, PROJECT_QUOTA_MB=50)
        assert client.post(f"/api/model/{session_id}/train", json=TRAIN).status_code == 200


class TestUnfinishedJobs:
    def test_a_session_with_too_many_jobs_waiting_is_refused_another(self, monkeypatch):
        session_id = _session()
        jobs.create(Path(get_settings().upload_dir) / session_id, "train", {})  # queued, never run
        _set(monkeypatch, MAX_UNFINISHED_JOBS_PER_SESSION=1)
        refused = client.post(f"/api/model/{session_id}/train", json=TRAIN)
        assert refused.status_code == 429
        assert refused.json()["code"] == "too_many_jobs"

    def test_finished_jobs_do_not_count(self, monkeypatch):
        session_id = _session()
        folder = jobs.create(Path(get_settings().upload_dir) / session_id, "train", {})
        jobs.update_state(folder, state=jobs.DONE)
        _set(monkeypatch, MAX_UNFINISHED_JOBS_PER_SESSION=1)
        assert client.post(f"/api/model/{session_id}/train", json=TRAIN).status_code == 200
