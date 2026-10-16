"""The streamed improvement loop.

The property that matters most is `test_the_streamed_result_is_the_saved_result`: the page
renders whatever the stream's last event says, and a refresh renders whatever was saved. If
those two could differ, a user who reloads mid-read would see a different model from the one
they were just looking at.
"""

from __future__ import annotations

import io
import json
import time

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services import improvement, modeling

client = TestClient(app)


def _csv(n: int = 300) -> bytes:
    rng = np.random.default_rng(17)
    signal = rng.normal(0, 1, n)
    buffer = io.StringIO()
    pd.DataFrame(
        {
            "signal": signal.round(4),
            "noise": rng.normal(0, 1, n).round(4),
            "region": rng.choice(["north", "south"], n),
            "outcome": np.where(rng.random(n) < 1 / (1 + np.exp(-2.4 * signal)), "yes", "no"),
        }
    ).to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def _session(payload: bytes | None = None) -> str:
    response = client.post(
        "/api/upload",
        files={"file": ("data.csv", io.BytesIO(payload or _csv()), "text/csv")},
    )
    session_id = response.json()["session_id"]
    goal = {
        "use_case": "ml_training",
        "expected_output": "predict who leaves",
        "target_column": "outcome",
        "task_type": "auto",
    }
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


def _stream(session_id: str, body: dict | None = None):
    response = client.post(
        f"/api/model/{session_id}/improve/stream",
        json=body or {"model_names": ["logistic_regression"], "budget": 4,
                      "include_deep_diagnostics": False},
    )
    events = [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    return response, events


class TestTheStream:
    def test_it_is_an_event_stream(self):
        response, _events = _stream(_session())
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    def test_it_ends_with_the_result_and_nothing_after(self):
        _response, events = _stream(_session())
        assert events[-1]["type"] == "result"
        assert sum(1 for e in events if e["type"] == "result") == 1

    def test_each_round_is_announced_before_it_is_reported(self):
        """A settings search can run for most of a minute. A stream that only speaks when a
        round ends looks exactly like one that has hung."""
        _response, events = _stream(_session())
        seen_stages: set[str] = set()
        for event in events:
            if event["type"] == "stage":
                seen_stages.add(event["name"])
                assert event["description"]
            elif event["type"] == "round":
                assert event["round"]["name"] in seen_stages, event["round"]["name"]

    def test_the_final_refit_is_announced_too(self):
        _response, events = _stream(_session())
        stages = [e["name"] for e in events if e["type"] == "stage"]
        assert stages[0] == "baseline"
        assert stages[-1] == "rebuilding"

    def test_the_streamed_rounds_are_the_rounds_in_the_result(self):
        _response, events = _stream(_session())
        streamed = [e["round"] for e in events if e["type"] == "round"]
        final = events[-1]["result"]["improvement"]["rounds"]
        assert streamed == final

    def test_the_streamed_result_is_the_saved_result(self):
        session_id = _session()
        _response, events = _stream(session_id)
        saved = client.get(f"/api/model/{session_id}/result").json()
        assert events[-1]["result"] == saved

    def test_the_run_reaches_the_history(self):
        session_id = _session()
        _stream(session_id)
        runs = client.get(f"/api/model/{session_id}/history").json()["runs"]
        assert len(runs) == 1 and runs[0]["tuned"]


class TestWhatIsRefusedBeforeTheStreamOpens:
    """Refusals arrive as ordinary JSON errors with a status code, not as an event inside a
    200 -- the page already knows how to show those."""

    def test_an_unreadable_trade_off_is_a_422(self):
        response, events = _stream(
            _session(),
            {"model_names": ["logistic_regression"], "budget": 4,
             "operating_point": {"objective": "min_recall"}},
        )
        assert response.status_code == 422
        assert events == []
        assert "target rate" in response.json()["detail"]

    def test_a_blocked_verdict_is_refused_before_any_fitting(self):
        rng = np.random.default_rng(1)
        n = 300
        frame = pd.DataFrame(
            {
                "signal": rng.normal(0, 1, n),
                # Almost nothing labelled: readiness says blocked.
                "outcome": ["yes", "no"] * 10 + [None] * (n - 20),
            }
        )
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        response, events = _stream(_session(buffer.getvalue().encode()))
        assert response.status_code == 422
        assert events == []


class TestFailuresMidRun:
    def test_a_modelling_failure_becomes_an_error_event_with_the_reason(self, monkeypatch):
        def fails(*_args, **_kwargs):
            raise modeling.ModelingError("There are too few rows in the smallest class.")

        monkeypatch.setattr(improvement, "improve", fails)
        response, events = _stream(_session())
        assert response.status_code == 200  # the stream had already opened
        assert events[-1] == {
            "type": "error",
            "detail": "There are too few rows in the smallest class.",
        }

    def test_an_unexpected_failure_says_so_rather_than_hanging(self, monkeypatch):
        def breaks(*_args, **_kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(improvement, "improve", breaks)
        _response, events = _stream(_session())
        assert events[-1]["type"] == "error"
        # A stable message and the request id to quote, never the exception's own text (#111).
        assert "stopped unexpectedly" in events[-1]["detail"] and "disk full" not in events[-1]["detail"]


class TestKeepalive:
    def test_a_long_silence_is_filled_with_a_comment_line(self, monkeypatch):
        """Proxies close connections that look idle. A comment line is ignored by every SSE
        client and keeps the connection looking alive."""
        from app.api import job_runner

        real = improvement.improve

        def slow(*args, **kwargs):
            time.sleep(0.4)
            return real(*args, **kwargs)

        monkeypatch.setattr(improvement, "improve", slow)
        monkeypatch.setattr(job_runner, "KEEPALIVE_SECONDS", 0.05)
        response, events = _stream(_session())
        assert ": still working" in response.text
        assert events[-1]["type"] == "result"  # and it still finishes properly
