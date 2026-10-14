"""A typed instruction may still fill a column from a whole-file statistic.

With a target named, the app no longer *proposes* a median fill for a model input (the
pipeline does that on the training rows alone). Someone who types "fill the blank spends with
the median" has made their own call, so it runs -- with the warning, so they know what it costs.
The same guard serves both doors; `source` is the only thing that differs.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import routes_chat
from app.main import app
from app.services.session_store import session_dir
from app.config import get_settings

client = TestClient(app)


class FakeGraph:
    def __init__(self, values):
        self.values = values
        self.updates = []

    async def aget_state(self, _config):
        return SimpleNamespace(values=dict(self.values), next=())

    async def aupdate_state(self, _config, update):
        self.updates.append(update)


class FakeClient:
    def __init__(self, reply: str):
        self.reply = reply

    async def ainvoke(self, _messages):
        return SimpleNamespace(content=self.reply)


@pytest.fixture
def graph_slot():
    had = hasattr(app.state, "graph")
    previous = getattr(app.state, "graph", None)
    yield
    if had:
        app.state.graph = previous
    elif hasattr(app.state, "graph"):
        del app.state.graph


def _session(target: str | None) -> tuple[str, dict]:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "spend": np.where(rng.random(80) < 0.2, np.nan, rng.normal(50, 10, 80).round(2)),
            "plan": rng.choice(["basic", "pro"], 80),
            "churned": rng.choice(["yes", "no"], 80),
        }
    )
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    session_id = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    directory = session_dir(get_settings().upload_dir, session_id)
    frame.to_csv(directory / "cleaned.csv", index=False)

    goal = {
        "use_case": "ml_training" if target else "reporting",
        "expected_output": "predict churn",
        "critical_columns": [],
        "cleaning_aggression": "moderate",
        "target_column": target,
        "task_type": "auto",
    }
    values = {
        "session_id": session_id,
        "file_path": str(directory / "original.csv"),
        "file_name": "d.csv",
        "file_size_bytes": 100,
        "goal_context": goal,
        "action_log": [],
    }
    return session_id, values


def _chat(session_id: str, monkeypatch, operation: str = "fill_median"):
    reply = (
        '{"operation": "%s", "params": {"column": "spend"}, "explanation": "Filled the blanks."}'
        % operation
    )
    monkeypatch.setattr(routes_chat, "get_llm_client", lambda *_a, **_k: FakeClient(reply))
    return client.post(f"/api/chat/{session_id}", json={"message": "fill the blank spends"})


@pytest.mark.parametrize("operation", ["fill_median", "cap_outliers"])
def test_chat_still_runs_a_whole_file_fix_on_an_input_with_the_warning(
    graph_slot, monkeypatch, operation
):
    session_id, values = _session("churned")
    app.state.graph = FakeGraph(values)

    response = _chat(session_id, monkeypatch, operation)

    body = response.json()
    assert response.status_code == 200, body
    assert body["applied"] is True
    assert "whole file" in body["message"]
    cleaned = pd.read_csv(session_dir(get_settings().upload_dir, session_id) / "cleaned.csv")
    if operation == "fill_median":
        assert cleaned["spend"].isna().sum() == 0


def test_without_a_target_chat_runs_it_with_no_warning(graph_slot, monkeypatch):
    session_id, values = _session(None)
    app.state.graph = FakeGraph(values)

    body = _chat(session_id, monkeypatch).json()

    assert body["applied"] is True
    assert "whole file" not in body["message"]
