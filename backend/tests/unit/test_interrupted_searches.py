"""A search the process never finished leaves an `interrupted` history entry, not nothing.

Each search writes a marker as it starts, adds rounds as they finish and removes it on the way out, however it ended.
A marker whose heartbeat has stopped can only belong to a search whose process died; the next read of the history
records it with the rounds that had finished.
"""

from __future__ import annotations

import io
import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import improvement, modeling, search_markers
from app.services.session_store import session_dir

client = TestClient(app)


def session() -> tuple[str, object]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    df = pd.DataFrame({"x": x, "z": rng.normal(size=300), "outcome": np.where(x + rng.normal(0, 1, 300) > 0, "yes", "no")})
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    sid = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    return sid, session_dir(get_settings().upload_dir, sid)


def markers(directory) -> list:
    root = directory / search_markers.SEARCHES_DIR
    return list(root.glob("*.json")) if root.exists() else []


def leave_marker(directory, *, age: float, rounds: list[dict]) -> None:
    """What a search killed with its process leaves behind."""
    root = directory / search_markers.SEARCHES_DIR
    root.mkdir(parents=True, exist_ok=True)
    started = time.time() - age - 60
    (root / "dead12345678.json").write_text(json.dumps({
        "search_id": "dead12345678", "route": "improve_stream", "request": {"budget": 8, "model_names": []},
        "target_column": "outcome", "task": "classification", "started_at": started,
        "heartbeat_at": time.time() - age, "rounds": rounds,
    }))


ROUNDS = [{"name": "baseline", "label": "Which model family", "accepted": True},
          {"name": "hyperparameters", "label": "Settings", "accepted": False}]


class TestAKilledSearch:
    def test_a_stale_marker_becomes_an_interrupted_entry_with_its_rounds(self):
        sid, directory = session()
        leave_marker(directory, age=search_markers.STALE_AFTER_SECONDS + 60, rounds=ROUNDS)
        entry = client.get(f"/api/model/{sid}/history").json()["runs"][0]
        assert entry["status"] == "interrupted" and entry["code"] == "interrupted"
        assert [r["name"] for r in entry["rounds_completed"]] == ["baseline", "hyperparameters"]
        assert "2 rounds had finished: Which model family, Settings." in entry["reason"]
        assert entry["request"]["budget"] == 8 and entry["runtime_seconds"] >= 60
        assert markers(directory) == []

    def test_it_is_recorded_once(self):
        sid, directory = session()
        leave_marker(directory, age=search_markers.STALE_AFTER_SECONDS + 60, rounds=[])
        client.get(f"/api/model/{sid}/history")
        runs = client.get(f"/api/model/{sid}/history").json()["runs"]
        assert [r["status"] for r in runs] == ["interrupted"]
        assert "No round had finished." in runs[0]["reason"]

    def test_a_search_still_beating_is_left_alone(self):
        sid, directory = session()
        leave_marker(directory, age=5, rounds=ROUNDS)
        assert client.get(f"/api/model/{sid}/history").json()["runs"] == []
        assert len(markers(directory)) == 1

    def test_an_unreadable_marker_counts_as_interrupted(self):
        sid, directory = session()
        root = directory / search_markers.SEARCHES_DIR
        root.mkdir(parents=True)
        (root / "broken.json").write_text("{half a marker")
        assert client.get(f"/api/model/{sid}/history").json()["runs"][0]["status"] == "interrupted"


class TestASearchThatEnds:
    def test_a_completed_search_leaves_no_marker(self):
        sid, directory = session()
        response = client.post(
            f"/api/model/{sid}/improve",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False, "budget": 4},
        )
        assert response.status_code == 200
        assert markers(directory) == []
        assert client.get(f"/api/model/{sid}/history").json()["runs"][0]["status"] == "succeeded"

    def test_a_failed_search_leaves_no_marker_and_only_its_failure(self, monkeypatch):
        def crash(*args, **kwargs):
            raise modeling.ModelingError("No usable columns.", code="no_usable_columns")

        monkeypatch.setattr(improvement, "improve", crash)
        sid, directory = session()
        assert client.post(f"/api/model/{sid}/improve", json={"budget": 4}).status_code == 422
        assert markers(directory) == []
        assert [r["status"] for r in client.get(f"/api/model/{sid}/history").json()["runs"]] == ["failed"]

    def test_the_marker_holds_each_round_while_the_search_runs(self, monkeypatch):
        seen: list[list[str]] = []
        real = improvement.improve

        def watching(*args, on_progress=None, **kwargs):
            def progress(entry):
                on_progress(entry)
                (path,) = markers(directory)
                seen.append([r["name"] for r in json.loads(path.read_text())["rounds"]])

            return real(*args, on_progress=progress, **kwargs)

        monkeypatch.setattr(improvement, "improve", watching)
        sid, directory = session()
        client.post(
            f"/api/model/{sid}/improve",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False, "budget": 4},
        )
        assert seen and seen[0] == ["baseline"] and seen[1][:2] == ["baseline", seen[1][1]]
        assert markers(directory) == []


class TestTheMarker:
    def test_two_searches_in_one_session_have_separate_markers(self, tmp_path):
        with search_markers.SearchMarker(tmp_path, route="improve", request={}, target="y", task="classification") as a:
            with search_markers.SearchMarker(tmp_path, route="improve", request={}, target="y", task="classification") as b:
                assert a.path != b.path and a.path.exists() and b.path.exists()
            assert not b.path.exists() and a.path.exists()
        assert not a.path.exists()

    def test_the_heartbeat_keeps_a_long_round_from_looking_dead(self, tmp_path, monkeypatch):
        monkeypatch.setattr(search_markers, "HEARTBEAT_SECONDS", 0.05)
        with search_markers.SearchMarker(tmp_path, route="improve", request={}, target="y", task="c") as marker:
            first = json.loads(marker.path.read_text())["heartbeat_at"]
            time.sleep(0.3)
            assert json.loads(marker.path.read_text())["heartbeat_at"] > first

    def test_the_marker_is_removed_when_the_search_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            with search_markers.SearchMarker(tmp_path, route="improve", request={}, target="y", task="c") as marker:
                raise RuntimeError("boom")
        assert not marker.path.exists()
