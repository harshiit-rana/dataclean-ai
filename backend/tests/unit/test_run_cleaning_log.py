"""Each run keeps the fixes it was trained after, so its clean.py does not drift when the log changes later.

The action log lives in the graph and changes after the fact -- an undo, a chat instruction. A run recorded the data's
fingerprint but not which approved fixes made that data, so the clean.py exported beside its model was written from
whatever the log said at export time. A run now snapshots its applied fixes; the bundle writes clean.py from the
current run's snapshot, and any stored run's own clean.py can be fetched.
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
from app.services import model_store
from app.services.session_store import session_dir

client = TestClient(app)
TRAIN = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}


class FakeGraph:
    def __init__(self, action_log):
        self.action_log = action_log

    async def aget_state(self, _config):
        return SimpleNamespace(values={"action_log": self.action_log}, next=())


def log(*, dedupe_status: str = "applied") -> list[dict]:
    return [
        {"action_id": "a1", "operation": "fill_median", "column_name": "spend", "operation_params": {},
         "status": "applied"},
        {"action_id": "a2", "operation": "drop_duplicate_rows", "column_name": "DATASET", "operation_params": {},
         "status": dedupe_status},
    ]


def frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    signal = rng.normal(0, 1, n)
    return pd.DataFrame({
        "signal": signal.round(4),
        "spend": np.where(rng.random(n) < 0.1, np.nan, rng.normal(50, 10, n).round(2)),
        "churned": np.where(rng.random(n) < 1 / (1 + np.exp(-2.5 * signal)), "yes", "no"),
    })


@pytest.fixture
def graph(monkeypatch):
    fake = FakeGraph(log())
    monkeypatch.setattr(app.state, "graph", fake, raising=False)
    return fake


def session() -> tuple[str, object]:
    df = frame()
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    sid = client.post("/api/upload", files={"file": ("t.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
    directory = session_dir(get_settings().upload_dir, sid)
    df.assign(spend=df.spend.fillna(df.spend.median())).drop_duplicates().to_csv(directory / "cleaned.csv", index=False)
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "churned", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    return sid, directory


def bundle_clean_script(sid: str) -> str:
    response = client.get(f"/api/export/{sid}/zip")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        return archive.read("clean.py").decode()


class TestTheSnapshot:
    def test_a_run_records_the_fixes_it_was_trained_after(self, graph):
        sid, _ = session()
        body = client.post(f"/api/model/{sid}/train", json=TRAIN).json()
        actions = body["run_metadata"]["actions"]
        assert [a["action_id"] for a in actions] == ["a1", "a2"]
        assert actions[0]["operation"] == "fill_median" and actions[0]["column_name"] == "spend"
        entry = client.get(f"/api/model/{sid}/history").json()["runs"][0]
        assert entry["applied_action_ids"] == ["a1", "a2"]

    def test_only_applied_fixes_are_kept(self, graph):
        graph.action_log = log(dedupe_status="rejected")
        sid, _ = session()
        body = client.post(f"/api/model/{sid}/train", json=TRAIN).json()
        assert body["run_metadata"]["applied_action_ids"] == ["a1"]


class TestTheScriptFollowsTheRun:
    def test_a_later_change_to_the_log_does_not_change_the_run_s_clean_py(self, graph):
        sid, _ = session()
        run_id = client.post(f"/api/model/{sid}/train", json=TRAIN).json()["run_id"]
        graph.action_log = log(dedupe_status="undone")

        live = client.get(f"/api/export/{sid}/clean-script").text
        own = client.get(f"/api/model/{sid}/runs/{run_id}/clean-script")
        assert own.status_code == 200
        assert "drop_duplicates" not in live
        assert "drop_duplicates" in own.text and "drop_duplicates" in bundle_clean_script(sid)

    def test_a_new_run_after_the_change_gets_the_new_log(self, graph):
        sid, _ = session()
        client.post(f"/api/model/{sid}/train", json=TRAIN)
        graph.action_log = log(dedupe_status="undone")
        body = client.post(f"/api/model/{sid}/train", json=TRAIN).json()
        assert body["run_metadata"]["applied_action_ids"] == ["a1"]
        assert "drop_duplicates" not in bundle_clean_script(sid)

    def test_without_a_current_run_the_bundle_uses_the_log_as_it_stands(self, graph):
        sid, directory = session()
        client.post(f"/api/model/{sid}/train", json=TRAIN)
        graph.action_log = log(dedupe_status="undone")
        model_store.clear(directory)
        assert "drop_duplicates" not in bundle_clean_script(sid)

    def test_a_run_with_no_snapshot_has_no_clean_py_of_its_own(self, graph):
        sid, _ = session()
        response = client.get(f"/api/model/{sid}/runs/nosuchrun/clean-script")
        assert response.status_code >= 400 and "no record of the fixes" in response.json()["detail"]


class TestTheCard:
    def test_the_card_names_the_data_version_and_the_number_of_fixes(self, graph):
        from app.services.model_card import build_model_card

        sid, directory = session()
        client.post(f"/api/model/{sid}/train", json=TRAIN)
        card = build_model_card(session_id=sid, file_name="t.csv", result=model_store.load_result(directory),
                                summary="A.", interpretation="B.")
        assert "after 2 approved fixes" in card and "| Data version |" in card
