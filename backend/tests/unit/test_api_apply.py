"""POST /api/apply -- a finished session used on a new file, end to end.

The model is trained for real through the API. The action log lives in the graph checkpoint,
which these tests replace with a fake that returns a fixed log -- the same approach as the
profile-reload tests.
"""

from __future__ import annotations

import csv
import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class FakeGraph:
    def __init__(self, action_log):
        self.action_log = action_log

    async def aget_state(self, _config):
        return SimpleNamespace(values={"action_log": self.action_log}, next=())


@pytest.fixture
def graph_slot():
    had = hasattr(app.state, "graph")
    previous = getattr(app.state, "graph", None)
    yield
    if had:
        app.state.graph = previous
    elif hasattr(app.state, "graph"):
        del app.state.graph


def _frame(n: int, seed: int, *, with_target: bool = True, blanks: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, n)
    frame = pd.DataFrame(
        {
            "signal": signal.round(4),
            "spend": np.where(rng.random(n) < blanks, np.nan, rng.normal(50, 10, n).round(2)),
            "plan": rng.choice(["basic", "pro"], n),
        }
    )
    if with_target:
        frame["churned"] = np.where(rng.random(n) < 1 / (1 + np.exp(-2.5 * signal)), "yes", "no")
    return frame


def _csv(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


LOG = [
    {"action_id": "a1", "operation": "fill_median", "column_name": "spend",
     "operation_params": {}, "status": "applied"},
    {"action_id": "a2", "operation": "drop_duplicate_rows", "column_name": "DATASET",
     "operation_params": {}, "status": "applied"},
]


def _session(*, train: bool = True, target: str | None = "churned") -> str:
    session_id = client.post(
        "/api/upload", files={"file": ("train.csv", io.BytesIO(_csv(_frame(300, 1))), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "predict churn", "task_type": "auto"}
    if target:
        goal["target_column"] = target
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    if train:
        response = client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False},
        )
        assert response.status_code == 200, response.text
    return session_id


def _apply(session_id: str, frame: pd.DataFrame, name: str = "march.csv"):
    return client.post(
        f"/api/apply/{session_id}", files={"file": (name, io.BytesIO(_csv(frame)), "text/csv")}
    )


def _new_rows() -> pd.DataFrame:
    """No answers, one blank to fill, one exact duplicate to drop."""
    frame = _frame(60, 7, with_target=False, blanks=0.0)
    frame.loc[5, "spend"] = np.nan
    frame.loc[59] = frame.loc[10]
    return frame


class TestWithAModel:
    def test_every_uploaded_row_comes_back_scored_or_explained(self, graph_slot):
        app.state.graph = FakeGraph(LOG)
        session_id = _session()
        response = _apply(session_id, _new_rows())
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["rows_in"] == 60
        assert body["rows_kept"] == 59
        assert body["dropped"] == [{"row": 60, "reason": "it duplicated an earlier row"}]
        assert body["predictions"]["count"] == 59

        download = client.get(body["downloads"]["predictions"])
        assert download.status_code == 200
        rows = list(csv.DictReader(io.StringIO(download.text)))
        assert len(rows) == 60  # every uploaded row, including the dropped one
        assert rows[59]["predicted_churned"] == ""
        assert rows[59]["dataclean_note"] == "no prediction: it duplicated an earlier row"
        assert rows[0]["predicted_churned"] in {"yes", "no"}
        assert rows[0]["signal"]  # the uploaded values ride along, for joining back

    def test_the_blank_is_filled_with_the_training_median(self, graph_slot):
        app.state.graph = FakeGraph(LOG)
        session_id = _session()
        body = _apply(session_id, _new_rows()).json()
        learned = body["steps"][0]["learned"]["fill_value"]
        training_median = _frame(300, 1)["spend"].median()
        assert learned == pytest.approx(training_median)
        cleaned = pd.read_csv(io.StringIO(client.get(body["downloads"]["cleaned"]).text))
        assert cleaned.loc[5, "spend"] == pytest.approx(training_median)

    def test_the_missing_target_is_expected_rather_than_an_error(self, graph_slot):
        log = LOG + [{"action_id": "a3", "operation": "drop_rows_with_nulls",
                      "column_name": "churned", "operation_params": {}, "status": "applied"}]
        app.state.graph = FakeGraph(log)
        body = _apply(_session(), _new_rows()).json()
        step = next(s for s in body["steps"] if s["column"] == "churned")
        assert step["status"] == "skipped"
        assert "expected" in step["note"]
        assert body["answers"] is None

    def test_drift_is_reported(self, graph_slot):
        app.state.graph = FakeGraph(LOG)
        body = _apply(_session(), _new_rows()).json()
        assert body["drift"]["verdict"] in {"similar", "shifted"}
        assert {c["column"] for c in body["drift"]["columns"]} >= {"signal", "spend"}

    def test_a_file_that_has_moved_is_called_different(self, graph_slot):
        app.state.graph = FakeGraph(LOG)
        moved = _new_rows()
        moved["spend"] = moved["spend"] + 200
        body = _apply(_session(), moved).json()
        assert body["drift"]["verdict"] == "different"
        assert "'spend'" in body["drift"]["headline"]

    def test_real_answers_in_the_file_are_checked(self, graph_slot):
        app.state.graph = FakeGraph(LOG)
        body = _apply(_session(), _frame(200, 11)).json()
        assert body["answers"]["rows"] == 200
        assert body["answers"]["now"] is not None
        assert body["answers"]["metric"] == "roc_auc"

    def test_too_few_answers_is_not_a_score(self, graph_slot):
        app.state.graph = FakeGraph(LOG)
        body = _apply(_session(), _frame(12, 11)).json()
        assert body["answers"]["now"] is None
        assert "isn't enough" in body["answers"]["headline"]

    def test_a_missing_input_is_refused_by_name(self, graph_slot):
        app.state.graph = FakeGraph(LOG)
        response = _apply(_session(), _new_rows().drop(columns=["signal"]))
        assert response.status_code == 422
        assert "signal" in response.json()["detail"]


class TestWithoutAModel:
    def test_cleaning_alone_still_applies(self, graph_slot):
        app.state.graph = FakeGraph(LOG)
        body = _apply(_session(train=False), _new_rows()).json()
        assert body["rows_kept"] == 59
        assert body["predictions"] is None
        assert "predictions" not in body["downloads"]
        assert client.get(body["downloads"]["cleaned"]).status_code == 200

    def test_nothing_to_apply_says_so(self, graph_slot):
        app.state.graph = FakeGraph([])
        response = _apply(_session(train=False), _new_rows())
        assert response.status_code == 409
        assert "nothing to apply" in response.json()["detail"].lower()


class TestDownloadsAreClosed:
    @pytest.mark.parametrize(
        "apply_id,kind",
        [("..%2F..%2Fmeta", "cleaned"), ("abcdef123456", "original"), ("ABCDEF123456", "cleaned")],
    )
    def test_only_files_this_route_wrote_can_be_fetched(self, graph_slot, apply_id, kind):
        app.state.graph = FakeGraph(LOG)
        session_id = _session(train=False)
        response = client.get(f"/api/apply/{session_id}/{apply_id}/{kind}")
        assert response.status_code == 404

    def test_a_non_csv_upload_is_refused(self, graph_slot):
        app.state.graph = FakeGraph(LOG)
        response = client.post(
            f"/api/apply/{_session(train=False)}",
            files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code in (400, 415, 422)
