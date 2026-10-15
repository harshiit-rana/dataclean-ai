"""The audit report now describes the model, so changing the model makes a written report stale.

Marked rather than rewritten, like every other change: rebuilding costs an LLM call, and a user retraining
a few times should pay once, when the report is asked for.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.config import get_settings
from app.graph.nodes.audit_agent import (
    STALE_MARKER,
    _model_for_report,
    is_report_stale,
    mark_report_stale_if_written,
)
from app.main import app
from app.services.session_store import session_dir

client = TestClient(app)
FAST = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}


def frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x.round(4), "z": rng.normal(size=n).round(4),
                         "outcome": np.where(x + rng.normal(size=n) > 0, "yes", "no")})


def session():
    buffer = io.StringIO()
    frame().to_csv(buffer, index=False)
    sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    return sid, session_dir(get_settings().upload_dir, sid)


class TestTheMarker:
    def test_an_unwritten_report_has_nothing_to_go_stale(self, tmp_path):
        mark_report_stale_if_written(tmp_path)
        assert not is_report_stale(tmp_path)

    def test_a_written_report_is_marked(self, tmp_path):
        (tmp_path / "audit_report.md").write_text("# report", encoding="utf-8")
        mark_report_stale_if_written(tmp_path)
        assert is_report_stale(tmp_path) and (tmp_path / STALE_MARKER).exists()


class TestTrainingMarksTheReportStale:
    def test_training_after_a_report_was_written_marks_it_stale(self):
        sid, directory = session()
        (directory / "audit_report.md").write_text("# report", encoding="utf-8")
        assert not is_report_stale(directory)
        assert client.post(f"/api/model/{sid}/train", json=FAST).status_code == 200
        assert is_report_stale(directory)

    def test_training_with_no_report_yet_leaves_no_marker(self):
        sid, directory = session()
        assert client.post(f"/api/model/{sid}/train", json=FAST).status_code == 200
        assert not is_report_stale(directory)

    def test_retraining_marks_it_stale_again(self):
        sid, directory = session()
        assert client.post(f"/api/model/{sid}/train", json=FAST).status_code == 200
        (directory / "audit_report.md").write_text("# report", encoding="utf-8")
        (directory / STALE_MARKER).unlink(missing_ok=True)
        assert client.post(f"/api/model/{sid}/train", json=FAST).status_code == 200
        assert is_report_stale(directory)


class TestTheReportReadsTheCurrentModel:
    def test_no_model_means_no_model_for_the_report(self, tmp_path):
        assert _model_for_report(tmp_path) is None

    def test_the_stored_model_comes_with_its_later_evidence(self):
        sid, directory = session()
        assert client.post(f"/api/model/{sid}/train", json=FAST).status_code == 200
        model = _model_for_report(directory)
        assert model["best_model"] == "logistic_regression"
        assert "new_data" in model["evaluation"] and "final" in model["evaluation"]
