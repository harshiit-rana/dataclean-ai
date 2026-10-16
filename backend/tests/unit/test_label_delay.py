"""Answers that arrive late are reported for the period they cover, and never passed off as current performance."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services import label_delay

client = TestClient(app)


def dated(n: int, seed: int, answered_share: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    df = pd.DataFrame({
        "when": pd.date_range("2026-01-01", periods=n, freq="D").astype(str),
        "x": x.round(4),
        "churned": np.where(x + rng.normal(0, 1, n) > 0, "yes", "no"),
    })
    unanswered = int(n * (1 - answered_share))
    if unanswered:
        df.loc[n - unanswered:, "churned"] = None  # the most recent rows have no answer yet
    return df


class TestCoverage:
    def test_answers_bunched_in_the_older_period_are_flagged(self):
        report, excluded = label_delay.coverage(dated(100, 1, answered_share=0.4), "churned", "when")
        assert report["answered"]["to"] < report["unanswered"]["from"]
        assert report["share_answered"] == 0.4 and excluded == set()
        assert any("mostly older" in w for w in report["warnings"])
        assert any("Only 40%" in w for w in report["warnings"])

    def test_a_fully_answered_file_has_no_warning(self):
        report, _ = label_delay.coverage(dated(100, 1), "churned", "when")
        assert report["warnings"] == [] and report["unanswered"] is None

    def test_an_outcome_window_leaves_out_the_recent_rows_and_counts_them(self):
        report, excluded = label_delay.coverage(dated(100, 1), "churned", "when", window_days=30)
        assert len(excluded) == 30 and report["excluded_by_window"] == 30

    def test_the_model_s_own_time_column_is_preferred(self):
        frame = dated(10, 1).assign(other=pd.date_range("2020-01-01", periods=10).astype(str))
        assert label_delay.date_column(frame, {"split": {"time_column": "other"}, "target_column": "churned"}) == "other"
        assert label_delay.date_column(frame, {"target_column": "churned"}) == "when"
        assert label_delay.date_column(frame.drop(columns=["when", "other"]), {"target_column": "churned"}) is None


async def _state(_config):
    return SimpleNamespace(values={"action_log": []}, next=())


def csv(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def test_the_apply_check_reports_coverage_and_honours_the_window(monkeypatch):
    monkeypatch.setattr(app.state, "graph", SimpleNamespace(aget_state=_state), raising=False)
    sid = client.post("/api/upload", files={"file": ("t.csv", io.BytesIO(csv(dated(300, 1))), "text/csv")}).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "churned", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    assert client.post(f"/api/model/{sid}/train", json={
        "model_names": ["logistic_regression"], "include_deep_diagnostics": False}).status_code == 200
    new = dated(200, 9, answered_share=0.6)
    new["when"] = pd.date_range("2027-01-01", periods=200, freq="D").astype(str)

    body = client.post(f"/api/apply/{sid}", files={"file": ("n.csv", io.BytesIO(csv(new)), "text/csv")}).json()
    assert body["answer_coverage"]["date_column"] == "when"
    assert any("mostly older" in w for w in body["answer_coverage"]["warnings"])
    rows_without_window = body["answers"]["rows"]

    windowed = client.post(f"/api/apply/{sid}", data={"outcome_window_days": "150"},
                           files={"file": ("n.csv", io.BytesIO(csv(new)), "text/csv")}).json()
    assert windowed["answer_coverage"]["excluded_by_window"] > 0
    assert windowed["answers"]["rows"] == rows_without_window - windowed["answer_coverage"]["excluded_by_window"]
