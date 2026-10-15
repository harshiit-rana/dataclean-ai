"""The champion and the challenger, scored on the same newly labelled rows, each with its own cleaning and inputs."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import evaluation_ledger as ledger
from app.services import model_store
from app.services.session_store import session_dir

client = TestClient(app)
DROP_BLANK_SPEND = {"action_id": "a1", "operation": "drop_rows_with_nulls", "column_name": "spend",
                    "operation_params": {}, "status": "applied"}


class FakeGraph:
    def __init__(self):
        self.action_log: list[dict] = []

    async def aget_state(self, _config):
        return SimpleNamespace(values={"action_log": self.action_log}, next=())


def rows(n: int, seed: int, blanks: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, n)
    return pd.DataFrame({
        "signal": signal.round(4),
        "spend": np.where(rng.random(n) < blanks, np.nan, rng.normal(50, 10, n).round(2)),
        "plan": rng.choice(["basic", "pro"], n),
        "churned": np.where(rng.random(n) < 1 / (1 + np.exp(-2.5 * signal)), "yes", "no"),
    })


def csv(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


@pytest.fixture
def comparison(monkeypatch):
    """A champion trained before a fix that drops rows with a blank spend, a challenger trained after it."""
    graph = FakeGraph()
    monkeypatch.setattr(app.state, "graph", graph, raising=False)
    original = rows(400, seed=1)
    sid = client.post("/api/upload", files={"file": ("t.csv", io.BytesIO(csv(original)), "text/csv")}).json()["session_id"]
    directory = session_dir(get_settings().upload_dir, sid)
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "churned", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    train = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}
    champion = client.post(f"/api/model/{sid}/train", json=train).json()["run_id"]
    graph.action_log = [DROP_BLANK_SPEND]
    original.dropna(subset=["spend"]).to_csv(directory / "cleaned.csv", index=False)
    challenger = client.post(f"/api/model/{sid}/train", json=train).json()["run_id"]
    return sid, directory, champion, challenger, original


def start(sid, champion, challenger, **criteria):
    body = {"deciding_metric": "roc_auc", "min_improvement": 0.0, "constraints": [], **criteria}
    response = client.post(f"/api/model/{sid}/comparisons", json={
        "champion_run_id": champion, "challenger_run_id": challenger, "criteria": body})
    assert response.status_code == 200, response.text
    return response.json()["comparison_id"]


def evaluate(sid, cid, frame):
    return client.post(f"/api/model/{sid}/comparisons/{cid}/evaluate",
                       files={"file": ("eval.csv", io.BytesIO(csv(frame)), "text/csv")})


class TestTheSameRows:
    def test_both_are_scored_on_identical_rows_and_the_rest_are_counted(self, comparison):
        sid, _, champion, challenger, _ = comparison
        cid = start(sid, champion, challenger)
        fresh = rows(200, seed=9, blanks=0.15)
        body = evaluate(sid, cid, fresh).json()
        evaluation = body["evaluation"]
        blank_spend = int(fresh["spend"].isna().sum())
        # The challenger's own cleaning drops the rows with a blank spend; the champion's does not.
        assert evaluation["rows_excluded"]["only_one_model_could_score"] == blank_spend
        assert evaluation["rows_compared"] == len(fresh) - blank_spend
        assert set(evaluation["metrics"]) == {"champion", "challenger"}
        assert "roc_auc" in evaluation["metrics"]["champion"] and "brier" in evaluation["metrics"]["challenger"]

    def test_rows_from_the_training_file_are_left_out(self, comparison):
        sid, _, champion, challenger, original = comparison
        cid = start(sid, champion, challenger)
        mixed = pd.concat([rows(100, seed=9, blanks=0.0), original.dropna(subset=["spend"]).head(30)],
                          ignore_index=True)
        evaluation = evaluate(sid, cid, mixed).json()["evaluation"]
        assert evaluation["rows_excluded"]["in_the_training_file"] == 30
        assert evaluation["rows_compared"] == 100

    def test_rows_without_an_answer_are_left_out(self, comparison):
        sid, _, champion, challenger, _ = comparison
        cid = start(sid, champion, challenger)
        fresh = rows(100, seed=9, blanks=0.0)
        fresh.loc[:9, "churned"] = None
        evaluation = evaluate(sid, cid, fresh).json()["evaluation"]
        assert evaluation["rows_excluded"]["no_answer"] == 10 and evaluation["rows_compared"] == 90


class TestTheCriteria:
    def test_each_criterion_reports_pass_or_fail_with_its_evidence(self, comparison):
        sid, _, champion, challenger, _ = comparison
        cid = start(sid, champion, challenger, min_improvement=0.5, constraints=[
            {"kind": "metric_at_least", "metric": "recall", "value": 0.0},
            {"kind": "metric_at_least", "metric": "accuracy", "value": 1.01},
            {"kind": "no_worse_than_champion", "metric": "brier", "tolerance": 1.0},
            {"kind": "no_new_required_columns"},
        ])
        evaluation = evaluate(sid, cid, rows(200, seed=9, blanks=0.0)).json()["evaluation"]
        passed = [c["passed"] for c in evaluation["constraints"]]
        assert passed == [True, False, True, True]
        assert evaluation["constraints"][1]["evidence"].startswith("accuracy ")
        assert evaluation["deciding"]["passed"] is False and "needing at least 0.5" in evaluation["deciding"]["evidence"]
        assert evaluation["deciding"]["improvement_low"] <= evaluation["deciding"]["improvement"] <= evaluation["deciding"]["improvement_high"]
        assert evaluation["all_passed"] is False

    def test_the_comparison_is_frozen_once_evaluated(self, comparison):
        sid, _, champion, challenger, _ = comparison
        cid = start(sid, champion, challenger)
        evaluate(sid, cid, rows(100, seed=9, blanks=0.0))
        criteria = {"criteria": {"deciding_metric": "accuracy", "min_improvement": 0.0, "constraints": []}}
        assert client.put(f"/api/model/{sid}/comparisons/{cid}/criteria", json=criteria).status_code == 409
        again = evaluate(sid, cid, rows(100, seed=10, blanks=0.0))
        assert again.status_code == 409 and again.json()["code"] == "already_evaluated"


class TestRefusals:
    def test_a_file_without_answers_is_refused(self, comparison):
        sid, _, champion, challenger, _ = comparison
        cid = start(sid, champion, challenger)
        response = evaluate(sid, cid, rows(50, seed=9).drop(columns=["churned"]))
        assert response.status_code == 409 and response.json()["code"] == "no_answers"

    def test_a_file_a_model_cannot_score_is_refused_with_the_reason(self, comparison):
        sid, _, champion, challenger, _ = comparison
        cid = start(sid, champion, challenger)
        response = evaluate(sid, cid, rows(50, seed=9).drop(columns=["plan"]))
        assert response.status_code == 409 and response.json()["code"] == "cannot_score"
        assert "plan" in response.json()["detail"]


class TestTheLedger:
    def test_each_model_s_evaluation_is_recorded_and_no_look_is_counted(self, comparison):
        sid, directory, champion, challenger, _ = comparison
        split = model_store.load_result(directory)["split"]
        looks = ledger.look_count(directory, None, split)
        cid = start(sid, champion, challenger)
        evaluate(sid, cid, rows(100, seed=9, blanks=0.0))
        assert ledger.look_count(directory, None, split) == looks
        recorded = [e for e in ledger.looks(directory) if e["event"] == ledger.NEW_DATA and e.get("route") == "compare"]
        assert {e["run_id"] for e in recorded} == {champion, challenger}
