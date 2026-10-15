"""Promotion criteria: a closed set of typed constraints, fixed once the comparison is evaluated."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import get_settings
from app.main import app
from app.services import promotion
from app.services.session_store import session_dir

client = TestClient(app)

CRITERIA = {
    "deciding_metric": "roc_auc",
    "min_improvement": 0.01,
    "constraints": [
        {"kind": "metric_at_least", "metric": "recall", "value": 0.6},
        {"kind": "no_worse_than_champion", "metric": "brier", "tolerance": 0.01},
        {"kind": "no_new_required_columns"},
    ],
}


class TestTheCriteria:
    def test_a_valid_set_is_accepted(self):
        criteria = promotion.Criteria.model_validate(CRITERIA)
        assert [c.kind for c in criteria.constraints] == [
            "metric_at_least", "no_worse_than_champion", "no_new_required_columns"]

    @pytest.mark.parametrize("broken", [
        {**CRITERIA, "deciding_metric": "vibes"},
        {**CRITERIA, "min_improvement": -0.1},
        {**CRITERIA, "constraints": [{"kind": "run_this_code", "expression": "1 + 1"}]},
        {**CRITERIA, "constraints": [{"kind": "metric_at_least", "metric": "recall", "value": 0.6, "extra": 1}]},
        {**CRITERIA, "constraints": [{"kind": "no_worse_than_champion", "metric": "f1", "tolerance": -1}]},
        {**CRITERIA, "surprise": True},
    ])
    def test_anything_outside_the_closed_set_is_refused(self, broken):
        with pytest.raises(ValidationError):
            promotion.Criteria.model_validate(broken)


@pytest.fixture
def two_runs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    df = pd.DataFrame({"x": x, "z": rng.normal(size=300), "outcome": np.where(x + rng.normal(0, 1, 300) > 0, "yes", "no")})
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    train = {"include_deep_diagnostics": False}
    champion = client.post(f"/api/model/{sid}/train", json={**train, "model_names": ["logistic_regression"]}).json()["run_id"]
    challenger = client.post(f"/api/model/{sid}/train", json={**train, "model_names": ["random_forest_classifier"]}).json()["run_id"]
    return sid, champion, challenger


def start(sid, **body):
    return client.post(f"/api/model/{sid}/comparisons", json={"criteria": CRITERIA, **body})


class TestComparisons:
    def test_a_comparison_names_both_runs_and_keeps_its_criteria(self, two_runs):
        sid, champion, challenger = two_runs
        body = start(sid, champion_run_id=champion, challenger_run_id=challenger).json()
        assert body["status"] == "open" and body["champion_run_id"] == champion
        assert body["criteria"]["constraints"][0] == {"kind": "metric_at_least", "metric": "recall", "value": 0.6}
        assert client.get(f"/api/model/{sid}/comparisons/{body['comparison_id']}").json() == body

    def test_the_model_in_use_is_the_champion_by_default(self, two_runs):
        sid, champion, challenger = two_runs
        client.post(f"/api/model/{sid}/runs/{champion}/restore")
        body = start(sid, challenger_run_id=challenger).json()
        assert body["champion_run_id"] == champion

    def test_a_run_against_itself_is_refused(self, two_runs):
        sid, champion, _ = two_runs
        response = start(sid, champion_run_id=champion, challenger_run_id=champion)
        assert response.status_code == 409 and response.json()["code"] == "same_run"

    def test_an_unknown_run_is_not_found(self, two_runs):
        sid, champion, _ = two_runs
        response = start(sid, champion_run_id=champion, challenger_run_id="nope")
        assert response.status_code == 404 and response.json()["code"] == "run_not_found"

    def test_criteria_with_code_in_them_never_reach_the_store(self, two_runs):
        sid, champion, challenger = two_runs
        response = client.post(f"/api/model/{sid}/comparisons", json={
            "champion_run_id": champion, "challenger_run_id": challenger,
            "criteria": {**CRITERIA, "constraints": [{"kind": "python", "code": "__import__('os')"}]},
        })
        assert response.status_code == 422
        assert client.get(f"/api/model/{sid}/comparisons").json()["comparisons"] == []

    def test_criteria_can_change_while_open_and_are_frozen_once_evaluated(self, two_runs):
        sid, champion, challenger = two_runs
        cid = start(sid, champion_run_id=champion, challenger_run_id=challenger).json()["comparison_id"]
        changed = {**CRITERIA, "min_improvement": 0.05}
        response = client.put(f"/api/model/{sid}/comparisons/{cid}/criteria", json={"criteria": changed})
        assert response.status_code == 200 and response.json()["criteria"]["min_improvement"] == 0.05

        promotion.update(session_dir(get_settings().upload_dir, sid), cid, status="evaluated")
        frozen = client.put(f"/api/model/{sid}/comparisons/{cid}/criteria", json={"criteria": CRITERIA})
        assert frozen.status_code == 409 and frozen.json()["code"] == "criteria_frozen"
        assert client.get(f"/api/model/{sid}/comparisons/{cid}").json()["criteria"]["min_improvement"] == 0.05

    def test_comparisons_are_listed_newest_first(self, two_runs):
        sid, champion, challenger = two_runs
        first = start(sid, champion_run_id=champion, challenger_run_id=challenger).json()["comparison_id"]
        second = start(sid, champion_run_id=challenger, challenger_run_id=champion).json()["comparison_id"]
        listed = [c["comparison_id"] for c in client.get(f"/api/model/{sid}/comparisons").json()["comparisons"]]
        assert set(listed) == {first, second}

    def test_an_unknown_comparison_is_not_found(self, two_runs):
        sid, *_ = two_runs
        response = client.get(f"/api/model/{sid}/comparisons/nope")
        assert response.status_code == 404 and response.json()["code"] == "comparison_not_found"
