"""Re-running a recorded configuration and saying whether its numbers come back.

A run records its request, software and seeds (#63); a reproduction proves the record is enough. On the same data,
code and libraries every model here is seeded, so a re-run matches to the last digit. When it does not, the likely
causes are listed. A reproduction is not saved and is not a look at the held-out rows (ADR 0001).
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import evaluation_ledger as ledger
from app.services import model_store, run_reproduction
from app.services.session_store import session_dir

client = TestClient(app)
LR = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}


def frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "outcome": np.where(x + rng.normal(0, 1, n) > 0, "yes", "no")})


@pytest.fixture
def session():
    buffer = io.StringIO()
    frame().to_csv(buffer, index=False)
    sid = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    return sid, session_dir(get_settings().upload_dir, sid)


def reproduce(sid: str, run_id: str):
    return client.post(f"/api/model/{sid}/runs/{run_id}/reproduce")


class TestReproducing:
    def test_a_deterministic_train_reproduces_exactly(self, session):
        sid, directory = session
        run_id = client.post(f"/api/model/{sid}/train", json=LR).json()["run_id"]
        body = reproduce(sid, run_id).json()
        assert body["reproduced"] is True and body["likely_causes"] == []
        assert body["differences"] and all(d["difference"] == 0 for d in body["differences"])
        entry = client.get(f"/api/model/{sid}/history").json()["runs"][0]
        assert entry["run_id"] == run_id and entry["reproduction"]["reproduced"] is True

    def test_a_search_reproduces_exactly(self, session):
        sid, _ = session
        run_id = client.post(
            f"/api/model/{sid}/improve", json={**LR, "model_names": ["logistic_regression", "random_forest_classifier"],
                                               "budget": 4},
        ).json()["run_id"]
        assert reproduce(sid, run_id).json()["reproduced"] is True

    def test_a_reproduction_is_not_a_look_and_saves_no_run(self, session):
        sid, directory = session
        run_id = client.post(f"/api/model/{sid}/train", json=LR).json()["run_id"]
        split = model_store.load_result(directory)["split"]
        before_looks = ledger.look_count(directory, None, split)
        before_runs = len(client.get(f"/api/model/{sid}/history").json()["runs"])
        reproduce(sid, run_id)
        assert ledger.look_count(directory, None, split) == before_looks
        assert len(client.get(f"/api/model/{sid}/history").json()["runs"]) == before_runs
        assert model_store.current_run_id(directory) == run_id
        assert ledger.looks(directory)[-1]["event"] == ledger.REPRODUCTION

    def test_a_run_from_another_library_version_names_the_difference(self, session):
        sid, directory = session
        run_id = client.post(f"/api/model/{sid}/train", json=LR).json()["run_id"]
        entry = client.get(f"/api/model/{sid}/history").json()["runs"][0]
        model_store.update_history_entry(
            directory, run_id,
            environment={**entry["environment"], "scikit_learn": "0.24.2"},
            metrics={**entry["metrics"], "accuracy": entry["metrics"]["accuracy"] + 0.01},
        )
        body = reproduce(sid, run_id).json()
        assert body["reproduced"] is False
        assert any("scikit-learn was 0.24.2 when the run was made" in cause for cause in body["likely_causes"])
        assert "gave different numbers" in body["summary"]


class TestRefusing:
    def test_a_run_on_data_that_has_changed_is_refused(self, session):
        sid, directory = session
        run_id = client.post(f"/api/model/{sid}/train", json=LR).json()["run_id"]
        changed = frame()
        changed.loc[0, "x"] = 99.0
        changed.to_csv(directory / "cleaned.csv", index=False)
        response = reproduce(sid, run_id)
        assert response.status_code == 409 and response.json()["code"] == "data_changed"

    def test_a_run_with_no_recorded_request_is_refused(self, session):
        sid, directory = session
        run_id = client.post(f"/api/model/{sid}/train", json=LR).json()["run_id"]
        model_store.update_history_entry(directory, run_id, request=None)
        response = reproduce(sid, run_id)
        assert response.status_code == 409 and response.json()["code"] == "request_not_recorded"

    def test_an_unknown_run_is_not_found(self, session):
        sid, _ = session
        assert reproduce(sid, "nope").status_code == 404


class TestTheComparison:
    def test_a_different_model_chosen_does_not_reproduce(self):
        entry = {"run_id": "r", "best_model": "a", "metrics": {"roc_auc": 0.8}, "environment": {"python": "3"}}
        result = {"best_model": "b", "results": [{"model": "b", "metrics": {"roc_auc": 0.8}}]}
        found = run_reproduction.compare(entry, result)
        assert found["reproduced"] is False and any("different model was chosen" in c for c in found["likely_causes"])

    def test_a_time_limited_search_is_named(self):
        entry = {"run_id": "r", "best_model": "a", "metrics": {"roc_auc": 0.8}, "environment": {"python": "3"},
                 "request": {"time_limit_seconds": 30}}
        result = {"best_model": "a", "results": [{"model": "a", "metrics": {"roc_auc": 0.79}}]}
        assert any("time limit" in c for c in run_reproduction.compare(entry, result)["likely_causes"])

    def test_a_difference_inside_the_tolerance_reproduces(self):
        entry = {"run_id": "r", "best_model": "a", "metrics": {"roc_auc": 0.8}}
        result = {"best_model": "a", "results": [{"model": "a", "metrics": {"roc_auc": 0.8 + 1e-9}}]}
        assert run_reproduction.compare(entry, result)["reproduced"] is True
