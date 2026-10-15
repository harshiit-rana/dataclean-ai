"""Rolling back to the champion the last promotion replaced, checked and recorded."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import model_store, promotion
from app.services.session_store import session_dir

client = TestClient(app)


def rows(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, n)
    return pd.DataFrame({"signal": signal.round(4), "noise": rng.normal(0, 1, n).round(4),
                         "churned": np.where(rng.random(n) < 1 / (1 + np.exp(-3 * signal)), "yes", "no")})


def csv(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


@pytest.fixture
def promoted():
    """A strong challenger promoted over a weak champion."""
    sid = client.post("/api/upload", files={"file": ("t.csv", io.BytesIO(csv(rows(400, 1))), "text/csv")}).json()["session_id"]
    directory = session_dir(get_settings().upload_dir, sid)
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "churned", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    train = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}
    champion = client.post(f"/api/model/{sid}/train", json={**train, "excluded_columns": ["signal"]}).json()["run_id"]
    challenger = client.post(f"/api/model/{sid}/train", json=train).json()["run_id"]
    client.post(f"/api/model/{sid}/runs/{champion}/restore")
    cid = client.post(f"/api/model/{sid}/comparisons", json={
        "champion_run_id": champion, "challenger_run_id": challenger,
        "criteria": {"deciding_metric": "roc_auc"}}).json()["comparison_id"]
    client.post(f"/api/model/{sid}/comparisons/{cid}/evaluate",
                files={"file": ("e.csv", io.BytesIO(csv(rows(300, 9))), "text/csv")})
    assert client.post(f"/api/model/{sid}/comparisons/{cid}/decision", json={"decision": "promote"}).json()["decision"] == "promoted"
    return sid, directory, champion, challenger


def roll_back(sid, reason="It flagged too many good customers."):
    return client.post(f"/api/model/{sid}/rollback", json={"reason": reason})


class TestRollingBack:
    def test_the_previous_champion_is_put_back_exactly_and_the_rollback_is_recorded(self, promoted):
        sid, directory, champion, challenger = promoted
        before = model_store.file_sha256(model_store.run_dir(directory, champion) / model_store.PIPELINE_FILENAME)
        entry = roll_back(sid).json()
        assert entry["decision"] == "rolled_back" and entry["from_run_id"] == challenger and entry["to_run_id"] == champion
        assert entry["reason"] == "It flagged too many good customers."
        assert model_store.current_run_id(directory) == champion
        assert model_store.file_sha256(model_store.pipeline_path(directory)) == before
        assert [d["decision"] for d in client.get(f"/api/model/{sid}/decisions").json()["decisions"]] == [
            "promoted", "rolled_back"]

    def test_a_promotion_is_rolled_back_once(self, promoted):
        sid, *_ = promoted
        roll_back(sid)
        again = roll_back(sid)
        assert again.status_code == 409 and again.json()["code"] == "nothing_to_roll_back"

    def test_without_a_promotion_there_is_nothing_to_roll_back(self):
        sid = client.post("/api/upload", files={"file": ("t.csv", io.BytesIO(csv(rows(100, 1))), "text/csv")}).json()["session_id"]
        response = roll_back(sid)
        assert response.status_code == 409 and response.json()["code"] == "nothing_to_roll_back"

    def test_a_champion_trained_before_the_data_changed_is_not_put_back(self, promoted):
        sid, directory, champion, _ = promoted
        changed = rows(400, 1)
        changed.loc[0, "noise"] = 99.0
        changed.to_csv(directory / "cleaned.csv", index=False)
        response = roll_back(sid)
        assert response.status_code == 409 and response.json()["code"] == "data_changed"
        assert [d["decision"] for d in promotion.decisions(directory)] == ["promoted"]

    def test_a_champion_whose_model_file_changed_is_not_put_back(self, promoted):
        sid, directory, champion, challenger = promoted
        model = model_store.run_dir(directory, champion) / model_store.PIPELINE_FILENAME
        model.write_bytes(model.read_bytes() + b"x")
        response = roll_back(sid)
        assert response.status_code == 409 and response.json()["code"] == "model_changed"
        assert model_store.current_run_id(directory) == challenger

    def test_a_champion_no_longer_stored_cannot_be_put_back(self, promoted):
        import shutil

        sid, directory, champion, _ = promoted
        shutil.rmtree(model_store.run_dir(directory, champion))
        response = roll_back(sid)
        assert response.status_code == 404 and response.json()["code"] == "run_not_found"
