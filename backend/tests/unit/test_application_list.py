"""Earlier applications of the model to new files are listed, newest first, and their files can be downloaded."""

from __future__ import annotations

import io
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


async def _state(_config):
    return SimpleNamespace(values={"action_log": []}, next=())


def frame(n: int, seed: int, target: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x.round(4), "z": rng.normal(size=n).round(4)})
    if target:
        df["outcome"] = np.where(x + rng.normal(0, 1, n) > 0, "yes", "no")
    return df


def csv(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def test_applications_are_listed_newest_first_and_downloadable(monkeypatch):
    monkeypatch.setattr(app.state, "graph", SimpleNamespace(aget_state=_state), raising=False)
    sid = client.post("/api/upload", files={"file": ("t.csv", io.BytesIO(csv(frame(300, 1))), "text/csv")}).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    client.post(f"/api/model/{sid}/train", json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False})
    assert client.get(f"/api/apply/{sid}").json() == {"applications": []}

    ids = []
    for name, seed in (("march.csv", 7), ("april.csv", 8)):
        response = client.post(f"/api/apply/{sid}", files={"file": (name, io.BytesIO(csv(frame(40, seed, False))), "text/csv")})
        assert response.status_code == 200, response.text
        ids.append(response.json()["apply_id"])
        time.sleep(1.1)  # the listing is ordered by the second each was made

    listed = client.get(f"/api/apply/{sid}").json()["applications"]
    assert [a["apply_id"] for a in listed] == ids[::-1]
    assert listed[0]["file_name"] == "april.csv" and listed[0]["rows_in"] == 40 and listed[0]["predictions"] is True
    assert client.get(listed[1]["downloads"]["predictions"]).status_code == 200
    assert client.get(listed[1]["downloads"]["cleaned"]).status_code == 200


def test_an_unknown_session_is_not_found():
    assert client.get("/api/apply/no-such-session").status_code == 404
