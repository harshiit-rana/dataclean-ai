"""The whole replacement flow over the API, as the epic's walkthrough describes it.

A champion trained on months 1-3, a challenger on months 1-4 (the data grew), both evaluated on month 5, which neither
has seen, against criteria fixed first; the challenger promoted and put in use, the decision recorded; and a rollback
refused, because the old champion was trained on data that is no longer the data.
"""

from __future__ import annotations

import io
import pathlib
import re

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import model_store
from app.services.session_store import session_dir

client = TestClient(app)


def month(m: int, n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(m)
    signal = rng.normal(0, 1, n)
    return pd.DataFrame({
        "month": m,
        "signal": signal.round(4),
        "spend": rng.normal(50, 10, n).round(2),
        "churned": np.where(rng.random(n) < 1 / (1 + np.exp(-3 * signal)), "yes", "no"),
    })


def csv(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def test_champion_on_months_one_to_three_challenger_on_one_to_four_evaluated_on_month_five():
    months_1_4 = pd.concat([month(m) for m in (1, 2, 3, 4)], ignore_index=True)
    sid = client.post("/api/upload", files={"file": ("m.csv", io.BytesIO(csv(months_1_4)), "text/csv")}).json()["session_id"]
    directory = session_dir(get_settings().upload_dir, sid)
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "churned", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    train = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False, "excluded_columns": ["month"]}

    # The champion, when the data held months 1-3.
    months_1_4[months_1_4.month <= 3].to_csv(directory / "cleaned.csv", index=False)
    champion = client.post(f"/api/model/{sid}/train", json=train).json()["run_id"]
    # Month 4 arrives: the challenger trains on months 1-4.
    (directory / "cleaned.csv").unlink()
    challenger = client.post(f"/api/model/{sid}/train", json=train).json()["run_id"]

    # Criteria first, then month 5.
    started = client.post(f"/api/model/{sid}/comparisons", json={
        "champion_run_id": champion, "challenger_run_id": challenger,
        "criteria": {"deciding_metric": "roc_auc", "min_improvement": 0.0, "constraints": [
            {"kind": "no_new_required_columns"},
            {"kind": "no_worse_than_champion", "metric": "accuracy", "tolerance": 0.05},
        ]},
    })
    assert started.status_code == 200, started.text
    cid = started.json()["comparison_id"]
    evaluated = client.post(f"/api/model/{sid}/comparisons/{cid}/evaluate",
                            files={"file": ("month5.csv", io.BytesIO(csv(month(5))), "text/csv")}).json()
    evaluation = evaluated["evaluation"]
    assert evaluation["rows_compared"] == 150 and evaluation["rows_excluded"]["in_the_training_file"] == 0
    assert [c["passed"] for c in evaluation["constraints"]] == [True, True]

    # Promote -- with an override if the noise of 150 rows hid the improvement; either way it is recorded.
    body = {"decision": "promote", "reason": "Month 4 is the newest behaviour."}
    if not evaluation["all_passed"]:
        body["override"] = True
    decided = client.post(f"/api/model/{sid}/comparisons/{cid}/decision", json=body).json()
    assert decided["decision"] == "promoted" and decided["override"] == (not evaluation["all_passed"])
    assert model_store.current_run_id(directory) == challenger
    assert client.get(f"/api/model/{sid}/result").json()["run_id"] == challenger

    # The old champion learned from months 1-3: not the data any more, so it cannot be put back.
    rollback = client.post(f"/api/model/{sid}/rollback", json={"reason": "Trying"})
    assert rollback.status_code == 409 and rollback.json()["code"] == "data_changed"
    assert [d["decision"] for d in client.get(f"/api/model/{sid}/decisions").json()["decisions"]] == ["promoted"]


def test_no_automatic_promotion_path_exists():
    """The only place a run is put in use as a promotion is the decision route, which a person calls."""
    app_dir = pathlib.Path(__file__).resolve().parents[2] / "app"
    places = []
    for path in app_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"set_current\([^)]*promoted=", text):
            places.append(path.name)
    assert places == ["routes_promotion.py"]
    source = (app_dir / "api" / "routes_promotion.py").read_text(encoding="utf-8")
    assert "schedule" not in source.lower() and "background" not in source.lower()
