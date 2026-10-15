"""Promotion and rejection, recorded with their evidence; an override needs a reason and says so."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import model_store, promotion
from app.services.session_store import session_dir

client = TestClient(app)
REASON = "Recall matters more to the clinic than the Brier score this quarter."


def rows(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, n)
    return pd.DataFrame({
        "signal": signal.round(4),
        "noise": rng.normal(0, 1, n).round(4),
        "churned": np.where(rng.random(n) < 1 / (1 + np.exp(-3 * signal)), "yes", "no"),
    })


def csv(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


@pytest.fixture
def session():
    """A weak champion (without the informative column) and a strong challenger (with it)."""
    sid = client.post("/api/upload", files={"file": ("t.csv", io.BytesIO(csv(rows(400, 1))), "text/csv")}).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "churned", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    train = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}
    champion = client.post(f"/api/model/{sid}/train", json={**train, "excluded_columns": ["signal"]}).json()["run_id"]
    challenger = client.post(f"/api/model/{sid}/train", json=train).json()["run_id"]
    client.post(f"/api/model/{sid}/runs/{champion}/restore")
    return sid, session_dir(get_settings().upload_dir, sid), champion, challenger


def evaluated(sid, champion, challenger, **criteria) -> str:
    body = {"deciding_metric": "roc_auc", "min_improvement": 0.0, "constraints": [], **criteria}
    cid = client.post(f"/api/model/{sid}/comparisons", json={
        "champion_run_id": champion, "challenger_run_id": challenger, "criteria": body}).json()["comparison_id"]
    response = client.post(f"/api/model/{sid}/comparisons/{cid}/evaluate",
                           files={"file": ("e.csv", io.BytesIO(csv(rows(300, 9))), "text/csv")})
    assert response.status_code == 200, response.text
    return cid


def decide(sid, cid, **body):
    return client.post(f"/api/model/{sid}/comparisons/{cid}/decision", json=body)


class TestPromoting:
    def test_a_challenger_that_meets_every_criterion_is_promoted_and_put_in_use(self, session):
        sid, directory, champion, challenger = session
        cid = evaluated(sid, champion, challenger, constraints=[{"kind": "no_new_required_columns"}])
        # The challenger needs a column the champion did not: that constraint fails...
        assert decide(sid, cid, decision="promote").json()["code"] == "criteria_failed"
        cid = evaluated(sid, champion, challenger)
        entry = decide(sid, cid, decision="promote", reason="Clearly better.").json()
        assert entry["decision"] == "promoted" and entry["override"] is False and entry["all_passed"] is True
        assert entry["rows_compared"] > 0 and entry["deciding"]["passed"] is True
        assert model_store.current_run_id(directory) == challenger
        assert client.get(f"/api/model/{sid}/comparisons/{cid}").json()["status"] == "decided"
        again = decide(sid, cid, decision="reject")
        assert again.status_code == 409 and again.json()["code"] == "already_decided"

    def test_a_failed_criterion_blocks_promotion_unless_overridden_with_a_reason(self, session):
        sid, directory, champion, challenger = session
        cid = evaluated(sid, champion, challenger, min_improvement=0.9)
        refused = decide(sid, cid, decision="promote")
        assert refused.status_code == 409 and refused.json()["code"] == "criteria_failed"
        no_reason = decide(sid, cid, decision="promote", override=True)
        assert no_reason.status_code == 409 and no_reason.json()["code"] == "override_needs_reason"
        entry = decide(sid, cid, decision="promote", override=True, reason=REASON).json()
        assert entry["override"] is True and entry["reason"] == REASON
        assert entry["failed_criteria"] == ["deciding_metric"]
        assert model_store.current_run_id(directory) == challenger

    def test_a_rejection_is_recorded_and_changes_nothing(self, session):
        sid, directory, champion, challenger = session
        cid = evaluated(sid, champion, challenger)
        entry = decide(sid, cid, decision="reject", reason="Not this quarter.").json()
        assert entry["decision"] == "rejected" and model_store.current_run_id(directory) == champion

    def test_there_is_nothing_to_decide_before_an_evaluation(self, session):
        sid, _, champion, challenger = session
        cid = client.post(f"/api/model/{sid}/comparisons", json={
            "champion_run_id": champion, "challenger_run_id": challenger,
            "criteria": {"deciding_metric": "roc_auc"}}).json()["comparison_id"]
        response = decide(sid, cid, decision="promote")
        assert response.status_code == 409 and response.json()["code"] == "not_evaluated"

    def test_a_challenger_trained_before_the_data_changed_cannot_be_put_in_use(self, session):
        sid, directory, champion, challenger = session
        cid = evaluated(sid, champion, challenger)
        changed = rows(400, 1)
        changed.loc[0, "noise"] = 99.0
        changed.to_csv(directory / "cleaned.csv", index=False)
        response = decide(sid, cid, decision="promote")
        assert response.status_code == 409 and response.json()["code"] == "data_changed"
        assert promotion.decisions(directory) == []  # nothing recorded for a promotion that did not happen


class TestTheRecord:
    def test_decisions_accumulate_and_are_never_truncated(self, session):
        sid, directory, champion, challenger = session
        for _ in range(3):
            decide(sid, evaluated(sid, champion, challenger), decision="reject")
        listed = client.get(f"/api/model/{sid}/decisions").json()["decisions"]
        assert [d["decision"] for d in listed] == ["rejected"] * 3
        assert all(d["criteria"]["deciding_metric"] == "roc_auc" and d["file_fingerprint"] for d in listed)

    def test_the_promoted_model_s_card_says_how_it_got_there_and_the_reason_never_reaches_the_llm(self, session):
        from app.services.model_card import build_model_card
        from app.services.prompts.model_prompts import build_model_card_messages

        sid, directory, champion, challenger = session
        decide(sid, evaluated(sid, champion, challenger, min_improvement=0.9),
               decision="promote", override=True, reason=REASON)
        result = model_store.load_result(directory)
        result["promotion"] = promotion.promotion_of(directory, result["run_id"])
        card = build_model_card(session_id=sid, file_name="t.csv", result=result, summary="A.", interpretation="B.")
        assert "## How it came to be the model in use" in card and "**Promoted by override**" in card
        assert REASON in card
        messages = build_model_card_messages(result, "t.csv", None)
        assert all(REASON not in str(m.content) for m in messages)
