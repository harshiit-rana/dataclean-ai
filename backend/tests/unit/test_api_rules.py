"""The approval boundary for business rules.

The split between "proposed" and "in force" is the entire design of this feature. A rule
applied without being read can quietly mark a fifth of someone's rows as errors, which is
precisely the failure the whole app is built to prevent -- so these tests care less about
whether the engine counts correctly (that is `test_business_rules.py`) and more about whether
anything can take effect without a person saying so.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _messy_orders(n: int = 300) -> bytes:
    rng = np.random.default_rng(4)
    order = pd.Series(
        pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    )
    delivery = pd.Series(order + pd.to_timedelta(rng.integers(1, 9, n), unit="D"))
    delivery.iloc[:6] = order.iloc[:6] - pd.Timedelta(days=3)
    age = pd.Series(rng.integers(18, 80, n).astype(float))
    age.iloc[:3] = [-5.0, -1.0, -40.0]

    buffer = io.StringIO()
    pd.DataFrame(
        {
            "age": age,
            "order_date": order,
            "delivery_date": delivery,
            "churn": rng.choice(["yes", "no"], n),
        }
    ).to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def _session(**goal_overrides) -> str:
    response = client.post(
        "/api/upload", files={"file": ("orders.csv", io.BytesIO(_messy_orders()), "text/csv")}
    )
    session_id = response.json()["session_id"]
    goal = {
        "use_case": "ml_training",
        "expected_output": "predict churn",
        "target_column": "churn",
        "task_type": "auto",
        **goal_overrides,
    }
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


class TestNothingTakesEffectWithoutApproval:
    def test_proposals_arrive_with_nothing_in_force(self):
        body = client.get(f"/api/rules/{_session()}").json()
        assert body["proposed"], "the file has impossible ages and backwards deliveries"
        assert body["rules"] == []
        assert body["checks"] == []

    def test_a_proposal_changes_no_verdict_until_it_is_approved(self):
        session_id = _session()
        before = client.get(f"/api/readiness/{session_id}").json()
        assert not [c for c in before["checks"] if c["name"].startswith("rule_")]

    def test_approving_puts_it_in_force(self):
        session_id = _session()
        proposed = client.get(f"/api/rules/{session_id}").json()["proposed"]
        saved = client.post(f"/api/rules/{session_id}", json={"rules": proposed[:1]}).json()
        assert len(saved["rules"]) == 1
        assert len(saved["checks"]) == 1

    def test_an_approved_rule_is_not_proposed_again(self):
        session_id = _session()
        proposed = client.get(f"/api/rules/{session_id}").json()["proposed"]
        saved = client.post(f"/api/rules/{session_id}", json={"rules": proposed}).json()
        assert saved["proposed"] == []


class TestApprovedRulesReachTheVerdict:
    def test_a_broken_rule_becomes_a_readiness_check(self):
        """The point of making a rule a Check: the panel, the summary and the audit report all
        pick it up without knowing rules exist."""
        session_id = _session()
        proposed = client.get(f"/api/rules/{session_id}").json()["proposed"]
        client.post(f"/api/rules/{session_id}", json={"rules": proposed})

        readiness = client.get(f"/api/readiness/{session_id}").json()
        rule_checks = [c for c in readiness["checks"] if c["name"].startswith("rule_")]
        assert len(rule_checks) == len(proposed)
        assert any(c["status"] == "failed" for c in rule_checks)

    def test_the_summary_carries_the_violation(self):
        session_id = _session()
        proposed = client.get(f"/api/rules/{session_id}").json()["proposed"]
        client.post(f"/api/rules/{session_id}", json={"rules": proposed})
        readiness = client.get(f"/api/readiness/{session_id}").json()
        assert any("break" in warning for warning in readiness["warnings"])

    def test_a_broken_rule_does_not_block_a_model(self):
        """A violation is a finding, not a fix. Which rows are wrong -- and whether they should
        be dropped or corrected -- is a decision, and blocking would make it for the user."""
        session_id = _session()
        proposed = client.get(f"/api/rules/{session_id}").json()["proposed"]
        client.post(f"/api/rules/{session_id}", json={"rules": proposed})
        assert client.get(f"/api/readiness/{session_id}").json()["verdict"] != "blocked"

    def test_rules_are_checked_even_with_no_target(self):
        """`age = -5` is wrong whether or not anyone is predicting anything, so this must not
        sit behind the target-dependent checks."""
        session_id = _session(target_column=None)
        proposed = client.get(f"/api/rules/{session_id}").json()["proposed"]
        client.post(f"/api/rules/{session_id}", json={"rules": proposed})

        readiness = client.get(f"/api/readiness/{session_id}").json()
        rule_checks = [c for c in readiness["checks"] if c["name"].startswith("rule_")]
        assert rule_checks and any(c["status"] == "failed" for c in rule_checks)
        assert readiness["verdict"] == "not_applicable"  # still no model asked for
        assert any("break" in warning for warning in readiness["warnings"])


class TestRulesPersist:
    def test_they_survive_a_second_request(self):
        session_id = _session()
        proposed = client.get(f"/api/rules/{session_id}").json()["proposed"]
        client.post(f"/api/rules/{session_id}", json={"rules": proposed[:2]})
        assert len(client.get(f"/api/rules/{session_id}").json()["rules"]) == 2

    def test_saving_replaces_rather_than_appends(self):
        session_id = _session()
        proposed = client.get(f"/api/rules/{session_id}").json()["proposed"]
        client.post(f"/api/rules/{session_id}", json={"rules": proposed})
        client.post(f"/api/rules/{session_id}", json={"rules": []})
        assert client.get(f"/api/rules/{session_id}").json()["rules"] == []

    def test_an_unusable_rule_is_refused_rather_than_dropped(self):
        """Silently discarding it would leave the user believing a rule is in force when it
        is not -- worse than the error itself."""
        session_id = _session()
        response = client.post(
            f"/api/rules/{session_id}", json={"rules": [{"kind": "range", "column": "age"}]}
        )
        assert response.status_code == 422
        assert "minimum" in response.json()["detail"]
