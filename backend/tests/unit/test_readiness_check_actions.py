"""Readiness checks say where to go to act on a finding.

A check's `actions` are places to navigate to -- never something applied. The page maps each kind
to a route; approval stays explicit there.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.graph_state import GoalContext
from app.services import ml_readiness, readiness_checks
from app.services.readiness_checks import ACTION_KINDS, MAX_ACTIONS_PER_CHECK, Check, action


def goal(target: str | None = "outcome") -> GoalContext:
    return GoalContext(use_case="ml_training", expected_output="x", target_column=target)


def frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "outcome": np.where(x + rng.normal(size=n) > 0, "yes", "no")})


def checks(df: pd.DataFrame, target: str | None = "outcome") -> dict[str, dict]:
    report = ml_readiness.assess_readiness(df, goal(target))
    return {c["name"]: c for c in report["checks"]}


class TestTheAction:
    def test_it_is_a_kind_a_label_and_a_column(self):
        assert action("exclude_column", "Leave it out", "leaky") == {
            "kind": "exclude_column", "label": "Leave it out", "column": "leaky",
        }

    def test_a_kind_without_a_column(self):
        assert action("choose_target", "Choose")["column"] is None

    def test_an_unknown_kind_is_refused_so_a_typo_cannot_make_a_button_that_goes_nowhere(self):
        with pytest.raises(ValueError):
            action("apply_the_fix", "Do it")

    def test_every_kind_is_navigation(self):
        assert set(ACTION_KINDS) == {"exclude_column", "use_group_split", "choose_target"}

    def test_a_check_serialises_its_actions_as_a_list(self):
        check = Check(name="n", label="l", status="failed", finding="f", actions=(action("choose_target", "Go"),))
        assert check.as_dict()["actions"] == [{"kind": "choose_target", "label": "Go", "column": None}]

    def test_a_check_without_actions_has_an_empty_list(self):
        assert Check(name="n", label="l", status="passed", finding="f").as_dict()["actions"] == []


class TestLeakage:
    def test_a_leaking_column_can_be_left_out(self):
        df = frame().assign(leaky=lambda d: np.where(d.outcome == "yes", 1.0, 0.0))
        leakage = checks(df)["leakage"]
        assert leakage["status"] == "failed"
        assert leakage["actions"] == [
            {"kind": "exclude_column", "label": "Leave 'leaky' out of the model", "column": "leaky"}
        ]

    def test_a_clean_file_has_no_leakage_action(self):
        assert checks(frame())["leakage"]["actions"] == []


class TestRepeatedEntities:
    def entities(self):
        rng = np.random.default_rng(4)
        n = 300
        return frame(n).assign(customer=rng.choice([f"c{i}" for i in range(40)], n))

    def test_a_repeating_column_offers_a_grouped_split_on_it(self):
        found = checks(self.entities())["repeated_entities"]
        assert found["status"] == "warning"
        assert found["actions"] == [
            {"kind": "use_group_split", "label": "Split by 'customer'", "column": "customer"}
        ]

    def test_no_more_than_three_are_offered(self):
        rng = np.random.default_rng(5)
        n = 300
        df = frame(n)
        for name in ("a_id", "b_id", "c_id", "d_id", "e_id"):
            df[name] = rng.choice([f"{name}{i}" for i in range(40)], n)
        found = checks(df)["repeated_entities"]
        assert len(found["actions"]) == MAX_ACTIONS_PER_CHECK

    def test_a_file_with_no_repeated_entity_offers_nothing(self):
        assert checks(frame())["repeated_entities"]["actions"] == []


class TestTheTarget:
    def test_a_target_that_is_gone_can_be_chosen_again(self):
        found = checks(frame().drop(columns="outcome"))["target_present"]
        assert found["status"] == "failed"
        assert found["actions"] == [{"kind": "choose_target", "label": "Choose a column to predict", "column": None}]

    def test_no_target_named_offers_to_choose_one(self):
        found = checks(frame(), target=None)["target_named"]
        assert found["status"] == "not_run"
        assert [a["kind"] for a in found["actions"]] == ["choose_target"]

    def test_a_target_that_is_there_offers_nothing(self):
        assert checks(frame())["target_present"]["actions"] == []


class TestOverHttp:
    def test_the_readiness_response_carries_the_actions(self):
        client = TestClient(app)
        df = frame().assign(leaky=lambda d: np.where(d.outcome == "yes", 1.0, 0.0))
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
        payload = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
        assert client.post(f"/api/goal/{sid}", json=payload).status_code == 200
        body = client.get(f"/api/readiness/{sid}").json()
        leakage = next(c for c in body["checks"] if c["name"] == "leakage")
        assert leakage["actions"][0]["kind"] == "exclude_column" and leakage["actions"][0]["column"] == "leaky"


def test_every_check_the_module_builds_has_the_field():
    for check in checks(frame()).values():
        assert isinstance(check["actions"], list)


def test_readiness_checks_module_exposes_the_helper():
    assert readiness_checks.action is action
