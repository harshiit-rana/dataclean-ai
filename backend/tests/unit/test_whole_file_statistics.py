"""Runs trained on data that had whole-file statistics applied say so.

A fill with a column's median, or an outlier limit, is worked out over every row -- the rows later
held out for testing included -- before any split. Sessions cleaned before the app stopped
proposing these, and fixes typed in chat, still carry them, and every run trained on that data
carries a small leak. The run, the card and the page should say so rather than present the score
as clean.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import action_log

client = TestClient(app)
WHOLE_FILE = ["fill_median", "fill_mean", "fill_mode", "cap_outliers", "drop_outlier_rows"]


def entry(operation, column, status="applied", **extra):
    return {
        "action_id": f"a-{operation}-{column}", "column_name": column, "operation": operation,
        "operation_params": {"column": column}, "status": status, **extra,
    }


class TestWhichFixesAreFlagged:
    @pytest.mark.parametrize("operation", WHOLE_FILE)
    def test_each_whole_file_operation_on_an_input_is_flagged(self, operation):
        found = action_log.whole_file_statistics([entry(operation, "salary")], ["salary", "age"])
        assert found == [{"operation": operation, "column": "salary"}]

    def test_a_fix_on_a_column_the_model_does_not_use_is_not(self):
        assert action_log.whole_file_statistics([entry("fill_median", "employee_id")], ["salary"]) == []

    def test_an_undone_fix_is_not(self):
        log = [entry("fill_median", "salary", status="undone")]
        assert action_log.whole_file_statistics(log, ["salary"]) == []

    @pytest.mark.parametrize("status", ["rejected", "superseded"])
    def test_a_fix_that_never_ran_is_not(self, status):
        assert action_log.whole_file_statistics([entry("fill_median", "salary", status=status)], ["salary"]) == []

    def test_row_wise_fixes_are_not(self):
        log = [entry("trim_whitespace", "salary"), entry("drop_rows_with_nulls", "salary"),
               entry("standardize_case", "salary"), entry("fill_constant", "salary")]
        assert action_log.whole_file_statistics(log, ["salary"]) == []

    def test_a_chat_instruction_counts_as_much_as_a_card(self):
        log = [entry("fill_mean", "salary", source="conversational", card_id=None)]
        assert action_log.whole_file_statistics(log, ["salary"]) == [
            {"operation": "fill_mean", "column": "salary"}
        ]

    def test_several_are_all_listed(self):
        log = [entry("fill_median", "salary"), entry("cap_outliers", "age"), entry("fill_mode", "region")]
        assert len(action_log.whole_file_statistics(log, ["salary", "age", "region"])) == 3

    def test_nothing_to_report_is_an_empty_list(self):
        assert action_log.whole_file_statistics([], ["salary"]) == []
        assert action_log.whole_file_statistics(None, ["salary"]) == []


class TestReadingTheLog:
    def test_no_graph_is_an_empty_log(self):
        import asyncio

        assert asyncio.run(action_log.read(None, "s")) == []

    def test_an_unreadable_checkpoint_is_an_empty_log_not_an_error(self):
        import asyncio

        class Broken:
            async def aget_state(self, _config):
                raise ConnectionError("redis is down")

        assert asyncio.run(action_log.read(Broken(), "s")) == []


class FakeGraph:
    def __init__(self, log):
        self.log = log

    async def aget_state(self, _config):
        return SimpleNamespace(values={"action_log": self.log}, next=())


class BrokenGraph:
    async def aget_state(self, _config):
        raise ConnectionError("redis is down")


@pytest.fixture
def graph_slot():
    had = hasattr(app.state, "graph")
    previous = getattr(app.state, "graph", None)
    yield
    if had:
        app.state.graph = previous
    elif hasattr(app.state, "graph"):
        del app.state.graph


def _session() -> str:
    rng = np.random.default_rng(17)
    n = 300
    signal = rng.normal(0, 1, n)
    buffer = io.StringIO()
    pd.DataFrame(
        {
            "signal": signal.round(4),
            "spend": rng.normal(50, 10, n).round(2),
            "employee_id": [f"E{i:05d}" for i in range(n)],
            "outcome": np.where(rng.random(n) < 1 / (1 + np.exp(-2.4 * signal)), "yes", "no"),
        }
    ).to_csv(buffer, index=False)
    session_id = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome",
            "task_type": "auto"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


FAST = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}


class TestOnTheResult:
    def test_a_session_with_an_applied_fill_on_an_input_produces_the_flag(self, graph_slot):
        app.state.graph = FakeGraph([entry("fill_median", "spend")])
        body = client.post(f"/api/model/{_session()}/train", json=FAST).json()
        assert body["evaluation"]["whole_file_statistics"] == [
            {"operation": "fill_median", "column": "spend"}
        ]

    def test_one_on_an_excluded_column_does_not(self, graph_slot):
        app.state.graph = FakeGraph([entry("fill_median", "spend")])
        body = client.post(
            f"/api/model/{_session()}/train", json={**FAST, "excluded_columns": ["spend"]}
        ).json()
        assert body["evaluation"]["whole_file_statistics"] == []

    def test_one_on_an_identifier_the_app_leaves_out_does_not(self, graph_slot):
        app.state.graph = FakeGraph([entry("fill_mode", "employee_id")])
        body = client.post(f"/api/model/{_session()}/train", json=FAST).json()
        assert body["evaluation"]["whole_file_statistics"] == []

    def test_an_undone_fix_is_not_flagged(self, graph_slot):
        app.state.graph = FakeGraph([entry("fill_median", "spend", status="undone")])
        body = client.post(f"/api/model/{_session()}/train", json=FAST).json()
        assert body["evaluation"]["whole_file_statistics"] == []

    def test_a_clean_session_has_an_empty_list(self, graph_slot):
        app.state.graph = FakeGraph([])
        body = client.post(f"/api/model/{_session()}/train", json=FAST).json()
        assert body["evaluation"]["whole_file_statistics"] == []

    def test_the_improvement_runs_carry_it_too(self, graph_slot):
        app.state.graph = FakeGraph([entry("cap_outliers", "spend")])
        body = client.post(
            f"/api/model/{_session()}/improve", json={**FAST, "budget": 4}
        ).json()
        assert body["evaluation"]["whole_file_statistics"] == [
            {"operation": "cap_outliers", "column": "spend"}
        ]

    def test_the_stored_result_keeps_it(self, graph_slot):
        app.state.graph = FakeGraph([entry("fill_median", "spend")])
        session_id = _session()
        client.post(f"/api/model/{session_id}/train", json=FAST)
        stored = client.get(f"/api/model/{session_id}/result").json()
        assert stored["evaluation"]["whole_file_statistics"][0]["column"] == "spend"


class TestItNeverFailsATraining:
    def test_training_works_with_no_graph_at_all(self, graph_slot):
        if hasattr(app.state, "graph"):
            del app.state.graph
        response = client.post(f"/api/model/{_session()}/train", json=FAST)
        assert response.status_code == 200
        assert response.json()["evaluation"]["whole_file_statistics"] == []

    def test_an_unreadable_checkpoint_does_not_fail_the_train(self, graph_slot):
        """Adding a Redis read to the train route must not make training depend on Redis."""
        app.state.graph = BrokenGraph()
        response = client.post(f"/api/model/{_session()}/train", json=FAST)
        assert response.status_code == 200
        assert response.json()["evaluation"]["whole_file_statistics"] == []


class TestOnTheCard:
    def test_the_card_names_each_operation_and_column(self, graph_slot):
        app.state.graph = FakeGraph([entry("fill_median", "spend"), entry("cap_outliers", "signal")])
        session_id = _session()
        client.post(f"/api/model/{session_id}/train", json=FAST)
        card = client.get(f"/api/model/{session_id}/card").json()["markdown"]
        assert "`fill_median` on `spend`" in card
        assert "`cap_outliers` on `signal`" in card
        assert "every row" in card and "held-out" in card

    def test_the_card_is_silent_when_there_is_nothing_to_say(self, graph_slot):
        app.state.graph = FakeGraph([])
        session_id = _session()
        client.post(f"/api/model/{session_id}/train", json=FAST)
        card = client.get(f"/api/model/{session_id}/card").json()["markdown"]
        assert "worked out over every row" not in card
