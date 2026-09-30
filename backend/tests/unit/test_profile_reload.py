"""Starting profiling for a session that has already run.

The Review page starts profiling when it mounts, so a reload -- or a browser restoring the
tab -- used to start the whole graph again: profiling re-run, every card rewritten at one LLM
call each, and any decisions not yet submitted lost. These pin what a second POST should do,
against a fake graph that records whether it was asked to run.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ISSUES = [
    {"issue_id": "i1", "column_name": "salary", "issue_type": "missing_values",
     "severity": "high", "description": "12 blanks", "stats": {}, "priority_score": 0.9},
    {"issue_id": "i2", "column_name": "city", "issue_type": "inconsistent_format",
     "severity": "low", "description": "'paris' vs 'Paris'", "stats": {}, "priority_score": 0.3},
]
CARDS = [
    {"card_id": "c1", "issue_id": "i1", "column_name": "salary", "found": "Blanks.",
     "proposed_action": "Fill with the median.", "reasoning": "r", "tradeoff": "t",
     "operation": "fill_median", "operation_params": {"column": "salary"}},
    {"card_id": "c2", "issue_id": "i2", "column_name": "city", "found": "Casing.",
     "proposed_action": "Lower-case it.", "reasoning": "r", "tradeoff": "t",
     "operation": "standardize_case", "operation_params": {"column": "city", "case": "lower"}},
]


class FakeGraph:
    def __init__(self, state=None, state_error: Exception | None = None):
        self.state = state
        self.state_error = state_error
        self.runs = 0

    async def aget_state(self, _config):
        if self.state_error:
            raise self.state_error
        return self.state

    async def astream(self, _input, _config, stream_mode=None):
        self.runs += 1
        yield ("custom", {"event": "agent_start", "agent": "profiling_agent", "message": "fresh"})


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
    payload = b"salary,city\n1,paris\n2,Paris\n,rome\n"
    session_id = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(payload), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "reporting", "expected_output": "a clean table"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


def _profile(session_id: str):
    response = client.post(f"/api/profile/{session_id}")
    events = [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    return response, events


def _state(next_nodes, **values):
    return SimpleNamespace(values=values, next=tuple(next_nodes))


def test_a_reload_while_paused_replays_the_saved_cards_without_running_anything(graph_slot):
    graph = FakeGraph(_state(["human_approval"], issues=ISSUES, suggestion_cards=CARDS))
    app.state.graph = graph
    response, events = _profile(_session())

    assert response.status_code == 200
    assert graph.runs == 0  # no profiling, no card writing, no LLM calls
    assert [e["event"] for e in events] == [
        "profiling_complete", "card_complete", "card_complete",
        "all_cards_complete", "interrupt_reached",
    ]


def test_the_replay_carries_the_same_cards_in_the_same_order(graph_slot):
    app.state.graph = FakeGraph(_state(["human_approval"], issues=ISSUES, suggestion_cards=CARDS))
    _response, events = _profile(_session())
    cards = [e["card"] for e in events if e["event"] == "card_complete"]
    assert [c["card_id"] for c in cards] == ["c1", "c2"]
    assert cards[0]["operation_params"] == {"column": "salary"}


def test_the_replay_carries_the_issue_counts_the_page_shows(graph_slot):
    app.state.graph = FakeGraph(_state(["human_approval"], issues=ISSUES, suggestion_cards=CARDS))
    _response, events = _profile(_session())
    summary = events[0]
    assert (summary["issue_count"], summary["high"], summary["medium"], summary["low"]) == (2, 1, 0, 1)
    assert [i["column_name"] for i in summary["issues"]] == ["salary", "city"]


def test_a_session_already_cleaned_is_refused_rather_than_started_over(graph_slot):
    graph = FakeGraph(_state([], issues=ISSUES, suggestion_cards=CARDS, action_log=[{}]))
    app.state.graph = graph
    response, events = _profile(_session())
    assert response.status_code == 409
    assert events == []
    assert graph.runs == 0
    assert "cleaning page" in response.json()["detail"]


def test_a_session_that_never_ran_is_started(graph_slot):
    graph = FakeGraph(_state([]))  # no values at all
    app.state.graph = graph
    response, _events = _profile(_session())
    assert response.status_code == 200
    assert graph.runs == 1


def test_a_run_that_died_part_way_is_started_again(graph_slot):
    """Restarting is how a dropped profiling run recovers, and there is nothing saved to
    replay yet."""
    graph = FakeGraph(_state(["explanation_agent"], issues=ISSUES))
    app.state.graph = graph
    _profile(_session())
    assert graph.runs == 1


def test_unreadable_state_is_no_reason_to_refuse_a_fresh_start(graph_slot):
    graph = FakeGraph(state_error=RuntimeError("redis hiccup"))
    app.state.graph = graph
    response, _events = _profile(_session())
    assert response.status_code == 200
    assert graph.runs == 1
