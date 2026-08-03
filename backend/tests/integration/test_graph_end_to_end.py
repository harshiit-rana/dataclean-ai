"""End-to-end graph test through the human_approval interrupt, using MemorySaver instead
of Redis (PRD §15.2). Claude calls are monkeypatched to fake, deterministic clients so this
test doesn't need a real ANTHROPIC_API_KEY or network access.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.graph.checkpointer import thread_config
from app.graph.workflow import build_graph


class _FakeInvokeClient:
    """Stands in for the Profiling Agent's non-streaming ChatAnthropic call. Returning an
    empty JSON array means every issue keeps its deterministic baseline priority_score --
    a valid, non-crashing outcome the real node already handles (see _enrich_with_llm)."""

    async def ainvoke(self, messages):
        return SimpleNamespace(content="[]")


class _FakeStreamClient:
    """Stands in for the Explanation Agent's streaming ChatAnthropic call. Yields the whole
    card as a single chunk in the exact marker format parse_card_sections expects."""

    async def astream(self, messages):
        text = (
            "FOUND: This column has missing values.\n"
            "PROPOSED_ACTION: Fill missing values with the column median.\n"
            "REASONING: Keeps every row usable for training.\n"
            "TRADEOFF: Slightly reduces variance in the column.\n"
        )
        yield SimpleNamespace(content=text)


@pytest.fixture
def graph(monkeypatch):
    monkeypatch.setattr(
        "app.graph.nodes.profiling_agent.get_claude_client",
        lambda settings, streaming=False: _FakeInvokeClient(),
    )
    monkeypatch.setattr(
        "app.graph.nodes.explanation_agent.get_claude_client",
        lambda settings, streaming=False: _FakeStreamClient(),
    )
    return build_graph(MemorySaver())


@pytest.mark.asyncio
async def test_graph_pauses_at_human_approval(graph, fixtures_dir):
    session_id = "test-session-1"
    input_state = {
        "session_id": session_id,
        "file_path": str(fixtures_dir / "missing_values.csv"),
        "file_name": "missing_values.csv",
        "file_size_bytes": 500,
        "goal_context": {
            "use_case": "reporting",
            "expected_output": "a clean dataset for a monthly report",
            "critical_columns": ["name"],
            "cleaning_aggression": "moderate",
        },
    }
    config = thread_config(session_id)

    events = [event async for event in graph.astream(input_state, config, stream_mode=["custom"])]
    assert len(events) > 0

    state = await graph.aget_state(config)
    assert state.next == ("human_approval",)

    values = state.values
    assert len(values["issues"]) >= 3
    assert len(values["suggestion_cards"]) == len(values["issues"])
    assert values["current_agent"] == "explanation_agent"
    for card in values["suggestion_cards"]:
        assert card.found
        assert card.proposed_action
