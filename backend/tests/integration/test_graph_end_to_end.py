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
    """Stands in for the Explanation Agent's streaming LLM call. Yields the whole card as a
    single chunk in the exact marker format parse_card_sections expects, including the
    machine-readable OPERATION line the cleaning_agent executes."""

    async def astream(self, messages):
        # Echo back the column this call is actually about, so the emitted operation targets
        # a real column (the prompt embeds it as "- column: <name>").
        prompt = str(messages[-1].content)
        column = ""
        for line in prompt.splitlines():
            if line.startswith("- column:"):
                column = line.split(":", 1)[1].strip()
                break

        text = (
            "FOUND:\n"
            "This column has missing values.\n"
            "PROPOSED_ACTION:\n"
            "Drop the rows where this column is empty.\n"
            "REASONING:\n"
            "Keeps every remaining row usable for reporting.\n"
            "TRADEOFF:\n"
            "Loses the dropped rows entirely.\n"
            "OPERATION:\n"
            f'drop_rows_with_nulls(column="{column}")\n'
        )
        yield SimpleNamespace(content=text)


@pytest.fixture
def graph(monkeypatch):
    monkeypatch.setattr(
        "app.graph.nodes.profiling_agent.get_llm_client",
        lambda settings, streaming=False: _FakeInvokeClient(),
    )
    monkeypatch.setattr(
        "app.graph.nodes.explanation_agent.get_llm_client",
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
        assert card["found"]
        assert card["proposed_action"]


@pytest.mark.asyncio
async def test_resume_applies_only_approved_fixes(graph, fixtures_dir, tmp_path):
    """The whole point of the app: approving a card actually transforms the data, rejecting
    one leaves it alone, and the original file is never modified."""
    import shutil

    import pandas as pd
    from langgraph.types import Command

    # Work on a copy so the fixture on disk can be asserted unchanged afterwards.
    source = fixtures_dir / "missing_values.csv"
    working = tmp_path / "original.csv"
    shutil.copy(source, working)
    original_bytes = working.read_bytes()

    session_id = "test-session-resume"
    config = thread_config(session_id)
    await graph.ainvoke(
        {
            "session_id": session_id,
            "file_path": str(working),
            "file_name": "missing_values.csv",
            "file_size_bytes": 500,
            "goal_context": {
                "use_case": "reporting",
                "expected_output": "a clean dataset",
                "critical_columns": [],
                "cleaning_aggression": "moderate",
            },
        },
        config,
    )

    state = await graph.aget_state(config)
    cards = state.values["suggestion_cards"]
    assert len(cards) >= 2
    # every card got a usable operation parsed off the OPERATION line
    assert all(c["operation"] == "drop_rows_with_nulls" for c in cards)

    approved, rejected = cards[0], cards[1]
    approved_id, rejected_id = approved["card_id"], rejected["card_id"]
    await graph.ainvoke(
        Command(
            resume={
                "approved_action_ids": [approved_id],
                "skipped_action_ids": [rejected_id],
                "skip_reasons": {rejected_id: "I need those rows."},
            }
        ),
        config,
    )

    final = (await graph.aget_state(config)).values
    log = final["action_log"]
    by_status = {e["status"] for e in log}
    assert "applied" in by_status and "rejected" in by_status

    applied = [e for e in log if e["status"] == "applied"]
    assert len(applied) == 1
    assert applied[0]["card_id"] == approved_id
    assert applied[0]["before_stats"]["null_count"] > 0
    assert applied[0]["after_stats"]["null_count"] == 0
    assert applied[0]["after_stats"]["row_count"] < applied[0]["before_stats"]["row_count"]

    # the rejection reason is preserved verbatim (FR-30)
    rejected_entry = next(e for e in log if e["card_id"] == rejected_id)
    assert rejected_entry["rejection_reason"] == "I need those rows."

    # cleaned output written to a separate file; the uploaded copy is untouched
    cleaned_path = final["cleaned_file_path"]
    assert cleaned_path.endswith("cleaned.csv")
    assert working.read_bytes() == original_bytes
    cleaned = pd.read_csv(cleaned_path)
    assert cleaned[approved["column_name"]].isna().sum() == 0
