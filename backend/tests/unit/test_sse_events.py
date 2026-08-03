"""SSE event encoding tests. PRD §10.2."""

from __future__ import annotations

from app.models.sse_events import (
    AgentStartEvent,
    CardChunkEvent,
    InterruptReachedEvent,
    encode_sse,
)


def test_encode_sse_wire_format():
    payload = AgentStartEvent(agent="profiling_agent", message="Profiling your data...")
    frame = encode_sse(payload)

    assert frame.startswith("event: agent_start\n")
    assert frame.endswith("\n\n")
    assert '"agent":"profiling_agent"' in frame.replace(" ", "")


def test_encode_sse_uses_event_literal_by_default():
    payload = CardChunkEvent(card_id="card_1", chunk="Found:")
    frame = encode_sse(payload)
    assert frame.startswith("event: card_chunk\n")


def test_encode_sse_explicit_event_name_override():
    payload = InterruptReachedEvent(session_id="s1")
    frame = encode_sse(payload, event_name="custom_name")
    assert frame.startswith("event: custom_name\n")


def test_each_frame_is_a_single_sse_message():
    payload = AgentStartEvent(agent="x", message="y")
    frame = encode_sse(payload)
    # exactly one "event:" line and one "data:" line, per the SSE wire format
    assert frame.count("event:") == 1
    assert frame.count("data:") == 1
