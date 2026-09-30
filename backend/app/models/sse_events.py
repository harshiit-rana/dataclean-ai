"""SSE event payload models. PRD §10.2, plus interrupt_reached from §13.2's prose spec
(mentioned there but omitted from the §10.2 table)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from app.models.graph_state import ColumnIssue
from app.models.suggestion_card import SuggestionCard


class AgentStartEvent(BaseModel):
    event: Literal["agent_start"] = "agent_start"
    agent: str
    message: str


class ProfilingCompleteEvent(BaseModel):
    event: Literal["profiling_complete"] = "profiling_complete"
    issue_count: int
    high: int
    medium: int
    low: int
    # Full issue list so the Review page's data table can annotate each column with what
    # was actually found in it. Sent over SSE rather than fetched from the checkpointer
    # afterwards, since the stream is the one path already proven to work end to end.
    issues: list[ColumnIssue] = []


class CardChunkEvent(BaseModel):
    event: Literal["card_chunk"] = "card_chunk"
    card_id: str
    chunk: str


class CardCompleteEvent(BaseModel):
    event: Literal["card_complete"] = "card_complete"
    card: SuggestionCard


class AllCardsCompleteEvent(BaseModel):
    event: Literal["all_cards_complete"] = "all_cards_complete"
    total_cards: int


class InterruptReachedEvent(BaseModel):
    """PRD §13.2: sent when the graph hits the human_approval interrupt node."""

    event: Literal["interrupt_reached"] = "interrupt_reached"
    session_id: str


# --- Events for the /api/resume stream --------------------------------------------------
# The cleaning run that follows approval. ThinkingEvent and CodePreviewEvent were defined for
# a code-generating chat path that was never built -- chat maps onto whitelisted operations
# instead (routes_chat.py) and answers with plain JSON -- so nothing emits those two today.


class ActionStartEvent(BaseModel):
    event: Literal["action_start"] = "action_start"
    action_id: str
    operation: str
    column_name: str


class ActionCompleteEvent(BaseModel):
    event: Literal["action_complete"] = "action_complete"
    action_id: str
    before_stats: dict
    after_stats: dict


class AllActionsCompleteEvent(BaseModel):
    event: Literal["all_actions_complete"] = "all_actions_complete"
    actions_applied: int
    actions_skipped: int


class ThinkingEvent(BaseModel):
    event: Literal["thinking"] = "thinking"
    message: str


class CodePreviewEvent(BaseModel):
    event: Literal["code_preview"] = "code_preview"
    code: str


class ErrorEvent(BaseModel):
    event: Literal["error"] = "error"
    error: str
    detail: str


def encode_sse(payload: BaseModel, event_name: Optional[str] = None) -> str:
    """Serialize a payload model to a single `text/event-stream` frame."""
    name = event_name or getattr(payload, "event", "message")
    return f"event: {name}\ndata: {payload.model_dump_json()}\n\n"
