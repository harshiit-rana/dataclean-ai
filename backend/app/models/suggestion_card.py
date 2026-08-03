"""SuggestionCard and ActionLogEntry.

Referenced throughout PRD §9.1/§10.2/§12 but never fully tabled the way GraphState/
GoalContext/ColumnIssue are in §9. Field shapes here are inferred from the Explanation
Agent spec (§12.2: Found/Proposed Action/Reasoning/Tradeoff/card id) and the Cleaning
Agent spec (§12.3: before/after stats, operation, column) plus US-13's audit requirements.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class SuggestionCard(BaseModel):
    """LLM-generated card presenting one issue + proposed fix. PRD §12.2."""

    card_id: str
    issue_id: str
    column_name: str
    found: str  # 1-2 plain-English sentences, no jargon, no statistics notation
    proposed_action: str  # e.g. "Drop the 340 rows where revenue is null"
    reasoning: str  # must reference the stated use case
    tradeoff: str  # what is lost or risked, specific to the column/data


class ActionLogEntry(BaseModel):
    """One applied, rejected, or undone operation, accumulated in GraphState.action_log."""

    action_id: str
    card_id: Optional[str] = None  # None for conversational-instruction-originated actions
    column_name: str
    operation: str
    status: Literal["applied", "rejected", "undone"]
    rejection_reason: Optional[str] = None  # verbatim, never paraphrased (FR-30)
    before_stats: Optional[dict] = None
    after_stats: Optional[dict] = None
    source: Literal["suggestion_card", "conversational"] = "suggestion_card"
