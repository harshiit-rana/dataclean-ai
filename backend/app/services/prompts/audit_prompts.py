"""Placeholder for the Audit Agent's prompts. NOT IMPLEMENTED.

TODO (PRD §12.4): needs a prompt that, given the full action_log + issues + goal_context,
produces (1) an Executive Summary paragraph and (2) a plain-English disposition summary per
issue. Per FR-30, rejection_reason text must be reproduced verbatim in the final report --
the LLM must NOT be asked to paraphrase it. See app/graph/nodes/audit_agent.py.
"""

from __future__ import annotations

from app.models.graph_state import GoalContext
from app.models.suggestion_card import ActionLogEntry


def build_audit_report_messages(
    action_log: list[ActionLogEntry], goal_context: GoalContext | None
):
    raise NotImplementedError("Audit Agent prompts are not implemented yet -- see PRD §12.4")
