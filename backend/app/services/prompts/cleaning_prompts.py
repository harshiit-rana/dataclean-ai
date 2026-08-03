"""Placeholder for the Cleaning Agent's prompts. NOT IMPLEMENTED.

TODO (PRD §12.3): two prompt builders are needed here --
  1. translate an approved SuggestionCard's proposed_action into executable pandas code
  2. translate a free-text conversational instruction (FR-25) into executable pandas code
Both must support a retry-with-error-context variant for FR-27 (retry once if generated
code raises an exception in the sandboxed subprocess, per §14.2 "Invalid pandas code").
See app/graph/nodes/cleaning_agent.py for the node that will call these.
"""

from __future__ import annotations

from app.models.graph_state import GoalContext
from app.models.suggestion_card import SuggestionCard


def build_cleaning_code_messages(card: SuggestionCard, goal_context: GoalContext | None):
    raise NotImplementedError("Cleaning Agent prompts are not implemented yet -- see PRD §12.3")


def build_conversational_cleaning_messages(instruction: str, goal_context: GoalContext | None):
    raise NotImplementedError("Cleaning Agent prompts are not implemented yet -- see PRD §12.3, FR-25")
