"""conversational_loop node. PRD §13.1 / FR-37..40 -- STUB.

Will handle further plain-English cleaning instructions the user types after the initial
suggestion-card pass (e.g. "also drop rows where age is negative"). Reads the latest entry
appended to `state.conversation_history`, calls Claude via
`app.services.prompts.cleaning_prompts.build_conversational_cleaning_messages` to generate
pandas code, executes it the same way `cleaning_agent` does, and appends a new
ActionLogEntry with source="conversational". Loops back to itself (or to human_approval,
per §13.1's graph diagram) until the user signals they're done, at which point the graph
proceeds to audit_agent.

TODO: implement per PRD §13.1, FR-37..40.
"""

from __future__ import annotations

from app.models.graph_state import GraphState


async def conversational_loop(state: GraphState) -> dict:
    raise NotImplementedError("conversational_loop is not yet implemented -- see PRD §13.1")
