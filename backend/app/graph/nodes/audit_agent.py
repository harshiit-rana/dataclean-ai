"""audit_agent node. PRD §12.4 -- STUB.

Will call Claude via `app.services.prompts.audit_prompts.build_audit_report_messages` to
turn `state.action_log` + `state.goal_context` into a human-readable markdown audit report,
then render it to PDF with WeasyPrint (§8.3, FR-41..44) and write the cleaned DataFrame to
`state.cleaned_file_path`. This is the terminal node of the graph.

TODO: implement per PRD §12.4, FR-41..44.
"""

from __future__ import annotations

from app.models.graph_state import GraphState


async def audit_agent(state: GraphState) -> dict:
    raise NotImplementedError("audit_agent is not yet implemented -- see PRD §12.4")
