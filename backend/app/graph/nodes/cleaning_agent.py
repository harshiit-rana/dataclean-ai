"""cleaning_agent node. PRD §12.3 -- STUB.

Will take `state.approved_action_ids` (from human_approval) and, for each approved
suggestion card, call Claude to generate pandas cleaning code (via
`app.services.prompts.cleaning_prompts.build_cleaning_code_messages`), execute it against
the working DataFrame in a sandboxed manner, and append an ActionLogEntry per action
(FR-28..36). Rejected cards get an ActionLogEntry with status="rejected" and the supplied
skip_reason, no code executed. Emits `action_start` / `action_complete` / `code_preview` /
`all_actions_complete` SSE events via get_stream_writer(), mirroring explanation_agent's
streaming style.

TODO: implement per PRD §12.3, FR-28..36, §14.2 (sandboxed exec error handling).
"""

from __future__ import annotations

from app.models.graph_state import GraphState


async def cleaning_agent(state: GraphState) -> dict:
    raise NotImplementedError("cleaning_agent is not yet implemented -- see PRD §12.3")
