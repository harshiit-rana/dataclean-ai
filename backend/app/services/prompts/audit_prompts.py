"""Prompts for the Audit Agent node. PRD §12.4.

Deliberately narrow scope: the model writes only the *prose* (the executive summary and a
plain-English disposition line per action). Everything structural -- headings, tables,
before/after statistics, session metadata, and the user's rejection reasons -- is assembled
in code by audit_agent.py.

Two reasons:
- §12.4 is explicit that the agent must not "change, summarize, or paraphrase the user's
  rejection reasons" (FR-30). Inserting them in code makes that structurally impossible
  rather than a prompt instruction the model might drift from -- the reasons are not even
  sent to the model.
- One small call instead of asking the model to format an entire document keeps token cost
  down and removes a whole class of formatting failures.
"""

from __future__ import annotations

import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.models.graph_state import GoalContext
from app.services.prompts.untrusted import DATA_NOTICE, file_data

SYSTEM_PROMPT = """You are the Audit Agent inside DataClean AI. You are documenting a data \
cleaning session for a report the user may attach to a model card, show to stakeholders, or \
cite in an incident post-mortem. Write plainly and factually. No marketing language, no \
praise, no filler.

Return ONLY a JSON object in exactly this shape, with no markdown fences and no text around \
it:

{
  "executive_summary": "One paragraph, 2-4 sentences.",
  "dispositions": {"<action_id>": "One sentence."}
}

executive_summary: what dataset was cleaned, what the user was trying to accomplish, how \
many changes were applied versus declined, and the overall effect on the data. State facts \
from the action log; do not speculate about data you were not shown.

dispositions: one entry per action_id given below. For applied actions, say what changed and \
what it means for the user's stated goal. For declined actions, say only that it was \
declined and what that leaves unresolved -- do NOT restate, justify, or editorialize about \
why the user declined it.

Each action carries the issue_type it was addressing. Describe the action in terms of that \
reason, not just its mechanical effect. In particular:
- target_leakage: the column was removed because it gave away the value being predicted, \
not because of how many rows were empty.
- target_missing: rows were dropped because the value being predicted was absent and \
cannot be filled in.
- target_imbalance: a fact about how lopsided the answers are, not a change to the data."""


def build_audit_report_messages(
    action_log: list[dict],
    goal_context: GoalContext | None,
    file_name: str,
    issue_count: int,
) -> list[BaseMessage]:
    goal_block = (
        f"use_case={goal_context.use_case}, expected_output={goal_context.expected_output}, "
        f"critical_columns={goal_context.critical_columns}, "
        f"cleaning_aggression={goal_context.cleaning_aggression}"
        if goal_context
        else "not specified"
    )

    # Only the fields the model needs to write prose. rejection_reason is withheld on
    # purpose so it cannot leak into paraphrased output (FR-30).
    actions = [
        {
            "action_id": entry["action_id"],
            "column_name": entry["column_name"],
            "issue_type": entry.get("issue_type"),
            "operation": entry["operation"],
            "status": entry["status"],
            "before_stats": entry.get("before_stats"),
            "after_stats": entry.get("after_stats"),
        }
        for entry in action_log
    ]

    human = (
        f"Dataset:\n{file_data(file_name)}\n"
        f"Goal context: {goal_block}\n"
        f"Issues detected during profiling: {issue_count}\n\n"
        f"Action log:\n{file_data(json.dumps(actions, indent=2))}\n\n"
        "Write the report JSON now."
    )
    return [SystemMessage(content=SYSTEM_PROMPT + "\n\n" + DATA_NOTICE), HumanMessage(content=human)]


def build_retry_messages(
    action_log: list[dict],
    goal_context: GoalContext | None,
    file_name: str,
    issue_count: int,
    previous_error: str,
) -> list[BaseMessage]:
    """PRD FR-27/§14.2: one retry with the parse error fed back in."""
    messages = build_audit_report_messages(action_log, goal_context, file_name, issue_count)
    messages.append(
        HumanMessage(
            content=(
                f"That response could not be parsed as JSON ({previous_error}). Return ONLY "
                "the JSON object described above -- no markdown fences, no commentary."
            )
        )
    )
    return messages
