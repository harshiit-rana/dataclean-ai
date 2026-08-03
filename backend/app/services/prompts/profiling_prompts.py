"""Prompt for the Profiling Agent node. PRD §12.1.

One call covers all three LLM tasks: (1) infer likely cause per issue, (2) adjust priority
scores using goal_context and critical_columns, (3) flag issues that may indicate upstream
pipeline problems. likely_cause and pipeline_flag are written back into ColumnIssue.stats
(a generic dict) rather than requiring new model fields, since PRD §9.3 does not tabulate
those as top-level ColumnIssue fields.
"""

from __future__ import annotations

import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.models.graph_state import ColumnIssue, GoalContext

SYSTEM_PROMPT = """You are the Profiling Agent inside DataClean AI, a human-in-the-loop CSV \
cleaning tool. You receive a list of statistically-detected data issues and must:

1. Infer the likely real-world cause of each issue from its column name and stats.
2. Adjust each issue's priority_score (0.0-1.0) based on the user's stated goal and which \
columns they marked as critical -- issues in critical columns, or that block the stated \
use case, should score higher.
3. Flag any issue whose pattern suggests an upstream data pipeline problem (e.g. a sudden \
block of nulls, a format that changed mid-file) rather than an isolated data-quality issue.

Respond with ONLY a JSON array, one object per issue_id you were given, in this exact shape:
[{"issue_id": "...", "priority_score": 0.0, "likely_cause": "...", "pipeline_flag": false, \
"pipeline_note": "..."}]
No prose before or after the JSON. pipeline_note may be an empty string when pipeline_flag \
is false."""


def build_profiling_agent_messages(
    issues: list[ColumnIssue], goal_context: GoalContext | None
) -> list[BaseMessage]:
    goal_block = (
        goal_context.model_dump_json()
        if goal_context
        else '{"use_case": "unspecified", "critical_columns": []}'
    )
    issues_block = json.dumps([i.model_dump() for i in issues])
    human = (
        f"Goal context:\n{goal_block}\n\n"
        f"Detected issues:\n{issues_block}\n\n"
        "Return the JSON array now."
    )
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human)]


def build_retry_messages(
    issues: list[ColumnIssue], goal_context: GoalContext | None, previous_error: str
) -> list[BaseMessage]:
    """FR-27/§14.2: retry once with explicit JSON-formatting instruction on parse failure."""
    messages = build_profiling_agent_messages(issues, goal_context)
    messages.append(
        HumanMessage(
            content=(
                f"Your previous response failed to parse as JSON ({previous_error}). "
                "Respond again with ONLY the JSON array, no markdown code fences, no prose."
            )
        )
    )
    return messages
