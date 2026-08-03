"""Prompt for the Explanation Agent node. PRD §12.2.

One call per issue (rather than a single batched call for all issues) so each card can be
token-streamed to the frontend independently via card_chunk/card_complete SSE events, per
§10.2 and §FR-23. The response uses fixed section markers so the node can parse the final
text into a structured SuggestionCard once streaming completes, while forwarding raw deltas
to the client live in the meantime.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.models.graph_state import ColumnIssue, GoalContext
from app.services.operations import operation_catalog

SECTION_MARKERS = ("FOUND:", "PROPOSED_ACTION:", "REASONING:", "TRADEOFF:", "OPERATION:")

SYSTEM_PROMPT = """You are the Explanation Agent inside DataClean AI. For the single data \
issue you are given, write a suggestion card a non-statistician can act on. Use plain \
English, no jargon, no statistical notation (no "IQR", "p-value", "std dev", etc).

Respond in exactly this format, five sections, each starting on its own line with the \
literal marker shown:

FOUND:
One or two plain-English sentences describing what was found.

PROPOSED_ACTION:
The specific operation to perform, phrased concretely, e.g. "Drop the 340 rows where \
revenue is null."

REASONING:
Why this action makes sense given the user's stated use case. Must explicitly reference \
the use case.

TRADEOFF:
What the user gives up or risks if this action is applied. Must be specific to this \
column and this data, not generic.

OPERATION:
A single machine-readable call describing the fix, chosen ONLY from the list below. Use \
the exact operation name, with double-quoted argument values, and nothing else on the \
line -- no backticks, no explanation.

{operation_catalog}

Argument rules:
- Every operation except drop_duplicate_rows takes column="<the column name>".
- standardize_case also takes case="lower" | "upper" | "title".
- standardize_dates also takes format="%Y-%m-%d" (or another strftime format).
- fill_constant also takes value="<the fill value>".

Example OPERATION lines:
fill_median(column="revenue")
drop_duplicate_rows()
standardize_case(column="country", case="upper")

The OPERATION must match what PROPOSED_ACTION describes. No other text before FOUND: or \
after the OPERATION line."""


def build_explanation_messages(
    issue: ColumnIssue, goal_context: GoalContext | None
) -> list[BaseMessage]:
    goal_block = (
        f"use_case={goal_context.use_case}, expected_output={goal_context.expected_output}, "
        f"critical_columns={goal_context.critical_columns}, "
        f"cleaning_aggression={goal_context.cleaning_aggression}"
        if goal_context
        else "use_case=unspecified (no goal context was provided)"
    )
    human = (
        f"Goal context: {goal_block}\n\n"
        f"Issue:\n"
        f"- column: {issue.column_name}\n"
        f"- type: {issue.issue_type}\n"
        f"- severity: {issue.severity}\n"
        f"- description: {issue.description}\n"
        f"- stats: {issue.stats}\n\n"
        "Write the suggestion card now."
    )
    system = SYSTEM_PROMPT.format(operation_catalog=operation_catalog())
    return [SystemMessage(content=system), HumanMessage(content=human)]


def parse_card_sections(text: str) -> dict[str, str]:
    """Splits the marker-delimited response into {found, proposed_action, reasoning, tradeoff}."""
    key_by_marker = {
        "FOUND:": "found",
        "PROPOSED_ACTION:": "proposed_action",
        "REASONING:": "reasoning",
        "TRADEOFF:": "tradeoff",
        "OPERATION:": "operation",
    }
    sections: dict[str, str] = {}
    current_key = None
    buffer: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped in key_by_marker:
            if current_key:
                sections[current_key] = "\n".join(buffer).strip()
            current_key = key_by_marker[stripped]
            buffer = []
        else:
            buffer.append(line)
    if current_key:
        sections[current_key] = "\n".join(buffer).strip()

    return sections
