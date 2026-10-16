"""Prompt for free-form cleaning instructions. PRD §12.3, FR-37..40.

The PRD calls for translating instructions into executable pandas code run in a sandbox.
This instead maps the instruction onto the same whitelist the rest of the app uses (see
services/operations.py) -- the model selects an operation and its arguments rather than
writing code.

Why: a real sandbox (no network, no filesystem, memory and time limits) is genuinely hard to
get right, and a restricted-globals exec() is not one. Since the whitelist covers filtering,
renaming, replacing, rounding, type coercion, formatting and null handling, code generation
buys very little coverage in exchange for an arbitrary-code-execution surface in a tool
people self-host.

When an instruction genuinely can't be expressed as a whitelisted operation, the model says
so and the user gets told plainly -- which is a better outcome than silently running
approximate code against their data.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.models.graph_state import GoalContext
from app.services.operations import operation_catalog
from app.services.prompts.untrusted import DATA_NOTICE, file_data

SYSTEM_PROMPT = """You translate a user's plain-English data cleaning instruction into ONE \
operation from the fixed list below. You do not write code.

Return ONLY a JSON object, no markdown fences, no commentary:

{
  "operation": "<name from the list, or null if none fits>",
  "params": {...},
  "explanation": "One sentence, plain English, describing exactly what will happen.",
  "refusal": "<only when operation is null: why this can't be done, in plain English>"
}

Available operations:
{operation_catalog}

Parameters:
- Every operation takes column="<exact column name from the list of columns given>".
- filter_rows: comparison=one of eq, ne, gt, gte, lt, lte, contains, not_contains, is_null, \
not_null; value=<the value to compare against, omit for is_null/not_null>; mode="keep" to \
keep matching rows or "drop" to remove them.
- standardize_case: case=one of lower, upper, title.
- standardize_dates: format=a strftime pattern such as "%Y-%m-%d".
- fill_constant: value=<the value to fill with>.
- rename_column: new_name=<the new column name>.
- round_numeric: decimals=<whole number>.
- replace_values: find=<value to replace>, replace=<replacement value>.

Rules:
- Use only column names that appear in the provided column list. If the user names a column \
that does not exist, set operation to null and say which column is missing.
- If the instruction asks for more than one change, handle only the first and say so in the \
explanation.
- If nothing in the list expresses the instruction, set operation to null and explain what \
you cannot do. Never invent an operation name."""


def build_conversational_cleaning_messages(
    instruction: str,
    columns: list[str],
    goal_context: GoalContext | None,
) -> list[BaseMessage]:
    goal_block = (
        f"use_case={goal_context.use_case}, expected_output={goal_context.expected_output}, "
        f"critical_columns={goal_context.critical_columns}"
        if goal_context
        else "not specified"
    )
    system = (
        SYSTEM_PROMPT.replace("{operation_catalog}", operation_catalog(include_conversational=True))
        + "\n\n"
        + DATA_NOTICE
    )
    human = (
        f"Columns in the data:\n{file_data(', '.join(columns))}\n"
        f"Goal context: {goal_block}\n\n"
        f"Instruction: {instruction}\n\n"
        "Return the JSON now."
    )
    return [SystemMessage(content=system), HumanMessage(content=human)]


def build_retry_messages(
    instruction: str,
    columns: list[str],
    goal_context: GoalContext | None,
    previous_error: str,
) -> list[BaseMessage]:
    """PRD FR-27/§14.2: one retry with the error fed back."""
    messages = build_conversational_cleaning_messages(instruction, columns, goal_context)
    messages.append(
        HumanMessage(
            content=(
                f"That response could not be used ({previous_error}). Return ONLY the JSON "
                "object described above, using an operation name from the list."
            )
        )
    )
    return messages
