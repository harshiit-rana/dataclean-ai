"""Prompt for the model card's prose. Narrow by design, like audit_prompts.py.

The model writes two paragraphs of interpretation. Every number, table, verdict and warning
in the card is assembled in code from the evaluation, for the same reason the audit report
works that way: a metric that passes through a language model can come out subtly different
from the one that was measured, and a model card is a document people quote.

The verdict in particular is never delegated. Whether a model beat what luck could produce is
a threshold comparison against the test set's size -- decided in modeling.assess_result, and
handed to the prompt as a finding the prose must not contradict.
"""

from __future__ import annotations

import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

SYSTEM_PROMPT = """You are writing the interpretation section of a model card -- a document \
the reader may attach to a report, show to a manager, or return to in six months to remember \
what was built. Write plainly and factually. No marketing language, no praise, no filler, no \
hedging about "further analysis".

Return ONLY a JSON object in exactly this shape, no markdown fences, no text around it:

{
  "summary": "2-4 sentences: what was built, from what, and whether it works.",
  "interpretation": "2-5 sentences reading the numbers for someone who is not a data \
scientist.",
  "cautions": ["Short, specific warning.", "Another."]
}

Rules:
- The verdict you are given is settled. If it says the model is no better than guessing, say \
so plainly in the summary. Do not soften it, do not suggest the numbers are promising, and \
do not recommend tuning as a fix -- when there is no signal in the columns, no amount of \
tuning creates one.
- Explain what the headline metric means in terms of the actual outcome being predicted, not \
in terms of statistics. "Of the people it flags, about two thirds really do leave" beats \
"precision is 0.66".
- cautions: 2-4 entries, each specific to this data. Draw on the class balance, the test-set \
size, the cross-validation spread and the columns that were excluded. Do not include generic \
advice that would apply to any model."""


def build_model_card_messages(result: dict, file_name: str, goal_context) -> list[BaseMessage]:
    best = next(
        (r for r in result.get("results", []) if r.get("model") == result.get("best_model")),
        {},
    )
    deep = result.get("deep_diagnostics") or {}

    # Only what's needed to write two paragraphs. The curve data is large, and none of it
    # would change the prose.
    payload = {
        "target_column": result.get("target_column"),
        "task": result.get("task"),
        "chosen_model": best.get("label"),
        "metrics": best.get("metrics", {}),
        "baseline": result.get("baseline"),
        "verdict": (result.get("assessment") or {}).get("verdict"),
        "verdict_explanation": (result.get("assessment") or {}).get("headline"),
        "split": result.get("split"),
        "features_used": (
            result.get("features", {}).get("numeric", [])
            + result.get("features", {}).get("categorical", [])
        ),
        "columns_left_out": result.get("features", {}).get("dropped", []),
        "cross_validation": deep.get("cross_validation"),
        "top_columns": best.get("column_importance", [])[:5],
        "class_counts": best.get("diagnostics", {}).get("class_distribution"),
    }

    goal_block = (
        f"use_case={goal_context.use_case}, expected_output={goal_context.expected_output}"
        if goal_context
        else "not specified"
    )

    human = (
        f"Dataset: {file_name}\n"
        f"What the user said they wanted: {goal_block}\n\n"
        f"Evaluation:\n{json.dumps(payload, indent=2, default=str)}\n\n"
        "Write the model card JSON now."
    )
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human)]


def build_retry_messages(result: dict, file_name: str, goal_context, previous_error: str):
    messages = build_model_card_messages(result, file_name, goal_context)
    messages.append(
        HumanMessage(
            content=(
                f"That response could not be parsed as JSON ({previous_error}). Return ONLY "
                "the JSON object described above -- no markdown fences, no commentary."
            )
        )
    )
    return messages
