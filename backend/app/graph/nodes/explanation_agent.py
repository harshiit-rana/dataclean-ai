"""explanation_agent node. PRD §12.2 -- REAL implementation.

One streamed Claude call per issue so each suggestion card can be sent to the frontend
token-by-token (FR-23) via card_chunk events, then finalized as a structured SuggestionCard
via card_complete once the model finishes.
"""

from __future__ import annotations

import logging
import uuid

from langgraph.config import get_stream_writer

from app.config import get_settings
from app.models.graph_state import ColumnIssue, GraphState
from app.models.suggestion_card import SuggestionCard
from app.services import conflicts
from app.services.llm_client import get_llm_client
from app.services.ml_readiness import enforce_target_safety
from app.services.operations import OPERATIONS, parse_operation
from app.services.prompts.explanation_prompts import (
    build_explanation_messages,
    parse_card_sections,
)

logger = logging.getLogger(__name__)


def _new_card_id() -> str:
    return f"card_{uuid.uuid4().hex[:8]}"


def _readable_failure(exc: Exception, written: int, total: int) -> str:
    """What to tell the user when card generation stops part-way.

    Leads with what they still have. A run that produced four of twelve cards is a partial
    success, and saying so is more useful than reporting an error code."""
    got = (
        f"The {written} suggestion{'' if written == 1 else 's'} already written "
        f"{'is' if written == 1 else 'are'} ready to review."
        if written
        else "No suggestions were written."
    )
    text = str(exc)
    if "RateLimit" in type(exc).__name__ or "429" in text or "rate_limit" in text:
        return (
            f"The AI provider's rate limit was reached after {written} of {total} issues. "
            f"{got} Wait for the limit to reset and run the review again for the rest, or "
            f"switch provider in .env."
        )
    return f"{got} The provider reported: {text[:250]}"


async def explanation_agent(state: GraphState) -> dict:
    writer = get_stream_writer()
    writer(
        {"event": "agent_start", "agent": "explanation_agent", "message": "Writing suggestions..."}
    )

    settings = get_settings()
    client = get_llm_client(settings, streaming=True)

    cards: list[SuggestionCard] = []
    for issue_data in state.issues:
        # state holds plain dicts (see GraphState docstring); validate back into the model
        # for type-safe access here.
        issue = ColumnIssue.model_validate(issue_data)
        card_id = _new_card_id()
        messages = build_explanation_messages(issue, state.goal_context)

        full_text = ""
        try:
            async for chunk in client.astream(messages):
                delta = chunk.content if isinstance(chunk.content, str) else ""
                if not delta:
                    continue
                full_text += delta
                writer({"event": "card_chunk", "card_id": card_id, "chunk": delta})
        except Exception as exc:  # noqa: BLE001 -- one failure must not lose the whole run
            # Almost always a rate limit, and the next call would hit it too. Stop here and
            # hand back the cards already written: reviewing four fixes beats reviewing none,
            # and the profiling work is preserved in the checkpoint either way.
            logger.exception("Card generation failed after %d cards", len(cards))
            writer(
                {
                    "event": "error",
                    "error": "Couldn't write the rest of the suggestions",
                    "detail": _readable_failure(exc, len(cards), len(state.issues)),
                }
            )
            break

        sections = parse_card_sections(full_text)
        parsed_operation = parse_operation(sections.get("operation", ""))

        if parsed_operation is not None:
            op_name, op_params = parsed_operation
            # Models sometimes emit a column-scoped operation with the column argument
            # omitted (observed: `drop_outlier_rows()`). Each card is about exactly one
            # column, so fill it in rather than letting the fix fail at execution time.
            if OPERATIONS[op_name].needs_column and not op_params.get("column"):
                logger.info(
                    "Operation %s for card %s had no column; defaulting to the issue's "
                    "column %r",
                    op_name,
                    card_id,
                    issue.column_name,
                )
                op_params = {**op_params, "column": issue.column_name}
                parsed_operation = (op_name, op_params)

        # Deterministic guard rather than a prompt instruction: a fix that would impute the
        # prediction target has to be impossible, not discouraged. Runs here, at card-build
        # time, so the user approves the corrected operation rather than the original.
        warning: str | None = None
        if parsed_operation is not None:
            op_name, op_params = parsed_operation
            safe_name, safe_params, warning = enforce_target_safety(
                op_name, op_params, state.goal_context
            )
            if safe_name != op_name:
                logger.info(
                    "Card %s: %s on the target column was rewritten to %s",
                    card_id,
                    op_name,
                    safe_name or "no executable fix",
                )
            parsed_operation = (safe_name, safe_params) if safe_name else None

        if parsed_operation is None:
            logger.warning(
                "No usable OPERATION for card %s (issue %s, column %s); card will be "
                "informational only",
                card_id,
                issue.issue_id,
                issue.column_name,
            )

        card = SuggestionCard(
            card_id=card_id,
            issue_id=issue.issue_id,
            column_name=issue.column_name,
            found=sections.get("found", ""),
            proposed_action=sections.get("proposed_action", ""),
            reasoning=sections.get("reasoning", ""),
            tradeoff=sections.get("tradeoff", ""),
            operation=parsed_operation[0] if parsed_operation else None,
            operation_params=parsed_operation[1] if parsed_operation else {},
            warning=warning,
        )
        cards.append(card)
        writer({"event": "card_complete", "card": card.model_dump()})

    # Cards are written one issue at a time, so none of them can see the others. Now that
    # they all exist, flag the ones another card makes unnecessary -- a leaky column gets
    # flagged for leakage *and* for its nulls and its type, and only the first is worth doing.
    payload = [card.model_dump() for card in cards]
    detected = conflicts.describe_conflicts(payload)
    for card, data in zip(cards, payload):
        conflict = detected.get(card.card_id)
        # A target-safety warning outranks this one: it explains a fix that was rewritten,
        # which the user needs to read before approving.
        if conflict and not card.warning:
            card.warning = conflict
            data["warning"] = conflict

    writer({"event": "all_cards_complete", "total_cards": len(cards)})

    return {
        "suggestion_cards": payload,
        "current_agent": "explanation_agent",
    }
