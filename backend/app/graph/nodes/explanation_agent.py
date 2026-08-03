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
from app.services.llm_client import get_llm_client
from app.services.operations import OPERATIONS, parse_operation
from app.services.prompts.explanation_prompts import (
    build_explanation_messages,
    parse_card_sections,
)

logger = logging.getLogger(__name__)


def _new_card_id() -> str:
    return f"card_{uuid.uuid4().hex[:8]}"


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
        async for chunk in client.astream(messages):
            delta = chunk.content if isinstance(chunk.content, str) else ""
            if not delta:
                continue
            full_text += delta
            writer({"event": "card_chunk", "card_id": card_id, "chunk": delta})

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
        )
        cards.append(card)
        writer({"event": "card_complete", "card": card.model_dump()})

    writer({"event": "all_cards_complete", "total_cards": len(cards)})

    return {
        "suggestion_cards": [card.model_dump() for card in cards],
        "current_agent": "explanation_agent",
    }
