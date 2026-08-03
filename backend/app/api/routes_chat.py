"""POST /api/chat/{session_id} -- free-form cleaning instructions. PRD FR-37..40.

Maps the instruction onto a whitelisted operation (see prompts/cleaning_prompts.py for why
that beats generating code), applies it to the cleaned file, records it in the action log
with source="conversational", and regenerates the audit report so the document keeps
matching the data.

Implemented as a plain endpoint rather than the conversational_loop graph node: the graph
has already run to completion by this point, and re-entering it per message would mean
re-running profiling and explanation for no benefit. The node stays unwired.
"""

from __future__ import annotations

import json
import logging
import uuid

import pandas as pd
from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.core.errors import AppError, SessionNotFoundError
from app.dependencies import get_graph
from app.graph.checkpointer import thread_config
from app.graph.nodes.audit_agent import generate_audit_artifacts
from app.models.api_schemas import ChatRequest
from app.models.graph_state import GraphState
from app.models.suggestion_card import ActionLogEntry
from app.services import operations
from app.services.llm_client import get_llm_client
from app.services.prompts.cleaning_prompts import (
    build_conversational_cleaning_messages,
    build_retry_messages,
)
from app.services.session_store import session_dir

logger = logging.getLogger(__name__)

router = APIRouter()


class NoCleanedFileError(AppError):
    status_code = 409

    def __init__(self):
        super().__init__(
            "Nothing to change yet",
            "Approve some fixes first — there's no cleaned file to apply instructions to.",
        )


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


async def _interpret(instruction: str, columns: list[str], goal_context, client) -> dict:
    """Returns the model's parsed intent, or a refusal dict if it can't be interpreted."""
    messages = build_conversational_cleaning_messages(instruction, columns, goal_context)
    last_error = ""

    for attempt in range(2):  # initial try + one retry (FR-27)
        try:
            response = await client.ainvoke(messages)
            raw = response.content if isinstance(response.content, str) else str(response.content)
            parsed = json.loads(_strip_code_fences(raw))

            name = parsed.get("operation")
            if name is None:
                return {"operation": None, "refusal": parsed.get("refusal") or parsed.get("explanation")}
            if name not in operations.OPERATIONS:
                raise ValueError(f"unknown operation {name!r}")

            return {
                "operation": name,
                "params": parsed.get("params") or {},
                "explanation": parsed.get("explanation") or "",
            }
        except Exception as exc:  # noqa: BLE001 -- a bad response must not 500 the endpoint
            last_error = str(exc)
            logger.warning("Chat instruction parse failed (attempt %d): %s", attempt + 1, exc)
            messages = build_retry_messages(instruction, columns, goal_context, last_error)

    return {
        "operation": None,
        "refusal": "I couldn't turn that into a change I know how to make safely. Try "
        "rephrasing it, for example \"drop rows where age is negative\".",
    }


@router.post("/api/chat/{session_id}")
async def send_chat_message(
    session_id: str,
    chat: ChatRequest,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_graph),
) -> dict:
    config = thread_config(session_id)
    state = await graph.aget_state(config)
    if not state.values:
        raise SessionNotFoundError(session_id)

    directory = session_dir(settings.upload_dir, session_id)
    cleaned_path = directory / "cleaned.csv"
    if not cleaned_path.exists():
        raise NoCleanedFileError()

    df = pd.read_csv(cleaned_path)
    values = dict(state.values)
    goal_context = GraphState.model_validate(values).goal_context

    client = get_llm_client(settings, streaming=False)
    intent = await _interpret(chat.message, [str(c) for c in df.columns], goal_context, client)

    if intent["operation"] is None:
        return {
            "applied": False,
            "message": intent["refusal"],
            "actions": values.get("action_log", []),
        }

    params = dict(intent["params"])
    if operations.OPERATIONS[intent["operation"]].needs_column and not params.get("column"):
        return {
            "applied": False,
            "message": "I couldn't tell which column you meant. Try naming it explicitly.",
            "actions": values.get("action_log", []),
        }

    try:
        new_df, before_stats, after_stats = operations.apply_operation(
            df, intent["operation"], params
        )
    except (operations.OperationFailedError, operations.UnknownOperationError) as exc:
        return {
            "applied": False,
            "message": f"That change couldn't be applied: {exc}",
            "actions": values.get("action_log", []),
        }

    new_df.to_csv(cleaned_path, index=False)

    entry = ActionLogEntry(
        action_id=f"action_{uuid.uuid4().hex[:8]}",
        card_id=None,  # conversational actions have no originating suggestion card
        column_name=str(params.get("column", "DATASET")),
        operation=intent["operation"],
        operation_params=params,
        status="applied",
        before_stats=before_stats,
        after_stats=after_stats,
        source="conversational",
    )
    updated_log = list(values.get("action_log", [])) + [entry.model_dump()]

    await graph.aupdate_state(
        config,
        {
            "action_log": updated_log,
            "conversation_history": [
                {"role": "user", "content": chat.message},
                {"role": "assistant", "content": intent["explanation"]},
            ],
        },
    )

    values["action_log"] = updated_log
    try:
        await generate_audit_artifacts(GraphState.model_validate(values))
    except Exception:  # noqa: BLE001 -- the change itself already succeeded
        logger.exception("Audit report could not be regenerated after a chat instruction")

    return {
        "applied": True,
        "message": intent["explanation"],
        "action_id": entry.action_id,
        "before_stats": before_stats,
        "after_stats": after_stats,
        "actions": updated_log,
    }
