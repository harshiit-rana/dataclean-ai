"""cleaning_agent node. PRD §12.3.

Executes the fixes the user approved at the human_approval interrupt, in card order, and
writes the result to a NEW file (`cleaned.csv`) in the session directory. The user's
uploaded copy (`original.csv`) is never modified, and their local file is never touched at
all -- only the browser-uploaded copy ever existed server-side.

Each approved card carries a machine-readable `operation` chosen by the Explanation Agent
from the whitelist in services/operations.py, so no LLM call and no code execution happens
here: this node is deterministic pandas. See operations.py's docstring for why that's
preferred over the PRD's generate-and-sandbox approach.

A single failing operation is logged against that action and skipped -- it never aborts the
remaining approved fixes.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

import pandas as pd
from langgraph.config import get_stream_writer

from app.models.graph_state import GraphState
from app.models.suggestion_card import ActionLogEntry, SuggestionCard
from app.services import conflicts, operations, session_store

logger = logging.getLogger(__name__)


def _new_action_id() -> str:
    return f"action_{uuid.uuid4().hex[:8]}"


async def cleaning_agent(state: GraphState) -> dict:
    writer = get_stream_writer()
    writer({"event": "agent_start", "agent": "cleaning_agent", "message": "Applying your approved fixes..."})

    df = pd.read_csv(state.file_path)
    approved = set(state.approved_action_ids)
    skipped = set(state.skipped_action_ids)

    # Carried onto each log entry so the audit report can describe *why* a change was made,
    # not just what it did to the row counts.
    issue_types = {issue["issue_id"]: issue.get("issue_type") for issue in state.issues}

    action_log: list[ActionLogEntry] = []
    applied_count = 0

    for card_data in state.suggestion_cards:
        # state holds plain dicts (see GraphState docstring); validate for typed access.
        card = SuggestionCard.model_validate(card_data)
        issue_type = issue_types.get(card.issue_id)
        if card.card_id in skipped or card.card_id not in approved:
            action_log.append(
                ActionLogEntry(
                    action_id=_new_action_id(),
                    card_id=card.card_id,
                    column_name=card.column_name,
                    issue_type=issue_type,
                    operation=card.operation or "none",
                    status="rejected",
                    # verbatim, never paraphrased (FR-30)
                    rejection_reason=state.skip_reasons.get(card.card_id),
                )
            )
            continue

        if not card.operation:
            logger.warning("Approved card %s has no executable operation; skipping", card.card_id)
            action_log.append(
                ActionLogEntry(
                    action_id=_new_action_id(),
                    card_id=card.card_id,
                    column_name=card.column_name,
                    issue_type=issue_type,
                    operation="none",
                    status="rejected",
                    failure_reason="This suggestion had no executable fix attached.",
                )
            )
            continue

        # An earlier approved fix may have removed the column this one was going to change --
        # a leaky column gets flagged for leakage *and* for its nulls and its type, and the
        # first fix drops it. That is the app working correctly, so it is recorded as
        # superseded rather than reported to the user as a failure.
        superseded = conflicts.superseded_reason(
            df, card.operation, card.operation_params, card.column_name
        )
        if superseded:
            action_log.append(
                ActionLogEntry(
                    action_id=_new_action_id(),
                    card_id=card.card_id,
                    column_name=card.column_name,
                    issue_type=issue_type,
                    operation=card.operation,
                    operation_params=card.operation_params,
                    status="superseded",
                    failure_reason=superseded,
                )
            )
            continue

        action_id = _new_action_id()
        writer(
            {
                "event": "action_start",
                "action_id": action_id,
                "operation": card.operation,
                "column_name": card.column_name,
            }
        )

        try:
            df, before_stats, after_stats = operations.apply_operation(
                df, card.operation, card.operation_params
            )
        except (operations.OperationFailedError, operations.UnknownOperationError) as exc:
            logger.warning("Operation %s failed for card %s: %s", card.operation, card.card_id, exc)
            action_log.append(
                ActionLogEntry(
                    action_id=action_id,
                    card_id=card.card_id,
                    column_name=card.column_name,
                    issue_type=issue_type,
                    operation=card.operation,
                    status="rejected",
                    failure_reason=f"Could not be applied: {exc}",
                )
            )
            writer({"event": "error", "error": "Fix could not be applied", "detail": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 -- one bad fix must not kill the whole run
            logger.exception("Unexpected failure applying %s for card %s", card.operation, card.card_id)
            action_log.append(
                ActionLogEntry(
                    action_id=action_id,
                    card_id=card.card_id,
                    column_name=card.column_name,
                    issue_type=issue_type,
                    operation=card.operation,
                    status="rejected",
                    failure_reason=f"Could not be applied: {exc}",
                )
            )
            writer({"event": "error", "error": "Fix could not be applied", "detail": str(exc)})
            continue

        applied_count += 1
        action_log.append(
            ActionLogEntry(
                action_id=action_id,
                card_id=card.card_id,
                column_name=card.column_name,
                issue_type=issue_type,
                operation=card.operation,
                operation_params=card.operation_params,
                status="applied",
                before_stats=before_stats,
                after_stats=after_stats,
            )
        )
        writer(
            {
                "event": "action_complete",
                "action_id": action_id,
                "before_stats": before_stats,
                "after_stats": after_stats,
            }
        )

    cleaned_path = Path(state.file_path).parent / "cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    # A run that applied nothing leaves the data as it was and adds no version.
    await asyncio.to_thread(
        session_store.record_dataset_version_in, cleaned_path.parent, "cleaning_run", None,
        [entry.model_dump() for entry in action_log],
    )

    superseded_count = sum(1 for entry in action_log if entry.status == "superseded")
    writer(
        {
            "event": "all_actions_complete",
            "actions_applied": applied_count,
            # Superseded fixes are reported separately -- lumping them into "skipped" makes a
            # run that went exactly right look like it partly failed.
            "actions_skipped": len(action_log) - applied_count - superseded_count,
            "actions_superseded": superseded_count,
        }
    )

    return {
        "cleaned_file_path": str(cleaned_path),
        "action_log": [entry.model_dump() for entry in action_log],
        "current_agent": "cleaning_agent",
    }
