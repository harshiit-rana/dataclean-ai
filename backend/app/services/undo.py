"""Undoing an applied cleaning action. PRD US-10.

Approach: replay, not snapshots. The PRD suggests saving a column snapshot before each
operation as a reversible diff, but every operation in services/operations.py is
deterministic, so the original file plus the action log is already a complete description of
the current state. Undoing means re-deriving the data from `original.csv` through every
still-applied action, skipping the undone one.

Replay is strictly better here than stored diffs:
- nothing extra to persist, so no storage cost and no snapshots to fall out of sync
- correct for operations that change the row count (dropping rows/duplicates), which a
  per-column snapshot cannot restore
- gives the exactly right semantics: "what would this data look like if I had never
  approved that fix", including for actions applied before it

Undone entries stay in the log with status="undone" so the audit report can still show that
the change was applied and then reverted.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from app.core.errors import AppError
from app.services import operations

logger = logging.getLogger(__name__)


class ActionNotFoundError(AppError):
    status_code = 404

    def __init__(self, action_id: str):
        super().__init__("Action not found", f"No action {action_id!r} exists in this session.")


class ActionNotUndoableError(AppError):
    status_code = 409

    def __init__(self, status: str):
        super().__init__(
            "Nothing to undo",
            f"That action was never applied (status: {status}), so there is nothing to revert.",
        )


def undo_action(
    *,
    original_path: Path,
    cleaned_path: Path,
    action_log: list[dict],
    action_id: str,
) -> list[dict]:
    """Reverts one applied action and rewrites `cleaned_path`. Returns the updated log.

    Raises ActionNotFoundError / ActionNotUndoableError; the caller maps those to responses.
    """
    target = next((entry for entry in action_log if entry["action_id"] == action_id), None)
    if target is None:
        raise ActionNotFoundError(action_id)
    if target["status"] != "applied":
        raise ActionNotUndoableError(target["status"])

    updated_log = [
        {**entry, "status": "undone"} if entry["action_id"] == action_id else dict(entry)
        for entry in action_log
    ]

    df = pd.read_csv(original_path)
    for entry in updated_log:
        if entry["status"] != "applied":
            continue
        params = dict(entry.get("operation_params") or {})
        params.setdefault("column", entry["column_name"])
        try:
            df, _before, _after = operations.apply_operation(df, entry["operation"], params)
        except (operations.OperationFailedError, operations.UnknownOperationError) as exc:
            # An action that succeeded originally can legitimately fail on replay once an
            # earlier one is removed (e.g. a fill that depended on a prior type cast). Mark
            # it undone too rather than silently producing data that contradicts the log.
            logger.warning(
                "Replay of %s (%s) failed after undoing %s: %s",
                entry["action_id"],
                entry["operation"],
                action_id,
                exc,
            )
            entry["status"] = "undone"
            entry["rejection_reason"] = (
                f"Automatically reverted: this fix no longer applies once the undone change "
                f"was removed ({exc})"
            )

    df.to_csv(cleaned_path, index=False)
    return updated_log
