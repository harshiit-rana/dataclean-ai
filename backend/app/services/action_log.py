"""Reading the session's action log, and what it says about the data a model was trained on.

The log lives in the graph checkpoint (Redis), not in the session directory, so reading it is a
network call that can fail. Everything here treats "could not read it" as "no fixes to report"
rather than as an error: a log that is unavailable must never stop a model being trained, and
the cost of that choice is a missing caution, which the caller can live with.
"""

from __future__ import annotations

import logging

from app.graph.checkpointer import thread_config
from app.services.ml_readiness import DATASET_STATISTIC_OPERATIONS

logger = logging.getLogger(__name__)


async def read(graph, session_id: str) -> list[dict]:
    """The session's action log, or [] when it was never cleaned or cannot be read.

    A model trained straight from the upload is still worth applying and still worth training,
    so a missing or unreadable log is an empty one.
    """
    if graph is None:
        return []
    try:
        state = await graph.aget_state(thread_config(session_id))
    except Exception:  # noqa: BLE001 -- no readable log means no fixes, not a failed request
        logger.warning("Could not read the action log for session %s", session_id, exc_info=True)
        return []
    return list((state.values or {}).get("action_log", [])) if state else []


def whole_file_statistics(action_log: list[dict] | None, input_columns: list[str]) -> list[dict]:
    """Applied fixes that computed their value from every row, on columns the model uses.

    A fill with the column's median, or an outlier limit, is worked out over the whole file --
    the rows later held out for testing included -- before any split. Once applied, the
    pipeline's own training-only imputer has nothing left to do, so every run trained on that
    data carries the leak. The app no longer proposes these for a model input, but sessions
    cleaned before that, and fixes typed in chat, still do.

    Undone fixes are not listed (they are no longer in the data), and neither are fixes on
    columns the model does not use, which cannot have reached it.
    """
    used = set(input_columns)
    found: list[dict] = []
    for entry in action_log or []:
        if entry.get("status") != "applied":
            continue
        operation = entry.get("operation")
        column = entry.get("column_name")
        if operation in DATASET_STATISTIC_OPERATIONS and column in used:
            found.append({"operation": operation, "column": column})
    return found
