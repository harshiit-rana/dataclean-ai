"""POST /api/profile/{session_id} -- REAL, SSE stream.

Starts the graph (goal_intake -> profiling_agent -> explanation_agent -> human_approval)
and streams progress as SSE. The response ends once the graph pauses at the human_approval
interrupt (an `interrupt_reached` event, emitted by stream_graph_as_sse) or completes.

Called again for a session that already ran -- a reload of the Review page -- it replays the
saved cards rather than starting over, or refuses if the session has already been cleaned.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends

from app.api.sse import sse_response
from app.config import Settings, get_settings
from app.core.errors import AppError, SessionNotFoundError
from app.dependencies import get_graph
from app.graph.checkpointer import thread_config
from app.models.graph_state import ColumnIssue
from app.models.sse_events import (
    AllCardsCompleteEvent,
    CardCompleteEvent,
    InterruptReachedEvent,
    ProfilingCompleteEvent,
    encode_sse,
)
from app.models.suggestion_card import SuggestionCard
from app.services.session_store import read_session_meta, session_dir
from app.services.sse_broker import stream_graph_as_sse

logger = logging.getLogger(__name__)

router = APIRouter()


class GoalContextMissingError(AppError):
    def __init__(self):
        super().__init__(
            "Goal context missing",
            "Submit POST /api/goal/{session_id} before starting profiling.",
        )


class AlreadyCleanedError(AppError):
    status_code = 409

    def __init__(self):
        super().__init__(
            "Already cleaned",
            "This session has already been reviewed and cleaned. Its results are on the "
            "cleaning page -- profiling it again would throw your decisions away.",
        )


def _paused_for_review(state) -> bool:
    return bool(state and state.values and "human_approval" in (state.next or ()))


def _finished(state) -> bool:
    return bool(state and state.values and not state.next)


async def _replay_review(values: dict, session_id: str) -> AsyncIterator[str]:
    """The events a first run sends, rebuilt from the saved state. No LLM calls.

    The Review page starts profiling when it mounts, so a reload -- or a browser restoring the
    tab -- used to start the whole graph again: profiling re-run, every card rewritten at one
    LLM call each, and any decisions not yet submitted lost. The cards already exist in the
    checkpoint the graph paused on. Sending them back is what a reload should mean.
    """
    issues = [ColumnIssue.model_validate(i) for i in values.get("issues") or []]
    counts = {level: sum(1 for i in issues if i.severity == level) for level in ("high", "medium", "low")}
    yield encode_sse(
        ProfilingCompleteEvent(issue_count=len(issues), issues=issues, **counts)
    )
    cards = values.get("suggestion_cards") or []
    for card in cards:
        yield encode_sse(CardCompleteEvent(card=SuggestionCard.model_validate(card)))
    yield encode_sse(AllCardsCompleteEvent(total_cards=len(cards)))
    yield encode_sse(InterruptReachedEvent(session_id=session_id))


@router.post("/api/profile/{session_id}")
async def start_profiling(
    session_id: str,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_graph),
):
    meta = read_session_meta(settings.upload_dir, session_id)
    if meta is None:
        raise SessionNotFoundError(session_id)
    if meta.get("goal_context") is None:
        raise GoalContextMissingError()

    file_path = session_dir(settings.upload_dir, session_id) / "original.csv"
    input_state = {
        "session_id": session_id,
        "file_path": str(file_path),
        "file_name": meta["file_name"],
        "file_size_bytes": meta["file_size_bytes"],
        "goal_context": meta["goal_context"],
    }
    config = thread_config(session_id)

    # A reload of the Review page lands here too. Only a session that has never run, or whose
    # run died part-way, is started from the top -- restarting is the recovery path for the
    # second, and there is nothing else to show for the first.
    try:
        existing = await graph.aget_state(config)
    except Exception:  # noqa: BLE001 -- unreadable state is no reason to refuse a fresh start
        logger.warning("Could not read existing state for session %s", session_id, exc_info=True)
        existing = None

    if _paused_for_review(existing):
        return sse_response(_replay_review(dict(existing.values), session_id))
    if _finished(existing):
        raise AlreadyCleanedError()

    return sse_response(stream_graph_as_sse(graph, input_state, config, session_id))
