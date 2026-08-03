"""POST /api/chat/{session_id} -- STUB, correct signature. PRD FR-37..40.

Will append the message to `state.conversation_history` and route into
graph/nodes/conversational_loop.py, streaming the resulting action(s) as SSE.

TODO: implement per PRD FR-37..40, once conversational_loop (§13.1) is implemented.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.api_schemas import ChatRequest

router = APIRouter()


@router.post("/api/chat/{session_id}")
async def send_chat_message(session_id: str, chat: ChatRequest):
    raise HTTPException(status_code=501, detail="Chat is not yet implemented -- see PRD FR-37..40")
