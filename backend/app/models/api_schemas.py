"""Request/response models for the API surface. PRD §10.1, error shape §10.3."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from app.models.graph_state import CleaningAggression, TaskType, UseCase


class ErrorResponse(BaseModel):
    """PRD §10.3. All error responses follow this shape."""

    error: str
    detail: str


class UploadResponse(BaseModel):
    session_id: str
    file_name: str
    row_count: int
    column_count: int
    # Returned so the goal form can offer a real target-column picker instead of asking the
    # user to type a column name and hope they spell it the same way.
    columns: list[str] = []


class GoalContextRequest(BaseModel):
    use_case: UseCase
    expected_output: str
    critical_columns: list[str] = []
    cleaning_aggression: CleaningAggression = "moderate"
    target_column: Optional[str] = None
    task_type: TaskType = "auto"


class GoalContextResponse(BaseModel):
    session_id: str
    accepted: bool


class PreviewColumn(BaseModel):
    name: str
    dtype: str
    null_count: int
    unique_count: int
    is_critical: bool
    is_target: bool = False  # the column the user wants to predict, if any


class PreviewResponse(BaseModel):
    """Actual CSV contents for the Review page's data table (see routes_preview.py)."""

    session_id: str
    columns: list[PreviewColumn]
    rows: list[dict]
    total_rows: int
    total_columns: int
    preview_row_count: int


# --- Stub-endpoint request/response shapes -------------------------------------------
# These endpoints are not implemented yet (see routes_resume/chat/undo/export). The
# schemas below exist now so the route signatures and the frontend API client can be
# written against a stable contract ahead of the real implementation.


class ResumeRequest(BaseModel):
    """PRD §13.2 resume payload."""

    approved_action_ids: list[str]
    skipped_action_ids: list[str] = []
    skip_reasons: dict[str, str] = {}


class ChatRequest(BaseModel):
    """PRD FR-37: plain-English cleaning instruction."""

    message: str


class UndoRequest(BaseModel):
    """PRD US-10. If action_id is omitted, undo the most recent action."""

    action_id: Optional[str] = None


class ReadinessResponse(BaseModel):
    """GET /api/readiness/{session_id}. Whether the data as it currently stands can support
    the model the user said they wanted -- computed on demand from the file on disk, so it
    stays correct after an undo or a conversational edit."""

    session_id: str
    target_column: Optional[str]
    task_type: Optional[str]
    row_count: int
    column_count: int
    verdict: str  # ready | warnings | blocked | not_applicable
    blockers: list[str]
    warnings: list[str]
    notes: list[str]
    target_summary: Optional[dict]
    remaining_issues: dict


class SessionStatusResponse(BaseModel):
    session_id: str
    current_agent: Optional[str]
    error_message: Optional[str]
    issue_count: int
    approved_count: int
    rejected_count: int
