"""Core state models carried through the LangGraph pipeline. PRD §9."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

from app.models.suggestion_card import ActionLogEntry, SuggestionCard  # noqa: F401 -- re-exported for callers that validate state dicts back into models

UseCase = Literal["ml_training", "reporting", "analysis", "pipeline_validation"]
CleaningAggression = Literal["conservative", "moderate", "thorough"]
TaskType = Literal["auto", "classification", "regression"]
IssueType = Literal[
    "missing_values",
    "duplicates",
    "type_mismatch",
    "outliers",
    "inconsistent_format",
    "high_cardinality",
    # Only detectable once GoalContext.target_column is set -- see services/ml_readiness.py.
    "target_missing",
    "target_imbalance",
    "target_leakage",
]
Severity = Literal["high", "medium", "low"]


class GoalContext(BaseModel):
    """PRD §9.2. User's stated goal, injected into every agent's system prompt."""

    use_case: UseCase
    expected_output: str
    critical_columns: list[str] = Field(default_factory=list)
    cleaning_aggression: CleaningAggression = "moderate"

    # The column the user ultimately wants to predict, if they're heading for a model.
    # Unlike the fields above -- which only tint the wording of prompts -- this one changes
    # what the tool *does*: it unlocks the leakage/imbalance detectors and hard-blocks fixes
    # that would corrupt the target. Optional throughout; empty means "not building a model"
    # and every target-aware code path is skipped.
    target_column: Optional[str] = None
    task_type: TaskType = "auto"  # "auto" infers from the target column's own values


class ColumnIssue(BaseModel):
    """PRD §9.3. A single detected problem, produced by the Profiling Agent (FR-11..19)."""

    issue_id: str
    column_name: str  # "DATASET" for dataset-level issues
    issue_type: IssueType
    severity: Severity
    description: str
    stats: dict = Field(default_factory=dict)
    priority_score: float = Field(ge=0.0, le=1.0)


class GraphState(BaseModel):
    """PRD §9.1. Central state object carried through the LangGraph pipeline.

    session_id doubles as the LangGraph thread_id. Fields accumulated across nodes
    conversation_history uses an operator.add reducer so successive node writes append rather
    than overwrite, per LangGraph's Annotated-reducer convention. action_log deliberately
    does not -- see the note on that field.

    IMPORTANT -- why the collections below are `list[dict]` and not `list[ColumnIssue]` /
    `list[SuggestionCard]` / `list[ActionLogEntry]`:

    The checkpointer serializes nested Pydantic models into LangChain's envelope form
    (`{"lc": 2, "type": "constructor", "id": [...], "kwargs": {...}}`). Whether those get
    reconstructed on read is version-dependent -- when they aren't, the raw envelope dict
    is handed straight to GraphState validation and every field reads as missing, so
    resuming a checkpointed session dies with a wall of "Field required" errors.

    Keeping state JSON-native makes round-tripping independent of the serializer's type
    registry. Nodes validate into the real models at the edges (ColumnIssue.model_validate
    etc.) for type-safe logic, then store model_dump() back into state.
    """

    session_id: str
    file_path: str
    cleaned_file_path: Optional[str] = None
    file_name: str
    file_size_bytes: int

    goal_context: Optional[GoalContext] = None

    raw_profile: Optional[dict] = None
    issues: list[dict] = Field(default_factory=list)  # ColumnIssue.model_dump()
    suggestion_cards: list[dict] = Field(default_factory=list)  # SuggestionCard.model_dump()

    approved_action_ids: list[str] = Field(default_factory=list)
    skipped_action_ids: list[str] = Field(default_factory=list)
    skip_reasons: dict[str, str] = Field(default_factory=dict)

    # ActionLogEntry.model_dump(). Deliberately NOT an operator.add reducer: undo (US-10)
    # has to rewrite entries in place to mark them "undone", and an append-only reducer makes
    # replacement impossible. Writers return the full list instead (read-modify-write), which
    # is explicit and keeps correction possible.
    action_log: list[dict] = Field(default_factory=list)
    conversation_history: Annotated[list[dict], operator.add] = Field(default_factory=list)

    audit_markdown_path: Optional[str] = None
    audit_pdf_path: Optional[str] = None  # None when WeasyPrint's OS libs are unavailable

    current_agent: Optional[str] = None
    error_message: Optional[str] = None
