"""Core state models carried through the LangGraph pipeline. PRD §9."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

from app.models.suggestion_card import ActionLogEntry, SuggestionCard

UseCase = Literal["ml_training", "reporting", "analysis", "pipeline_validation"]
CleaningAggression = Literal["conservative", "moderate", "thorough"]
IssueType = Literal[
    "missing_values",
    "duplicates",
    "type_mismatch",
    "outliers",
    "inconsistent_format",
    "high_cardinality",
]
Severity = Literal["high", "medium", "low"]


class GoalContext(BaseModel):
    """PRD §9.2. User's stated goal, injected into every agent's system prompt."""

    use_case: UseCase
    expected_output: str
    critical_columns: list[str] = Field(default_factory=list)
    cleaning_aggression: CleaningAggression = "moderate"


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
    (action_log, conversation_history) use operator.add reducers so parallel/sequential
    node writes append rather than overwrite, per LangGraph's Annotated-reducer convention.
    """

    session_id: str
    file_path: str
    cleaned_file_path: Optional[str] = None
    file_name: str
    file_size_bytes: int

    goal_context: Optional[GoalContext] = None

    raw_profile: Optional[dict] = None
    issues: list[ColumnIssue] = Field(default_factory=list)
    suggestion_cards: list[SuggestionCard] = Field(default_factory=list)

    approved_action_ids: list[str] = Field(default_factory=list)
    skipped_action_ids: list[str] = Field(default_factory=list)
    skip_reasons: dict[str, str] = Field(default_factory=dict)

    action_log: Annotated[list[ActionLogEntry], operator.add] = Field(default_factory=list)
    conversation_history: Annotated[list[dict], operator.add] = Field(default_factory=list)

    current_agent: Optional[str] = None
    error_message: Optional[str] = None
