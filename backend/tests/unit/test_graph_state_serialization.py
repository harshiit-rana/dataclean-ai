"""GraphState serialization/deserialization tests. PRD §15.2, §9.1."""

from __future__ import annotations

from app.models.graph_state import ColumnIssue, GoalContext, GraphState
from app.models.suggestion_card import ActionLogEntry, SuggestionCard


def _sample_state() -> GraphState:
    return GraphState(
        session_id="11111111-1111-1111-1111-111111111111",
        file_path="/tmp/uploads/session/original.csv",
        file_name="data.csv",
        file_size_bytes=1234,
        goal_context=GoalContext(
            use_case="ml_training",
            expected_output="a clean numeric dataset for model training",
            critical_columns=["salary", "age"],
            cleaning_aggression="moderate",
        ),
        issues=[
            ColumnIssue(
                issue_id="issue_abc123",
                column_name="salary",
                issue_type="outliers",
                severity="high",
                description="Column 'salary' has 1 outlier value.",
                stats={"outlier_count": 1},
                priority_score=0.9,
            )
        ],
        suggestion_cards=[
            SuggestionCard(
                card_id="card_abc123",
                issue_id="issue_abc123",
                column_name="salary",
                found="One extreme salary value.",
                proposed_action="Cap the value at the 99th percentile.",
                reasoning="Keeps the row without letting it skew training.",
                tradeoff="Slightly understates the true maximum.",
            )
        ],
        approved_action_ids=["card_abc123"],
        action_log=[
            ActionLogEntry(
                action_id="action_1",
                card_id="card_abc123",
                column_name="salary",
                operation="cap_outliers",
                status="applied",
                source="suggestion_card",
            )
        ],
        current_agent="human_approval",
    )


def test_round_trip_serialization():
    state = _sample_state()
    payload = state.model_dump_json()
    restored = GraphState.model_validate_json(payload)

    assert restored.session_id == state.session_id
    assert restored.goal_context.use_case == "ml_training"
    assert restored.issues[0].issue_id == "issue_abc123"
    assert restored.suggestion_cards[0].card_id == "card_abc123"
    assert restored.action_log[0].status == "applied"
    assert restored.approved_action_ids == ["card_abc123"]


def test_defaults_for_minimal_state():
    state = GraphState(
        session_id="s1",
        file_path="/tmp/x.csv",
        file_name="x.csv",
        file_size_bytes=10,
    )
    assert state.goal_context is None
    assert state.issues == []
    assert state.suggestion_cards == []
    assert state.action_log == []
    assert state.conversation_history == []
