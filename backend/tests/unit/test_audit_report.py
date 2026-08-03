"""Audit report assembly. PRD §12.4, FR-30, FR-44."""

from __future__ import annotations

import json

import pytest

from app.services.audit_report import build_markdown_report
from app.services.prompts.audit_prompts import build_audit_report_messages

APPLIED = {
    "action_id": "a1",
    "column_name": "salary",
    "operation": "fill_median",
    "status": "applied",
    "before_stats": {"row_count": 20, "null_count": 2, "unique_value_count": 17},
    "after_stats": {"row_count": 20, "null_count": 0, "unique_value_count": 18},
}
DECLINED = {
    "action_id": "a2",
    "column_name": "employee_id",
    "operation": "drop_column",
    "status": "rejected",
    "rejection_reason": "I need this to join with payroll.",
}


def _report(**overrides) -> str:
    kwargs = dict(
        session_id="abc-123",
        file_name="employee_records.csv",
        goal_context={
            "use_case": "analysis",
            "expected_output": "clean data",
            "critical_columns": ["salary"],
            "cleaning_aggression": "moderate",
        },
        issues=[
            {
                "column_name": "salary",
                "issue_type": "missing_values",
                "severity": "medium",
                "description": "Column 'salary' has 2 missing values.",
            }
        ],
        action_log=[APPLIED, DECLINED],
        executive_summary="Cleaned a 20-row employee dataset for analysis.",
        dispositions={"a1": "Filled the two missing salaries.", "a2": "Declined."},
    )
    kwargs.update(overrides)
    return build_markdown_report(**kwargs)


def test_includes_session_metadata():
    """FR-44: session id, file name and timestamp must always be present."""
    report = _report()
    assert "abc-123" in report
    assert "employee_records.csv" in report
    assert "**Generated:**" in report
    assert "UTC" in report


def test_rejection_reason_appears_verbatim():
    """FR-30: the user's words, not a paraphrase."""
    report = _report()
    assert "> I need this to join with payroll." in report
    assert "verbatim" in report.lower()


def test_rejection_reason_is_never_sent_to_the_model():
    """The strongest guarantee for FR-30: the model cannot paraphrase what it never sees."""
    messages = build_audit_report_messages([APPLIED, DECLINED], None, "employee_records.csv", 1)
    prompt_text = "\n".join(str(m.content) for m in messages)
    assert "join with payroll" not in prompt_text
    assert "rejection_reason" not in prompt_text
    # but the action ids it must write dispositions for are present
    assert "a1" in prompt_text and "a2" in prompt_text


def test_applied_and_declined_are_separated():
    report = _report()
    applied_at = report.index("## Changes applied")
    declined_at = report.index("## Changes declined")
    assert applied_at < declined_at
    assert report.index("salary — Filled missing values") < declined_at
    assert report.index("employee_id — Removed the column") > declined_at


def test_before_after_stats_render_as_a_table():
    report = _report()
    assert "| Statistic | Before | After |" in report
    assert "| Missing values | 2 | 0 |" in report


def test_survives_llm_returning_nothing():
    """A failed LLM call must still produce a usable report (NFR-08)."""
    report = _report(executive_summary="", dispositions={})
    assert "No summary was generated" in report
    assert "> I need this to join with payroll." in report  # verbatim reason survives
    assert "| Missing values | 2 | 0 |" in report


def test_handles_a_session_with_no_changes():
    report = _report(action_log=[], dispositions={})
    assert "_No changes were applied to this dataset._" in report
    assert "_Every proposed change was applied._" in report


def test_pipe_in_description_does_not_break_the_table():
    report = _report(
        issues=[
            {
                "column_name": "notes",
                "issue_type": "inconsistent_format",
                "severity": "low",
                "description": "Values use a | separator.",
            }
        ]
    )
    assert r"Values use a \| separator." in report


@pytest.mark.parametrize("entry", [APPLIED, DECLINED])
def test_prompt_payload_is_json_serializable(entry):
    messages = build_audit_report_messages([entry], None, "f.csv", 0)
    # the human message embeds a JSON block; make sure it parses
    body = str(messages[-1].content)
    start = body.index("[")
    end = body.rindex("]") + 1
    assert json.loads(body[start:end])


UNDONE = {
    "action_id": "a3",
    "column_name": "salary",
    "operation": "drop_outlier_rows",
    "status": "undone",
}


def test_undone_changes_are_reported_separately_from_declined():
    """An undone change was applied and then reverted -- materially different history from
    one that was never applied, so the report must not conflate them (US-13)."""
    report = _report(action_log=[APPLIED, DECLINED, UNDONE], dispositions={})

    assert "## Changes applied then undone" in report
    assert "| Changes applied then undone | 1 |" in report
    assert "not** reflected in the exported data" in report

    # the undone entry must live in its own section, not inside the declined one
    declined_section = report[
        report.index("## Changes declined") : report.index("## Changes applied then undone")
    ]
    assert "Dropped rows with extreme values" not in declined_section


def test_no_undone_section_when_nothing_was_undone():
    report = _report()
    assert "## Changes applied then undone" not in report
    assert "| Changes applied then undone |" not in report
