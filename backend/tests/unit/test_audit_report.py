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
# Also status="rejected", but nobody decided it -- the operation raised.
FAILED = {
    "action_id": "a3",
    "column_name": "hire_date",
    "operation": "cast_to_datetime",
    "status": "rejected",
    "failure_reason": "Could not be applied: Column 'hire_date' is not in the data.",
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


class TestFailedFixesAreNotAttributedToTheUser:
    """A fix the app couldn't run is not a fix the user turned down. Reporting them together
    printed the app's own error message under "Reason given by the user, verbatim"."""

    def test_failure_text_never_appears_as_the_users_words(self):
        report = _report(action_log=[APPLIED, DECLINED, FAILED])
        quoted = [line for line in report.splitlines() if line.startswith("> ")]
        assert quoted == ["> I need this to join with payroll."]

    def test_failed_fixes_get_their_own_section(self):
        report = _report(action_log=[APPLIED, DECLINED, FAILED])
        assert "## Fixes that could not be applied" in report
        assert "Column 'hire_date' is not in the data." in report

    def test_failures_are_excluded_from_the_declined_count(self):
        report = _report(action_log=[APPLIED, DECLINED, FAILED])
        assert "| Changes declined | 1 |" in report
        assert "| Fixes that could not be applied | 1 |" in report

    def test_the_section_is_absent_when_nothing_failed(self):
        assert "## Fixes that could not be applied" not in _report()


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


class TestBusinessRulesReachTheDocument:
    """A rule is a fact about the data, not about a model. A session cleaning a file for a
    report still has to hand over "3 rows have a negative age" -- and the readiness section
    that normally carries rule checks is omitted entirely when no target was named."""

    RULE_CHECK = {
        "name": "rule_range_age",
        "label": "Rule: 'age' is never below 0",
        "status": "failed",
        "finding": "3 of 300 rows (1.0%) break age >= 0.",
        "evidence": "These are impossible values, not unusual ones.",
        "remedy": "",
        "blocked_by": None,
        "severity": "warning",
    }
    SKIPPED_CHECK = {
        "name": "target_named",
        "label": "A column to predict",
        "status": "not_run",
        "finding": "No prediction target was set for this session.",
        "evidence": "",
        "remedy": "",
        "blocked_by": None,
        "severity": None,
    }

    def _readiness(self, **overrides) -> dict:
        base = {
            "target_column": None,
            "task_type": None,
            "verdict": "not_applicable",
            "blockers": [],
            "warnings": ["3 of 300 rows (1.0%) break age >= 0."],
            "notes": [],
            "checks": [self.RULE_CHECK, self.SKIPPED_CHECK],
        }
        base.update(overrides)
        return base

    def test_a_broken_rule_appears_without_a_target(self):
        report = _report(readiness=self._readiness())
        assert "## Business rules" in report
        assert "break age >= 0" in report

    def test_the_section_carries_only_rules(self):
        """The model-readiness checks are meaningless here -- there is no model."""
        report = _report(readiness=self._readiness())
        section = report[report.index("## Business rules") :]
        section = section[: section.index("## Changes applied")]
        assert "A column to predict" not in section

    def test_nothing_is_printed_when_no_rules_were_set(self):
        report = _report(readiness=self._readiness(checks=[self.SKIPPED_CHECK]))
        assert "## Business rules" not in report

    def test_a_session_with_a_target_uses_the_readiness_section_instead(self):
        """Not duplicated -- with a target, a rule check is just another check in the table."""
        report = _report(
            readiness=self._readiness(target_column="churn", task_type="classification")
        )
        assert "## Business rules" not in report
        assert "break age >= 0" in report


class TestTheModelReachesTheAuditReport:
    """A reviewer reading only the audit report should learn that a model was built and how it was
    evaluated -- and learn nothing about models for a session that never trained one."""

    @staticmethod
    def _model(**overrides) -> dict:
        base = {
            "task": "classification",
            "target_column": "churned",
            "best_model": "random_forest_classifier",
            "primary_metric": "roc_auc",
            "split": {"strategy": "random", "train_rows": 800, "test_rows": 200, "test_fraction": 0.2},
            "selection": {"method": "cross_validation", "metric": "roc_auc", "folds": 5, "repeats": 1,
                          "chosen_on": "cross-validated roc_auc"},
            "assessment": {"verdict": "useful", "headline": "It separates churners from the rest."},
            "evaluation": {"look_count": 1, "status": "independent", "message": ""},
            "results": [
                {"model": "random_forest_classifier", "label": "Random forest", "metrics": {"roc_auc": 0.81},
                 "metrics_intervals": {"metrics": {"roc_auc": {"low": 0.74, "high": 0.87}}}},
            ],
        }
        base.update(overrides)
        return base

    def test_the_section_names_the_model_the_verdict_and_the_score_with_its_range(self):
        report = _report(model=self._model())
        assert "## Model built from this data" in report
        section = report[report.index("## Model built from this data") :]
        assert "predict **churned** (classification)" in section
        assert "| Model chosen | Random forest |" in section
        assert "| ROC AUC on the held-out rows | 0.81 (0.74–0.87) |" in section
        assert "| Verdict | This model works |" in section
        assert "It separates churners from the rest." in section

    def test_it_carries_the_evaluation_protocol_facts_the_card_prints(self):
        report = _report(model=self._model())
        assert "**How these numbers were obtained**" in report
        assert "800 rows to train on and 200 held back for testing (20%), at random" in report
        assert "By cross-validated ROC AUC over 5 folds of the training rows" in report
        assert "first time these rows were used to evaluate a model" in report

    def test_the_facts_are_the_cards_own_lines(self):
        from app.services import model_card

        model = self._model()
        for bullet in model_card.protocol_facts(model, model_card.AUDIT_PROTOCOL_LABELS):
            assert bullet in _report(model=model)

    def test_a_reused_holdout_is_stated(self):
        model = self._model(evaluation={"look_count": 4, "status": "reused",
                                        "message": "These held-out rows have now been used 4 times."})
        assert "These held-out rows have now been used 4 times." in _report(model=model)

    def test_the_reader_is_pointed_at_the_model_card(self):
        assert "(`model_card.md` in the export bundle)" in _report(model=self._model())

    def test_there_is_no_model_section_when_no_model_was_trained(self):
        assert "Model built from this data" not in _report()
        assert "Model built from this data" not in _report(model=None)
        assert "Model built from this data" not in _report(model={})

    def test_the_section_comes_before_the_changes(self):
        report = _report(model=self._model())
        assert report.index("## Model built from this data") < report.index("## Changes applied")

    def test_a_sparse_old_result_does_not_break_the_report(self):
        report = _report(model={"best_model": "m", "results": [{"model": "m", "label": "M", "metrics": {}}]})
        assert "| Model chosen | M |" in report
