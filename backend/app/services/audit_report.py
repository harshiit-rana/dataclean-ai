"""Audit report assembly and PDF rendering. PRD §12.4, FR-41..44.

The markdown document is built here in code, with only the prose slotted in from the LLM
(see prompts/audit_prompts.py for why). That guarantees FR-30 -- rejection reasons are
copied verbatim from the action log and never pass through the model -- and FR-44, since
session id, file name and timestamp are always present regardless of what the model returns.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_OPERATION_LABELS = {
    "fill_median": "Filled missing values with the median",
    "fill_mean": "Filled missing values with the mean",
    "fill_mode": "Filled missing values with the most common value",
    "fill_constant": "Filled missing values with a fixed value",
    "drop_rows_with_nulls": "Dropped rows with missing values",
    "drop_duplicate_rows": "Removed duplicate rows",
    "cast_to_numeric": "Converted to numbers",
    "cast_to_datetime": "Converted to dates",
    "standardize_case": "Made capitalization consistent",
    "trim_whitespace": "Removed leading/trailing spaces",
    "standardize_dates": "Standardized date formats",
    "cap_outliers": "Clamped extreme values to the normal range",
    "drop_outlier_rows": "Dropped rows with extreme values",
    "drop_column": "Removed the column",
    "none": "No executable fix was attached",
}

_STAT_LABELS = [
    ("row_count", "Rows"),
    ("null_count", "Missing values"),
    ("unique_value_count", "Distinct values"),
    ("min", "Minimum"),
    ("max", "Maximum"),
    ("mean", "Mean"),
    ("std", "Std deviation"),
]


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".") if value % 1 else f"{int(value):,}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _stats_table(before: dict | None, after: dict | None) -> list[str]:
    if not before and not after:
        return []
    before, after = before or {}, after or {}
    rows = ["", "| Statistic | Before | After |", "| --- | --- | --- |"]
    for key, label in _STAT_LABELS:
        if key not in before and key not in after:
            continue
        rows.append(f"| {label} | {_fmt(before.get(key))} | {_fmt(after.get(key))} |")
    return rows if len(rows) > 3 else []


_VERDICT_HEADLINES = {
    "ready": "**Ready.** Nothing found that would stop a model being trained on this data.",
    "warnings": "**Usable, with caveats.** Nothing blocking, but read the points below "
    "before you trust a score from this data.",
    "blocked": "**Not ready.** The points below have to be resolved before this data can "
    "support the model that was asked for.",
}


def _rules_section(readiness: dict | None) -> list[str]:
    """Business rules, for a session that never named a target.

    The readiness section below covers these when there is a model in play, since a rule
    check is just another check. But a rule is a fact about the data rather than about a
    model, and someone cleaning a file for a report still needs "142 rows have a negative age"
    to reach the document they hand over. Without this, approving a rule on a
    no-target session produced findings that appeared on screen and nowhere else.
    """
    if not readiness or readiness.get("target_column"):
        return []

    rule_checks = [c for c in readiness.get("checks") or [] if c.get("name", "").startswith("rule_")]
    if not rule_checks:
        return []

    return ["", "## Business rules", "", *_checks_table(rule_checks)]


def _readiness_section(readiness: dict | None) -> list[str]:
    """Included only when the session named a column to predict -- for a reporting or
    analysis session there is no model to be ready for, and an empty section would just be
    noise in the document. Rule checks from such a session are carried by `_rules_section`."""
    if not readiness or not readiness.get("target_column"):
        return []

    lines = [
        "",
        "## Model readiness",
        "",
        f"Assessed against **{readiness['target_column']}** as the column to predict"
        f" ({readiness.get('task_type') or 'task type undetermined'}).",
        "",
        _VERDICT_HEADLINES.get(readiness.get("verdict", ""), ""),
        "",
    ]
    for label, key in (("Blocking", "blockers"), ("Worth knowing", "warnings"), ("", "notes")):
        items = readiness.get(key) or []
        if not items:
            continue
        if label:
            lines += [f"**{label}:**", ""]
        lines += [f"- {item}" for item in items]
        lines.append("")

    lines += _checks_table(readiness.get("checks") or [])
    return lines


_STATUS_MARKS = {
    "passed": "Passed",
    "warning": "Worth knowing",
    "failed": "Failed",
    "not_run": "Not checked",
}


def _checks_table(checks: list[dict]) -> list[str]:
    """Every check and what it concluded, including the ones that did not run.

    The rest of this report records what *changed*. This records what was *concluded*, which
    is the part someone has to defend months later -- and a verdict of "ready" is a claim of
    absence, which a reader cannot confirm by looking at the data. Checks that could not run
    are listed for the same reason: "we did not look" and "we looked and it was fine" are
    different statements, and without this they appear identically as silence.
    """
    if not checks:
        return []

    lines = ["**What was checked:**", "", "| Check | Result | Finding |", "| --- | --- | --- |"]
    for check in checks:
        mark = _STATUS_MARKS.get(check.get("status", ""), check.get("status", ""))
        finding = str(check.get("finding", "")).replace("|", "\\|")
        if check.get("blocked_by"):
            finding = f"{finding} (blocked by: {check['blocked_by']})"
        lines.append(f"| {check.get('label', check.get('name', ''))} | {mark} | {finding} |")
    lines.append("")
    return lines


def _model_section(model: dict | None) -> list[str]:
    """What was built from this data and how it was evaluated, in a few lines, with a pointer to the
    model card. Only when a model exists: a report for a session that never trained one says nothing
    about models.

    Written from the stored result and the same protocol facts the card prints, so the two cannot
    describe the evaluation differently. No figure passes through a language model.
    """
    if not model or not model.get("best_model"):
        return []
    from app.services import model_card, model_store

    best = next((r for r in model.get("results", []) if r.get("model") == model.get("best_model")), {})
    assessment = model.get("assessment") or {}
    metric = model.get("primary_metric")
    value = (best.get("metrics") or {}).get(metric)
    interval = ((best.get("metrics_intervals") or {}).get("metrics") or {}).get(metric)
    label = model_card._METRIC_LABELS.get(metric or "", (metric or "score",))[0]
    split = model.get("split") or {}

    lines = [
        "",
        "## Model built from this data",
        "",
        f"A model was built to predict **{model.get('target_column')}** ({model.get('task')}) from the cleaned "
        f"data. The numbers below are for that model; the full write-up is the model card.",
        "",
        "| | |",
        "| --- | --- |",
        f"| Model chosen | {best.get('label', model.get('best_model'))} |",
        f"| Rows trained on / held back | {model_card._fmt(split.get('train_rows'))} / {model_card._fmt(split.get('test_rows'))} |",
    ]
    if value is not None:
        lines.append(
            f"| {label} on the held-out rows | {model_card._fmt_with_interval(value, interval)} |"
        )
    verdict = model_card._VERDICT_HEADINGS.get(assessment.get("verdict", ""))
    if verdict:
        lines.append(f"| Verdict | {verdict} |")
    lines += ["", assessment.get("headline", ""), ""] if assessment.get("headline") else [""]

    facts = model_card.protocol_facts(model, model_card.AUDIT_PROTOCOL_LABELS)
    if facts:
        lines += ["**How these numbers were obtained**", ""] + facts + [""]
    lines.append(
        f"The model card (`{model_store.CARD_FILENAME}` in the export bundle) has the results for each answer, "
        f"the cautions to read before relying on the model, and how to reproduce it."
    )
    return lines


def build_markdown_report(
    *,
    session_id: str,
    file_name: str,
    goal_context: dict | None,
    issues: list[dict],
    action_log: list[dict],
    executive_summary: str,
    dispositions: dict[str, str],
    readiness: dict | None = None,
    model: dict | None = None,
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    applied = [a for a in action_log if a["status"] == "applied"]
    undone = [a for a in action_log if a["status"] == "undone"]
    superseded = [a for a in action_log if a["status"] == "superseded"]
    rejected = [a for a in action_log if a["status"] == "rejected"]
    # A fix the app couldn't run is not a fix the user turned down. Reporting them together
    # inflates the declined count and, worse, prints the app's own error message under a
    # heading that says the user said it.
    failed = [a for a in rejected if a.get("failure_reason")]
    declined = [a for a in rejected if not a.get("failure_reason")]

    lines: list[str] = [
        "# Data Cleaning Audit Report",
        "",
        # FR-44: session id, file name and timestamp always appear.
        f"**Dataset:** {file_name}  ",
        f"**Session ID:** {session_id}  ",
        f"**Generated:** {generated_at}",
        "",
        "## Executive summary",
        "",
        executive_summary or "_No summary was generated for this session._",
        "",
        "## Session overview",
        "",
        "| | |",
        "| --- | --- |",
        f"| Issues detected | {len(issues)} |",
        f"| Changes applied | {len(applied)} |",
        f"| Changes declined | {len(declined)} |",
    ]
    if failed:
        lines.append(f"| Fixes that could not be applied | {len(failed)} |")
    if superseded:
        lines.append(f"| Fixes that were no longer needed | {len(superseded)} |")
    if undone:
        lines.append(f"| Changes applied then undone | {len(undone)} |")

    if goal_context:
        lines += [
            f"| Stated use case | {goal_context.get('use_case', '—')} |",
            f"| Desired outcome | {goal_context.get('expected_output', '—')} |",
            f"| Critical columns | {', '.join(goal_context.get('critical_columns') or []) or '—'} |",
            f"| Cleaning aggression | {goal_context.get('cleaning_aggression', '—')} |",
        ]
        if goal_context.get("target_column"):
            lines.append(f"| Column to be predicted | {goal_context['target_column']} |")

    lines += _readiness_section(readiness)
    lines += _rules_section(readiness)
    lines += _model_section(model)

    lines += ["", "## Changes applied", ""]
    if not applied:
        lines.append("_No changes were applied to this dataset._")
    for entry in applied:
        label = _OPERATION_LABELS.get(entry["operation"], entry["operation"])
        lines += [f"### {entry['column_name']} — {label}", ""]
        disposition = dispositions.get(entry["action_id"])
        if disposition:
            lines += [disposition, ""]
        lines += _stats_table(entry.get("before_stats"), entry.get("after_stats"))
        lines.append("")

    lines += ["## Changes declined", ""]
    if not declined:
        lines.append("_Every proposed change was applied._")
    for entry in declined:
        label = _OPERATION_LABELS.get(entry["operation"], entry["operation"])
        lines += [f"### {entry['column_name']} — {label}", ""]
        disposition = dispositions.get(entry["action_id"])
        if disposition:
            lines += [disposition, ""]
        # FR-30: verbatim, straight from the action log, never through the model.
        reason = entry.get("rejection_reason")
        if reason:
            lines += ["**Reason given by the user, verbatim:**", "", f"> {reason}", ""]
        else:
            lines += ["_No reason was recorded._", ""]

    # Attributed to the app, not the user. These were never anyone's decision.
    if failed:
        lines += [
            "## Fixes that could not be applied",
            "",
            "These were proposed but did not run. The issue they addressed is still present "
            "in the data.",
            "",
        ]
        for entry in failed:
            label = _OPERATION_LABELS.get(entry["operation"], entry["operation"])
            lines += [
                f"### {entry['column_name']} — {label}",
                "",
                str(entry["failure_reason"]),
                "",
            ]

    # Approved, and neither applied nor failed: an earlier approved fix removed what this one
    # was going to change. Reported apart from failures so the record doesn't suggest
    # something went wrong.
    if superseded:
        lines += [
            "## Fixes that were no longer needed",
            "",
            "These were approved, but an earlier change had already dealt with what they "
            "addressed.",
            "",
        ]
        for entry in superseded:
            label = _OPERATION_LABELS.get(entry["operation"], entry["operation"])
            lines += [
                f"### {entry['column_name']} — {label}",
                "",
                str(entry.get("failure_reason") or "No longer applicable."),
                "",
            ]

    # Kept distinct from declined changes: these were applied to the data and then reverted,
    # which is a materially different history than never having been applied at all.
    if undone:
        lines += ["## Changes applied then undone", ""]
        for entry in undone:
            label = _OPERATION_LABELS.get(entry["operation"], entry["operation"])
            lines += [
                f"### {entry['column_name']} — {label}",
                "",
                "This change was applied during the session and later reverted by the user. "
                "It is **not** reflected in the exported data.",
                "",
            ]
            note = entry.get("rejection_reason") or entry.get("failure_reason")
            if note:
                lines += [f"_{note}_", ""]

    lines += [
        "## Issues detected during profiling",
        "",
        "| Column | Issue | Severity | Description |",
        "| --- | --- | --- | --- |",
    ]
    for issue in issues:
        description = str(issue.get("description", "")).replace("|", "\\|")
        lines.append(
            f"| {issue.get('column_name', '—')} | {issue.get('issue_type', '—')} "
            f"| {issue.get('severity', '—')} | {description} |"
        )

    lines += [
        "",
        "---",
        "",
        "_Generated by DataClean AI. Every change in this report was individually approved "
        "by the user before it was applied._",
    ]
    return "\n".join(lines)


_PDF_CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: 'Liberation Sans', 'DejaVu Sans', sans-serif; font-size: 10.5pt;
       line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 20pt; margin-bottom: 0.2em; }
h2 { font-size: 14pt; margin-top: 1.4em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }
h3 { font-size: 11.5pt; margin-top: 1.1em; }
table { border-collapse: collapse; width: 100%; margin: 0.6em 0; }
th, td { border: 1px solid #ddd; padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #f5f5f5; }
blockquote { margin: 0.5em 0; padding: 0.5em 1em; border-left: 3px solid #bbb;
             background: #fafafa; font-style: italic; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
"""


def render_pdf(markdown_text: str, output_path: Path) -> Path:
    """Markdown -> HTML -> PDF via WeasyPrint (FR-42).

    WeasyPrint needs Pango/Cairo/GDK-Pixbuf at the OS level; those are installed in the
    backend image (see backend/Dockerfile). Raises if rendering fails -- the caller decides
    whether to degrade to markdown-only.
    """
    import markdown as markdown_lib
    from weasyprint import CSS, HTML

    html_body = markdown_lib.markdown(markdown_text, extensions=["tables"])
    html = f"<!doctype html><html><head><meta charset='utf-8'></head><body>{html_body}</body></html>"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html).write_pdf(output_path, stylesheets=[CSS(string=_PDF_CSS)])
    return output_path
