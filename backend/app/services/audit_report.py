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


def build_markdown_report(
    *,
    session_id: str,
    file_name: str,
    goal_context: dict | None,
    issues: list[dict],
    action_log: list[dict],
    executive_summary: str,
    dispositions: dict[str, str],
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    applied = [a for a in action_log if a["status"] == "applied"]
    declined = [a for a in action_log if a["status"] != "applied"]

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

    if goal_context:
        lines += [
            f"| Stated use case | {goal_context.get('use_case', '—')} |",
            f"| Desired outcome | {goal_context.get('expected_output', '—')} |",
            f"| Critical columns | {', '.join(goal_context.get('critical_columns') or []) or '—'} |",
            f"| Cleaning aggression | {goal_context.get('cleaning_aggression', '—')} |",
        ]

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
