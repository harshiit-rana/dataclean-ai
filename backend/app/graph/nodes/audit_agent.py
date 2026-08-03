"""audit_agent node. PRD §12.4, FR-41..44.

Turns the action log into a markdown report and renders it to PDF. One LLM call produces
only the prose (executive summary + per-action dispositions); the document structure,
statistics, session metadata and the user's verbatim rejection reasons are assembled in
code -- see services/audit_report.py and prompts/audit_prompts.py.

Degrades rather than fails: if the LLM call or JSON parse fails the report is still produced
with its deterministic content, and if PDF rendering fails the markdown is still written to
disk. A cleaning session that already changed the user's data must not end with nothing to
show for it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langgraph.config import get_stream_writer

from app.config import get_settings
from app.models.graph_state import GraphState
from app.services.audit_report import build_markdown_report, render_pdf
from app.services.llm_client import get_llm_client
from app.services.prompts.audit_prompts import build_audit_report_messages, build_retry_messages

logger = logging.getLogger(__name__)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


async def _write_prose(state: GraphState, client) -> tuple[str, dict[str, str]]:
    """Returns (executive_summary, dispositions). Empty values on failure -- the report is
    still worth producing without them (NFR-08)."""
    args = (state.action_log, state.goal_context, state.file_name, len(state.issues))
    messages = build_audit_report_messages(*args)

    for attempt in range(2):  # initial try + one retry (FR-27, §14.2)
        try:
            response = await client.ainvoke(messages)
            raw = response.content if isinstance(response.content, str) else str(response.content)
            parsed = json.loads(_strip_code_fences(raw))
            return (
                str(parsed.get("executive_summary", "")),
                {str(k): str(v) for k, v in (parsed.get("dispositions") or {}).items()},
            )
        except Exception as exc:  # noqa: BLE001 -- must never sink the report
            logger.warning("Audit Agent LLM call/parse failed (attempt %d): %s", attempt + 1, exc)
            messages = build_retry_messages(*args, str(exc))

    return "", {}


async def audit_agent(state: GraphState) -> dict:
    writer = get_stream_writer()
    writer({"event": "agent_start", "agent": "audit_agent", "message": "Writing your audit report..."})

    settings = get_settings()
    client = get_llm_client(settings, streaming=False)
    executive_summary, dispositions = await _write_prose(state, client)

    markdown_text = build_markdown_report(
        session_id=state.session_id,
        file_name=state.file_name,
        goal_context=state.goal_context.model_dump() if state.goal_context else None,
        issues=state.issues,
        action_log=state.action_log,
        executive_summary=executive_summary,
        dispositions=dispositions,
    )

    session_dir = Path(state.file_path).parent
    markdown_path = session_dir / "audit_report.md"
    markdown_path.write_text(markdown_text, encoding="utf-8")

    pdf_path: Path | None = None
    try:
        pdf_path = render_pdf(markdown_text, session_dir / "audit_report.pdf")
    except Exception as exc:  # noqa: BLE001 -- WeasyPrint needs OS libs that may be absent
        logger.exception("PDF rendering failed; markdown report was still written")
        writer(
            {
                "event": "error",
                "error": "PDF could not be generated",
                "detail": f"The report was written as markdown instead. ({exc})",
            }
        )

    writer({"event": "audit_complete", "has_pdf": pdf_path is not None})

    return {
        "audit_markdown_path": str(markdown_path),
        "audit_pdf_path": str(pdf_path) if pdf_path else None,
        "current_agent": "audit_agent",
    }
