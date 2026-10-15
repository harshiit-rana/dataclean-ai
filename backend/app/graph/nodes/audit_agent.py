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


def _no_op(_event: dict) -> None:
    pass


# Writing the report costs an LLM call. Chat instructions and undos both change the data it
# describes, and a user adjusting their data can easily do a dozen of those -- regenerating
# each time bills for a document nobody has asked to see yet. Instead the report is marked
# stale and rewritten when it is actually requested.
STALE_MARKER = "audit_report.stale"


def mark_report_stale(directory: Path) -> None:
    (directory / STALE_MARKER).write_text("", encoding="utf-8")


def mark_report_stale_if_written(directory: Path) -> None:
    """For changes to the model: a report that was never written has nothing to go stale, and will
    describe the model as it is when it is."""
    if (directory / "audit_report.md").exists():
        mark_report_stale(directory)


def is_report_stale(directory: Path) -> bool:
    return (directory / STALE_MARKER).exists()


def clear_stale_marker(directory: Path) -> None:
    (directory / STALE_MARKER).unlink(missing_ok=True)


def _readiness_for_report(state: GraphState) -> dict | None:
    """Assessed against the cleaned file so the report describes the data being exported,
    not the data that was uploaded. Never allowed to sink the report.

    Runs without a target as well as with one. It used to return early when no column was
    being predicted, which was right when this only produced model-readiness -- but business
    rules are checked here too, and a rule is a fact about the data rather than about a model.
    A rule broken in a session that never named a target still has to reach the document the
    user hands to someone else. The report itself decides which sections to print.
    """
    has_target = bool(state.goal_context and (state.goal_context.target_column or "").strip())
    try:
        import pandas as pd

        from app.services.business_rules import InvalidRule, Rule
        from app.services.ml_readiness import assess_readiness

        path = state.cleaned_file_path or state.file_path
        rules = []
        meta_path = Path(path).parent / "meta.json"
        if meta_path.exists():
            stored = json.loads(meta_path.read_text()).get("business_rules") or []
            for payload in stored:
                try:
                    rules.append(Rule.from_dict(payload))
                except (InvalidRule, TypeError):
                    continue

        if not has_target and not rules:
            return None
        return assess_readiness(pd.read_csv(path), state.goal_context, rules=rules)
    except Exception:  # noqa: BLE001
        logger.exception("Model-readiness assessment failed; report will omit that section")
        return None


def _model_for_report(directory: Path) -> dict | None:
    """The current model's result, with what happened to it since, or None when there is none.
    Never allowed to sink the report."""
    try:
        from app.services import evaluation_ledger, model_store

        result = model_store.load_result(directory)
        if not result:
            return None
        return evaluation_ledger.attach_later_evidence(directory, result)
    except Exception:  # noqa: BLE001
        logger.exception("The model could not be read for the report; it will omit that section")
        return None


def _formula_cells(directory: Path) -> int:
    """How many cells of the cleaned file the CSV download escapes for a spreadsheet. Never sinks the report."""
    try:
        import pandas as pd

        from app.services import spreadsheet_safe

        cleaned = directory / "cleaned.csv"
        if not cleaned.exists():
            return 0
        return spreadsheet_safe.count_risky(pd.read_csv(cleaned, dtype=str, keep_default_na=False))
    except Exception:  # noqa: BLE001
        logger.warning("Formula-leading cells could not be counted for the report", exc_info=True)
        return 0


async def generate_audit_artifacts(state: GraphState, emit=_no_op) -> dict:
    """Writes audit_report.md (and audit_report.pdf when WeasyPrint can run) for this
    session, returning the paths.

    Split out from the node so it can also be called outside a graph run -- undoing an
    action changes the data and the action log, which makes any previously written report
    wrong, so the undo endpoint regenerates it through here. `emit` takes the node's stream
    writer during a graph run and defaults to a no-op elsewhere (get_stream_writer() is only
    valid inside one).
    """
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
        readiness=_readiness_for_report(state),
        model=_model_for_report(Path(state.file_path).parent),
        formula_cells=_formula_cells(Path(state.file_path).parent),
    )

    session_dir = Path(state.file_path).parent
    markdown_path = session_dir / "audit_report.md"
    markdown_path.write_text(markdown_text, encoding="utf-8")
    clear_stale_marker(session_dir)

    pdf_path: Path | None = None
    try:
        pdf_path = render_pdf(markdown_text, session_dir / "audit_report.pdf")
    except Exception as exc:  # noqa: BLE001 -- WeasyPrint needs OS libs that may be absent
        logger.exception("PDF rendering failed; markdown report was still written")
        emit(
            {
                "event": "error",
                "error": "PDF could not be generated",
                "detail": f"The report was written as markdown instead. ({exc})",
            }
        )

    emit({"event": "audit_complete", "has_pdf": pdf_path is not None})

    return {
        "audit_markdown_path": str(markdown_path),
        "audit_pdf_path": str(pdf_path) if pdf_path else None,
    }


async def audit_agent(state: GraphState) -> dict:
    writer = get_stream_writer()
    writer({"event": "agent_start", "agent": "audit_agent", "message": "Writing your audit report..."})

    artifacts = await generate_audit_artifacts(state, emit=writer)
    return {**artifacts, "current_agent": "audit_agent"}
