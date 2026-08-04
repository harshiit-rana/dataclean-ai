"""profiling_agent node. PRD §12.1 -- REAL implementation.

Runs the deterministic pandas-based detectors (app/services/profiling.py) to get a
baseline issues list, then calls Claude once to add semantic interpretation and
goal-aware priority adjustment (the agent's three LLM tasks), retrying once with an
explicit JSON-formatting instruction on parse failure (FR-27, §14.2).
"""

from __future__ import annotations

import json
import logging

import pandas as pd
from langgraph.config import get_stream_writer

from app.config import get_settings
from app.models.graph_state import GraphState
from app.services import ml_readiness, profiling
from app.services.llm_client import get_llm_client
from app.services.prompts.profiling_prompts import (
    build_profiling_agent_messages,
    build_retry_messages,
)

logger = logging.getLogger(__name__)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


async def _enrich_with_llm(issues, goal_context, client) -> dict:
    """Returns {issue_id: update_dict}, or {} if both attempts fail (issues keep their
    deterministic baseline priority_score in that case -- a degraded but non-crashing
    outcome, per NFR-08)."""
    messages = build_profiling_agent_messages(issues, goal_context)
    last_error = ""
    for attempt in range(2):  # initial try + one retry
        try:
            response = await client.ainvoke(messages)
            raw = response.content if isinstance(response.content, str) else str(response.content)
            parsed = json.loads(_strip_code_fences(raw))
            return {item["issue_id"]: item for item in parsed}
        except Exception as exc:  # noqa: BLE001 -- broad by design, this must never crash the session
            last_error = str(exc)
            logger.warning("Profiling Agent LLM call/parse failed (attempt %d): %s", attempt + 1, exc)
            messages = build_retry_messages(issues, goal_context, last_error)
    return {}


async def profiling_agent(state: GraphState) -> dict:
    writer = get_stream_writer()
    writer({"event": "agent_start", "agent": "profiling_agent", "message": "Profiling your data..."})

    df = pd.read_csv(state.file_path)
    issues = profiling.detect_issues(df)
    # Adds the findings that only exist once a prediction target is known (leakage, unusable
    # target). A no-op when the user didn't name one.
    issues = ml_readiness.augment_issues(df, issues, state.goal_context)
    raw_profile = profiling.run_ydata_profile(df)

    if issues:
        settings = get_settings()
        client = get_llm_client(settings, streaming=False)
        updates = await _enrich_with_llm(issues, state.goal_context, client)
        for issue in issues:
            update = updates.get(issue.issue_id)
            if not update:
                continue
            issue.priority_score = float(update.get("priority_score", issue.priority_score))
            issue.stats = {
                **issue.stats,
                "likely_cause": update.get("likely_cause", ""),
                "pipeline_flag": bool(update.get("pipeline_flag", False)),
                "pipeline_note": update.get("pipeline_note", ""),
            }
        issues.sort(key=lambda i: i.priority_score, reverse=True)

    high = sum(1 for i in issues if i.severity == "high")
    medium = sum(1 for i in issues if i.severity == "medium")
    low = sum(1 for i in issues if i.severity == "low")
    writer(
        {
            "event": "profiling_complete",
            "issue_count": len(issues),
            "high": high,
            "medium": medium,
            "low": low,
            "issues": [issue.model_dump() for issue in issues],
        }
    )

    # Stored as plain dicts -- see GraphState's docstring on checkpoint round-tripping.
    return {
        "issues": [issue.model_dump() for issue in issues],
        "raw_profile": raw_profile,
        "current_agent": "profiling_agent",
    }
