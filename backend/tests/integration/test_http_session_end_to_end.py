"""A whole session over HTTP: upload -> goal -> profile -> resume -> readiness -> plan -> train ->
improve -> export -> apply, on the churn sample, with a fake LLM.

The graph integration test drives LangGraph directly through the approval interrupt, and the route
tests use fake graphs or skip the graph. Nothing exercised the whole journey through the real
routes, so a wiring defect between two steps -- a response shape the next step depends on, a file
one route writes and another reads -- was only ever found by running real data by hand, which is
how this project has found most of its bugs. This is that run, kept.

The fake card writer answers like a naive model would for each kind of issue, including the
whole-file fills and outlier caps the app no longer lets through for a model input, so the journey
also proves that over HTTP. The model is trained for real; only the LLM is fake.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from app.config import get_settings
from app.graph.workflow import build_graph
from app.main import app
from app.services import evaluation_ledger as ledger
from app.services import model_store
from app.services.session_store import session_dir

SAMPLE = Path(__file__).resolve().parents[3] / "samples" / "subscription_churn.csv"
BUILD_ROWS = 900  # the model is built on these; the rest of the file arrives later
FAST = {
    "model_names": ["logistic_regression", "random_forest_classifier"],
    "include_deep_diagnostics": False,
}

# What a model that has not been told better proposes for each kind of issue. The fills and caps
# are the point: with a prediction target named they must come out as cards with no operation.
_WHOLE_FILE_BY_ISSUE = {"missing_values": "fill_median", "outliers": "cap_outliers"}


class FakeProfilingClient:
    async def ainvoke(self, messages):
        return SimpleNamespace(content="[]")  # every issue keeps its deterministic priority


class FakeCardWriter:
    """Writes each card in the marker format the node parses, with an OPERATION line chosen by
    issue type from the prompt itself, as the real prompt contract describes."""

    async def astream(self, messages):
        fields = {}
        for line in str(messages[-1].content).splitlines():
            for key in ("column", "type", "description"):
                if line.startswith(f"- {key}:"):
                    fields[key] = line.split(":", 1)[1].strip()
        column, kind, description = fields.get("column", ""), fields.get("type", ""), fields.get("description", "")

        if kind == "duplicates":
            operation = "drop_duplicate_rows()"
        elif kind in _WHOLE_FILE_BY_ISSUE:
            operation = f'{_WHOLE_FILE_BY_ISSUE[kind]}(column="{column}")'
        elif kind == "type_mismatch":
            operation = f'cast_to_datetime(column="{column}")'
        elif kind == "inconsistent_format" and "date" in column:
            operation = f'standardize_dates(column="{column}", format="%Y-%m-%d")'
        elif kind == "inconsistent_format" and "space" in description.lower():
            operation = f'trim_whitespace(column="{column}")'
        elif kind == "inconsistent_format":
            operation = f'standardize_case(column="{column}", case="lower")'
        elif kind == "high_cardinality":
            operation = f'drop_column(column="{column}")'
        else:
            operation = ""

        text = (
            "FOUND:\nSomething in this column needs attention.\n"
            "PROPOSED_ACTION:\nFix it.\n"
            "REASONING:\nIt matters for the stated goal.\n"
            "TRADEOFF:\nSome information is changed.\n"
            f"OPERATION:\n{operation}\n"
        )
        yield SimpleNamespace(content=text)


class FakeProseClient:
    """One reply that satisfies both the audit report and the model card writer."""

    async def ainvoke(self, messages):
        return SimpleNamespace(
            content=json.dumps(
                {
                    "executive_summary": "The data was cleaned as approved.",
                    "dispositions": {},
                    "summary": "A model was built from the cleaned data.",
                    "interpretation": "It finds a real signal.",
                    "cautions": [],
                }
            )
        )


@pytest.fixture
def journey(monkeypatch):
    for target, fake in (
        ("app.graph.nodes.profiling_agent.get_llm_client", FakeProfilingClient()),
        ("app.graph.nodes.explanation_agent.get_llm_client", FakeCardWriter()),
        ("app.graph.nodes.audit_agent.get_llm_client", FakeProseClient()),
        ("app.services.llm_client.get_llm_client", FakeProseClient()),
    ):
        monkeypatch.setattr(target, lambda *_a, _fake=fake, **_k: _fake)

    had = hasattr(app.state, "graph")
    previous = getattr(app.state, "graph", None)
    app.state.graph = build_graph(MemorySaver())
    yield TestClient(app)
    if had:
        app.state.graph = previous
    else:
        del app.state.graph


def events(response) -> list[dict]:
    """The `data:` payloads of an SSE response, in order."""
    out = []
    for frame in response.text.split("\n\n"):
        for line in frame.splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[len("data: "):]))
    return out


def csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def test_a_whole_session_over_http(journey):
    client = journey
    sample = pd.read_csv(SAMPLE)
    build, later = sample.iloc[:BUILD_ROWS], sample.iloc[BUILD_ROWS:]
    upload = csv_bytes(build)

    # --- upload ----------------------------------------------------------------------------
    uploaded = client.post("/api/upload", files={"file": ("churn.csv", io.BytesIO(upload), "text/csv")})
    assert uploaded.status_code == 200, uploaded.text
    body = uploaded.json()
    session_id = body["session_id"]
    assert body["row_count"] == BUILD_ROWS and "churned" in body["columns"]
    directory = session_dir(get_settings().upload_dir, session_id)
    assert (directory / "original.csv").read_bytes() == upload

    # --- goal ------------------------------------------------------------------------------
    goal = client.post(
        f"/api/goal/{session_id}",
        json={
            "use_case": "ml_training", "expected_output": "predict who churns",
            "critical_columns": [], "cleaning_aggression": "moderate",
            "target_column": "churned", "task_type": "auto",
        },
    )
    assert goal.status_code == 200, goal.text

    # --- profile: cards stream, the graph pauses for approval ------------------------------
    profiled = client.post(f"/api/profile/{session_id}")
    assert profiled.status_code == 200, profiled.text[:300]
    stream = events(profiled)
    names = [e["event"] for e in stream]
    assert "profiling_complete" in names and names[-1] == "interrupt_reached"
    cards = [e["card"] for e in stream if e["event"] == "card_complete"]
    issues = next(e for e in stream if e["event"] == "profiling_complete")["issues"]
    assert cards and len(cards) == len(issues)
    assert {"found", "proposed_action", "reasoning", "tradeoff"} <= set(cards[0])

    # With a target named, no card may carry a fix that reads a statistic from every row.
    whole_file = {"fill_median", "fill_mean", "fill_mode", "cap_outliers", "drop_outlier_rows"}
    proposed = [c for c in cards if c["column_name"] != "churned"]
    assert not [c for c in proposed if c["operation"] in whole_file]
    left_to_model = [c for c in cards if c.get("warning_kind") == "left_to_model"]
    assert left_to_model and all(c["operation"] is None for c in left_to_model)

    # --- resume: only what was approved runs ------------------------------------------------
    approved = [c["card_id"] for c in cards if c["operation"]]
    skipped = [c["card_id"] for c in cards if not c["operation"]]
    resumed = client.post(
        f"/api/resume/{session_id}",
        json={"approved_action_ids": approved, "skipped_action_ids": skipped,
              "skip_reasons": {c: "Left for the model." for c in skipped}},
    )
    assert resumed.status_code == 200, resumed.text[:300]
    finished = [e for e in events(resumed) if e["event"] == "all_actions_complete"]
    assert finished and finished[0]["actions_applied"] >= 1

    assert (directory / "original.csv").read_bytes() == upload  # the upload is never modified
    cleaned = pd.read_csv(directory / "cleaned.csv")
    assert 0 < len(cleaned) <= BUILD_ROWS and "customer_id" not in cleaned.columns

    actions = client.get(f"/api/actions/{session_id}").json()["actions"]
    assert any(a["status"] == "applied" for a in actions)
    assert all(a["operation"] not in whole_file for a in actions if a["status"] == "applied")

    status = client.get(f"/api/session/{session_id}")
    assert status.status_code == 200
    assert [v["event"] for v in status.json()["dataset_versions"]] == ["upload", "cleaning_run"]

    # --- readiness -------------------------------------------------------------------------
    readiness = client.get(f"/api/readiness/{session_id}")
    assert readiness.status_code == 200
    assert readiness.json()["verdict"] in {"ready", "warnings"}
    assert readiness.json()["checks"]

    # --- plan ------------------------------------------------------------------------------
    plan = client.get(f"/api/model/{session_id}/plan")
    assert plan.status_code == 200, plan.text
    plan = plan.json()
    assert plan["target_column"] == "churned" and plan["task_type"] == "classification"
    assert not plan["blockers"] and not plan["trained"]
    assert plan["exclusions_basis"].startswith("Leakage checked on the")

    # --- train: chosen on folds, test set opened once, first look ---------------------------
    trained = client.post(f"/api/model/{session_id}/train", json=FAST)
    assert trained.status_code == 200, trained.text[:300]
    result = trained.json()
    assert result["selection"]["method"] == "cross_validation"
    assert "test set was not consulted" in result["selection"]["chosen_on"]
    assert result["evaluation"]["status"] == "independent" and result["evaluation"]["look_count"] == 1
    assert result["assessment"]["verdict"] in {"useful", "marginal", "no_better_than_guessing", "not_enough_test_data"}
    assert all(r["cv_score"] is not None for r in result["results"] if not r["failed"])
    assert ledger.look_count(directory, None, result["split"]) == 1
    assert model_store.pipeline_path(directory) is not None

    # --- improve: a second look, still no test-set selection --------------------------------
    improved = client.post(f"/api/model/{session_id}/improve", json={**FAST, "budget": 4})
    assert improved.status_code == 200, improved.text[:300]
    again = improved.json()
    assert again["improvement"]["rounds"]
    assert again["evaluation"]["look_count"] == 2 and again["evaluation"]["status"] == "reused"
    assert "no longer an independent estimate" in again["assessment"]["headline"]
    assert client.get(f"/api/model/{session_id}/result").json()["run_id"] == again["run_id"]
    runs = client.get(f"/api/model/{session_id}/history").json()["runs"]
    assert [r["evaluation"]["look_count"] for r in runs] == [2, 1]

    # --- export ----------------------------------------------------------------------------
    card = client.get(f"/api/model/{session_id}/card")
    assert card.status_code == 200 and "Held-out evaluation" in card.json()["markdown"]
    script = client.get(f"/api/model/{session_id}/script")
    assert script.status_code == 200 and "TARGET = 'churned'" in script.text
    assert client.get(f"/api/model/{session_id}/download").status_code == 200
    assert client.get(f"/api/export/{session_id}/clean-script").status_code == 200

    bundle = client.get(f"/api/export/{session_id}/zip")
    assert bundle.status_code == 200, bundle.text[:200]
    members = " ".join(zipfile.ZipFile(io.BytesIO(bundle.content)).namelist())
    assert "cleaned.csv" in members and "train_model.py" in members and "clean.py" in members

    # --- apply: the rest of the file, with real answers -------------------------------------
    applied = client.post(
        f"/api/apply/{session_id}",
        files={"file": ("later.csv", io.BytesIO(csv_bytes(later)), "text/csv")},
    )
    assert applied.status_code == 200, applied.text[:300]
    out = applied.json()
    assert out["rows_in"] == len(later)
    assert out["rows_kept"] + out["dropped_count"] == out["rows_in"]  # no row vanishes
    assert out["predictions"]["count"] == out["rows_kept"]
    assert out["drift"]["verdict"] in {"stable", "shifted", "different", "not_enough_rows"} or out["drift"]
    assert out["answers"]["now"] is not None and out["answers"]["counted_as_independent"] is True

    predictions = client.get(out["downloads"]["predictions"])
    assert predictions.status_code == 200
    assert len(pd.read_csv(io.StringIO(predictions.text))) == len(later)  # every row, in order

    # Scoring new rows is a different event from a look at the held-out rows.
    final = client.get(f"/api/model/{session_id}/result").json()
    assert final["evaluation"]["new_data"]["count"] == 1
    assert ledger.look_count(directory, None, final["split"]) == 2
