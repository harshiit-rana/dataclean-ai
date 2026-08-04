"""GET /api/readiness/{session_id} tests.

The endpoint reads whichever file is current -- cleaned.csv once cleaning has run, the
upload otherwise -- so these check that it follows the data rather than a cached verdict.
"""

from __future__ import annotations

import csv
import io

from fastapi.testclient import TestClient

from app.main import app
from app.services.session_store import session_dir

client = TestClient(app)


def _csv_bytes(rows: list[dict]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _upload(payload: bytes) -> str:
    response = client.post(
        "/api/upload", files={"file": ("data.csv", io.BytesIO(payload), "text/csv")}
    )
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


def _set_goal(session_id: str, **overrides) -> None:
    body = {
        "use_case": "ml_training",
        "expected_output": "one row per applicant",
        **overrides,
    }
    assert client.post(f"/api/goal/{session_id}", json=body).status_code == 200


def _rows(n: int = 80, *, blank_targets: int = 0) -> list[dict]:
    # `region` cycles on 3 and `approved` on 2 so the two stay independent -- a feature that
    # tracks the target exactly is a leak, and would (correctly) drag the verdict down.
    rows = [
        {
            "score": str(600 + i),
            "region": ["north", "south", "east"][i % 3],
            "approved": ["yes", "no"][i % 2],
        }
        for i in range(n)
    ]
    for row in rows[:blank_targets]:
        row["approved"] = ""
    return rows


def test_upload_response_lists_columns():
    """The goal form needs these to offer a target picker instead of a free-text box."""
    response = client.post(
        "/api/upload", files={"file": ("data.csv", io.BytesIO(_csv_bytes(_rows())), "text/csv")}
    )
    assert response.json()["columns"] == ["score", "region", "approved"]


def test_session_without_a_target_is_not_applicable():
    session_id = _upload(_csv_bytes(_rows()))
    _set_goal(session_id, use_case="reporting")

    body = client.get(f"/api/readiness/{session_id}").json()
    assert body["verdict"] == "not_applicable"
    assert body["target_column"] is None


def test_clean_dataset_with_a_target_is_ready():
    session_id = _upload(_csv_bytes(_rows()))
    _set_goal(session_id, target_column="approved")

    body = client.get(f"/api/readiness/{session_id}").json()
    assert body["verdict"] == "ready"
    assert body["target_column"] == "approved"
    assert body["task_type"] == "classification"
    assert body["blockers"] == []


def test_missing_target_values_block():
    session_id = _upload(_csv_bytes(_rows(blank_targets=12)))
    _set_goal(session_id, target_column="approved")

    body = client.get(f"/api/readiness/{session_id}").json()
    assert body["verdict"] == "blocked"
    assert any("no value for 'approved'" in blocker for blocker in body["blockers"])


def test_verdict_follows_the_cleaned_file():
    """Cleaning, undo and chat all rewrite cleaned.csv after the graph has finished. A
    verdict computed once at the end of the run would be stale exactly when it matters."""
    session_id = _upload(_csv_bytes(_rows(blank_targets=12)))
    _set_goal(session_id, target_column="approved")
    assert client.get(f"/api/readiness/{session_id}").json()["verdict"] == "blocked"

    from app.config import get_settings

    cleaned = session_dir(get_settings().upload_dir, session_id) / "cleaned.csv"
    cleaned.write_bytes(_csv_bytes(_rows()))  # the rows with no answer have been dropped

    assert client.get(f"/api/readiness/{session_id}").json()["verdict"] == "ready"


def test_unknown_session_is_404():
    assert client.get("/api/readiness/not-a-session").status_code == 404
