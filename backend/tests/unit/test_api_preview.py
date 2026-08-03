"""GET /api/preview/{session_id} tests -- the Review page's data table depends on this."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _upload(fixtures_dir, name: str) -> str:
    with open(fixtures_dir / name, "rb") as f:
        response = client.post("/api/upload", files={"file": (name, f, "text/csv")})
    return response.json()["session_id"]


def test_preview_returns_rows_and_columns(fixtures_dir):
    session_id = _upload(fixtures_dir, "missing_values.csv")
    body = client.get(f"/api/preview/{session_id}").json()

    assert body["total_rows"] == 10
    assert body["total_columns"] == 5
    assert body["preview_row_count"] == 10
    assert [c["name"] for c in body["columns"]] == ["id", "name", "age", "city", "email"]
    assert body["rows"][0]["name"] == "Alice"


def test_preview_serializes_nulls_as_none(fixtures_dir):
    """NaN is not valid JSON -- missing cells must come back as null, not NaN."""
    session_id = _upload(fixtures_dir, "missing_values.csv")
    body = client.get(f"/api/preview/{session_id}").json()

    name_column = next(c for c in body["columns"] if c["name"] == "name")
    assert name_column["null_count"] == 4
    assert any(row["name"] is None for row in body["rows"])


def test_preview_flags_critical_columns(fixtures_dir):
    session_id = _upload(fixtures_dir, "missing_values.csv")
    client.post(
        f"/api/goal/{session_id}",
        json={
            "use_case": "analysis",
            "expected_output": "clean data",
            "critical_columns": ["email"],
            "cleaning_aggression": "moderate",
        },
    )
    body = client.get(f"/api/preview/{session_id}").json()

    by_name = {c["name"]: c for c in body["columns"]}
    assert by_name["email"]["is_critical"] is True
    assert by_name["age"]["is_critical"] is False


def test_preview_unknown_session_returns_404():
    response = client.get("/api/preview/does-not-exist")
    assert response.status_code == 404
