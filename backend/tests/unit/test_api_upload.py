"""POST /api/upload tests. PRD FR-01..09.

Uses TestClient without entering it as a context manager, so the app's lifespan (which
opens a real Redis connection for the checkpointer, see main.py) never runs -- /api/upload
doesn't touch the graph or Redis, so this keeps the test isolated and fast.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_upload_valid_csv_returns_session(fixtures_dir):
    with open(fixtures_dir / "missing_values.csv", "rb") as f:
        response = client.post(
            "/api/upload", files={"file": ("missing_values.csv", f, "text/csv")}
        )

    assert response.status_code == 200
    body = response.json()
    assert "session_id" in body
    assert body["file_name"] == "missing_values.csv"
    assert body["row_count"] == 10
    assert body["column_count"] == 5


def test_upload_rejects_non_csv():
    response = client.post(
        "/api/upload", files={"file": ("notes.txt", b"hello world", "text/plain")}
    )
    assert response.status_code == 400
    assert response.json()["error"] == "Unsupported file type"


def test_upload_rejects_empty_file():
    response = client.post("/api/upload", files={"file": ("empty.csv", b"", "text/csv")})
    assert response.status_code == 400
    assert response.json()["error"] == "Empty file"


def test_upload_rejects_header_only_file():
    response = client.post(
        "/api/upload", files={"file": ("header_only.csv", b"a,b,c\n", "text/csv")}
    )
    assert response.status_code == 400
    assert response.json()["error"] == "No data rows"
