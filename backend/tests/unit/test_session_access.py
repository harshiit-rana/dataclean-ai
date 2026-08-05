"""Per-session access tokens.

Two things are being checked here and they matter for different reasons. The pure functions
in services/session_access.py are checked because a hash that isn't a hash is a silent
failure -- nothing breaks, the token just stops protecting anything. The dependency is
checked through the real app because the guard is attached at the router in main.py, so the
only honest test of "is this route protected" is to call the route.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services.session_access import (
    HEADER_NAME,
    META_KEY,
    hash_token,
    issue_token,
    token_matches,
)
from app.services.session_store import session_dir

client = TestClient(app)

CSV = b"name,score\nada,9\ngrace,7\n"


# --- the token itself ---------------------------------------------------------------


def test_each_token_is_unique():
    tokens = {issue_token()[0] for _ in range(50)}
    assert len(tokens) == 50


def test_only_the_hash_is_meant_to_be_stored():
    token, stored = issue_token()
    assert stored != token
    assert stored == hash_token(token)
    # A hash that still contains the token would defeat the point.
    assert token not in stored


def test_a_matching_token_is_accepted():
    token, stored = issue_token()
    assert token_matches(token, stored) is True


@pytest.mark.parametrize(
    "token, stored",
    [
        ("wrong-token", hash_token("right-token")),
        (None, hash_token("right-token")),
        ("", hash_token("right-token")),
        ("right-token", None),
        ("right-token", ""),
        (None, None),
    ],
)
def test_anything_short_of_a_match_is_rejected(token, stored):
    assert token_matches(token, stored) is False


# --- the guard, through the real app ------------------------------------------------


def _upload() -> tuple[str, str]:
    response = client.post(
        "/api/upload", files={"file": ("data.csv", io.BytesIO(CSV), "text/csv")}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    return body["session_id"], body["session_token"]


def _require_tokens(monkeypatch, required: bool = True) -> None:
    monkeypatch.setenv("REQUIRE_SESSION_TOKENS", str(required).lower())
    get_settings.cache_clear()


def test_upload_hands_back_a_token_and_stores_only_its_hash(tmp_path):
    session_id, token = _upload()
    assert token

    meta = json.loads((session_dir(str(tmp_path), session_id) / "meta.json").read_text())
    assert meta[META_KEY] == hash_token(token)
    assert token not in json.dumps(meta)


def test_development_does_not_require_a_token():
    """The default. `docker compose up` and a browser has to keep working."""
    session_id, _ = _upload()
    assert client.get(f"/api/preview/{session_id}").status_code == 200


def test_the_right_token_gets_in(monkeypatch):
    session_id, token = _upload()
    _require_tokens(monkeypatch)

    response = client.get(
        f"/api/preview/{session_id}", headers={HEADER_NAME: token}
    )
    assert response.status_code == 200


def test_the_session_id_alone_is_not_enough(monkeypatch):
    """The whole point: a leaked id must not be a leaked dataset."""
    session_id, _ = _upload()
    _require_tokens(monkeypatch)

    response = client.get(f"/api/preview/{session_id}")
    assert response.status_code == 403


def test_someone_elses_token_is_not_enough(monkeypatch):
    session_id, _ = _upload()
    _, other_token = _upload()
    _require_tokens(monkeypatch)

    response = client.get(
        f"/api/preview/{session_id}", headers={HEADER_NAME: other_token}
    )
    assert response.status_code == 403


def test_the_denial_does_not_say_which_way_it_failed(monkeypatch):
    """Distinguishing "no token" from "wrong token" would help someone probing."""
    session_id, _ = _upload()
    _require_tokens(monkeypatch)

    missing = client.get(f"/api/preview/{session_id}")
    wrong = client.get(f"/api/preview/{session_id}", headers={HEADER_NAME: "nope"})
    assert missing.status_code == wrong.status_code == 403
    assert missing.json() == wrong.json()


def test_an_unknown_session_is_still_a_404(monkeypatch):
    _require_tokens(monkeypatch)
    response = client.get("/api/preview/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_a_session_from_before_tokens_were_switched_on_still_works(monkeypatch, tmp_path):
    """Turning the setting on must not strand work already in progress."""
    session_id, _ = _upload()
    meta_path = session_dir(str(tmp_path), session_id) / "meta.json"
    meta = json.loads(meta_path.read_text())
    del meta[META_KEY]
    meta_path.write_text(json.dumps(meta))

    _require_tokens(monkeypatch)
    assert client.get(f"/api/preview/{session_id}").status_code == 200


def test_the_guard_is_on_every_router_not_just_one(monkeypatch):
    """Attached at the router in main.py, so this is really asking whether that stuck.

    One route per session-scoped router: a new router included without the dependency is
    exactly the mistake worth catching, and it shows up here as an unguarded path."""
    session_id, _ = _upload()
    _require_tokens(monkeypatch)

    for path in (
        f"/api/preview/{session_id}",
        f"/api/session/{session_id}",
        f"/api/readiness/{session_id}",
        f"/api/model/{session_id}/plan",
        f"/api/export/{session_id}/csv",
        f"/api/actions/{session_id}",
    ):
        assert client.get(path).status_code == 403, f"{path} is unguarded"


def test_the_token_gets_past_the_guard_on_every_route_it_is_for(monkeypatch):
    """The routes that don't need the compiled graph, so the only thing under test is the
    guard -- whatever each route answers afterwards is its own business."""
    session_id, token = _upload()
    _require_tokens(monkeypatch)

    for path in (
        f"/api/preview/{session_id}",
        f"/api/readiness/{session_id}",
        f"/api/model/{session_id}/plan",
        f"/api/model/{session_id}/result",
    ):
        assert (
            client.get(path, headers={HEADER_NAME: token}).status_code != 403
        ), f"{path} rejects its own token"


def test_upload_itself_needs_no_token(monkeypatch):
    """There's no session to authorise against yet -- this is where the token comes from."""
    _require_tokens(monkeypatch)
    response = client.post(
        "/api/upload", files={"file": ("data.csv", io.BytesIO(CSV), "text/csv")}
    )
    assert response.status_code == 200
    assert response.json()["session_token"]


def test_health_is_not_behind_the_guard(monkeypatch):
    _require_tokens(monkeypatch)
    assert client.get("/health").status_code == 200
