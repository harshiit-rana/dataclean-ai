"""Projects kept until deleted (#120, docs/decisions/0004-platform-architecture.md).

A project is a session its owner chose to keep. What has to hold: it is never swept, it always needs its token
even in development mode, its token opens nothing else, a rotated token stops working at once, and deleting it
removes every file and its checkpoint.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.graph import checkpointer as checkpointer_module
from app.main import app
from app.services.session_access import HEADER_NAME, META_KEY
from app.services.session_store import (
    is_kept,
    purge_expired_session_ids,
    session_dir,
)

client = TestClient(app)

CSV = b"name,score\nada,9\ngrace,7\nalan,8\n"
RETENTION = 3600


def upload() -> tuple[str, str]:
    body = client.post("/api/upload", files={"file": ("scores.csv", io.BytesIO(CSV), "text/csv")}).json()
    return body["session_id"], body["session_token"]


def auth(token: str | None) -> dict:
    return {HEADER_NAME: token} if token else {}


def keep(session_id: str, token: str, name: str = "Churn study"):
    return client.post(f"/api/session/{session_id}/keep", json={"name": name}, headers=auth(token))


def age(session_id: str, seconds: float) -> None:
    """Makes every file of the session look `seconds` old."""
    then = time.time() - seconds
    directory = Path(get_settings().upload_dir) / session_id
    for path in [directory, *directory.rglob("*")]:
        os.utime(path, (then, then))


class FakeCheckpointer:
    def __init__(self):
        self.deleted: list[str] = []

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)


class FakeGraph:
    def __init__(self):
        self.checkpointer = FakeCheckpointer()


@pytest.fixture
def graph(monkeypatch):
    fake = FakeGraph()
    monkeypatch.setattr(app.state, "graph", fake, raising=False)
    return fake


class TestKeeping:
    def test_a_kept_session_says_so_with_its_name(self):
        session_id, token = upload()
        body = keep(session_id, token, "  Churn   study ").json()
        assert body["kept"] is True and body["name"] == "Churn study" and body["kept_at"]
        # The browser already holds the token, so none is issued.
        assert body["token"] is None
        assert client.get(f"/api/session/{session_id}/project", headers=auth(token)).json()["kept"] is True

    def test_renaming_keeps_the_date_it_was_first_kept(self):
        session_id, token = upload()
        first = keep(session_id, token, "One").json()
        second = keep(session_id, token, "Two").json()
        assert second["name"] == "Two" and second["kept_at"] == first["kept_at"]

    @pytest.mark.parametrize("name", ["", "   ", "x" * 101])
    def test_a_name_is_required_and_bounded(self, name):
        session_id, token = upload()
        assert keep(session_id, token, name).status_code == 422

    def test_a_session_from_before_tokens_gets_one_when_kept(self):
        session_id, _ = upload()
        meta_path = session_dir(get_settings().upload_dir, session_id) / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta.pop(META_KEY)
        meta_path.write_text(json.dumps(meta))

        body = keep(session_id, None).json()
        assert body["token"]
        assert client.get(f"/api/session/{session_id}/project").status_code == 403
        assert client.get(f"/api/session/{session_id}/project", headers=auth(body["token"])).status_code == 200


class TestTheTokenIsAlwaysRequired:
    """conftest runs the app in development mode, where ephemeral sessions are open."""

    def test_an_ephemeral_session_is_open_in_development_mode(self):
        session_id, _ = upload()
        assert client.get(f"/api/session/{session_id}/project").status_code == 200

    def test_a_kept_project_is_not(self):
        session_id, token = upload()
        keep(session_id, token)
        assert client.get(f"/api/session/{session_id}/project").status_code == 403
        assert client.get(f"/api/session/{session_id}/project", headers=auth("wrong")).status_code == 403
        assert client.get(f"/api/session/{session_id}/project", headers=auth(token)).status_code == 200

    def test_one_projects_token_never_opens_another(self):
        first, first_token = upload()
        second, second_token = upload()
        keep(first, first_token, "First")
        keep(second, second_token, "Second")
        assert client.get(f"/api/session/{second}/project", headers=auth(first_token)).status_code == 403
        assert client.delete(f"/api/session/{second}", headers=auth(first_token)).status_code == 403
        assert (Path(get_settings().upload_dir) / second).is_dir()

    def test_a_rotated_token_replaces_the_old_one_at_once(self):
        session_id, token = upload()
        keep(session_id, token)
        new = client.post(f"/api/session/{session_id}/token/rotate", headers=auth(token)).json()["token"]
        assert new and new != token
        assert client.get(f"/api/session/{session_id}/project", headers=auth(token)).status_code == 403
        assert client.get(f"/api/session/{session_id}/project", headers=auth(new)).status_code == 200

    def test_only_the_hash_of_a_rotated_token_is_stored(self):
        session_id, token = upload()
        new = client.post(f"/api/session/{session_id}/token/rotate", headers=auth(token)).json()["token"]
        assert new not in (session_dir(get_settings().upload_dir, session_id) / "meta.json").read_text()


class TestTheSweep:
    def test_a_kept_project_outlives_the_retention_period(self):
        kept, token = upload()
        keep(kept, token)
        ephemeral, _ = upload()
        age(kept, RETENTION * 10)
        age(ephemeral, RETENTION * 10)

        removed = purge_expired_session_ids(get_settings().upload_dir, RETENTION)
        assert removed == [ephemeral]
        assert (Path(get_settings().upload_dir) / kept).is_dir()

    def test_a_sidecar_that_cannot_be_read_is_treated_as_kept(self, tmp_path):
        """Deleting a project over a parse error is the one mistake the sweep must not make."""
        (tmp_path / "meta.json").write_text("{ not json")
        assert is_kept(tmp_path) is True

    def test_no_sidecar_is_not_kept(self, tmp_path):
        assert is_kept(tmp_path) is False

    def test_the_sweep_deletes_the_checkpoints_of_what_it_removed(self, monkeypatch):
        from app import main

        ephemeral, _ = upload()
        age(ephemeral, RETENTION * 10)
        fake = FakeCheckpointer()
        settings = get_settings()
        monkeypatch.setattr(settings, "upload_retention_seconds", RETENTION)
        monkeypatch.setattr(settings, "cleanup_interval_seconds", 3600)

        async def one_pass():
            task = asyncio.create_task(main._sweep_expired_sessions(settings, fake))
            for _ in range(200):
                if fake.deleted:
                    break
                await asyncio.sleep(0.01)
            task.cancel()

        asyncio.run(one_pass())
        assert fake.deleted == [ephemeral]


class TestDeleting:
    def test_it_removes_every_file_and_the_checkpoint(self, graph):
        session_id, token = upload()
        keep(session_id, token)
        directory = Path(get_settings().upload_dir) / session_id
        (directory / "runs" / "r1").mkdir(parents=True)
        (directory / "runs" / "r1" / "model.joblib").write_bytes(b"x" * 100)

        body = client.delete(f"/api/session/{session_id}", headers=auth(token)).json()
        assert body["deleted"] == session_id
        assert body["files_removed"] >= 3 and body["bytes_removed"] >= 100
        assert body["checkpoint_deleted"] is True
        assert graph.checkpointer.deleted == [session_id]
        assert not directory.exists()
        assert client.get(f"/api/session/{session_id}/project", headers=auth(token)).status_code == 404

    def test_an_ephemeral_session_can_be_deleted_too(self):
        session_id, token = upload()
        assert client.delete(f"/api/session/{session_id}", headers=auth(token)).status_code == 200
        assert not (Path(get_settings().upload_dir) / session_id).exists()

    def test_without_redis_the_files_still_go(self):
        session_id, token = upload()
        body = client.delete(f"/api/session/{session_id}", headers=auth(token)).json()
        assert body["checkpoint_deleted"] is False
        assert not (Path(get_settings().upload_dir) / session_id).exists()

    def test_an_unknown_session_is_404(self):
        assert client.delete("/api/session/not-a-session").status_code == 404


class TestTheCheckpointHelpers:
    def test_a_failing_delete_is_reported_not_raised(self):
        class Broken:
            async def adelete_thread(self, thread_id):
                raise ConnectionError("redis is down")

        assert asyncio.run(checkpointer_module.delete_thread(Broken(), "s")) is False

    def test_persisting_keys_removes_their_ttl(self):
        class FakeRedis:
            def __init__(self):
                self.persisted = []

            async def scan_iter(self, match, count):
                for key in [b"checkpoint:s1:__empty__:a", b"checkpoint_write:s1:__empty__:a:t:0"]:
                    yield key

            async def persist(self, key):
                self.persisted.append(key)
                return True

        class Saver:
            _redis = FakeRedis()

        assert asyncio.run(checkpointer_module.persist_thread(Saver(), "s1")) == 2
