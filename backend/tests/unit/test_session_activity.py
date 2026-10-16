"""Reading a session keeps its files alive, in step with its checkpoint; an expired one is turned away alike."""

from __future__ import annotations

import io
import os
import shutil
import time

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import session_store
from app.services.session_store import purge_expired_sessions, session_dir

client = TestClient(app)
RETENTION = 3600


def upload() -> tuple[str, object]:
    sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(b"a,b\n1,2\n3,4\n"), "text/csv")}).json()["session_id"]
    return sid, session_dir(get_settings().upload_dir, sid)


def age(directory, seconds: float) -> None:
    old = time.time() - seconds
    for path in [directory, *directory.rglob("*")]:
        os.utime(path, (old, old))


class TestReadingCountsAsUse:
    def test_a_session_only_read_keeps_its_files(self):
        sid, directory = upload()
        age(directory, RETENTION * 2)
        assert client.get(f"/api/preview/{sid}").status_code == 200
        assert (directory / session_store.ACTIVITY_MARKER).exists()
        assert purge_expired_sessions(get_settings().upload_dir, RETENTION) == 0
        assert directory.exists()

    def test_a_session_nobody_touched_is_still_swept(self):
        _, directory = upload()
        age(directory, RETENTION * 2)
        assert purge_expired_sessions(get_settings().upload_dir, RETENTION) == 1
        assert not directory.exists()

    def test_the_marker_is_written_at_most_once_a_minute(self, tmp_path):
        session_store.touch_session(tmp_path)
        marker = tmp_path / session_store.ACTIVITY_MARKER
        stamp = time.time() - 30
        os.utime(marker, (stamp, stamp))
        session_store.touch_session(tmp_path)
        assert abs(marker.stat().st_mtime - stamp) < 1
        stamp = time.time() - 120
        os.utime(marker, (stamp, stamp))
        session_store.touch_session(tmp_path)
        assert marker.stat().st_mtime > stamp + 60


class TestAnExpiredSession:
    def test_every_route_answers_alike_when_the_files_are_gone(self):
        sid, directory = upload()
        shutil.rmtree(directory)
        paths = [f"/api/session/{sid}", f"/api/preview/{sid}", f"/api/model/{sid}/history",
                 f"/api/export/{sid}/csv", f"/api/model/{sid}/result"]
        answers = [client.get(path) for path in paths]
        assert {r.status_code for r in answers} == {404}
        assert {r.json()["error"] for r in answers} == {"Session expired"}
