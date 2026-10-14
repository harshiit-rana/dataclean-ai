"""Dataset versions: a name for each state the session's working data has been in.

`cleaned.csv` is rewritten in place by the cleaning run, by undo and by chat, so nothing stable
existed to attach a model, a test-set look or an export to. These pin the rules the evaluation
ledger will rely on: a version per change, none per non-change, an undo that lands on an old
state recorded as a new version, and bookkeeping that can never fail the work it describes.
"""

from __future__ import annotations

import io
import json
import threading
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import routes_chat
from app.config import get_settings
from app.main import app
from app.services import model_store, session_store
from app.services.session_store import (
    MAX_DATASET_VERSIONS,
    dataset_versions,
    record_dataset_version_in,
    session_dir,
)

client = TestClient(app)


@pytest.fixture
def session(tmp_path):
    """A session directory shaped like the app's: an upload, and a meta sidecar with other
    keys in it that versioning must never disturb."""
    pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]}).to_csv(
        tmp_path / "original.csv", index=False
    )
    (tmp_path / "meta.json").write_text(
        json.dumps({"file_name": "d.csv", "goal_context": {"use_case": "reporting"}, "token_hash": "h"})
    )
    return tmp_path


def _meta(directory):
    return json.loads((directory / "meta.json").read_text())


def _write_cleaned(directory, rows: dict) -> None:
    pd.DataFrame(rows).to_csv(directory / "cleaned.csv", index=False)


class TestRecording:
    def test_the_first_version_is_the_upload(self, session):
        entry = record_dataset_version_in(session, "upload")
        assert entry["version"] == 1 and entry["event"] == "upload"
        assert entry["rows"] == 4 and entry["columns"] == 2
        assert len(entry["fingerprint"]) == 64  # the full digest, not the display prefix

    def test_the_fingerprint_is_the_one_the_model_path_would_compute(self, session):
        """A run's fingerprint is taken from a re-read of the file; a version has to describe
        the same bytes or the two can never be matched up."""
        entry = record_dataset_version_in(session, "upload")
        assert entry["fingerprint"] == model_store.full_fingerprint(
            pd.read_csv(session / "original.csv")
        )

    def test_a_change_appends_one_version(self, session):
        record_dataset_version_in(session, "upload")
        _write_cleaned(session, {"a": [1, 2, 3], "b": ["x", "y", "x"]})
        entry = record_dataset_version_in(session, "cleaning_run")
        assert entry["version"] == 2 and entry["event"] == "cleaning_run"
        assert [v["version"] for v in dataset_versions_of(session)] == [1, 2]

    def test_no_change_appends_nothing(self, session):
        record_dataset_version_in(session, "upload")
        assert record_dataset_version_in(session, "cleaning_run") is None
        assert len(dataset_versions_of(session)) == 1

    def test_a_cleaning_run_that_applied_nothing_adds_no_version(self, session):
        """cleaned.csv is written even when no fix was approved, and equals the upload."""
        record_dataset_version_in(session, "upload")
        (session / "cleaned.csv").write_bytes((session / "original.csv").read_bytes())
        assert record_dataset_version_in(session, "cleaning_run") is None

    def test_an_undo_back_to_an_old_state_is_a_new_version_with_an_old_fingerprint(self, session):
        """The history is chronological: 'we went back to version 1' is itself an event."""
        first = record_dataset_version_in(session, "upload")
        _write_cleaned(session, {"a": [1, 2], "b": ["x", "y"]})
        record_dataset_version_in(session, "cleaning_run")
        (session / "cleaned.csv").write_bytes((session / "original.csv").read_bytes())
        restored = record_dataset_version_in(session, "undo:a1")

        assert restored["version"] == 3
        assert restored["fingerprint"] == first["fingerprint"]

    def test_versions_are_numbered_monotonically_and_persisted(self, session):
        record_dataset_version_in(session, "upload")
        for rows in (2, 3, 1):
            _write_cleaned(session, {"a": list(range(rows)), "b": ["x"] * rows})
            record_dataset_version_in(session, f"chat:{rows}")
        assert [v["version"] for v in _meta(session)["dataset_versions"]] == [1, 2, 3, 4]

    def test_other_meta_keys_survive(self, session):
        record_dataset_version_in(session, "upload")
        meta = _meta(session)
        assert meta["goal_context"] == {"use_case": "reporting"} and meta["token_hash"] == "h"

    def test_no_temporary_file_is_left_behind(self, session):
        record_dataset_version_in(session, "upload")
        assert not list(session.glob("*.tmp"))


class TestItNeverFailsTheWorkItDescribes:
    def test_an_unreadable_data_file_is_logged_not_raised(self, session):
        (session / "cleaned.csv").write_bytes(b"\x00\x01 not a csv \xff")
        (session / "original.csv").write_text("")
        assert record_dataset_version_in(session, "cleaning_run") is None

    def test_a_corrupt_meta_file_is_survived(self, session):
        (session / "meta.json").write_text("{ not json")
        assert record_dataset_version_in(session, "upload") is None

    def test_a_directory_with_no_meta_is_not_a_session(self, tmp_path):
        """A graph run outside the app (tests, scripts) has no sidecar. It must not conjure
        one, or a stray directory."""
        (tmp_path / "original.csv").write_text("a\n1\n2\n")
        assert record_dataset_version_in(tmp_path, "cleaning_run") is None
        assert not (tmp_path / "meta.json").exists()

    def test_a_failing_write_does_not_raise(self, session, monkeypatch):
        def explode(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(session_store, "_write_meta_file", explode)
        assert record_dataset_version_in(session, "upload") is None


class TestSessionsThatPredateVersioning:
    def test_the_first_change_states_version_one_truthfully(self, session):
        """No list yet, so the upload is seeded from original.csv rather than guessed."""
        _write_cleaned(session, {"a": [1, 2], "b": ["x", "y"]})
        entry = record_dataset_version_in(session, "chat:abc")
        versions = dataset_versions_of(session)
        assert [v["event"] for v in versions] == ["upload", "chat:abc"]
        assert entry["version"] == 2


class TestTheListIsBounded:
    def test_it_keeps_the_upload_and_the_newest_and_says_what_it_dropped(self, session, monkeypatch):
        monkeypatch.setattr(session_store, "MAX_DATASET_VERSIONS", 5)
        record_dataset_version_in(session, "upload")
        for i in range(1, 9):
            _write_cleaned(session, {"a": list(range(i)), "b": ["x"] * i})
            record_dataset_version_in(session, f"chat:{i}")

        versions = _meta(session)["dataset_versions"]
        assert len(versions) == 5
        assert versions[0]["version"] == 1 and versions[0]["event"] == "upload"
        assert versions[-1]["version"] == 9  # numbering never restarts
        assert _meta(session)["dataset_versions_dropped"] == 4

    def test_the_real_cap_is_two_hundred(self):
        assert MAX_DATASET_VERSIONS == 200


class TestConcurrentRequests:
    def test_two_threads_recording_different_data_lose_no_version(self, session):
        record_dataset_version_in(session, "upload")
        barrier = threading.Barrier(2)
        results: list = []

        def worker(rows: int, event: str):
            barrier.wait()
            # Each thread needs its own file state, which a shared cleaned.csv cannot give, so
            # hand the frame in directly: the race is on the meta list, not the data.
            frame = pd.DataFrame({"a": list(range(rows)), "b": ["x"] * rows})
            results.append(record_dataset_version_in(session, event, frame))

        threads = [
            threading.Thread(target=worker, args=(7, "chat:a")),
            threading.Thread(target=worker, args=(11, "chat:b")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        versions = _meta(session)["dataset_versions"]
        assert sorted(v["version"] for v in versions) == [1, 2, 3]
        assert {v["event"] for v in versions} == {"upload", "chat:a", "chat:b"}


def dataset_versions_of(directory) -> list[dict]:
    return _meta(directory)["dataset_versions"]


# --------------------------------------------------------------------- over the API


class FakeGraph:
    def __init__(self, values):
        self.values = values
        self.updated: dict = {}

    async def aget_state(self, _config):
        return SimpleNamespace(values=dict(self.values, **self.updated), next=())

    async def aupdate_state(self, _config, update):
        self.updated.update({k: v for k, v in update.items() if k == "action_log"})


class FakeClient:
    def __init__(self, reply):
        self.reply = reply

    async def ainvoke(self, _messages):
        return SimpleNamespace(content=self.reply)


@pytest.fixture
def graph_slot():
    had = hasattr(app.state, "graph")
    previous = getattr(app.state, "graph", None)
    yield
    if had:
        app.state.graph = previous
    elif hasattr(app.state, "graph"):
        del app.state.graph


def _upload(rows: int = 12) -> str:
    frame = pd.DataFrame(
        {"id": range(rows), "amount": [10.0 + i for i in range(rows)], "region": ["n", "s", "n"] * (rows // 3)}
    )
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]


class TestOverTheApi:
    def test_an_upload_creates_version_one(self):
        session_id = _upload()
        versions = dataset_versions(get_settings().upload_dir, session_id)
        assert [(v["version"], v["event"]) for v in versions] == [(1, "upload")]

    def test_get_session_returns_the_versions(self, graph_slot):
        session_id = _upload()
        app.state.graph = FakeGraph({"issues": [], "action_log": []})
        body = client.get(f"/api/session/{session_id}").json()
        assert [v["event"] for v in body["dataset_versions"]] == ["upload"]
        assert body["dataset_versions"][0]["rows"] == 12

    @staticmethod
    def _graph_for(session_id, directory, action_log=None):
        goal = {"use_case": "reporting", "expected_output": "x", "critical_columns": [],
                "cleaning_aggression": "moderate", "target_column": None, "task_type": "auto"}
        return FakeGraph(
            {"session_id": session_id, "file_path": str(directory / "original.csv"),
             "file_name": "d.csv", "file_size_bytes": 10, "goal_context": goal,
             "action_log": action_log or []}
        )

    def test_a_chat_instruction_that_changes_the_data_adds_a_version(self, graph_slot, monkeypatch):
        session_id = _upload()
        directory = session_dir(get_settings().upload_dir, session_id)
        original = pd.read_csv(directory / "original.csv")
        # What a cleaning run that kept a duplicate row would have left behind.
        pd.concat([original, original.iloc[[0]]], ignore_index=True).to_csv(
            directory / "cleaned.csv", index=False
        )
        record_dataset_version_in(directory, "cleaning_run")
        app.state.graph = self._graph_for(session_id, directory)
        monkeypatch.setattr(
            routes_chat,
            "get_llm_client",
            lambda *_a, **_k: FakeClient(
                '{"operation": "drop_duplicate_rows", "params": {}, "explanation": "x"}'
            ),
        )

        chat = client.post(f"/api/chat/{session_id}", json={"message": "remove the duplicates"})

        assert chat.status_code == 200 and chat.json()["applied"] is True
        versions = dataset_versions(get_settings().upload_dir, session_id)
        assert [v["version"] for v in versions] == [1, 2, 3]
        assert versions[2]["event"] == f"chat:{chat.json()['action_id']}"
        assert versions[2]["fingerprint"] == versions[0]["fingerprint"]  # back to the upload's data

    def test_an_undo_that_changes_the_data_adds_a_version(self, graph_slot):
        session_id = _upload()
        directory = session_dir(get_settings().upload_dir, session_id)
        original = pd.read_csv(directory / "original.csv")
        # A cleaning run that upper-cased the region column.
        original.assign(region=original["region"].str.upper()).to_csv(
            directory / "cleaned.csv", index=False
        )
        record_dataset_version_in(directory, "cleaning_run")
        app.state.graph = self._graph_for(
            session_id,
            directory,
            [
                {"action_id": "a1", "card_id": "c1", "column_name": "region",
                 "operation": "standardize_case", "operation_params": {"column": "region", "case": "upper"},
                 "status": "applied"}
            ],
        )

        undone = client.post(f"/api/undo/{session_id}", json={"action_id": "a1"})

        assert undone.status_code == 200, undone.text
        versions = dataset_versions(get_settings().upload_dir, session_id)
        assert [(v["version"], v["event"]) for v in versions] == [
            (1, "upload"), (2, "cleaning_run"), (3, "undo:a1"),
        ]
        assert versions[2]["fingerprint"] == versions[0]["fingerprint"]

    def test_a_chat_instruction_that_changes_nothing_adds_no_version(self, graph_slot, monkeypatch):
        session_id = _upload()
        directory = session_dir(get_settings().upload_dir, session_id)
        (directory / "cleaned.csv").write_bytes((directory / "original.csv").read_bytes())
        app.state.graph = self._graph_for(session_id, directory)
        monkeypatch.setattr(
            routes_chat,
            "get_llm_client",
            lambda *_a, **_k: FakeClient(
                '{"operation": "drop_duplicate_rows", "params": {}, "explanation": "x"}'
            ),
        )
        assert client.post(f"/api/chat/{session_id}", json={"message": "dedupe"}).status_code == 200
        assert len(dataset_versions(get_settings().upload_dir, session_id)) == 1
