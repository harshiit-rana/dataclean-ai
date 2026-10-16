"""Long work as jobs recorded on disk, run inline or by a worker (#103, docs/decisions/0004).

What has to hold: a job is claimed by exactly one runner; its record says how it ended and its events end with the
result or the error; a dead runner's job is marked interrupted; and under the worker runner the API only queues, the
worker does the work, a stream that goes away does not lose the search, and a page can pick the stream up again.
"""

from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import worker
from app.config import get_settings
from app.main import app
from app.services import jobs

client = TestClient(app)


def _csv(n: int = 300) -> bytes:
    rng = np.random.default_rng(17)
    signal = rng.normal(0, 1, n)
    buffer = io.StringIO()
    pd.DataFrame({
        "signal": signal.round(4),
        "noise": rng.normal(0, 1, n).round(4),
        "outcome": np.where(rng.random(n) < 1 / (1 + np.exp(-2.4 * signal)), "yes", "no"),
    }).to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def _session() -> str:
    session_id = client.post("/api/upload", files={"file": ("data.csv", io.BytesIO(_csv()), "text/csv")}).json()[
        "session_id"
    ]
    goal = {"use_case": "ml_training", "expected_output": "predict", "target_column": "outcome", "task_type": "auto"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


def _events(text: str) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in text.splitlines() if line.startswith("data: ")]


@pytest.fixture
def kinds(monkeypatch):
    """Throwaway job kinds for the record-keeping tests, removed afterwards."""
    registered: dict = {}
    monkeypatch.setattr(jobs, "HANDLERS", {**jobs.HANDLERS, **registered})
    return jobs.HANDLERS


class TestTheRecord:
    def test_only_one_runner_can_claim_a_job(self, tmp_path):
        folder = jobs.create(tmp_path, "nothing", {})
        assert jobs.claim(folder, "a") is True
        assert jobs.claim(folder, "b") is False
        assert (folder / jobs.CLAIM_FILE).read_text() == "a"

    def test_files_are_there_before_the_job_is(self, tmp_path):
        folder = jobs.create(tmp_path, "nothing", {}, files={"upload.csv": b"a,b\n1,2\n"})
        assert (folder / "upload.csv").read_bytes() == b"a,b\n1,2\n"
        assert jobs.read_state(folder)["state"] == jobs.QUEUED

    def test_a_finished_job_ends_with_its_result(self, tmp_path, kinds):
        def work(folder, payload, emit):
            emit({"type": "stage", "name": "one"})
            return {"answer": payload["x"] * 2}

        kinds["double"] = work
        folder = jobs.create(tmp_path, "double", {"x": 21})
        assert jobs.run(folder)["state"] == jobs.DONE
        events, _ = jobs.read_events(folder)
        assert [e["type"] for e in events] == ["stage", "result"]
        assert jobs.result_of(folder) == {"answer": 42}

    def test_a_failure_keeps_its_message_code_and_status(self, tmp_path, kinds):
        def work(folder, payload, emit):
            raise jobs.JobError("No usable columns.", "no_usable_columns", 422, "Model could not be trained")

        kinds["refuse"] = work
        folder = jobs.create(tmp_path, "refuse", {})
        assert jobs.run(folder)["state"] == jobs.FAILED
        events, _ = jobs.read_events(folder)
        assert events[-1] == {"type": "error", "detail": "No usable columns.", "code": "no_usable_columns"}
        with pytest.raises(jobs.JobError) as raised:
            jobs.result_of(folder)
        assert (raised.value.code, raised.value.status, raised.value.error) == (
            "no_usable_columns", 422, "Model could not be trained")

    def test_an_unexpected_failure_does_not_leak_its_text(self, tmp_path, kinds):
        def work(folder, payload, emit):
            raise RuntimeError("secret path C:/x/y")

        kinds["crash"] = work
        folder = jobs.create(tmp_path, "crash", {})
        jobs.run(folder)
        events, _ = jobs.read_events(folder)
        assert events[-1]["code"] == "job_error" and "secret" not in events[-1]["detail"]

    def test_a_half_written_line_waits_for_the_next_read(self, tmp_path):
        folder = jobs.create(tmp_path, "nothing", {})
        with (folder / jobs.EVENTS_FILE).open("a", encoding="utf-8") as handle:
            handle.write('{"type": "stage"}\n{"type": "ro')
        events, offset = jobs.read_events(folder)
        assert events == [{"type": "stage"}]
        with (folder / jobs.EVENTS_FILE).open("a", encoding="utf-8") as handle:
            handle.write('und"}\n')
        more, _ = jobs.read_events(folder, offset)
        assert more == [{"type": "round"}]


class TestDeadRunners:
    def test_a_running_job_without_a_heartbeat_is_marked_interrupted(self, tmp_path):
        session = tmp_path / "s1"
        folder = jobs.create(session, "nothing", {})
        jobs.claim(folder, "w")
        jobs.update_state(folder, state=jobs.RUNNING)
        jobs.beat(folder)
        assert jobs.reap_stale(str(tmp_path)) == []
        assert jobs.reap_stale(str(tmp_path), now=time.time() + jobs.STALE_AFTER_SECONDS + 1) == [folder]
        assert jobs.read_state(folder)["state"] == jobs.INTERRUPTED
        events, _ = jobs.read_events(folder)
        assert events[-1]["code"] == "job_interrupted"

    def test_a_restart_interrupts_the_inline_jobs_at_once(self, tmp_path):
        session = tmp_path / "s1"
        running = jobs.create(session, "nothing", {})
        jobs.claim(running, "inline")
        jobs.update_state(running, state=jobs.RUNNING, owner="inline")
        waiting = jobs.create(session, "nothing", {})  # queued for a worker, never claimed: left alone
        assert jobs.interrupt_orphans(str(tmp_path)) == [running]
        assert jobs.read_state(waiting)["state"] == jobs.QUEUED

    def test_the_queue_is_oldest_first_and_skips_claimed_jobs(self, tmp_path):
        session = tmp_path / "s1"
        first = jobs.create(session, "nothing", {})
        time.sleep(0.01)
        second = jobs.create(session, "nothing", {})
        time.sleep(0.01)
        third = jobs.create(session, "nothing", {})
        jobs.claim(second, "w")
        assert jobs.queued(str(tmp_path)) == [first, third]

    def test_deleting_a_session_cancels_its_unfinished_jobs(self, tmp_path):
        folder = jobs.create(tmp_path, "nothing", {})
        assert jobs.cancel_session_jobs(tmp_path) == 1
        assert jobs.read_state(folder)["state"] == jobs.CANCELLED


@pytest.fixture
def worker_mode(monkeypatch):
    """JOB_RUNNER=worker, with a worker running in a thread of the test, as the compose worker container would."""
    monkeypatch.setenv("JOB_RUNNER", "worker")
    get_settings.cache_clear()
    stop = threading.Event()
    upload_dir = get_settings().upload_dir

    def loop():
        while not stop.is_set():
            if not worker.run_once(upload_dir, "test-worker"):
                time.sleep(0.05)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    yield
    stop.set()
    thread.join(timeout=60)


TRAIN = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}
IMPROVE = {"model_names": ["logistic_regression"], "budget": 4, "include_deep_diagnostics": False}


def _jobs_of(session_id: str) -> list[Path]:
    return list(jobs.session_jobs(Path(get_settings().upload_dir) / session_id))


class TestTheWorkerRunner:
    def test_the_api_only_queues_and_the_worker_trains(self, worker_mode):
        session_id = _session()
        result = client.post(f"/api/model/{session_id}/train", json=TRAIN).json()
        assert result["best_model"] == "logistic_regression"
        (folder,) = _jobs_of(session_id)
        assert (folder / jobs.CLAIM_FILE).read_text() == "test-worker"
        assert jobs.read_state(folder)["state"] == jobs.DONE

    def test_a_failure_in_the_worker_comes_back_as_the_routes_own_error(self, worker_mode):
        session_id = _session()
        response = client.post(f"/api/model/{session_id}/train", json={**TRAIN, "excluded_columns": ["signal", "noise"]})
        assert response.status_code == 422
        assert response.json()["code"] == "no_usable_columns"

    def test_a_search_survives_its_stream_going_away(self, worker_mode):
        """The acceptance test of #103: the page (or the API) goes, the worker finishes and saves."""
        session_id = _session()
        with client.stream("POST", f"/api/model/{session_id}/improve/stream", json=IMPROVE) as response:
            first = next(response.iter_lines())
        job_id = json.loads(first[len("data: "):])["job_id"]
        folder = jobs.job_dir(Path(get_settings().upload_dir) / session_id, job_id)
        deadline = time.time() + 300
        while jobs.read_state(folder).get("state") not in jobs.FINISHED and time.time() < deadline:
            time.sleep(0.2)
        assert jobs.read_state(folder)["state"] == jobs.DONE
        saved = client.get(f"/api/model/{session_id}/result").json()
        assert saved["improvement"]["rounds"]

    def test_a_page_can_pick_the_stream_up_where_it_was(self, worker_mode):
        session_id = _session()
        whole = _events(client.post(f"/api/model/{session_id}/improve/stream", json=IMPROVE).text)
        job_id = whole[0]["job_id"]
        progress = [e for e in whole if e["type"] != "job"]
        again = _events(client.get(f"/api/model/{session_id}/jobs/{job_id}/stream", params={"after": 2}).text)
        assert again[0] == {"type": "job", "job_id": job_id}
        assert again[1:] == progress[2:]
        assert again[-1]["type"] == "result"
        state = client.get(f"/api/model/{session_id}/jobs/{job_id}").json()
        assert state["state"] == "done" and state["kind"] == "improve"


class TestTheJobRoutes:
    def test_an_unknown_job_is_404(self):
        session_id = _session()
        assert client.get(f"/api/model/{session_id}/jobs/abc123").status_code == 404
        assert client.get(f"/api/model/{session_id}/jobs/..%2F..%2Fmeta").status_code == 404

    def test_a_job_that_ended_without_a_closing_event_says_so(self):
        session_id = _session()
        folder = jobs.create(Path(get_settings().upload_dir) / session_id, "nothing", {})
        jobs.update_state(folder, state=jobs.CANCELLED)
        events = _events(client.get(f"/api/model/{session_id}/jobs/{folder.name}/stream").text)
        assert events[-1]["type"] == "error" and events[-1]["code"] == "job_cancelled"

    def test_inline_training_is_recorded_as_a_job_too(self):
        session_id = _session()
        client.post(f"/api/model/{session_id}/train", json=TRAIN)
        (folder,) = _jobs_of(session_id)
        assert jobs.read_state(folder)["state"] == jobs.DONE
        assert (folder / jobs.CLAIM_FILE).read_text() == "inline"

    def test_an_applied_file_is_not_kept_in_the_job_folder(self, monkeypatch):
        class FakeGraph:
            async def aget_state(self, _config):
                return SimpleNamespace(values={"action_log": []}, next=())

        monkeypatch.setattr(app.state, "graph", FakeGraph(), raising=False)
        session_id = _session()
        client.post(f"/api/model/{session_id}/train", json=TRAIN)
        new = pd.read_csv(io.BytesIO(_csv(80))).drop(columns="outcome").to_csv(index=False).encode()
        response = client.post(f"/api/apply/{session_id}", files={"file": ("new.csv", io.BytesIO(new), "text/csv")})
        assert response.status_code == 200, response.text
        apply_jobs = [f for f in _jobs_of(session_id) if jobs.read_state(f)["kind"] == "apply"]
        assert apply_jobs and not (apply_jobs[0] / "upload.csv").exists()
