"""Resuming an interrupted improvement search from where it stopped (#104).

The acceptance test of the issue: a search interrupted after round 2 resumes at round 3 and ends with the same result
as one that was never interrupted, with the same seeds; the trail says where it resumed. The journal replays the
finished work only while every call matches what was recorded, and never for different data.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import worker
from app.config import get_settings
from app.main import app
from app.services import improvement, jobs, model_store, search_journal, tuning

client = TestClient(app)

MODELS = ["logistic_regression", "random_forest_classifier"]


def _frame(n: int = 300, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, n)
    return pd.DataFrame({
        "signal": signal.round(4),
        "noise": rng.normal(0, 1, n).round(4),
        "region": rng.choice(["north", "south"], n),
        "outcome": np.where(rng.random(n) < 1 / (1 + np.exp(-2.4 * signal)), "yes", "no"),
    })


def _search(frame, journal=None, on_progress=None) -> dict:
    return improvement.improve(frame, target_column="outcome", task="classification", model_names=MODELS, budget=4,
                               journal=journal, on_progress=on_progress)


def _summary(found: dict) -> dict:
    best = found["best"]
    return {
        "model": best["model"], "params": best["params"], "cv_score": best["cv_score"],
        "rounds": [(r["name"], r["accepted"], r["score"], r["outcome"]) for r in found["rounds"]],
    }


class Interrupted(BaseException):
    """A process killed mid-search: not an error the handler can catch and record, as SIGKILL is not."""


def _interrupt_after(rounds: int):
    seen = []

    def on_progress(entry):
        seen.append(entry.name)
        if len(seen) == rounds:
            raise Interrupted(entry.name)

    return on_progress


class TestTheJournal:
    def test_a_recorded_call_is_replayed_without_running_again(self, tmp_path):
        calls = []

        def work(x, *, scale):
            calls.append(x)
            return x * scale

        first = search_journal.Journal(tmp_path / "a.json", fingerprint="f")
        assert [first.call("work", work, n, scale=2) for n in (1, 2, 3)] == [2, 4, 6]
        again = search_journal.Journal(tmp_path / "b.json", fingerprint="f", previous=tmp_path / "a.json")
        assert [again.call("work", work, n, scale=2) for n in (1, 2, 3)] == [2, 4, 6]
        assert calls == [1, 2, 3] and again.replayed == 3

    def test_the_first_call_that_differs_switches_replay_off_for_the_rest(self, tmp_path):
        calls = []

        def work(x):
            calls.append(x)
            return x

        first = search_journal.Journal(tmp_path / "a.json", fingerprint="f")
        for n in (1, 2, 3):
            first.call("work", work, n)
        again = search_journal.Journal(tmp_path / "b.json", fingerprint="f", previous=tmp_path / "a.json")
        for n in (1, 9, 3):
            again.call("work", work, n)
        assert again.replayed == 1 and again.diverged_at == 1
        assert calls == [1, 2, 3, 9, 3]  # the 3 after the divergence is computed afresh

    def test_a_journal_for_different_data_is_never_replayed(self, tmp_path):
        first = search_journal.Journal(tmp_path / "a.json", fingerprint="january")
        first.call("work", lambda x: x, 1)
        again = search_journal.Journal(tmp_path / "b.json", fingerprint="february", previous=tmp_path / "a.json")
        again.call("work", lambda x: x, 1)
        assert again.replayed == 0

    def test_the_data_itself_is_part_of_the_key(self):
        a = pd.DataFrame({"x": [1.0, 2.0]})
        b = pd.DataFrame({"x": [1.0, 2.5]})
        assert search_journal.signature("k", (a,), {}) != search_journal.signature("k", (b,), {})
        assert search_journal.signature("k", (a,), {"deadline": 1.0}) == search_journal.signature("k", (a,), {"deadline": 9.0})

    def test_a_search_result_comes_back_as_the_same_value(self):
        found = tuning.SearchResult(params={"model__C": np.float64(0.3), "model__class_weight": None}, score=0.81,
                                    std=float("nan"), metric="roc_auc", fits=np.int64(20), fold_scores=(0.8, 0.82))
        back = search_journal.decode(json.loads(json.dumps(search_journal.encode(found))))
        assert back.params == {"model__C": 0.3, "model__class_weight": None}
        assert back.fold_scores == (0.8, 0.82) and isinstance(back.fold_scores, tuple)
        assert back.fits == 20 and np.isnan(back.std)

    def test_it_is_json_not_a_pickle(self, tmp_path):
        journal = search_journal.Journal(tmp_path / search_journal.JOURNAL_FILE, fingerprint="f")
        journal.call("work", lambda: {"importance": (0.1, 0.02)})
        stored = json.loads((tmp_path / search_journal.JOURNAL_FILE).read_text(encoding="utf-8"))
        assert stored["fingerprint"] == "f" and len(stored["entries"]) == 1


class TestResumingASearch:
    def test_interrupted_after_round_two_it_resumes_and_ends_where_an_uninterrupted_run_does(self, tmp_path):
        frame = _frame()
        whole = _summary(_search(frame))

        interrupted = search_journal.Journal(tmp_path / "first.json", fingerprint="f")
        with pytest.raises(Interrupted):
            _search(frame, journal=interrupted, on_progress=_interrupt_after(2))
        recorded = len(interrupted.entries)
        assert recorded > 0

        resumed = search_journal.Journal(tmp_path / "second.json", fingerprint="f", previous=tmp_path / "first.json")
        again = _summary(_search(frame, journal=resumed))
        assert again == whole
        # Rounds 1 and 2 came from the journal: every call they made, and nothing was recomputed until round 3.
        assert resumed.replayed == recorded and resumed.diverged_at is None
        assert len(resumed.entries) > recorded

    def test_without_a_journal_nothing_changes(self):
        frame = _frame()
        assert _summary(_search(frame)) == _summary(_search(frame, journal=None))


class TestTheJobs:
    def _improve_job(self, session: Path, payload_extra: dict | None = None) -> Path:
        request = {"model_names": MODELS, "budget": 4, "include_deep_diagnostics": False}
        return jobs.create(session, "improve", {
            "request": request, "task": "classification", "action_log": [], "cost_limit": None, "route": "improve",
            **(payload_extra or {}),
        })

    def _session(self) -> Path:
        buffer = io.StringIO()
        _frame().to_csv(buffer, index=False)
        session_id = client.post(
            "/api/upload", files={"file": ("data.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
        ).json()["session_id"]
        goal = {"use_case": "ml_training", "expected_output": "p", "target_column": "outcome", "task_type": "auto"}
        client.post(f"/api/goal/{session_id}", json=goal)
        return Path(get_settings().upload_dir) / session_id

    def test_a_dead_runners_search_carries_on_as_a_new_job_that_replays_the_journal(self, monkeypatch):
        session = self._session()
        folder = self._improve_job(session)
        jobs.claim(folder, "dead-worker")

        # The first run dies after round 2: its handler is stopped by an emit that raises, as a killed process would.
        rounds_seen = []

        def dying(event):
            if event.get("type") == "round":
                rounds_seen.append(event)
                if len(rounds_seen) == 2:
                    raise Interrupted("killed")

        from app.api import routes_model

        state = jobs.update_state(folder, state=jobs.RUNNING, owner="dead-worker", started_epoch=time.time())
        with pytest.raises(Interrupted):
            routes_model._improve_job(folder, state["payload"], dying)
        assert (folder / search_journal.JOURNAL_FILE).exists()

        # Its heartbeat goes stale; the next worker marks it interrupted and resumes it.
        (resumed,) = jobs.reap_stale(str(session.parent), now=time.time() + jobs.STALE_AFTER_SECONDS + 5)
        assert jobs.read_state(folder)["state"] == jobs.INTERRUPTED
        events, _ = jobs.read_events(folder)
        assert events[-1]["resumed_as"] == resumed.name
        assert jobs.read_state(resumed)["payload"]["resume_of"] == folder.name

        assert worker.run_once(str(session.parent), "next-worker") is True
        result = jobs.result_of(resumed)
        assert result["improvement"]["resumed"]["from_job"] == folder.name
        assert result["improvement"]["resumed"]["reused"] > 0
        events, _ = jobs.read_events(resumed)
        assert events[0] == {"type": "stage", "name": "resuming",
                             "description": "Resuming the interrupted search: the rounds it finished are restored "
                                            "from its journal."}

    def test_a_job_that_keeps_dying_is_not_resumed_for_ever(self, tmp_path):
        folder = self._improve_job(tmp_path / "s1", {"resume_count": jobs.MAX_RESUMES})
        assert jobs.resume(folder) is None

    def test_only_searches_resume(self, tmp_path):
        folder = jobs.create(tmp_path / "s1", "train", {"request": {}})
        assert jobs.resume(folder) is None

    def test_a_time_limit_counts_the_time_already_spent(self, tmp_path):
        folder = self._improve_job(tmp_path / "s1")
        state = jobs.read_state(folder)
        state["payload"]["request"]["time_limit_seconds"] = 300
        jobs.update_state(folder, payload=state["payload"], started_epoch=time.time() - 120)
        resumed = jobs.resume(folder)
        assert 170 < jobs.read_state(resumed)["payload"]["request"]["time_limit_seconds"] <= 180

    def test_the_saved_journal_matches_the_data(self):
        assert model_store.fingerprint_frame(_frame()) == model_store.fingerprint_frame(_frame())
