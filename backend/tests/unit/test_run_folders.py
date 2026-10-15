"""Each run's artifacts in their own folder, with a pointer to the one in use.

Retraining used to overwrite the result and the fitted pipeline, so only the latest model existed and the history was
a record, never a restore point. Each run now writes `runs/<run_id>/`; `current.json` names the one in use, and the
routes read through it unchanged. A change to the data unsets the pointer and marks the runs, rather than deleting
them, and the oldest beyond a limit are pruned -- never the one in use.
"""

from __future__ import annotations

import io
import json
import os
import time
import zipfile

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import model_store
from app.services.session_store import purge_expired_sessions, session_dir

client = TestClient(app)
TRAIN = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}


def csv(n: int = 300, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "z": rng.normal(size=n), "outcome": np.where(x + rng.normal(0, 1, n) > 0, "yes", "no")})
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def session() -> tuple[str, object]:
    sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(csv()), "text/csv")}).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    return sid, session_dir(get_settings().upload_dir, sid)


def fake_result(run_id: str) -> dict:
    return {"run_id": run_id, "results": [], "best_model": None, "task": "classification"}


class TestOverHttp:
    def test_two_trains_make_two_run_folders_and_the_result_is_the_latest(self):
        sid, directory = session()
        first = client.post(f"/api/model/{sid}/train", json=TRAIN).json()
        second = client.post(f"/api/model/{sid}/train", json={**TRAIN, "model_names": ["random_forest_classifier"]}).json()
        assert set(model_store.stored_runs(directory)) == {first["run_id"], second["run_id"]}
        assert model_store.current_run_id(directory) == second["run_id"]
        assert client.get(f"/api/model/{sid}/result").json()["run_id"] == second["run_id"]
        assert model_store.pipeline_path(directory).parent.name == second["run_id"]
        assert not (directory / model_store.RESULT_FILENAME).exists()

    def test_the_model_download_and_the_bundle_carry_the_current_run_s_pipeline(self, monkeypatch):
        from types import SimpleNamespace

        class FakeGraph:
            async def aget_state(self, _config):
                return SimpleNamespace(values={"action_log": []}, next=())

        monkeypatch.setattr(app.state, "graph", FakeGraph(), raising=False)
        sid, directory = session()
        client.post(f"/api/model/{sid}/train", json=TRAIN)
        client.post(f"/api/model/{sid}/train", json={**TRAIN, "model_names": ["random_forest_classifier"]})
        current = model_store.pipeline_path(directory).read_bytes()
        assert client.get(f"/api/model/{sid}/download").content == current
        bundle = client.get(f"/api/export/{sid}/zip")
        assert bundle.status_code == 200
        with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
            assert archive.read(model_store.PIPELINE_FILENAME) == current

    def test_a_cleaning_undo_unsets_the_model_but_keeps_the_runs(self):
        sid, directory = session()
        run_id = client.post(f"/api/model/{sid}/train", json=TRAIN).json()["run_id"]
        model_store.clear(directory)
        assert model_store.load_result(directory) is None and model_store.pipeline_path(directory) is None
        assert client.get(f"/api/model/{sid}/result").status_code == 404
        assert model_store.stored_runs(directory) == [run_id]
        assert model_store.is_superseded(directory, run_id)
        assert model_store.load_run(directory, run_id)["run_id"] == run_id


class TestTheStore:
    def test_saving_points_at_the_new_run_and_drops_the_old_write_ups(self, tmp_path):
        (tmp_path / model_store.CARD_FILENAME).write_text("old card")
        (tmp_path / model_store.SCRIPT_FILENAME).write_text("old script")
        model_store.save_result(tmp_path, fake_result("aaa"))
        assert model_store.current_run_id(tmp_path) == "aaa"
        assert not (tmp_path / model_store.CARD_FILENAME).exists()
        assert not (tmp_path / model_store.SCRIPT_FILENAME).exists()

    def test_pruning_keeps_the_newest_and_never_the_current(self, tmp_path):
        for index, run_id in enumerate(["r1", "r2", "r3", "r4"]):
            model_store.save_result(tmp_path, fake_result(run_id), keep_runs=10)
            stamp = time.time() - 100 + index
            os.utime(model_store.run_dir(tmp_path, run_id) / model_store.RUN_RESULT_FILENAME, (stamp, stamp))
        model_store.set_current(tmp_path, "r1")  # an old run brought back is the one in use
        removed = model_store.prune_runs(tmp_path, 2)
        assert set(removed) == {"r2", "r3"}
        assert set(model_store.stored_runs(tmp_path)) == {"r1", "r4"}

    def test_saving_prunes_to_the_limit(self, tmp_path):
        for index, run_id in enumerate(["r1", "r2", "r3"]):
            model_store.save_result(tmp_path, fake_result(run_id), keep_runs=2)
            stamp = time.time() - 100 + index
            os.utime(model_store.run_dir(tmp_path, run_id) / model_store.RUN_RESULT_FILENAME, (stamp, stamp))
        assert len(model_store.stored_runs(tmp_path)) == 2 and model_store.current_run_id(tmp_path) == "r3"

    def test_a_limit_below_one_still_keeps_the_current_run(self, tmp_path):
        model_store.save_result(tmp_path, fake_result("only"), keep_runs=0)
        assert model_store.stored_runs(tmp_path) == ["only"]

    def test_a_session_from_before_run_folders_still_reads_and_the_next_save_replaces_it(self, tmp_path):
        (tmp_path / model_store.RESULT_FILENAME).write_text(json.dumps({"run_id": "legacy", "results": []}))
        (tmp_path / model_store.PIPELINE_FILENAME).write_bytes(b"old")
        assert model_store.load_result(tmp_path)["run_id"] == "legacy"
        assert model_store.pipeline_path(tmp_path) == tmp_path / model_store.PIPELINE_FILENAME
        model_store.save_result(tmp_path, fake_result("new"))
        assert model_store.load_result(tmp_path)["run_id"] == "new"
        assert not (tmp_path / model_store.RESULT_FILENAME).exists()
        assert not (tmp_path / model_store.PIPELINE_FILENAME).exists()

    def test_a_broken_pointer_reads_as_no_model(self, tmp_path):
        (tmp_path / model_store.CURRENT_FILENAME).write_text("not json")
        assert model_store.current_run_id(tmp_path) is None and model_store.load_result(tmp_path) is None

    def test_the_run_id_names_the_folder_and_the_history_entry(self, tmp_path):
        result = {"results": [], "best_model": None}
        model_store.save_result(tmp_path, result)
        entry = model_store.load_history(tmp_path)[0]
        assert entry["run_id"] == result["run_id"] == model_store.current_run_id(tmp_path)

    def test_the_retention_sweep_still_removes_everything(self, tmp_path):
        directory = tmp_path / "session-a"
        directory.mkdir()
        model_store.save_result(directory, fake_result("r1"))
        old = time.time() - 10_000
        for path in [directory, *directory.rglob("*")]:
            os.utime(path, (old, old))
        assert purge_expired_sessions(str(tmp_path), 60) == 1
        assert not directory.exists()
