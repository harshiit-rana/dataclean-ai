"""A saved model is checked before it is loaded: its checksum, and the scikit-learn that saved it.

`model.joblib` is a pickle. A file changed on disk could load and run anything; one loaded under another scikit-learn
could load and predict differently. Each saved model now has a `model.json` beside it with its SHA-256 and versions,
every load goes through `model_store.load_pipeline`, and the export bundle carries the sidecar.
"""

from __future__ import annotations

import io
import json
import pathlib
import re
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import model_export, model_store, modeling, run_metadata
from app.services.session_store import session_dir

client = TestClient(app)


class FakeGraph:
    async def aget_state(self, _config):
        return SimpleNamespace(values={"action_log": []}, next=())


def frame(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, n)
    return pd.DataFrame({"signal": signal.round(4), "plan": rng.choice(["basic", "pro"], n),
                         "churned": np.where(rng.random(n) < 1 / (1 + np.exp(-2.5 * signal)), "yes", "no")})


def csv(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


@pytest.fixture
def trained(monkeypatch):
    monkeypatch.setattr(app.state, "graph", FakeGraph(), raising=False)
    sid = client.post("/api/upload", files={"file": ("t.csv", io.BytesIO(csv(frame())), "text/csv")}).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "churned", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    response = client.post(f"/api/model/{sid}/train", json={"model_names": ["logistic_regression"],
                                                             "include_deep_diagnostics": False})
    assert response.status_code == 200
    return sid, session_dir(get_settings().upload_dir, sid)


def apply_rows(sid: str):
    rows = frame(40, seed=7).drop(columns=["churned"])
    return client.post(f"/api/apply/{sid}", files={"file": ("new.csv", io.BytesIO(csv(rows)), "text/csv")})


def sidecar(directory) -> pathlib.Path:
    return model_store.pipeline_path(directory).parent / model_store.MODEL_SIDECAR_FILENAME


class TestTheSidecar:
    def test_a_saved_model_records_its_checksum_and_versions(self, trained):
        _, directory = trained
        recorded = json.loads(sidecar(directory).read_text())
        assert recorded["sha256"] == model_store.file_sha256(model_store.pipeline_path(directory))
        assert recorded["scikit_learn"] == run_metadata.environment()["scikit_learn"]
        assert recorded["model"] == "logistic_regression" and recorded["run_id"]

    def test_the_bundle_carries_it(self, trained):
        sid, directory = trained
        bundle = client.get(f"/api/export/{sid}/zip")
        with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
            assert json.loads(archive.read(model_store.MODEL_SIDECAR_FILENAME)) == json.loads(sidecar(directory).read_text())


class TestLoading:
    def test_an_untouched_model_loads_and_scores(self, trained):
        sid, directory = trained
        pipeline, notes = model_store.load_pipeline(directory)
        assert pipeline is not None and notes == []
        assert apply_rows(sid).status_code == 200

    def test_a_modified_file_is_refused_with_a_clear_reason(self, trained):
        sid, directory = trained
        path = model_store.pipeline_path(directory)
        path.write_bytes(path.read_bytes() + b"tampered")
        with pytest.raises(model_store.ModelArtifactError) as caught:
            model_store.load_pipeline(directory)
        assert caught.value.code == "model_changed"
        response = apply_rows(sid)
        assert response.status_code == 409 and response.json()["code"] == "model_changed"
        assert "checksum does not match" in response.json()["detail"]

    def test_a_different_scikit_learn_minor_version_is_refused(self, trained, monkeypatch):
        sid, directory = trained
        recorded = json.loads(sidecar(directory).read_text())
        sidecar(directory).write_text(json.dumps({**recorded, "scikit_learn": "0.24.2"}))
        response = apply_rows(sid)
        assert response.status_code == 409 and response.json()["code"] == "model_version_mismatch"
        assert "saved with scikit-learn 0.24.2" in response.json()["detail"]

    def test_a_bug_fix_release_only_gets_a_note(self, trained):
        _, directory = trained
        here = run_metadata.environment()["scikit_learn"]
        major, minor = here.split(".")[:2]
        recorded = json.loads(sidecar(directory).read_text())
        sidecar(directory).write_text(json.dumps({**recorded, "scikit_learn": f"{major}.{minor}.999"}))
        pipeline, notes = model_store.load_pipeline(directory)
        assert pipeline is not None and "bug-fix release" in notes[0]

    def test_a_model_from_before_sidecars_loads_with_a_note(self, trained):
        _, directory = trained
        sidecar(directory).unlink()
        pipeline, notes = model_store.load_pipeline(directory)
        assert pipeline is not None and "before checksums were kept" in notes[0]

    def test_an_unreadable_sidecar_is_refused(self, trained):
        _, directory = trained
        sidecar(directory).write_text("{not json")
        with pytest.raises(model_store.ModelArtifactError):
            model_store.load_pipeline(directory)

    def test_no_stored_model_is_none(self, tmp_path):
        assert model_store.load_pipeline(tmp_path) == (None, [])


class TestEveryLoadGoesThroughTheCheck:
    def test_nothing_in_the_app_unpickles_a_model_except_load_pipeline(self):
        app_dir = pathlib.Path(__file__).resolve().parents[2] / "app"
        offenders = []
        for path in app_dir.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r"joblib\.load\b|pickle\.loads?\(", text):
                line = text[: match.start()].count("\n") + 1
                source_line = text.splitlines()[line - 1]
                # Script templates (train_model.py, predict.py) are text written for the user -- predict.py checks
                # model.json itself -- not the app loading a model.
                if path.name in {"model_export.py", "predict_export.py", "model_store.py"}:
                    continue
                offenders.append(f"{path.name}:{line}: {source_line.strip()}")
        assert offenders == []
        import inspect

        # Inside the store, every unpickling is preceded by the check in the same function.
        for loader in (model_store.load_pipeline, model_store.load_run_pipeline):
            source = inspect.getsource(loader)
            assert source.index("check_pipeline(path)") < source.index("joblib.load(path)")
        store = (app_dir / "services" / "model_store.py").read_text(encoding="utf-8")
        assert store.count("joblib.load(") == 2


class TestTheCardSnippet:
    def test_the_snippet_says_to_check_the_sidecar(self):
        result = modeling.train_candidates(
            frame(), target_column="churned", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False,
        )
        snippet = model_export.build_prediction_snippet(result)
        assert "model.json beside model.joblib records the file's SHA-256" in snippet
        assert "CostWeighted" not in snippet

    def test_a_cost_weighted_model_says_to_copy_its_class(self):
        weights = {"positive_label": "yes", "cost_of_missing": 5.0, "cost_of_false_alarm": 1.0}
        result = modeling.train_candidates(
            frame(), target_column="churned", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False, model_params={"logistic_regression": {"cost_weighting": weights}},
        )
        assert "contains a CostWeighted step" in model_export.build_prediction_snippet(result)
