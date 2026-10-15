"""predict.py predicts on a new file exactly as the app does, from the export bundle alone.

The acceptance test: apply a new file in the app, run the bundle's predict.py on the same file in a separate,
isolated Python, and require the two predictions files to be identical byte for byte -- the cleaning (rows dropped
with the same reason), the schema check, the derived inputs, the model and the cut-off all included.
"""

from __future__ import annotations

import ast
import io
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import predict_export
from app.services.session_store import session_dir

client = TestClient(app)

LOG = [
    {"action_id": "a1", "operation": "fill_median", "column_name": "spend", "operation_params": {}, "status": "applied"},
    {"action_id": "a2", "operation": "drop_duplicate_rows", "column_name": "DATASET", "operation_params": {},
     "status": "applied"},
]


class FakeGraph:
    async def aget_state(self, _config):
        return SimpleNamespace(values={"action_log": LOG}, next=())


def training_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    signal = rng.normal(0, 1, n)
    return pd.DataFrame({
        "signal": signal.round(4),
        "spend": np.where(rng.random(n) < 0.1, np.nan, rng.normal(50, 10, n).round(2)),
        "plan": rng.choice(["basic", "pro"], n),
        "joined": pd.date_range("2022-01-01", periods=n, freq="D").astype(str),
        "churned": np.where(rng.random(n) < 1 / (1 + np.exp(-2.5 * signal)), "yes", "no"),
    })


def new_rows() -> pd.DataFrame:
    frame = training_frame(60).drop(columns=["churned"])
    frame.loc[3, "spend"] = np.nan          # filled with the training median
    frame.loc[59] = frame.loc[10]           # an exact copy, dropped with its reason
    frame.loc[7, "plan"] = "enterprise"     # a category never seen in training
    frame.loc[8, "signal"] = 40.0           # far outside the training range
    return frame


def csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


@pytest.fixture
def exported(monkeypatch, tmp_path):
    """A trained session, a new file applied in the app, and the bundle unpacked into a folder of its own."""
    monkeypatch.setattr(app.state, "graph", FakeGraph(), raising=False)
    original = training_frame()
    sid = client.post("/api/upload", files={"file": ("t.csv", io.BytesIO(csv_bytes(original)), "text/csv")}).json()["session_id"]
    directory = session_dir(get_settings().upload_dir, sid)
    cleaned = original.assign(spend=original.spend.fillna(original.spend.median())).drop_duplicates()
    cleaned.to_csv(directory / "cleaned.csv", index=False)
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "churned", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    trained = client.post(f"/api/model/{sid}/train", json={
        "model_names": ["logistic_regression"], "include_deep_diagnostics": False,
        "feature_recipes": ["date_parts:joined"],
        "operating_point": {"objective": "cost", "cost_of_missing": 4, "cost_of_false_alarm": 1, "positive_label": "yes"},
    })
    assert trained.status_code == 200, trained.text

    applied = client.post(f"/api/apply/{sid}", files={"file": ("new.csv", io.BytesIO(csv_bytes(new_rows())), "text/csv")})
    assert applied.status_code == 200, applied.text
    app_file = directory / "applied" / applied.json()["apply_id"] / "predictions.csv"

    bundle = client.get(f"/api/export/{sid}/zip")
    folder = tmp_path / "bundle"
    with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
        archive.extractall(folder)
    (folder / "new.csv").write_bytes(csv_bytes(new_rows()))
    return folder, app_file, applied.json()


def run_predict(folder, source="new.csv"):
    # -I: an isolated Python that cannot see the backend's own code; predict.py finds the copy beside it.
    return subprocess.run([sys.executable, "-I", "predict.py", source, "out.csv"], cwd=folder,
                          capture_output=True, text=True, timeout=300)


class TestAcceptance:
    def test_predict_py_writes_the_app_s_predictions_file_byte_for_byte(self, exported):
        folder, app_file, response = exported
        assert (folder / "predict.py").exists() and (folder / "app" / "services" / "replay.py").exists()
        done = run_predict(folder)
        assert done.returncode == 0, done.stdout + done.stderr
        assert (folder / "out.csv").read_bytes() == app_file.read_bytes()
        # The cases worth having on purpose, all inside that equality:
        out = pd.read_csv(folder / "out.csv", keep_default_na=False)
        assert out["dataclean_note"].tolist()[59] == "no prediction: it duplicated an earlier row"
        assert "plan: a value not seen in training" in out["dataclean_validation"][7]
        assert out["predicted_churned"][3] in {"yes", "no"}  # the blank was filled, the row was scored
        assert response["validation"]["rows_with_findings"] >= 2

    def test_a_file_missing_a_column_is_refused_with_the_app_s_message(self, exported):
        folder, _, _ = exported
        new_rows().drop(columns=["plan"]).to_csv(folder / "short.csv", index=False)
        done = run_predict(folder, "short.csv")
        assert done.returncode != 0
        assert "This file is missing columns the model needs, so nothing could be scored: plan." in done.stderr

    def test_a_changed_model_file_is_not_loaded(self, exported):
        folder, _, _ = exported
        model = folder / "model.joblib"
        model.write_bytes(model.read_bytes() + b"x")
        done = run_predict(folder)
        assert done.returncode != 0 and "checksum does not match" in done.stderr


class TestTheRuntime:
    def test_it_needs_only_pandas_numpy_scikit_learn_joblib_and_itself(self):
        allowed = {"pandas", "numpy", "sklearn", "joblib", "app"}
        stdlib = set(sys.stdlib_module_names)
        files = predict_export.runtime_files()
        files["predict.py"] = predict_export.build_predict_script(
            {"target_column": "y", "task": "classification", "features": {}}, [], [], file_name="f.csv"
        )
        for name, text in files.items():
            for node in ast.walk(ast.parse(text)):
                modules = (
                    [a.name for a in node.names] if isinstance(node, ast.Import)
                    else [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
                    else []
                )
                for module in modules:
                    top = module.split(".")[0]
                    assert top in allowed or top in stdlib or top == "__future__", f"{name} imports {module}"

    def test_the_shipped_modules_are_the_app_s_own_text(self):
        import inspect

        from app.services import replay

        assert predict_export.runtime_files()["app/services/replay.py"] == inspect.getsource(replay)
