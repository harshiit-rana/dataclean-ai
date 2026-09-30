"""The export bundle's training script.

train_model.py used to be written only as a side effect of generating the model card, so a
user who trained a model, skipped the write-up and pressed "Export everything" got an archive
without the one artifact the model path exists to produce. These pin the on-demand write.
"""

from __future__ import annotations

import ast
import io
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.routes_export import _ensure_training_script
from app.config import get_settings
from app.main import app
from app.services import model_store, modeling
from app.services.session_store import session_dir


def _trained(directory) -> dict:
    rng = np.random.default_rng(5)
    n = 200
    signal = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "signal": signal,
            "outcome": np.where(rng.random(n) < 1 / (1 + np.exp(-2 * signal)), "yes", "no"),
        }
    )
    result = modeling.train_candidates(
        df, target_column="outcome", task="classification",
        model_names=["logistic_regression"], include_deep_diagnostics=False,
    )
    result["features"]["excluded"] = []
    return model_store.save_result(directory, result)


def test_the_script_is_written_when_a_model_exists_and_the_card_was_never_generated(tmp_path):
    _trained(tmp_path)
    script = tmp_path / model_store.SCRIPT_FILENAME
    assert not script.exists()  # save_result clears it; only the card used to write it

    _ensure_training_script(tmp_path)

    assert script.exists()
    ast.parse(script.read_text(encoding="utf-8"))


def test_an_existing_script_is_left_alone(tmp_path):
    """The card writes the script too; the bundle must not overwrite what it wrote."""
    _trained(tmp_path)
    script = tmp_path / model_store.SCRIPT_FILENAME
    script.write_text("# written by the card\n", encoding="utf-8")

    _ensure_training_script(tmp_path)

    assert script.read_text(encoding="utf-8") == "# written by the card\n"


def test_no_model_means_no_script_and_no_error(tmp_path):
    _ensure_training_script(tmp_path)
    assert not (tmp_path / model_store.SCRIPT_FILENAME).exists()


def test_an_unreadable_result_does_not_sink_the_export(tmp_path):
    """The rest of the archive -- the cleaned CSV, the audit report -- is still worth
    serving when the script cannot be built."""
    (tmp_path / model_store.RESULT_FILENAME).write_text(
        '{"best_model": "wishful_thinking"}', encoding="utf-8"
    )
    _ensure_training_script(tmp_path)  # must not raise
    assert not (tmp_path / model_store.SCRIPT_FILENAME).exists()


# --- clean.py -------------------------------------------------------------------------------

client = TestClient(app)

LOG = [{"action_id": "a1", "operation": "fill_median", "column_name": "spend",
        "operation_params": {}, "status": "applied"}]


class _FakeGraph:
    async def aget_state(self, _config):
        return SimpleNamespace(values={"action_log": LOG}, next=())


@pytest.fixture
def cleaned_session():
    had = hasattr(app.state, "graph")
    previous = getattr(app.state, "graph", None)
    app.state.graph = _FakeGraph()
    payload = "\n".join(["spend,plan", "10,a", ",b", "30,a", ""]).encode()
    session_id = client.post(
        "/api/upload", files={"file": ("jan.csv", io.BytesIO(payload), "text/csv")}
    ).json()["session_id"]
    directory = session_dir(get_settings().upload_dir, session_id)
    (directory / "cleaned.csv").write_text(
        "\n".join(["spend,plan", "10.0,a", "20.0,b", "30.0,a", ""]), encoding="utf-8"
    )
    yield session_id
    if had:
        app.state.graph = previous
    else:
        del app.state.graph


def test_the_cleaning_script_can_be_downloaded(cleaned_session):
    response = client.get(f"/api/export/{cleaned_session}/clean-script")
    assert response.status_code == 200
    assert 'filename="clean.py"' in response.headers["content-disposition"]
    assert ".fillna(20.0)" in response.text  # the median of 10 and 30, written in
    assert "jan.csv" in response.text


def test_the_bundle_carries_the_cleaning_script(cleaned_session):
    response = client.get(f"/api/export/{cleaned_session}/zip")
    assert response.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
    assert "clean.py" in names
