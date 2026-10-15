"""Every run saves a schema of what its model expects as input, built from the training rows only."""

from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import input_schema, model_store, modeling
from app.services.session_store import session_dir

client = TestClient(app)


def frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    return pd.DataFrame({
        "x": x,
        "plan": rng.choice(["basic", "pro", "team"], n),
        "spend": np.where(rng.random(n) < 0.1, np.nan, rng.normal(50, 5, n)),
        "joined": pd.date_range("2022-01-01", periods=n, freq="D").astype(str),
        "outcome": np.where(x + rng.normal(0, 1, n) > 0, "yes", "no"),
    })


def train(df=None, **kwargs) -> dict:
    return modeling.train_candidates(
        frame() if df is None else df, target_column="outcome", task="classification",
        model_names=["logistic_regression"], include_deep_diagnostics=False, **kwargs,
    )


class TestTheSchema:
    def test_every_run_has_one_with_its_columns_and_target(self):
        schema = train()["input_schema"]
        assert schema["schema_version"] == input_schema.SCHEMA_VERSION and schema["built_from"] == "training rows"
        by_name = {c["name"]: c for c in schema["columns"]}
        assert by_name["x"]["kind"] == "numeric" and by_name["x"]["range"]["min"] < by_name["x"]["range"]["max"]
        assert set(by_name["x"]["quantiles"]) == {"0.01", "0.25", "0.5", "0.75", "0.99"}
        assert by_name["plan"]["kind"] == "categorical" and set(by_name["plan"]["categories"]) == {"basic", "pro", "team"}
        assert by_name["spend"]["nullable"] is True and 0 < by_name["spend"]["blank_share"] < 0.3
        assert schema["target"] == {"name": "outcome", "task": "classification",
                                    "labels": schema["target"]["labels"]}
        assert set(schema["target"]["labels"]) == {"no", "yes"}

    def test_it_counts_the_training_rows_only(self):
        result = train()
        assert result["input_schema"]["rows"] == result["split"]["train_rows"]
        assert sum(result["input_schema"]["target"]["labels"].values()) == result["split"]["train_rows"]

    def test_a_column_only_a_derived_input_reads_is_included(self):
        result = train(feature_recipes_wanted=["date_parts:joined"])
        derived = result["features"]["derived"]
        assert derived, "the date recipe should be accepted on this data"
        joined = next(c for c in result["input_schema"]["columns"] if c["name"] == "joined")
        assert joined["kind"] == "date" and joined["used_directly"] is False
        assert joined["read_by"] == [derived[0]["id"]] and joined["range"]["min"].startswith("2022-01")

    def test_both_halves_of_an_interaction_are_sources(self):
        recipe = {"id": "interaction:product:x x spend", "kind": "interaction", "column": "x",
                  "outputs": ["x__x__spend"], "evidence": {"op": "product", "a": "x", "b": "spend"}}
        df = frame()
        schema = input_schema.build(df, {"numeric": ["x__x__spend"], "categorical": [], "frequent": [],
                                         "derived": [recipe]}, "outcome", "classification", df["outcome"])
        assert {c["name"] for c in schema["columns"]} == {"x", "spend"}
        assert all(c["read_by"] == [recipe["id"]] and not c["used_directly"] for c in schema["columns"])

    def test_categories_are_capped_and_say_so(self):
        df = pd.DataFrame({"code": [f"c{i % 80}" for i in range(400)], "outcome": ["a", "b"] * 200})
        schema = input_schema.build(df, {"numeric": [], "categorical": ["code"], "frequent": [], "derived": []},
                                    "outcome", "classification", df["outcome"])
        code = schema["columns"][0]
        assert len(code["categories"]) == input_schema.MAX_CATEGORIES and code["categories_truncated"] is True
        assert code["distinct"] == 80


class TestWhereItGoes:
    def test_it_is_saved_in_the_run_folder(self, tmp_path):
        result = train()
        model_store.save_result(tmp_path, result)
        saved = json.loads((model_store.run_dir(tmp_path, result["run_id"]) / model_store.SCHEMA_FILENAME).read_text())
        assert saved == result["input_schema"]

    def test_the_bundle_carries_it(self, monkeypatch):
        class FakeGraph:
            async def aget_state(self, _config):
                return SimpleNamespace(values={"action_log": []}, next=())

        monkeypatch.setattr(app.state, "graph", FakeGraph(), raising=False)
        buffer = io.StringIO()
        frame().to_csv(buffer, index=False)
        sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
        goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
        assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
        client.post(f"/api/model/{sid}/train", json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False})
        with zipfile.ZipFile(io.BytesIO(client.get(f"/api/export/{sid}/zip").content)) as archive:
            schema = json.loads(archive.read(model_store.SCHEMA_FILENAME))
        assert schema == model_store.load_result(session_dir(get_settings().upload_dir, sid))["input_schema"]

    def test_the_card_documents_it(self):
        from app.services.model_card import build_model_card

        card = build_model_card(session_id="s", file_name="f.csv", result=train(), summary="A.", interpretation="B.")
        assert "## What it expects as input" in card and "| plan | categorical |" in card
