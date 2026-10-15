"""Every run records what exactly produced it: software, seeds, request, configuration hash, runtime, data version.

A history entry used to say what the score was but not what made it, so two runs that differed in an unrecorded way
looked identical. These tests pin the record, the hash's two promises (equal for the same configuration, different
when anything in it changes) and that old entries without it still read.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services import model_store, run_metadata

client = TestClient(app)

FIELDS = ("environment", "seeds", "request", "config_hash", "runtime_seconds", "dataset_version", "selection")
VERSION = {"version": 2, "fingerprint": "abc", "rows": 300, "columns": 4, "at": "t", "event": "clean"}


def _csv(n: int = 300, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "z": rng.normal(size=n), "outcome": np.where(x + rng.normal(0, 1, n) > 0, "yes", "no")})
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def _session(payload: bytes | None = None) -> str:
    sid = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(payload or _csv()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    return sid


class TestTheRecord:
    def test_the_environment_names_python_and_the_libraries(self):
        env = run_metadata.environment()
        assert env["python"].count(".") == 2
        for key in ("scikit_learn", "pandas", "numpy", "joblib"):
            assert env[key], key

    def test_the_seeds_are_the_ones_in_the_code(self):
        from app.services import modeling, tuning

        seeds = run_metadata.seeds()
        assert seeds["split"] == modeling.RANDOM_STATE and seeds["confirmation"] == tuning.CONFIRMATION_SEED

    def test_describe_carries_every_field(self):
        result = {"selection": {"method": "cross_validation", "metric": "roc_auc", "rule": "highest_score",
                                "scores": {"a": {"score": 0.8, "std": 0.01, "folds": 5}}}}
        found = run_metadata.describe(request={"budget": 4}, route="train", dataset_version=VERSION,
                                      runtime_seconds=12.345, result=result)
        assert set(FIELDS) <= set(found)
        assert found["runtime_seconds"] == 12.35
        assert found["dataset_version"] == {"version": 2, "fingerprint": "abc", "rows": 300, "columns": 4, "event": "clean"}
        assert found["selection"] == {"method": "cross_validation", "metric": "roc_auc", "rule": "highest_score",
                                      "scores": {"a": 0.8}}


class TestTheHash:
    def test_the_same_configuration_on_the_same_data_hashes_the_same(self):
        one = run_metadata.config_hash({"budget": 4, "model_names": ["a"]}, VERSION, route="improve")
        two = run_metadata.config_hash({"model_names": ["a"], "budget": 4}, dict(VERSION), route="improve_stream")
        assert one == two and len(one) == 16

    def test_any_change_in_the_request_changes_it(self):
        base = run_metadata.config_hash({"budget": 4, "test_size": 0.2}, VERSION, route="improve")
        assert run_metadata.config_hash({"budget": 5, "test_size": 0.2}, VERSION, route="improve") != base
        assert run_metadata.config_hash({"budget": 4, "test_size": 0.3}, VERSION, route="improve") != base

    def test_different_data_or_a_different_route_changes_it(self):
        base = run_metadata.config_hash({}, VERSION, route="train")
        assert run_metadata.config_hash({}, {**VERSION, "fingerprint": "xyz"}, route="train") != base
        assert run_metadata.config_hash({}, VERSION, route="improve") != base

    def test_the_data_is_named_by_content_so_an_undo_hashes_like_the_state_it_restored(self):
        assert run_metadata.config_hash({}, VERSION, route="train") == run_metadata.config_hash(
            {}, {**VERSION, "version": 7, "event": "undo"}, route="train"
        )

    def test_a_changed_default_or_processing_limit_changes_it(self, monkeypatch):
        from app.services import modeling

        base = run_metadata.config_hash({}, VERSION, route="train")
        assert run_metadata.config_hash({}, VERSION, route="train", cost_limit=1000) != base
        monkeypatch.setattr(modeling, "DEFAULT_TEST_SIZE", 0.25)
        assert run_metadata.config_hash({}, VERSION, route="train") != base


class TestTheHistory:
    def test_an_old_entry_without_the_fields_still_summarises(self):
        entry = model_store.summarise({"results": [], "best_model": None})
        assert all(entry[key] is None for key in FIELDS)

    def test_a_train_records_everything(self):
        sid = _session()
        body = client.post(
            f"/api/model/{sid}/train", json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False}
        ).json()
        assert body["run_metadata"]["config_hash"]
        run = client.get(f"/api/model/{sid}/history").json()["runs"][0]
        for key in ("environment", "seeds", "request", "config_hash", "runtime_seconds", "dataset_version"):
            assert run[key] is not None, key
        assert run["request"]["model_names"] == ["logistic_regression"]
        assert run["runtime_seconds"] > 0
        assert run["dataset_version"]["rows"] == 300

    def test_two_identical_trains_share_a_hash_and_a_changed_request_does_not(self):
        sid = _session()
        request = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}
        client.post(f"/api/model/{sid}/train", json=request)
        client.post(f"/api/model/{sid}/train", json=request)
        client.post(f"/api/model/{sid}/train", json={**request, "test_size": 0.3})
        changed, second, first = client.get(f"/api/model/{sid}/history").json()["runs"][:3]
        assert first["config_hash"] == second["config_hash"] != changed["config_hash"]

    def test_an_improve_records_everything_including_how_it_chose(self):
        sid = _session()
        client.post(
            f"/api/model/{sid}/improve",
            json={"model_names": ["logistic_regression", "random_forest_classifier"], "include_deep_diagnostics": False,
                  "budget": 4},
        )
        run = client.get(f"/api/model/{sid}/history").json()["runs"][0]
        assert run["tuned"] is True
        for key in ("environment", "seeds", "request", "config_hash", "runtime_seconds", "dataset_version"):
            assert run[key] is not None, key
        assert run["request"]["budget"] == 4

    def test_metadata_that_cannot_be_gathered_never_fails_the_run(self, monkeypatch):
        def broken(**kwargs):
            raise RuntimeError("no metadata today")

        monkeypatch.setattr(run_metadata, "describe", broken)
        sid = _session()
        response = client.post(
            f"/api/model/{sid}/train", json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False}
        )
        assert response.status_code == 200 and response.json()["run_metadata"] is None
        assert client.get(f"/api/model/{sid}/history").json()["runs"][0]["config_hash"] is None
