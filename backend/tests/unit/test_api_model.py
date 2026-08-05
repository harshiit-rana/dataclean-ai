"""The model-building endpoints: plan, train, result.

The plan endpoint carries most of the weight here. The interesting decisions in a model build
happen before any fitting -- which columns count as inputs, how the data is split -- and the
point of the endpoint is that those arrive as visible proposals with reasons rather than as
defaults nobody saw.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import model_store
from app.services.session_store import session_dir

client = TestClient(app)


def _learnable_csv(n: int = 300, *, with_leak: bool = False, with_dates: bool = False) -> bytes:
    rng = np.random.default_rng(17)
    signal = rng.normal(0, 1, n)
    probability = 1 / (1 + np.exp(-2.4 * signal))
    outcome = np.where(rng.random(n) < probability, "yes", "no")

    data = {
        "employee_id": [f"E{i:05d}" for i in range(n)],
        "signal": signal.round(4),
        "noise": rng.normal(0, 1, n).round(4),
        "region": rng.choice(["north", "south"], n),
        "outcome": outcome,
    }
    if with_leak:
        # Varied values, not a single constant: a column whose *non-null* values are all
        # identical gets dropped as uninformative, which is a different code path and would
        # let this test pass without the exclusion mechanism working at all.
        months = ["january", "february", "march", "april"]
        data["exit_note"] = [
            f"left in {months[i % 4]}" if o == "yes" else None for i, o in enumerate(outcome)
        ]
    if with_dates:
        data["joined"] = pd.date_range("2021-01-01", periods=n, freq="D").astype(str)

    buffer = io.StringIO()
    pd.DataFrame(data).to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def _session(payload: bytes | None = None, **goal_overrides) -> str:
    payload = payload if payload is not None else _learnable_csv()
    response = client.post(
        "/api/upload", files={"file": ("data.csv", io.BytesIO(payload), "text/csv")}
    )
    session_id = response.json()["session_id"]

    goal = {
        "use_case": "ml_training",
        "expected_output": "predict who leaves",
        "target_column": "outcome",
        "task_type": "auto",
        **goal_overrides,
    }
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


class TestPlan:
    def test_describes_what_would_be_built(self):
        plan = client.get(f"/api/model/{_session()}/plan").json()
        assert plan["target_column"] == "outcome"
        assert plan["task_type"] == "classification"
        assert plan["row_count"] == 300
        assert not plan["trained"]

    def test_offers_only_models_that_fit_the_task(self):
        plan = client.get(f"/api/model/{_session()}/plan").json()
        names = {m["name"] for m in plan["candidate_models"]}
        assert "logistic_regression" in names
        assert "linear_regression" not in names  # a regressor can't predict a label

    def test_suggests_dropping_a_leaky_column_with_the_reason(self):
        session_id = _session(_learnable_csv(with_leak=True))
        plan = client.get(f"/api/model/{session_id}/plan").json()
        leaks = [s for s in plan["suggested_exclusions"] if s["column"] == "exit_note"]
        assert leaks and leaks[0]["kind"] == "leakage"
        assert "exit_note" in plan["excluded_columns"]  # pre-ticked, not silently applied

    def test_suggests_dropping_an_identifier(self):
        plan = client.get(f"/api/model/{_session()}/plan").json()
        identifiers = [s for s in plan["suggested_exclusions"] if s["column"] == "employee_id"]
        assert identifiers and identifiers[0]["kind"] == "identifier"

    def test_a_measurement_is_not_an_identifier_just_because_it_is_unique(self):
        """Found live: 809 employees have 809 different salaries, so the uniqueness rule
        called salary an identifier and pre-ticked the most informative column in the file
        for removal. A continuous measurement is *naturally* near-unique."""
        plan = client.get(f"/api/model/{_session()}/plan").json()
        excluded = set(plan["excluded_columns"])
        assert "signal" not in excluded
        assert "noise" not in excluded

    def test_a_whole_number_measurement_is_not_an_identifier_either(self):
        """Salaries in whole pounds are integers and near-unique, but scattered across a
        range far wider than the row count -- unlike a key, which runs 1001, 1002, 1003."""
        rng = np.random.default_rng(3)
        n = 300
        salary = rng.integers(30_000, 120_000, n)
        # Related to the target but nowhere near determining it, or it would be excluded as
        # leakage and the test would pass without the identifier rule doing anything.
        probability = 1 / (1 + np.exp(-(salary - 75_000) / 25_000))
        buffer = io.StringIO()
        pd.DataFrame(
            {
                "staff_number": range(5000, 5000 + n),  # a real key: dense and sequential
                "salary": salary,
                "outcome": np.where(rng.random(n) < probability, "yes", "no"),
            }
        ).to_csv(buffer, index=False)

        session_id = _session(buffer.getvalue().encode())
        plan = client.get(f"/api/model/{session_id}/plan").json()
        excluded = set(plan["excluded_columns"])
        assert "salary" not in excluded
        assert "staff_number" in excluded

    def test_suggestions_are_proposals_the_user_can_reject(self):
        """A leaky column is sometimes legitimately available at prediction time, and only
        the person who owns the data knows that. So training must accept fewer exclusions
        than were suggested."""
        session_id = _session(_learnable_csv(with_leak=True))
        result = client.post(
            f"/api/model/{session_id}/train",
            json={"excluded_columns": [], "model_names": ["logistic_regression"],
                  "include_deep_diagnostics": False},
        ).json()
        used = result["features"]["numeric"] + result["features"]["categorical"]
        assert "exit_note" in used

    def test_finds_date_columns_for_a_time_split(self):
        session_id = _session(_learnable_csv(with_dates=True))
        plan = client.get(f"/api/model/{session_id}/plan").json()
        assert "joined" in plan["date_columns"]
        assert {o["value"] for o in plan["split_options"]} == {"random", "time"}

    def test_a_session_without_a_target_is_refused_with_an_explanation(self):
        session_id = _session(target_column=None)
        response = client.get(f"/api/model/{session_id}/plan")
        assert response.status_code == 409
        assert "target" in response.json()["detail"].lower()

    def test_unknown_session_is_404(self):
        assert client.get("/api/model/not-a-session/plan").status_code == 404


class TestTraining:
    def test_trains_and_returns_metrics(self):
        result = client.post(
            f"/api/model/{_session()}/train",
            json={"excluded_columns": ["employee_id"], "include_deep_diagnostics": False},
        ).json()
        assert result["best_model"]
        best = next(r for r in result["results"] if r["model"] == result["best_model"])
        assert best["metrics"]["accuracy"] > 0.7  # the signal is genuinely there
        assert result["assessment"]["verdict"] == "useful"

    def test_response_carries_the_diagnostics(self):
        result = client.post(
            f"/api/model/{_session()}/train",
            json={"excluded_columns": ["employee_id"], "model_names": ["logistic_regression"],
                  "include_deep_diagnostics": False},
        ).json()
        charts = result["results"][0]["diagnostics"]
        assert charts["roc_curve"] and charts["confusion_cells"] and charts["thresholds"]

    def test_fitted_estimators_never_reach_the_response(self):
        """A sklearn Pipeline isn't JSON-serializable, and shipping one would be a mistake
        even if it were."""
        result = client.post(
            f"/api/model/{_session()}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False},
        ).json()
        assert all("pipeline" not in entry for entry in result["results"])

    def test_defaults_to_every_model_for_the_task(self):
        result = client.post(
            f"/api/model/{_session()}/train",
            json={"excluded_columns": ["employee_id"], "include_deep_diagnostics": False},
        ).json()
        assert len(result["results"]) == 3

    def test_excluding_everything_is_a_422_not_a_500(self):
        response = client.post(
            f"/api/model/{_session()}/train",
            json={"excluded_columns": ["employee_id", "signal", "noise", "region"]},
        )
        assert response.status_code == 422
        assert "no usable columns" in response.json()["detail"]

    def test_too_little_data_is_refused_with_a_readable_reason(self):
        response = client.post(f"/api/model/{_session(_learnable_csv(30))}/train", json={})
        assert response.status_code == 422
        assert "too few" in response.json()["detail"]

    def test_a_time_split_uses_the_named_column(self):
        session_id = _session(_learnable_csv(with_dates=True))
        result = client.post(
            f"/api/model/{session_id}/train",
            json={"excluded_columns": ["employee_id"], "split_strategy": "time",
                  "time_column": "joined", "model_names": ["logistic_regression"],
                  "include_deep_diagnostics": False},
        ).json()
        assert result["split"]["strategy"] == "time"
        assert result["split"]["time_column"] == "joined"

    def test_a_time_split_without_a_column_is_refused(self):
        response = client.post(
            f"/api/model/{_session()}/train", json={"split_strategy": "time"}
        )
        assert response.status_code == 422
        assert "date column" in response.json()["detail"]


class TestStoredResult:
    def test_result_survives_a_refresh(self):
        session_id = _session()
        client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False},
        )
        stored = client.get(f"/api/model/{session_id}/result").json()
        assert stored["best_model"] == "logistic_regression"

    def test_plan_reports_that_a_model_exists(self):
        session_id = _session()
        assert client.get(f"/api/model/{session_id}/plan").json()["trained"] is False
        client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False},
        )
        assert client.get(f"/api/model/{session_id}/plan").json()["trained"] is True

    def test_plan_remembers_the_last_exclusions(self):
        session_id = _session()
        client.post(
            f"/api/model/{session_id}/train",
            json={"excluded_columns": ["noise"], "model_names": ["logistic_regression"],
                  "include_deep_diagnostics": False},
        )
        assert client.get(f"/api/model/{session_id}/plan").json()["excluded_columns"] == ["noise"]

    def test_asking_for_a_result_before_training_is_404(self):
        assert client.get(f"/api/model/{_session()}/result").status_code == 404

    def test_the_fitted_pipeline_is_written_to_disk(self):
        session_id = _session()
        client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False},
        )
        directory = session_dir(get_settings().upload_dir, session_id)
        assert model_store.pipeline_path(directory) is not None

    def test_changing_the_data_discards_the_stored_model(self):
        """An evaluation describes the rows it was tested on. Once those change, showing the
        old score next to the new data is worse than showing nothing."""
        session_id = _session()
        directory = session_dir(get_settings().upload_dir, session_id)
        client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False},
        )
        assert model_store.load_result(directory) is not None

        model_store.clear(directory)
        assert model_store.load_result(directory) is None
        assert client.get(f"/api/model/{session_id}/result").status_code == 404


class TestCorrelations:
    def test_available_before_any_training(self):
        """It describes the data, not a model, and it's useful while choosing exclusions."""
        result = client.get(f"/api/model/{_session()}/correlations").json()
        assert set(result["columns"]) == {"signal", "noise"}
        assert result["cells"]
