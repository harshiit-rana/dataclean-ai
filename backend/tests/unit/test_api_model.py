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

    def test_improving_produces_an_ordinary_result(self):
        """The integration that matters: whatever the loop settles on goes back through the
        normal training path, so the charts, the model card, the joblib and the exported
        script all work on it without knowing it was tuned."""
        session_id = _session()
        result = client.post(
            f"/api/model/{session_id}/improve",
            json={"budget": 4, "include_deep_diagnostics": False},
        ).json()

        assert result["best_model"]
        assert result["results"][0]["metrics"]
        assert result["assessment"]["verdict"]
        assert result["improvement"]["rounds"][0]["name"] == "baseline"

    def test_the_tuned_settings_reach_the_exported_script(self):
        """A tuned model whose script quietly reverts to the defaults would report a score
        the script cannot reproduce -- which is the one promise the script makes."""
        session_id = _session()
        client.post(
            f"/api/model/{session_id}/improve",
            json={"budget": 8, "include_deep_diagnostics": False},
        )
        script = client.get(f"/api/model/{session_id}/script").text

        stored = client.get(f"/api/model/{session_id}/result").json()
        settings = stored["improvement"]["settings"]
        for key, value in settings.items():
            assert f"{key}=" in script, f"{key} is missing from the script"
            assert repr(value) in script

    def test_it_reports_when_a_goal_is_out_of_reach(self):
        """Rather than searching until something clears the bar by luck."""
        session_id = _session()
        result = client.post(
            f"/api/model/{session_id}/improve",
            json={"goal": 0.999, "budget": 4, "include_deep_diagnostics": False},
        ).json()
        assert result["improvement"]["stopped_because"] == "goal_out_of_reach"

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
        assert {o["value"] for o in plan["split_options"]} == {"random", "time", "group"}

    def test_a_session_without_a_target_is_refused_with_an_explanation(self):
        session_id = _session(target_column=None)
        response = client.get(f"/api/model/{session_id}/plan")
        assert response.status_code == 409
        assert "target" in response.json()["detail"].lower()

    def test_unknown_session_is_404(self):
        assert client.get("/api/model/not-a-session/plan").status_code == 404


class TestABlockedVerdictBinds:
    """Found live on the RSNA knee dataset: readiness returned "blocked" because 4,349 of
    4,407 rows had no label, and `/train` -- which never asked -- trained on the 58 that
    remained and reported the result as useful. A tool whose central claim is that it will say
    no cannot have a route that says no and then proceeds anyway."""

    @staticmethod
    def _mostly_unlabelled_csv(n: int = 400, labelled: int = 40) -> bytes:
        rng = np.random.default_rng(5)
        outcome = list(rng.choice(["yes", "no"], labelled)) + [None] * (n - labelled)
        buffer = io.StringIO()
        pd.DataFrame(
            {"signal": rng.normal(0, 1, n).round(4), "outcome": outcome}
        ).to_csv(buffer, index=False)
        return buffer.getvalue().encode()

    def test_training_is_refused_when_readiness_is_blocked(self):
        session_id = _session(self._mostly_unlabelled_csv())
        assert client.get(f"/api/readiness/{session_id}").json()["verdict"] == "blocked"

        response = client.post(
            f"/api/model/{session_id}/train", json={"include_deep_diagnostics": False}
        )
        assert response.status_code >= 400
        assert "nothing was trained" in response.text

    def test_the_refusal_repeats_the_blocking_reason(self):
        """"Cannot train" without the reason makes the user go hunting for a fact the app
        already has."""
        session_id = _session(self._mostly_unlabelled_csv())
        response = client.post(
            f"/api/model/{session_id}/train", json={"include_deep_diagnostics": False}
        )
        assert "outcome" in response.text

    def test_the_improvement_loop_is_refused_too(self):
        """The first version of this guard sat inline in `/train` only, which left `/improve`
        -- a route that ends by calling `train_candidates` itself -- as an unguarded way to
        reach the same result. A guard one of two doors enforces is not a guard."""
        session_id = _session(self._mostly_unlabelled_csv())
        response = client.post(f"/api/model/{session_id}/improve", json={})
        assert response.status_code >= 400
        assert "nothing was trained" in response.text

    def test_a_clean_dataset_is_still_trained(self):
        """The guard has to be narrow, or it becomes the thing that stops the app working."""
        response = client.post(
            f"/api/model/{_session()}/train",
            json={"excluded_columns": ["employee_id"], "include_deep_diagnostics": False},
        )
        assert response.status_code == 200


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


class TestGroupedSplitOverTheApi:
    """The readiness panel has been naming repeated-entity columns for a while and telling
    users the split could not act on them. These check that the plan now offers exactly those
    columns, and that choosing one reaches the split."""

    @staticmethod
    def _repeated_entities_csv(customers: int = 80, per: int = 12) -> bytes:
        rng = np.random.default_rng(7)
        buffer = io.StringIO()
        pd.DataFrame(
            {
                "customer_id": np.repeat([f"CUST-{i:03d}" for i in range(customers)], per),
                "account_age_days": np.repeat(
                    rng.integers(100, 9000, customers), per
                ).astype(float),
                "spend": rng.normal(50, 15, customers * per).round(2),
                "outcome": np.repeat(rng.choice(["yes", "no"], customers), per),
            }
        ).to_csv(buffer, index=False)
        return buffer.getvalue().encode()

    def test_the_plan_offers_the_repeating_column(self):
        plan = client.get(f"/api/model/{_session(self._repeated_entities_csv())}/plan").json()
        assert "customer_id" in plan["group_columns"]
        assert any(option["value"] == "group" for option in plan["split_options"])

    def test_a_file_with_no_repeated_entity_offers_nothing_to_group_on(self):
        """The option is still listed -- the form disables it and says why -- but proposing a
        column to group on when none repeats would be an invitation to a meaningless split."""
        assert client.get(f"/api/model/{_session()}/plan").json()["group_columns"] == []

    def test_choosing_it_reaches_the_split(self):
        session_id = _session(self._repeated_entities_csv())
        result = client.post(
            f"/api/model/{session_id}/train",
            json={
                "split_strategy": "group",
                "group_column": "customer_id",
                "model_names": ["logistic_regression"],
                "include_deep_diagnostics": False,
            },
        ).json()
        assert result["split"]["strategy"] == "group"
        assert result["split"]["group_column"] == "customer_id"
        assert result["split"]["train_groups"] + result["split"]["test_groups"] == 80

    def test_grouping_on_a_column_with_too_few_values_fails_with_the_reason(self):
        session_id = _session(self._repeated_entities_csv())
        response = client.post(
            f"/api/model/{session_id}/train",
            json={
                "split_strategy": "group",
                "group_column": "outcome",
                "include_deep_diagnostics": False,
            },
        )
        assert response.status_code == 422
        assert "too few" in response.text
