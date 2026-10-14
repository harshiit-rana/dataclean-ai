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
from app.services import ml_readiness, model_store, modeling
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


class TestLeakageIsMeasuredOnTrainingRows:
    """Whether a column gives away the answer is read from the answers, so it must not be
    read from the held-out ones. Flipping every held-out answer is the test the rest of the
    suite uses for this rule: if a decision moves, it was reading them."""

    @staticmethod
    def _held_out_rows(payload: bytes):
        df = pd.read_csv(io.BytesIO(payload))
        _, X_test, _, _ = modeling.split_data(df, "outcome", "classification")
        return df, X_test.index

    @staticmethod
    def _csv(df: pd.DataFrame) -> bytes:
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue().encode()

    def test_flipping_every_held_out_answer_changes_no_suggestion(self, monkeypatch):
        payload = _learnable_csv(with_leak=True)
        honest_df, test_index = self._held_out_rows(payload)
        flipped_df = honest_df.copy()
        flipped_df.loc[test_index, "outcome"] = flipped_df.loc[test_index, "outcome"].map(
            {"yes": "no", "no": "yes"}
        )

        # Flipping labels re-rolls a stratified split, which would compare two different
        # partitions. Pin the partition so the only thing that differs is the held-out answers.
        def fixed_partition(df, target, task, **_kwargs):
            labelled = df[df[target].notna()]
            train_index = labelled.index.difference(test_index)
            features = labelled.drop(columns=[target])
            return (
                features.loc[train_index],
                features.loc[test_index],
                labelled.loc[train_index, target],
                labelled.loc[test_index, target],
            )

        monkeypatch.setattr(modeling, "split_data", fixed_partition)

        honest = client.get(f"/api/model/{_session(payload)}/plan").json()
        rigged = client.get(f"/api/model/{_session(self._csv(flipped_df))}/plan").json()

        assert [s["column"] for s in honest["suggested_exclusions"]] == [
            s["column"] for s in rigged["suggested_exclusions"]
        ]
        assert "exit_note" in rigged["excluded_columns"]
        # The patch has to bite: measured on every row, the flipped file hides the leak. If
        # this stops being true the fixture no longer proves anything.
        assert ml_readiness.detect_leakage(flipped_df, "outcome", "classification") == []

    def test_a_column_that_leaks_on_the_training_rows_is_still_suggested(self):
        plan = client.get(f"/api/model/{_session(_learnable_csv(with_leak=True))}/plan").json()
        assert "exit_note" in [s["column"] for s in plan["suggested_exclusions"]]

    def test_the_plan_says_which_rows_the_check_read(self):
        plan = client.get(f"/api/model/{_session(_learnable_csv(with_leak=True))}/plan").json()
        assert "240 training rows" in plan["exclusions_basis"]
        assert "held-out rows are not read" in plan["exclusions_basis"]

    def test_a_file_too_small_to_split_falls_back_to_every_row_and_says_so(self):
        """Failing the plan would hide the reason the file cannot be trained on."""
        plan = client.get(f"/api/model/{_session(_learnable_csv(n=30, with_leak=True))}/plan").json()
        assert "every row" in plan["exclusions_basis"]
        assert "too small" in plan["exclusions_basis"]
        assert plan["blockers"] or plan["warnings"]


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


class TestRunHistoryOverTheApi:
    def test_retraining_no_longer_discards_the_previous_run(self):
        session_id = _session()
        client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False},
        )
        client.post(
            f"/api/model/{session_id}/train",
            json={
                "model_names": ["random_forest_classifier"],
                "excluded_columns": ["employee_id", "noise"],
                "include_deep_diagnostics": False,
            },
        )

        body = client.get(f"/api/model/{session_id}/history").json()
        assert len(body["runs"]) == 2
        assert body["runs"][0]["best_model"] == "random_forest_classifier"
        assert body["runs"][1]["best_model"] == "logistic_regression"
        assert body["runs"][0]["excluded_columns"] == ["employee_id", "noise"]

    def test_nothing_trained_yet_is_an_empty_list_not_an_error(self):
        body = client.get(f"/api/model/{_session()}/history").json()
        assert body["runs"] == []

    def test_a_run_on_the_current_data_matches_the_current_fingerprint(self):
        """This is what lets the client tell "worse than last time" from "different data"."""
        session_id = _session()
        client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False},
        )
        body = client.get(f"/api/model/{session_id}/history").json()
        assert body["current_fingerprint"]
        assert body["runs"][0]["data_fingerprint"] == body["current_fingerprint"]

    def test_the_history_carries_the_verdict_not_just_the_score(self):
        session_id = _session()
        client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False},
        )
        run = client.get(f"/api/model/{session_id}/history").json()["runs"][0]
        assert run["verdict"]
        assert run["primary_metric"] in run["metrics"]


class TestOperatingPointOverTheApi:
    def test_the_plan_names_the_answers(self):
        plan = client.get(f"/api/model/{_session()}/plan").json()
        assert plan["class_labels"] == ["no", "yes"]

    def test_a_stated_trade_off_moves_the_line(self):
        session_id = _session()
        response = client.post(
            f"/api/model/{session_id}/train",
            json={
                "model_names": ["logistic_regression"],
                "include_deep_diagnostics": False,
                "operating_point": {"objective": "cost", "cost_of_missing": 8, "cost_of_false_alarm": 1},
            },
        )
        assert response.status_code == 200, response.text
        point = response.json()["operating_point"]
        assert point["objective"] == "cost"
        assert point["threshold"] < 0.5
        assert point["observed"]["recall"] >= point["at_default"]["recall"]

    def test_a_trade_off_the_app_cannot_read_is_a_422(self):
        response = client.post(
            f"/api/model/{_session()}/train",
            json={"operating_point": {"objective": "min_recall"}},  # no rate
        )
        assert response.status_code == 422
        assert "target rate" in response.json()["detail"]

    def test_the_line_survives_the_improvement_loop(self):
        """Improve rebuilds the result through train_candidates; the cut-off has to ride
        along or improving a model silently resets the trade-off."""
        session_id = _session()
        response = client.post(
            f"/api/model/{session_id}/improve",
            json={
                "model_names": ["logistic_regression"],
                "include_deep_diagnostics": False,
                "budget": 4,
                "operating_point": {"objective": "min_recall", "target_rate": 0.85},
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["operating_point"]["objective"] == "min_recall"

    def test_the_history_records_where_the_line_sat(self):
        session_id = _session()
        client.post(
            f"/api/model/{session_id}/train",
            json={
                "model_names": ["logistic_regression"],
                "include_deep_diagnostics": False,
                "operating_point": {"objective": "f1"},
            },
        )
        runs = client.get(f"/api/model/{session_id}/history").json()["runs"]
        assert runs[0]["operating_point"]["objective"] == "f1"
        assert "observed" in runs[0]["operating_point"]


def _derivable_csv(n: int = 300) -> bytes:
    """A file whose date column is worth deriving from and whose text column is not an id."""
    rng = np.random.default_rng(23)
    words = ["the", "customer", "called", "about", "a", "billing", "issue", "again"]
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    buffer = io.StringIO()
    pd.DataFrame(
        {
            "signal": rng.normal(0, 1, n).round(4),
            "signup_date": dates.astype(str),
            "notes": [" ".join(rng.choice(words, size=int(rng.integers(6, 18)))) for _ in range(n)],
            "spend": np.where(rng.random(n) < 0.2, np.nan, rng.normal(50, 10, n).round(2)),
            "outcome": np.where(dates.weekday >= 5, "yes", "no"),
        }
    ).to_csv(buffer, index=False)
    return buffer.getvalue().encode()


class TestDerivedInputsOverTheApi:
    def test_the_plan_offers_them_with_the_evidence(self):
        session_id = _session(_derivable_csv())
        plan = client.get(f"/api/model/{session_id}/plan").json()
        offered = {f["id"]: f for f in plan["suggested_features"]}
        assert "date_parts:signup_date" in offered
        assert "text_length:notes" in offered
        assert "missing_flag:spend" in offered
        assert offered["date_parts:signup_date"]["evidence"]["span_days"] > 0
        assert "signup_date" in offered["date_parts:signup_date"]["reason"]

    def test_they_are_offered_but_off_until_chosen(self):
        """Offered with the reason attached, off by default. They were pre-ticked until the
        churn sample, where signup_date's parts duplicated a tenure column already in the
        file and made the default train worse. An extra input is a change; a change is the
        user's to opt into, and the loop's fourth round measures it either way."""
        session_id = _session(_derivable_csv())
        plan = client.get(f"/api/model/{session_id}/plan").json()
        assert plan["suggested_features"]
        assert plan["feature_recipes"] == []

        response = client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False,
                  "feature_recipes": []},
        )
        assert response.status_code == 200, response.text
        assert response.json()["features"]["derived"] == []
        assert not any("__" in c for c in response.json()["features"]["numeric"])

    def test_choosing_them_puts_them_in_the_model(self):
        session_id = _session(_derivable_csv())
        response = client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["random_forest_classifier"], "include_deep_diagnostics": False,
                  "feature_recipes": ["date_parts:signup_date"]},
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert "signup_date__weekday" in result["features"]["numeric"]
        assert [r["id"] for r in result["features"]["derived"]] == ["date_parts:signup_date"]
        # The target is the weekday, so this is the difference between a model and nothing.
        assert result["assessment"]["verdict"] == "useful"

    def test_the_plan_comes_back_showing_what_the_last_run_used(self):
        session_id = _session(_derivable_csv())
        client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False,
                  "feature_recipes": ["missing_flag:spend"]},
        )
        plan = client.get(f"/api/model/{session_id}/plan").json()
        assert plan["feature_recipes"] == ["missing_flag:spend"]

    def test_an_unknown_recipe_id_is_ignored_rather_than_fatal(self):
        """Ids are re-proposed from the data, not trusted. A stale id from an older plan is
        a reason to drop the recipe, not to fail a training run."""
        session_id = _session(_derivable_csv())
        response = client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False,
                  "feature_recipes": ["date_parts:a_column_that_does_not_exist"]},
        )
        assert response.status_code == 200
        assert response.json()["features"]["derived"] == []

    def test_the_history_records_which_derived_inputs_a_run_used(self):
        session_id = _session(_derivable_csv())
        client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False,
                  "feature_recipes": ["date_parts:signup_date"]},
        )
        runs = client.get(f"/api/model/{session_id}/history").json()["runs"]
        assert runs[0]["feature_recipes"] == ["date_parts:signup_date"]

    def test_a_date_column_is_not_pre_excluded_as_an_identifier(self):
        """The bug this whole feature exists to fix, caught at the plan. A date column is
        row-unique, so the identifier rule claimed it and pre-ticked it for removal -- which
        meant the recipe that would turn it into four usable inputs was never even offered.
        The RSNA run did the same to a column of radiology reports."""
        session_id = _session(_derivable_csv())
        plan = client.get(f"/api/model/{session_id}/plan").json()
        identified = {s["column"] for s in plan["suggested_exclusions"] if s["kind"] == "identifier"}
        assert "signup_date" not in identified
        assert "notes" not in identified
        assert "signup_date" not in plan["excluded_columns"]
        assert "notes" not in plan["excluded_columns"]

    def test_a_real_identifier_is_still_pre_excluded(self):
        """The rule still has to do its job -- this is a narrowing, not a removal."""
        plan = client.get(f"/api/model/{_session()}/plan").json()
        identified = {s["column"] for s in plan["suggested_exclusions"] if s["kind"] == "identifier"}
        assert "employee_id" in identified

    def test_the_improvement_loop_can_add_them_and_the_result_reflects_it(self):
        session_id = _session(_derivable_csv())
        response = client.post(
            f"/api/model/{session_id}/improve",
            json={"model_names": ["random_forest_classifier"], "include_deep_diagnostics": False,
                  "budget": 4, "feature_recipes": []},
        )
        assert response.status_code == 200, response.text
        result = response.json()
        round_4 = next(r for r in result["improvement"]["rounds"] if r["name"] == "derived_features")
        assert round_4["accepted"]
        # The rebuild has to be the configuration that won, not the one that was sent in.
        assert "signup_date__weekday" in result["features"]["numeric"]



class TestAKeptSettingSurvivesTheRebuild:
    """Found on the churn sample, the first data where the settings search kept anything.

    The route re-added the `model__` pipeline prefix before handing the settings to
    train_candidates, whose constructor rejected them, so "Improve it" failed outright. The
    existing tests passed because on their data the search never kept a setting -- the
    settings dict was empty and every check over it was vacuous. These force a kept setting
    so the path is actually exercised."""

    KEPT = {"C": 0.1, "class_weight": None}

    def _fake_improve(self, *args, **kwargs):
        return {
            "rounds": [
                {"name": "baseline", "label": "Starting point", "what_it_tried": "",
                 "score": 0.8, "std": 0.01, "gain": None, "accepted": True, "outcome": "",
                 "detail": {}},
                {"name": "hyperparameters", "label": "Model settings", "what_it_tried": "",
                 "score": 0.83, "std": 0.01, "gain": 0.03, "accepted": True, "outcome": "",
                 "detail": {}},
            ],
            "metric": "roc_auc",
            "goal": None,
            "stopped_because": "no_further_gains",
            "best": {
                "model": "logistic_regression",
                "label": "Logistic regression",
                "params": dict(self.KEPT),
                "cv_score": 0.83,
                "cv_std": 0.01,
                "feature_recipes": [],
            },
        }

    def test_improving_with_a_kept_setting_does_not_fail(self, monkeypatch):
        from app.services import improvement

        monkeypatch.setattr(improvement, "improve", self._fake_improve)
        response = client.post(
            f"/api/model/{_session()}/improve",
            json={"budget": 4, "include_deep_diagnostics": False},
        )
        assert response.status_code == 200, response.text
        chosen = response.json()["results"][0]
        assert chosen["failed"] is None
        assert chosen["tuned_params"] == self.KEPT

    def test_the_kept_setting_is_what_the_model_was_built_with(self, monkeypatch):
        from app.services import improvement, model_store

        monkeypatch.setattr(improvement, "improve", self._fake_improve)
        session_id = _session()
        client.post(
            f"/api/model/{session_id}/improve",
            json={"budget": 4, "include_deep_diagnostics": False},
        )
        import joblib

        directory = session_dir(get_settings().upload_dir, session_id)
        fitted = joblib.load(model_store.pipeline_path(directory))
        assert fitted.named_steps["model"].C == 0.1
        assert fitted.named_steps["model"].class_weight is None

    def test_the_kept_setting_reaches_the_script_as_a_plain_argument(self, monkeypatch):
        from app.services import improvement

        monkeypatch.setattr(improvement, "improve", self._fake_improve)
        session_id = _session()
        client.post(
            f"/api/model/{session_id}/improve",
            json={"budget": 4, "include_deep_diagnostics": False},
        )
        script = client.get(f"/api/model/{session_id}/script").text
        assert "C=0.1" in script
        assert "class_weight=None" in script
        assert "model__" not in script



def _people_csv(n: int = 300) -> bytes:
    rng = np.random.default_rng(29)
    signal = rng.normal(0, 1, n)
    buffer = io.StringIO()
    pd.DataFrame(
        {
            "First Name": rng.choice(["Ada", "Alan", "Grace", "Radia", "Tim"], n),
            "last_name": rng.choice(["Lovelace", "Turing", "Hopper", "Perlman"], n),
            "Email": [f"user{i % 150}@example.com" for i in range(n)],
            "product_name": rng.choice(["basic", "pro", "team"], n),
            "signal": signal.round(4),
            "outcome": np.where(rng.random(n) < 1 / (1 + np.exp(-2 * signal)), "yes", "no"),
        }
    ).to_csv(buffer, index=False)
    return buffer.getvalue().encode()


class TestPersonalDetailsAreNotInputs:
    """Found on the churn sample: `last_name` was the model's second most important input.
    With ~100 rows per surname its churn-rate spread was what chance produces, and the
    importance chart made chance look like a finding."""

    def _suggested(self) -> dict:
        plan = client.get(f"/api/model/{_session(_people_csv())}/plan").json()
        return plan, {s["column"]: s for s in plan["suggested_exclusions"]}

    def test_names_and_contact_details_are_suggested_for_exclusion(self):
        plan, suggested = self._suggested()
        for column in ("First Name", "last_name", "Email"):
            assert suggested[column]["kind"] == "personal", column
            assert column in plan["excluded_columns"]  # pre-ticked, not silently applied

    def test_the_reason_names_what_is_wrong_with_it(self):
        _plan, suggested = self._suggested()
        assert "who someone is" in suggested["last_name"]["reason"]
        assert "gender" in suggested["last_name"]["reason"]

    def test_a_thing_with_a_name_is_not_a_person(self):
        """`product_name` names a product. A closed list of words, not a guess at what
        sounds personal."""
        _plan, suggested = self._suggested()
        assert "product_name" not in suggested

    def test_the_user_can_keep_one_anyway(self):
        session_id = _session(_people_csv())
        response = client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False,
                  "excluded_columns": ["First Name", "Email"]},
        )
        assert response.status_code == 200, response.text
        assert "last_name" in response.json()["features"]["categorical"]
