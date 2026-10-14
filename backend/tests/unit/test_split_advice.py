"""A recommended way to divide the rows, with the reason, that the user can overrule.

The form defaulted to a random split even when the app had already found a column repeating across
rows (which needs a grouped split) or a clear time order (which usually needs a date split). Anyone who
does not understand splitting takes the default and gets a score the model cannot repeat. The rules are
a pure function of the file's shape and the stated goal, so they are tested here without the API.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import routes_model
from app.main import app
from app.services import model_store, validation_advice as advice

client = TestClient(app)


def entity_file(customers: int = 240, per: int = 5, seed: int = 0, column: str = "customer_id") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = customers * per
    return pd.DataFrame(
        {column: np.repeat([f"c{i:04d}" for i in range(customers)], per), "x": rng.normal(size=n),
         "y": rng.choice(["yes", "no"], n)}
    )


def drifting_file(n: int = 600, seed: int = 1, column: str = "created_on", start: float = 0.05, end: float = 0.6) -> pd.DataFrame:
    """The answer becomes more common as time passes."""
    rng = np.random.default_rng(seed)
    rate = np.linspace(start, end, n)
    return pd.DataFrame(
        {column: pd.date_range("2021-01-01", periods=n).astype(str), "x": rng.normal(size=n),
         "y": np.where(rng.random(n) < rate, "yes", "no")}
    )


def steady_file(n: int = 600, seed: int = 2, column: str = "created_on") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {column: pd.date_range("2021-01-01", periods=n).astype(str), "x": rng.normal(size=n),
         "y": rng.choice(["yes", "no"], n)}
    )


def recommend(df, goal="", **kwargs):
    target = "y"
    groups = routes_model._group_columns(df, target)
    dates = routes_model._date_columns(df)
    return advice.recommend_split(
        df, target, "classification", group_columns=groups, date_columns=dates, goal_text=goal,
        training_rows=kwargs.pop("training_rows", df), **kwargs,
    )


class TestRepeatedEntities:
    def test_a_repeated_entity_gets_a_grouped_recommendation_naming_the_column_and_the_evidence(self):
        found = recommend(entity_file())
        assert found["strategy"] == "group" and found["column"] == "customer_id"
        reason = found["reasons"][0]
        assert "'customer_id' repeats in 1,200 of 1,200 rows" in reason
        assert "240 distinct values" in reason and "about 5.0 rows each" in reason
        assert "same customer_id on both sides" in reason

    def test_the_evidence_counts_only_the_rows_whose_value_repeats(self):
        df = entity_file(customers=100, per=4)
        lonely = pd.DataFrame({"customer_id": [f"solo{i}" for i in range(40)], "x": 0.0, "y": "no"})
        found = recommend(pd.concat([df, lonely], ignore_index=True))
        assert "repeats in 400 of 440 rows" in found["reasons"][0]

    def test_an_identifier_shaped_name_is_high_confidence(self):
        assert recommend(entity_file(column="patient_id"))["confidence"] == "high"

    def test_a_plain_name_is_medium_and_gets_the_caution_about_categories(self):
        found = recommend(entity_file(column="plan_tier"))
        assert found["confidence"] == "medium"
        assert any("Check that 'plan_tier' really is one thing" in r for r in found["reasons"])

    def test_it_picks_the_most_repeated_column_and_offers_the_others(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame(
            {"customer_id": np.repeat([f"c{i}" for i in range(100)], 8),
             "store": np.repeat([f"s{i}" for i in range(50)], 16),
             "x": rng.normal(size=800), "y": rng.choice(["yes", "no"], 800)}
        )
        found = recommend(df)
        assert found["column"] == "store"
        assert {"strategy": "group", "column": "customer_id"} in found["alternatives"]

    def test_it_reuses_the_readiness_panels_test_so_a_region_is_not_an_entity(self):
        rng = np.random.default_rng(4)
        df = pd.DataFrame({"region": rng.choice(["n", "s", "e", "w"], 800), "x": rng.normal(size=800), "y": rng.choice(["yes", "no"], 800)})
        assert recommend(df)["strategy"] == "random"

    def test_a_measurement_that_happens_to_repeat_is_not_an_entity(self):
        rng = np.random.default_rng(5)
        df = pd.DataFrame({"days_since_login": rng.integers(0, 60, 900), "x": rng.normal(size=900), "y": rng.choice(["yes", "no"], 900)})
        assert recommend(df)["strategy"] == "random"

    def test_entities_win_over_dates_and_it_says_why(self):
        df = entity_file().assign(created_on=pd.date_range("2021-01-01", periods=1200).astype(str))
        found = recommend(df, goal="forecast next month")
        assert found["strategy"] == "group"
        assert any("divide on only one thing at a time" in r for r in found["reasons"])


class TestTimeOrder:
    def test_a_stated_interest_in_the_future_with_a_date_column_recommends_a_date_split(self):
        found = recommend(steady_file(), goal="Forecast which accounts will cancel next month")
        assert found["strategy"] == "time" and found["column"] == "created_on"
        assert "'forecast'" in found["reasons"][0].lower() and found["confidence"] == "medium"

    def test_an_answer_that_drifts_over_time_recommends_one_even_without_the_words(self):
        found = recommend(drifting_file())
        assert found["strategy"] == "time"
        assert any("changes over time" in r and "'yes' is" in r for r in found["reasons"])

    def test_both_together_is_high_confidence(self):
        assert recommend(drifting_file(), goal="a forecast")["confidence"] == "high"

    def test_a_date_column_with_nothing_else_does_not_trigger_a_date_split(self):
        found = recommend(steady_file(), goal="Which customers will churn")
        assert found["strategy"] == "random"
        assert any("nothing says the order matters" in r for r in found["reasons"])
        assert {"strategy": "time", "column": "created_on"} in found["alternatives"]

    def test_a_date_of_birth_is_never_used_to_order_the_rows(self):
        df = steady_file(column="date_of_birth")
        found = recommend(df, goal="forecast the future")
        assert found["strategy"] == "random"
        assert any("date of birth or a deadline" in r for r in found["reasons"])

    def test_a_deadline_is_skipped_too(self):
        assert recommend(steady_file(column="renewal_date"), goal="forecast")["strategy"] == "random"

    def test_an_event_time_beats_a_birth_date_when_both_are_there(self):
        df = steady_file().assign(date_of_birth=pd.date_range("1950-01-01", periods=600, freq="W").astype(str))
        found = recommend(df, goal="forecast")
        assert found["strategy"] == "time" and found["column"] == "created_on"

    def test_with_two_date_columns_the_one_the_answer_drifts_along_is_chosen(self):
        df = drifting_file(column="shipped_on").assign(noise_date=lambda d: d["shipped_on"].sample(frac=1, random_state=0).to_numpy())
        found = recommend(df)
        assert found["strategy"] == "time" and found["column"] == "shipped_on"

    def test_other_phrasings_of_the_future_count(self):
        for goal in ("predict next quarter's churn", "going forward we want to flag risk", "a time series of sales"):
            assert recommend(steady_file(), goal=goal)["strategy"] == "time", goal

    def test_ordinary_goals_do_not_(self):
        for goal in ("Find customers likely to churn", "Classify each ticket", "Predict the price of a house"):
            assert recommend(steady_file(), goal=goal)["strategy"] == "random", goal


class TestNeither:
    def test_a_plain_file_gets_random_with_the_reason(self):
        found = recommend(pd.DataFrame({"x": np.arange(300.0), "y": ["yes", "no"] * 150}))
        assert found["strategy"] == "random" and found["column"] is None
        assert "no repeated entities or time order found" in found["reasons"][0].lower()
        assert any("stratified" in r for r in found["reasons"])

    def test_a_regression_is_not_said_to_be_stratified(self):
        df = pd.DataFrame({"x": np.arange(300.0), "y": np.arange(300.0) * 2})
        found = advice.recommend_split(df, "y", "regression", group_columns=[], date_columns=[])
        assert found["strategy"] == "random" and not any("stratified" in r for r in found["reasons"])

    def test_the_confidence_is_medium_because_absence_of_evidence_is_not_proof(self):
        assert recommend(pd.DataFrame({"x": np.arange(300.0), "y": ["yes", "no"] * 150}))["confidence"] == "medium"


class TestTheDriftTest:
    def test_it_sees_a_real_trend_and_describes_it(self):
        found = advice.drift_over_time(drifting_file(), "created_on", "y", "classification")
        assert found["drifts"] is True and found["p"] < advice.DRIFT_P_VALUE
        assert "earliest third" in found["text"] and "latest" in found["text"]

    def test_it_sees_nothing_in_steady_data(self):
        assert advice.drift_over_time(steady_file(), "created_on", "y", "classification")["drifts"] is False

    def test_a_small_wobble_is_not_a_trend(self):
        assert advice.drift_over_time(drifting_file(start=0.28, end=0.32), "created_on", "y", "classification")["drifts"] is False

    def test_a_regression_trend_is_seen_too(self):
        rng = np.random.default_rng(6)
        n = 600
        df = pd.DataFrame({"when": pd.date_range("2021-01-01", periods=n).astype(str), "y": np.linspace(0, 10, n) + rng.normal(0, 1, n)})
        assert advice.drift_over_time(df, "when", "y", "regression")["drifts"] is True

    def test_too_few_rows_or_dates_give_no_answer(self):
        assert advice.drift_over_time(drifting_file(n=40), "created_on", "y", "classification") is None
        constant_date = drifting_file().assign(created_on="2021-01-01")
        assert advice.drift_over_time(constant_date, "created_on", "y", "classification") is None

    def test_it_reads_only_the_rows_it_is_given(self):
        """The caller hands it the training side of the default split, so the held-back rows never
        reach it. Give it a part of the file with no trend in it and it finds none."""
        df = drifting_file()
        assert advice.drift_over_time(df.iloc[:100], "created_on", "y", "classification")["drifts"] is False

    def test_rows_without_a_parseable_date_or_answer_are_ignored(self):
        df = drifting_file()
        df.loc[::7, "created_on"] = "not a date"
        df.loc[3::11, "y"] = None
        assert advice.drift_over_time(df, "created_on", "y", "classification")["drifts"] is True


def session(df: pd.DataFrame, expected_output: str = "x") -> str:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    session_id = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": expected_output, "target_column": "y", "task_type": "auto"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


BODY = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}


class TestOverTheApi:
    def test_the_plan_carries_the_recommendation(self):
        plan = client.get(f"/api/model/{session(entity_file())}/plan").json()
        rec = plan["recommended_split"]
        assert rec["strategy"] == "group" and rec["column"] == "customer_id" and rec["confidence"] == "high"
        assert rec["reasons"][0].startswith("'customer_id' repeats in 1,200 of 1,200 rows")

    def test_a_file_with_neither_gets_random(self):
        plan = client.get(f"/api/model/{session(pd.DataFrame({'x': np.arange(300.0), 'y': ['yes', 'no'] * 150}))}/plan").json()
        assert plan["recommended_split"]["strategy"] == "random"
        assert "no repeated entities or time order found" in plan["recommended_split"]["reasons"][0].lower()

    def test_the_goal_the_user_stated_reaches_the_rule(self):
        sid = session(steady_file(), expected_output="forecast which accounts cancel next month")
        assert client.get(f"/api/model/{sid}/plan").json()["recommended_split"]["strategy"] == "time"

    def test_following_it_is_recorded_as_not_overridden(self):
        sid = session(entity_file())
        body = client.post(f"/api/model/{sid}/train", json={**BODY, "split_strategy": "group", "group_column": "customer_id"}).json()
        assert body["split"]["overridden"] is False
        assert body["split"]["recommended"] == {"strategy": "group", "column": "customer_id", "confidence": "high"}

    def test_choosing_something_else_is_recorded_as_an_override(self):
        sid = session(entity_file())
        body = client.post(f"/api/model/{sid}/train", json=BODY).json()  # random, against the advice
        assert body["split"]["overridden"] is True

    def test_choosing_the_right_strategy_on_the_wrong_column_is_still_an_override(self):
        rng = np.random.default_rng(7)
        df = entity_file().assign(store=np.repeat([f"s{i}" for i in range(30)], 40)[:1200])
        sid = session(df)
        rec = client.get(f"/api/model/{sid}/plan").json()["recommended_split"]
        other = "customer_id" if rec["column"] != "customer_id" else "store"
        body = client.post(f"/api/model/{sid}/train", json={**BODY, "split_strategy": "group", "group_column": other}).json()
        assert body["split"]["overridden"] is True

    def test_the_run_history_says_so(self):
        sid = session(entity_file())
        client.post(f"/api/model/{sid}/train", json={**BODY, "split_strategy": "group", "group_column": "customer_id"})
        client.post(f"/api/model/{sid}/train", json=BODY)
        runs = client.get(f"/api/model/{sid}/history").json()["runs"]
        assert [r["split_overridden"] for r in runs] == [True, False]
        assert runs[0]["recommended_split"]["strategy"] == "group"

    def test_the_improvement_route_records_it_too(self):
        sid = session(entity_file(customers=120, per=8))
        body = client.post(f"/api/model/{sid}/improve", json={**BODY, "budget": 4}).json()
        assert body["split"]["overridden"] is True

    def test_advice_never_stops_a_plan_or_a_train(self, monkeypatch):
        def broken(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(advice, "recommend_split", broken)
        sid = session(entity_file())
        assert client.get(f"/api/model/{sid}/plan").json()["recommended_split"] is None
        trained = client.post(f"/api/model/{sid}/train", json=BODY)
        assert trained.status_code == 200 and "overridden" not in trained.json()["split"]

    def test_the_history_entry_for_an_old_run_has_neither_field_set(self):
        entry = model_store.summarise({"results": [], "split": {"strategy": "random"}, "run_id": "old"})
        assert entry["recommended_split"] is None and entry["split_overridden"] is None
