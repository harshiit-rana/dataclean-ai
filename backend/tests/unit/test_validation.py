"""A new file checked against its model's input schema, with column-level and row-level findings."""

from __future__ import annotations

import io
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import validation

client = TestClient(app)

SCHEMA = {
    "schema_version": 1,
    "columns": [
        {"name": "amount", "kind": "numeric", "nullable": False, "range": {"min": 0.0, "max": 100.0}},
        {"name": "plan", "kind": "categorical", "nullable": True, "categories": {"basic": 10, "pro": 5},
         "categories_truncated": False},
        {"name": "joined", "kind": "date", "nullable": True},
        {"name": "code", "kind": "categorical", "nullable": True, "categories": {"a": 1}, "categories_truncated": True},
    ],
    "target": {"name": "churned", "task": "classification", "labels": {"no": 10, "yes": 5}},
}


def new_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "amount": ["10", "abc", "250", None, "50"],
        "plan": ["basic", "gold", "pro", "gold", None],
        "joined": ["2024-01-01", "2024-02-01", "not a date", None, "2024-03-01"],
        "code": ["zzz", "a", "b", "c", "d"],
        "extra": [1, 2, 3, 4, 5],
        "churned": ["no", "yes", "no", "no", "yes"],
    })


def finding(report: dict, column: str, kind: str) -> dict:
    return next(f for f in report["findings"] if f["column"] == column and f["finding"] == kind)


class TestTheFindings:
    def test_a_value_that_is_not_a_number(self):
        report, _ = validation.validate(SCHEMA, new_rows())
        found = finding(report, "amount", "type")
        assert found["count"] == 1 and found["rows"] == [2] and found["values"] == ["abc"]
        assert found["severity"] == validation.WARNING

    def test_a_number_outside_the_training_range(self):
        found = finding(validation.validate(SCHEMA, new_rows())[0], "amount", "out_of_range")
        assert found["count"] == 1 and found["rows"] == [3] and "0 to 100" in found["message"]

    def test_a_blank_where_training_never_had_one(self):
        found = finding(validation.validate(SCHEMA, new_rows())[0], "amount", "blank_new")
        assert found["rows"] == [4]

    def test_an_unseen_category_with_its_values(self):
        found = finding(validation.validate(SCHEMA, new_rows())[0], "plan", "unseen_category")
        assert found["count"] == 2 and found["rows"] == [2, 4] and found["values"] == ["gold"]

    def test_a_value_that_is_not_a_date(self):
        found = finding(validation.validate(SCHEMA, new_rows())[0], "joined", "type")
        assert found["rows"] == [3]

    def test_a_category_list_that_was_capped_cannot_judge_unseen_values(self):
        report, _ = validation.validate(SCHEMA, new_rows())
        assert not [f for f in report["findings"] if f["column"] == "code"]

    def test_unexpected_columns_are_listed_and_the_target_is_not_one(self):
        report, _ = validation.validate(SCHEMA, new_rows())
        assert report["unexpected_columns"] == ["extra"] and report["blocking"] is False

    def test_a_missing_column_is_blocking(self):
        report, _ = validation.validate(SCHEMA, new_rows().drop(columns=["plan"]))
        assert report["blocking"] is True and report["missing_columns"] == ["plan"]
        assert "plan" in validation.describe_blocking(report)

    def test_each_row_gets_a_summary_of_its_findings(self):
        report, notes = validation.validate(SCHEMA, new_rows())
        assert notes[0] == ""
        assert notes[1] == "amount: not a number (treated as blank); plan: a value not seen in training"
        assert "joined: not a date" in notes[2] and "amount: outside the training range" in notes[2]
        assert report["rows_with_findings"] == 3

    def test_a_clean_file_has_no_findings(self):
        clean = pd.DataFrame({"amount": [1.0, 2.0], "plan": ["basic", "pro"], "joined": ["2024-01-01", None],
                              "code": ["a", "x"]})
        report, notes = validation.validate(SCHEMA, clean)
        assert report["findings"] == [] and (notes == "").all()


class TestSpeed:
    def test_a_hundred_thousand_rows_of_twenty_columns_check_in_under_a_second(self):
        rng = np.random.default_rng(0)
        n = 100_000
        schema = {"columns": []}
        data = {}
        for i in range(10):
            schema["columns"].append({"name": f"n{i}", "kind": "numeric", "nullable": False,
                                      "range": {"min": -2.0, "max": 2.0}})
            data[f"n{i}"] = rng.normal(size=n)
            schema["columns"].append({"name": f"c{i}", "kind": "categorical", "nullable": True,
                                      "categories": {"a": 1, "b": 1}, "categories_truncated": False})
            data[f"c{i}"] = rng.choice(["a", "b", "c"], n)
        frame = pd.DataFrame(data)
        validation.validate(schema, frame.head(100))  # warm up
        started = time.perf_counter()
        report, _ = validation.validate(schema, frame)
        assert time.perf_counter() - started < 1.0
        assert report["rows_checked"] == n and report["findings"]


class FakeGraph:
    async def aget_state(self, _config):
        return SimpleNamespace(values={"action_log": []}, next=())


@pytest.fixture
def trained_session(monkeypatch):
    monkeypatch.setattr(app.state, "graph", FakeGraph(), raising=False)
    rng = np.random.default_rng(1)
    n = 300
    signal = rng.normal(0, 1, n)
    df = pd.DataFrame({"signal": signal.round(4), "plan": rng.choice(["basic", "pro"], n),
                       "churned": np.where(rng.random(n) < 1 / (1 + np.exp(-2.5 * signal)), "yes", "no")})
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    sid = client.post("/api/upload", files={"file": ("t.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "churned", "task_type": "auto"}
    assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
    assert client.post(f"/api/model/{sid}/train", json={"model_names": ["logistic_regression"],
                                                        "include_deep_diagnostics": False}).status_code == 200
    return sid


def apply(sid: str, df: pd.DataFrame):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return client.post(f"/api/apply/{sid}", files={"file": ("new.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")})


class TestOverHttp:
    def test_the_apply_response_carries_the_validation_and_the_file_its_rows(self, trained_session):
        new = pd.DataFrame({"signal": [0.1, 99.0, -0.2], "plan": ["basic", "gold", "pro"], "other": [1, 2, 3]})
        response = apply(trained_session, new)
        assert response.status_code == 200, response.text
        report = response.json()["validation"]
        assert report["unexpected_columns"] == ["other"]
        assert {(f["column"], f["finding"]) for f in report["findings"]} == {
            ("signal", "out_of_range"), ("plan", "unseen_category")}
        predictions = client.get(response.json()["downloads"]["predictions"]).text
        rows = pd.read_csv(io.StringIO(predictions), keep_default_na=False)
        assert rows["dataclean_validation"].tolist()[0] == ""
        assert "signal: outside the training range" in rows["dataclean_validation"][1]
        assert "plan: a value not seen in training" in rows["dataclean_validation"][1]

    def test_a_file_missing_a_column_is_refused_with_the_column_named(self, trained_session):
        response = apply(trained_session, pd.DataFrame({"signal": [0.1, 0.2]}))
        assert response.status_code >= 400 and "plan" in response.json()["detail"]
