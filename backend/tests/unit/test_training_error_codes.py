"""A refused or failed training carries a stable reason code the page can act on.

The message explains the problem; the code lets the page offer a way out. Only the errors a page can
do something about carry one -- every other error is exactly the {error, detail} it always was.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import routes_model
from app.core.errors import AppError
from app.main import app
from app.models.graph_state import GoalContext
from app.services import ml_readiness, modeling

client = TestClient(app)


def frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    return pd.DataFrame({
        "x": x,
        "z": rng.normal(size=n),
        "when": pd.date_range("2022-01-01", periods=n, freq="D").astype(str),
        "outcome": np.where(x + rng.normal(size=n) > 0, "yes", "no"),
    })


def raised(**kwargs) -> modeling.ModelingError:
    with pytest.raises(modeling.ModelingError) as caught:
        modeling.train_candidates(
            kwargs.pop("df", frame()), target_column="outcome", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False, **kwargs,
        )
    return caught.value


class TestTheModelingErrorsCarryTheirCode:
    def test_too_few_labelled_rows(self):
        assert raised(df=frame().assign(outcome=lambda d: d.outcome.where(d.index < 20))).code == "too_few_labelled_rows"

    def test_no_usable_columns(self):
        df = pd.DataFrame({"constant": ["a"] * 120, "outcome": frame().outcome})
        assert raised(df=df).code == "no_usable_columns"

    def test_a_date_split_without_its_column(self):
        assert raised(split_strategy="time", time_column=None).code == "split_unusable"

    def test_a_date_column_that_is_not_dates(self):
        assert raised(split_strategy="time", time_column="z").code == "split_unusable"

    def test_a_grouped_split_on_too_few_groups(self):
        df = frame().assign(shop=lambda d: np.where(d.index % 2 == 0, "a", "b"))
        assert raised(df=df, split_strategy="group", group_column="shop").code == "split_unusable"

    def test_a_grouped_split_with_no_column(self):
        assert raised(split_strategy="group", group_column=None).code == "split_unusable"

    def test_an_unknown_model_has_no_code(self):
        with pytest.raises(modeling.ModelingError) as caught:
            modeling.build_pipeline("no_such_model", ["x"], [])
        assert caught.value.code is None

    def test_a_plain_message_still_works(self):
        error = modeling.ModelingError("just a message")
        assert str(error) == "just a message" and error.code is None


class TestTheRouteGuard:
    GOAL = GoalContext(use_case="ml_training", expected_output="x", target_column="outcome")

    def guard(self, df, goal=None):
        return routes_model._refuse_if_the_data_cannot_support_a_model(df, goal or self.GOAL, [])

    def test_a_target_that_is_gone(self):
        with pytest.raises(routes_model.TrainingFailedError) as caught:
            self.guard(frame().drop(columns="outcome"))
        assert caught.value.code == "target_missing"

    def test_a_target_with_no_values(self):
        with pytest.raises(routes_model.TrainingFailedError) as caught:
            self.guard(frame().assign(outcome=None))
        assert caught.value.code == "target_unusable"

    def test_a_blocked_readiness_verdict(self, monkeypatch):
        monkeypatch.setattr(
            ml_readiness, "assess_readiness", lambda *a, **k: {"verdict": "blocked", "blockers": ["Too few labels."]}
        )
        with pytest.raises(routes_model.TrainingFailedError) as caught:
            self.guard(frame())
        assert caught.value.code == "data_blocked" and "Too few labels." in caught.value.detail


class TestOverHttp:
    def session(self, df: pd.DataFrame) -> str:
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
        goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
        assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
        return sid

    def test_a_blocked_training_response_carries_the_code(self):
        rng = np.random.default_rng(5)
        n = 400
        outcome = list(rng.choice(["yes", "no"], 40)) + [None] * (n - 40)
        sid = self.session(pd.DataFrame({"signal": rng.normal(size=n).round(4), "outcome": outcome}))
        response = client.post(f"/api/model/{sid}/train", json={"include_deep_diagnostics": False})
        assert response.status_code == 422
        assert response.json()["code"] == "data_blocked"
        assert set(response.json()) == {"error", "detail", "code", "request_id"}

    def test_the_stream_refuses_a_blocked_run_with_the_code_too(self):
        rng = np.random.default_rng(5)
        n = 400
        outcome = list(rng.choice(["yes", "no"], 40)) + [None] * (n - 40)
        sid = self.session(pd.DataFrame({"signal": rng.normal(size=n).round(4), "outcome": outcome}))
        response = client.post(f"/api/model/{sid}/improve/stream", json={})
        assert response.status_code == 422 and response.json()["code"] == "data_blocked"

    def test_an_error_without_a_code_has_no_code_key(self):
        response = client.get("/api/model/does-not-exist/plan")
        assert response.status_code >= 400
        assert "code" not in response.json()

    def test_the_handler_leaves_a_codeless_error_exactly_as_it_was(self):
        error = AppError("Something", "went wrong")
        assert error.code is None
