"""A round that asks whether each input earns its place.

Every usable column is an input, and a weak or noisy one adds variance, most of all on a small file. Inputs are ranked by
how much shuffling each hurts the model on validation folds (never the test rows), and the model is re-scored without
the ones that did not matter; the smaller set is kept only under the same gain rule as every other round.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as sklearn_metrics
from fastapi.testclient import TestClient

from app.main import app
from app.services import improvement, model_card, modeling, tuning
from app.services.tuning import SearchResult

NOISE = [f"n{i}" for i in range(6)]


def frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    data = {"signal": x, **{c: rng.normal(size=n) for c in NOISE}}
    data["outcome"] = np.where(x + rng.normal(0, 0.8, n) > 0, "yes", "no")
    return pd.DataFrame(data)


def build_for(columns):
    return lambda: modeling.build_pipeline("logistic_regression", columns, [])


class TestTheImportance:
    def test_a_column_that_carries_the_answer_matters_and_noise_does_not(self):
        df = frame()
        columns = ["signal", *NOISE]
        found = improvement.fold_importance(
            build_for(columns), df[columns], df["outcome"], "classification", groups=None, ordered=False, columns=columns,
        )
        assert set(found) == set(columns)
        assert found["signal"][0] > 0.15
        assert all(abs(found[c][0]) < 0.05 for c in NOISE)
        assert all(found[c][1] >= 0 for c in columns)  # the spread is a spread

    def test_it_is_deterministic(self):
        df = frame()
        columns = ["signal", *NOISE]
        args = dict(groups=None, ordered=False, columns=columns)
        a = improvement.fold_importance(build_for(columns), df[columns], df["outcome"], "classification", **args)
        b = improvement.fold_importance(build_for(columns), df[columns], df["outcome"], "classification", **args)
        assert a == b

    def test_a_regression_is_ranked_too(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=200)
        df = pd.DataFrame({"signal": x, "n0": rng.normal(size=200), "n1": rng.normal(size=200)})
        y = pd.Series(2 * x + rng.normal(0, 0.3, 200))
        build = lambda: modeling.build_pipeline("linear_regression", list(df.columns), [])  # noqa: E731
        found = improvement.fold_importance(build, df, y, "regression", groups=None, ordered=False, columns=list(df.columns))
        assert found["signal"][0] > 0.5 and abs(found["n0"][0]) < 0.05

    def test_it_only_ever_scores_rows_the_model_was_not_fitted_on(self, monkeypatch):
        fitted, scored = [], []
        real = sklearn_metrics.get_scorer

        def spying(name):
            scorer = real(name)

            def wrapped(model, X, y):
                scored.append(set(X.index))
                return scorer(model, X, y)

            return wrapped

        class Recording:
            def __init__(self, inner):
                self.inner = inner

            def fit(self, X, y):
                fitted.append(set(X.index))
                self.inner.fit(X, y)
                return self

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def predict_proba(self, X):
                return self.inner.predict_proba(X)

            def predict(self, X):
                return self.inner.predict(X)

            def get_params(self, deep=True):
                return self.inner.get_params(deep)

        monkeypatch.setattr(sklearn_metrics, "get_scorer", spying)
        df = frame()
        columns = ["signal", *NOISE]
        improvement.fold_importance(
            lambda: Recording(build_for(columns)()), df[columns], df["outcome"], "classification",
            groups=None, ordered=False, columns=columns,
        )
        assert fitted and len(fitted) * 1 <= len(scored)
        # Scoring happens on each fold's held-out rows only: no row is in a fit and in its fold's scoring.
        folds = len(fitted)
        per_fold = len(scored) // folds
        for index in range(folds):
            for held in scored[index * per_fold:(index + 1) * per_fold]:
                assert not held & fitted[index]


class Controlled:
    """Fixes the importance ranking and makes any smaller input set score better (or not), to test the decision."""

    def __init__(self, monkeypatch, improved: bool, importances: dict | None = None):
        self.sets_scored: list[int] = []
        real_score = tuning.score_config

        def fake_importance(build, X, y, task, **kwargs):
            wanted = importances or {c: (0.0, 0.01) for c in NOISE}
            return {c: wanted.get(c, (0.2, 0.01)) for c in kwargs["columns"]}

        def score(build, X, y, task, **kwargs):
            if kwargs.get("confirmation") or X.shape[1] == 1 + len(NOISE):
                return real_score(build, X, y, task, **kwargs)
            self.sets_scored.append(X.shape[1])
            return SearchResult(params={}, score=0.99 if improved else 0.01, std=0.001, metric="roc_auc", fits=5)

        monkeypatch.setattr(improvement, "fold_importance", fake_importance)
        monkeypatch.setattr(tuning, "score_config", score)


def improve(df=None, **kwargs):
    return improvement.improve(
        frame() if df is None else df, target_column="outcome", task="classification",
        model_names=["logistic_regression"], budget=4, **kwargs,
    )


def selection_round(found):
    return next((r for r in found["rounds"] if r["name"] == "feature_selection"), None)


class TestTheDecision:
    def test_inputs_the_model_does_better_without_are_removed_with_their_importance(self, monkeypatch):
        controlled = Controlled(monkeypatch, improved=True)
        found = improve()
        round_ = selection_round(found)
        assert round_["accepted"] is True and "does better without 3 of its inputs" in round_["outcome"]
        removed = round_["detail"]["removed"]
        assert len(removed) == 3 and {r["column"] for r in removed} <= set(NOISE)
        assert all(set(r) == {"column", "importance", "spread"} for r in removed)
        assert found["best"]["removed_columns"] == removed
        assert controlled.sets_scored  # it re-scored the smaller set
        assert all(c not in found["features"]["numeric"] for c in (r["column"] for r in removed))

    def test_never_more_than_half_the_inputs_are_removed(self, monkeypatch):
        Controlled(monkeypatch, improved=True)
        found = improve()
        assert len(selection_round(found)["detail"]["removed"]) <= 7 // 2

    def test_a_smaller_set_that_does_not_clear_the_gain_rule_changes_nothing(self, monkeypatch):
        Controlled(monkeypatch, improved=False)
        found = improve()
        round_ = selection_round(found)
        assert round_["accepted"] is False and "all of them stayed" in round_["outcome"]
        assert found["best"]["removed_columns"] == []
        assert set(found["features"]["numeric"]) == {"signal", *NOISE}

    def test_a_group_that_fails_is_followed_by_a_smaller_one(self, monkeypatch):
        controlled = Controlled(monkeypatch, improved=False)
        improve()
        assert len(controlled.sets_scored) == 2  # three removable inputs, then the lowest third of them

    def test_when_every_input_is_missed_nothing_is_tried(self, monkeypatch):
        Controlled(monkeypatch, improved=True, importances={c: (0.1, 0.01) for c in ["signal", *NOISE]})
        round_ = selection_round(improve())
        assert round_["accepted"] is False and "none was removed" in round_["outcome"]
        assert round_["detail"]["importance"]

    def test_too_few_inputs_to_rank_means_no_round(self, monkeypatch):
        Controlled(monkeypatch, improved=True)
        found = improve(frame()[["signal", "n0", "n1", "outcome"]])
        assert selection_round(found) is None

    def test_a_goal_that_was_met_ends_the_search_before_this_round(self, monkeypatch):
        Controlled(monkeypatch, improved=True)
        assert selection_round(improve(goal=0.0)) is None

    def test_a_time_limit_that_has_passed_skips_it_and_says_so(self, monkeypatch):
        class Clock:
            now = 1000.0

            def monotonic(self):
                return self.now

        clock = Clock()
        monkeypatch.setattr(improvement, "time", clock)
        monkeypatch.setattr(tuning, "time", clock)
        real = improvement.tuning.score_config

        def slow(*args, **kwargs):
            clock.now += 100.0
            return real(*args, **kwargs)

        monkeypatch.setattr(tuning, "score_config", slow)
        found = improve(time_limit_seconds=50)
        assert "feature_selection" in found["skipped_for_time"]
        assert selection_round(found)["detail"] == {"skipped": "time_limit"}


class TestOnRealData:
    def test_removal_only_ever_takes_inputs_that_did_not_matter(self):
        found = improve(frame(300, seed=4))
        removed = [r["column"] for r in found["best"]["removed_columns"]]
        assert "signal" not in removed
        assert set(removed) <= set(NOISE)
        assert found["features"]["numeric"].count("signal") == 1


class TestTheRebuildAndTheCard:
    def session(self):
        client = TestClient(app)
        buffer = io.StringIO()
        frame().to_csv(buffer, index=False)
        sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
        goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
        assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
        return client, sid

    def test_removed_inputs_are_withheld_from_the_rebuilt_model_and_the_reason_is_theirs_not_the_users(self, monkeypatch):
        real = improvement.improve

        def with_removal(*args, **kwargs):
            found = real(*args, **kwargs)
            found["best"]["removed_columns"] = [{"column": "n1", "importance": -0.004, "spread": 0.01}]
            return found

        monkeypatch.setattr(improvement, "improve", with_removal)
        client, sid = self.session()
        body = client.post(
            f"/api/model/{sid}/improve",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False, "budget": 4,
                  "excluded_columns": ["n2"]},
        ).json()
        features = body["features"]
        assert "n1" not in features["numeric"] and "n2" not in features["numeric"]
        assert features["excluded"] == ["n2"]  # only what the user excluded
        assert features["removed_by_selection"] == [{"column": "n1", "importance": -0.004, "spread": 0.01}]
        reasons = {d["column"]: d["reason"] for d in features["dropped"]}
        assert reasons["n1"].startswith("Removed by feature selection: shuffling it on validation folds changed the score by +0.004")
        assert reasons["n2"] == "You excluded this column."

    def test_the_card_lists_them_with_their_own_reason(self):
        result = {
            "task": "classification", "best_model": "m", "results": [{"model": "m", "label": "M", "metrics": {}}],
            "features": {"numeric": ["a"], "categorical": [], "frequent": [], "excluded": [],
                         "removed_by_selection": [{"column": "n1", "importance": -0.004, "spread": 0.01}],
                         "dropped": [{"column": "n1", "reason": "Removed by feature selection: shuffling it did nothing."}]},
        }
        card = model_card.build_model_card(session_id="s", file_name="f.csv", result=result, summary="A.", interpretation="B.")
        assert "| n1 | Removed by feature selection: shuffling it did nothing. |" in card
