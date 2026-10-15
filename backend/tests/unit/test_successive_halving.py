"""Successive halving on the search-strategy interface.

Many settings are scored on a few folds, the promising third on more, the best on all of them. The resource is
folds, cut once by the same splitter random search uses, so grouped and forward-in-time folds hold at every rung;
the survivor's score is on every fold, which is what the loop compares with the incumbent.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
import sklearn.model_selection as sklearn_selection
from fastapi.testclient import TestClient

from app.main import app
from app.services import improvement, modeling, tuning
from app.services.tuning import SuccessiveHalvingSearch


def classification(n: int = 160, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    X = pd.DataFrame({"x": x, "z": rng.normal(size=n)})
    return X, pd.Series(np.where(x + rng.normal(0, 1, n) > 0, "yes", "no"))


def factory(name: str = "logistic_regression"):
    return lambda: modeling.build_pipeline(name, ["x", "z"], [])


def halving(X, y, task="classification", name="logistic_regression", **kwargs):
    return tuning.search(factory(name), name, X, y, task, strategy="halving", **kwargs)


class TestTheSchedule:
    @pytest.mark.parametrize(
        "folds,expected", [(5, [1, 2, 5]), (15, [2, 5, 15]), (3, [1, 3]), (2, [1, 2]), (10, [1, 3, 10])]
    )
    def test_the_last_rung_is_always_every_fold(self, folds, expected):
        assert SuccessiveHalvingSearch.fold_schedule(folds) == expected

    def test_it_is_registered_by_name(self):
        assert tuning.STRATEGIES["halving"].name == "halving"


class TestWhatItFindsAndWhatItCosts:
    def test_it_spends_fewer_fits_than_random_search_for_the_same_budget_and_finds_about_as_good(self):
        X, y = classification()
        random = tuning.search(factory(), "logistic_regression", X, y, "classification", budget=12)
        found = halving(X, y, budget=12)
        assert found.strategy == "halving"
        assert found.fits < random.fits
        assert found.fits <= 0.5 * random.fits  # 63 against 180 here
        assert found.score >= random.score - 0.03  # asserted loosely, as the issue asks
        assert found.planned > random.planned  # it looked at more settings for less

    def test_the_survivor_is_scored_on_every_fold_so_it_can_be_compared_with_the_incumbent(self):
        X, y = classification()
        found = halving(X, y, budget=12)
        # Re-score the winning settings on all the folds the usual way: the same mean and spread.
        splitter = tuning.cv_splitter("classification", y, tuning.CV_FOLDS, None, False, repeated=True)
        scores = sklearn_selection.cross_val_score(
            factory()().set_params(**found.params), X, y.astype(str), cv=splitter, scoring="roc_auc"
        )
        assert found.score == round(float(np.mean(scores)), 4)
        assert found.std == round(float(np.std(scores)), 4)

    def test_the_rungs_are_recorded_and_narrow_to_one(self):
        X, y = classification()
        found = halving(X, y, budget=12)
        assert [r["rung"] for r in found.rungs] == [1, 2, 3]
        assert [r["folds"] for r in found.rungs] == [2, 5, 15]
        settings = [r["settings"] for r in found.rungs]
        assert settings == sorted(settings, reverse=True) and settings[0] == found.planned
        assert found.rungs[-1]["kept"] == 1
        assert all(r["best_score"] is not None for r in found.rungs)

    def test_a_clearly_poor_setting_is_dropped_before_it_is_scored_on_every_fold(self, monkeypatch):
        """With one setting far worse than the rest, it must not reach the last rung."""
        monkeypatch.setitem(tuning.SEARCH_SPACES, "logistic_regression", {
            "model__C": [1e-9, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            "model__class_weight": [None, "balanced"],
        })
        X, y = classification()
        found = halving(X, y, budget=12)
        assert found.params["model__C"] != 1e-9
        assert found.fits < 16 * 15


class TestWhenThereIsNothingToHalve:
    def test_a_small_space_is_searched_at_random_and_says_so(self):
        # Ridge has five settings, fewer than the six needed to discard any.
        rng = np.random.default_rng(1)
        Xr = pd.DataFrame({"x": rng.normal(size=160), "z": rng.normal(size=160)})
        yr = pd.Series(2 * Xr["x"] + rng.normal(0, 1, 160))
        found = halving(Xr, yr, task="regression", name="linear_regression")
        assert found.strategy == "random"
        assert found.note.startswith("Searched at random: only 5 settings to try is too few to halve over.")

    def test_too_few_folds_is_searched_at_random_and_says_so(self):
        X, y = classification()
        found = halving(X, y, folds=2)
        # Two folds, repeated three times below 2,000 rows, is six: enough. Without the repeats it is not.
        assert found.strategy == "halving"
        big_x, big_y = classification(n=2500)
        few = halving(big_x, big_y, folds=2)
        assert few.strategy == "random" and "only 2 folds" in few.note

    def test_a_target_nothing_can_score_gives_nothing(self):
        X, y = classification()
        assert halving(X, pd.Series(["yes"] * len(y))) is None


class TestEveryRungKeepsTheKindOfFoldsItWasGiven:
    """Spy on every fold scored: whatever the rung, the folds must be group-aware or forward in time."""

    def spy(self, monkeypatch):
        seen: list[list[tuple[np.ndarray, np.ndarray]]] = []
        real = sklearn_selection.cross_val_score

        def recording(estimator, X, y=None, *, cv=None, **kwargs):
            seen.append(list(cv))
            return real(estimator, X, y, cv=cv, **kwargs)

        monkeypatch.setattr(sklearn_selection, "cross_val_score", recording)
        return seen

    def test_grouped_folds_never_split_a_group_at_any_rung(self, monkeypatch):
        X, y = classification(n=240)
        groups = pd.Series(np.repeat(np.arange(48), 5))
        seen = self.spy(monkeypatch)
        found = halving(X, y, groups=groups, budget=12)
        assert found is not None and found.strategy == "halving" and seen
        for folds in seen:
            for train, test in folds:
                assert not set(groups.iloc[train]) & set(groups.iloc[test])

    def test_forward_in_time_folds_never_train_on_the_future_at_any_rung(self, monkeypatch):
        X, y = classification(n=240)
        seen = self.spy(monkeypatch)
        found = halving(X, y, ordered=True, budget=12)
        assert found is not None and found.strategy == "halving" and seen
        for folds in seen:
            for train, test in folds:
                assert train.max() < test.min()

    def test_every_rung_scores_folds_the_splitter_made_and_none_twice(self, monkeypatch):
        X, y = classification()
        seen = self.spy(monkeypatch)
        halving(X, y, budget=12)
        splitter = tuning.cv_splitter("classification", y, tuning.CV_FOLDS, None, False, repeated=True)
        made = [tuple(test.tolist()) for _, test in splitter.split(X, y.astype(str))]
        for folds in seen:
            for _, test in folds:
                assert tuple(test.tolist()) in made


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


class TestTheDeadline:
    def run(self, monkeypatch, seconds_each=10.0, limit=65.0):
        clock = FakeClock()
        monkeypatch.setattr(tuning, "time", clock)
        real = sklearn_selection.cross_val_score

        def slow(*args, **kwargs):
            clock.now += seconds_each
            return real(*args, **kwargs)

        monkeypatch.setattr(sklearn_selection, "cross_val_score", slow)
        X, y = classification()
        return halving(X, y, budget=12, deadline=clock.now + limit)

    def test_it_stops_and_still_returns_a_score_on_every_fold(self, monkeypatch):
        found = self.run(monkeypatch)
        assert found.stopped_for_time is True and found.strategy == "halving"
        assert found.score is not None and 0 <= found.score <= 1
        assert len(found.rungs) < 3  # it did not get through every rung

    def test_it_does_far_less_than_a_full_search_before_stopping(self, monkeypatch):
        X, y = classification()
        full = halving(X, y, budget=12)
        stopped = self.run(monkeypatch)
        assert stopped.fits < full.fits


class TestInTheLoop:
    def frame(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=240)
        return pd.DataFrame({"x": x, "z": rng.normal(size=240), "outcome": np.where(x + rng.normal(size=240) > 0, "yes", "no")})

    def test_the_trail_shows_the_rungs_and_the_strategy(self):
        found = improvement.improve(
            self.frame(), target_column="outcome", task="classification",
            model_names=["logistic_regression"], budget=12, search_strategy="halving",
        )
        assert found["search_strategy"] == "halving"
        settings = next(r for r in found["rounds"] if r["name"] == "hyperparameters")
        assert settings["detail"]["strategy"] == "halving"
        assert [r["rung"] for r in settings["detail"]["rungs"]] == [1, 2, 3]

    def test_a_setting_is_only_accepted_on_its_full_score(self):
        found = improvement.improve(
            self.frame(), target_column="outcome", task="classification",
            model_names=["logistic_regression"], budget=12, search_strategy="halving",
        )
        settings = next(r for r in found["rounds"] if r["name"] == "hyperparameters")
        # The score the gain rule saw is the last rung's folds, not an earlier rung's.
        assert settings["score"] is not None
        assert settings["detail"]["rungs"][-1]["folds"] == max(r["folds"] for r in settings["detail"]["rungs"])

    def test_a_small_file_is_still_searched_at_random_by_default(self):
        found = improvement.improve(
            self.frame(), target_column="outcome", task="classification",
            model_names=["logistic_regression"], budget=4,
        )
        assert found["search_strategy"] == "auto" and found["search_strategies"] == ["random"]


class TestOverHttp:
    def test_the_request_can_ask_for_halving_and_refuses_an_unknown_strategy(self):
        client = TestClient(app)
        buffer = io.StringIO()
        TestInTheLoop().frame().to_csv(buffer, index=False)
        sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
        goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
        assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
        assert client.post(f"/api/model/{sid}/improve", json={"search_strategy": "bayesian"}).status_code == 422
        ok = client.post(
            f"/api/model/{sid}/improve",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False, "budget": 12,
                  "search_strategy": "halving"},
        )
        assert ok.status_code == 200, ok.text
        assert ok.json()["improvement"]["search_strategy"] == "halving"
