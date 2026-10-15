"""The settings search is an explicit loop that can be stopped by a deadline.

It replaced a single `RandomizedSearchCV` call, which cannot be interrupted part-way. Without a deadline it
must give exactly what that call gave -- same settings drawn, same winner, same score and spread -- so
nothing that depended on the old search moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import modeling, tuning


def classification(n: int = 160, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    X = pd.DataFrame({"x": x, "z": rng.normal(size=n)})
    return X, pd.Series(np.where(x + rng.normal(0, 1, n) > 0, "yes", "no"))


def regression(n: int = 160, seed: int = 1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    X = pd.DataFrame({"x": x, "z": rng.normal(size=n)})
    return X, pd.Series(2 * x + rng.normal(0, 1, n))


def factory(name: str):
    return lambda: modeling.build_pipeline(name, ["x", "z"], [])


def reference(name: str, X, y, task: str, budget: int):
    """The search as it was: one `RandomizedSearchCV`."""
    from sklearn.model_selection import RandomizedSearchCV

    space = tuning.SEARCH_SPACES[name]
    combinations = int(np.prod([len(v) for v in space.values()]))
    iterations = min(budget, combinations)
    splitter = tuning.cv_splitter(task, y, tuning.CV_FOLDS, None, False, repeated=True)
    y_input = pd.Series(y).astype(str) if task == "classification" else y
    searcher = RandomizedSearchCV(
        factory(name)(), space, n_iter=iterations, cv=splitter, scoring=tuning.scoring_for(task, y),
        random_state=42, n_jobs=1, refit=False, error_score=np.nan,
    )
    searcher.fit(X, y_input)
    index = int(searcher.best_index_)
    return (
        dict(searcher.best_params_),
        round(float(searcher.best_score_), 4),
        round(float(searcher.cv_results_["std_test_score"][index]), 4),
        iterations * splitter.get_n_splits(),
    )


class TestSameAsRandomizedSearch:
    @pytest.mark.parametrize(
        "name,task,data,budget",
        [
            ("logistic_regression", "classification", classification, 12),
            ("logistic_regression", "classification", classification, 5),
            ("linear_regression", "regression", regression, 12),
            ("random_forest_classifier", "classification", classification, 3),
        ],
    )
    def test_the_winner_its_score_its_spread_and_the_fit_count_are_unchanged(self, name, task, data, budget):
        X, y = data()
        found = tuning.search(factory(name), name, X, y, task, budget=budget)
        params, score, std, fits = reference(name, X, y, task, budget)
        assert found.params == params
        assert (found.score, found.std, found.fits) == (score, std, fits)
        assert found.stopped_for_time is False and found.tried == found.planned

    def test_the_budget_is_never_more_than_the_space_holds(self):
        X, y = regression()
        found = tuning.search(factory("linear_regression"), "linear_regression", X, y, "regression", budget=500)
        assert found.planned == 5  # five values of alpha

    def test_a_model_with_no_space_has_nothing_to_search(self):
        X, y = classification()
        assert tuning.search(factory("logistic_regression"), "no_such_model", X, y, "classification") is None


class FakeClock:
    """A clock that moves one second every time it is read."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self) -> float:
        self.now += 1.0
        return self.now


class TestTheDeadline:
    def run(self, monkeypatch, deadline, budget=12):
        monkeypatch.setattr(tuning, "time", FakeClock())
        X, y = classification()
        return tuning.search(
            factory("logistic_regression"), "logistic_regression", X, y, "classification",
            budget=budget, deadline=deadline,
        )

    def test_it_stops_when_the_deadline_passes_and_says_so(self, monkeypatch):
        # Read before settings 2, 3, 4: at 1, 2 and 3 seconds; the deadline of 2.5 passes on the third read.
        found = self.run(monkeypatch, deadline=2.5)
        assert found.stopped_for_time is True
        assert found.tried == 3 and found.planned == 12
        assert found.fits == 3 * tuning.CV_FOLDS * found.repeats

    def test_the_settings_it_reached_are_a_prefix_of_the_full_search(self, monkeypatch):
        X, y = classification()
        stopped = self.run(monkeypatch, deadline=2.5)
        full = tuning.search(factory("logistic_regression"), "logistic_regression", X, y, "classification", budget=12)
        # The best of the first three cannot beat the best of all twelve.
        assert stopped.score <= full.score

    def test_a_deadline_already_gone_still_tries_one_setting(self, monkeypatch):
        found = self.run(monkeypatch, deadline=-1.0)
        assert found.tried == 1 and found.stopped_for_time is True

    def test_a_distant_deadline_changes_nothing(self, monkeypatch):
        X, y = classification()
        monkeypatch.setattr(tuning, "time", FakeClock())
        far = tuning.search(factory("logistic_regression"), "logistic_regression", X, y, "classification", deadline=1e9)
        monkeypatch.undo()
        none = tuning.search(factory("logistic_regression"), "logistic_regression", X, y, "classification")
        assert (far.params, far.score, far.std, far.fits) == (none.params, none.score, none.std, none.fits)
        assert far.stopped_for_time is False

    def test_no_deadline_never_reads_the_clock(self, monkeypatch):
        class Boom:
            def monotonic(self):
                raise AssertionError("the clock was read")

        monkeypatch.setattr(tuning, "time", Boom())
        X, y = classification()
        assert tuning.search(factory("logistic_regression"), "logistic_regression", X, y, "classification") is not None
