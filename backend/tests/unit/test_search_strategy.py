"""The search-strategy interface, introduced with random search as its first implementation.

No behaviour changed: `test_search_time_limit.py` holds random search to what `RandomizedSearchCV` gave. These
tests are about the seam -- what a strategy is handed, what comes back, and that the name travels to the trail
and the run history so a second strategy can be told apart from the first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import improvement, model_store, modeling, tuning


def data(n: int = 160, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    X = pd.DataFrame({"x": x, "z": rng.normal(size=n)})
    return X, pd.Series(np.where(x + rng.normal(0, 1, n) > 0, "yes", "no"))


def factory(name: str = "logistic_regression"):
    return lambda: modeling.build_pipeline(name, ["x", "z"], [])


class TestTheRegistry:
    def test_random_search_is_the_default_and_the_only_one_so_far(self):
        assert tuning.DEFAULT_STRATEGY == "random"
        assert set(tuning.STRATEGIES) == {"random"}
        assert tuning.STRATEGIES["random"].name == "random"

    def test_an_unknown_strategy_is_refused_with_the_known_names(self):
        X, y = data()
        with pytest.raises(ValueError, match="Unknown search strategy 'bayesian'.*random"):
            tuning.search(factory(), "logistic_regression", X, y, "classification", strategy="bayesian")


class TestRandomSearchIsWhatSearchAlwaysDid:
    def test_asking_for_it_by_name_or_by_default_or_directly_gives_the_same_answer(self):
        X, y = data()
        by_default = tuning.search(factory(), "logistic_regression", X, y, "classification")
        by_name = tuning.search(factory(), "logistic_regression", X, y, "classification", strategy="random")
        direct = tuning.search(factory(), "logistic_regression", X, y, "classification", strategy=tuning.RandomSearch())
        assert by_default == by_name == direct
        assert by_default.strategy == "random"

    def test_running_it_on_its_own_matches_the_wrapper(self):
        X, y = data()
        splitter = tuning.cv_splitter("classification", y, tuning.CV_FOLDS, None, False, repeated=True)
        raw = tuning.RandomSearch().run(
            factory(), tuning.SEARCH_SPACES["logistic_regression"], X, y.astype(str),
            splitter=splitter, metric="roc_auc", budget=tuning.DEFAULT_BUDGET,
        )
        assert raw == tuning.search(factory(), "logistic_regression", X, y, "classification")


class RecordingStrategy:
    name = "recording"

    def __init__(self):
        self.seen = None

    def run(self, build, space, X, y, *, splitter, metric, budget, deadline=None, groups=None):
        self.seen = dict(space=space, X=X, y=y, splitter=splitter, metric=metric, budget=budget,
                         deadline=deadline, groups=groups, estimator=build())
        return tuning.SearchResult(params={"model__C": 1.0}, score=0.5, std=0.1, metric=metric, fits=0, strategy=self.name)


class TestWhatAStrategyIsHanded:
    def test_the_decided_things_arrive_and_its_answer_comes_back_untouched(self):
        X, y = data()
        strategy = RecordingStrategy()
        found = tuning.search(
            factory(), "logistic_regression", X, y, "classification",
            budget=7, deadline=123.0, strategy=strategy,
        )
        seen = strategy.seen
        assert found.strategy == "recording" and found.params == {"model__C": 1.0}
        assert seen["space"] is tuning.SEARCH_SPACES["logistic_regression"]
        assert seen["metric"] == "roc_auc" and seen["budget"] == 7 and seen["deadline"] == 123.0
        assert seen["splitter"].get_n_splits() > 0 and seen["groups"] is None
        # A classification target arrives as text, as the scorer needs it.
        assert set(seen["y"]) == {"yes", "no"} and seen["y"].dtype == object
        assert seen["estimator"] is not None

    def test_groups_reach_the_strategy(self):
        X, y = data(n=120)
        groups = pd.Series(np.repeat(np.arange(30), 4))
        strategy = RecordingStrategy()
        tuning.search(factory(), "logistic_regression", X, y, "classification", groups=groups, strategy=strategy)
        assert strategy.seen["groups"] is groups

    def test_a_model_with_no_space_or_folds_never_reaches_the_strategy(self):
        X, y = data()
        strategy = RecordingStrategy()
        assert tuning.search(factory(), "no_such_model", X, y, "classification", strategy=strategy) is None
        assert strategy.seen is None


class TestTheNameTravels:
    def frame(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=200)
        return pd.DataFrame({"x": x, "z": rng.normal(size=200), "outcome": np.where(x + rng.normal(size=200) > 0, "yes", "no")})

    def test_each_settings_round_names_the_strategy_it_used(self):
        found = improvement.improve(
            self.frame(), target_column="outcome", task="classification",
            model_names=["logistic_regression", "random_forest_classifier"], budget=4,
        )
        assert found["search_strategy"] == "random"
        searched = [r for r in found["rounds"] if r["name"] in {"hyperparameters", "other_models"} and r["score"] is not None]
        assert searched and all(r["detail"].get("strategy") == "random" for r in searched if r["name"] == "hyperparameters")

    def test_the_run_history_keeps_it(self):
        result = {"results": [], "split": {}, "improvement": {"search_strategy": "random", "stopped_because": "no_further_gains",
                                                               "settings": {}}}
        assert model_store.summarise(result)["search_strategy"] == "random"
        assert model_store.summarise({"results": [], "split": {}})["search_strategy"] is None
