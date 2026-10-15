"""A settings search's winner is re-scored on fresh folds before it can replace the incumbent.

The search returns the best of up to a dozen noisy estimates, so its score on the folds it won on is biased upward and
a setting that is no better can clear the gain rule by being the luckiest. The winner, and the incumbent it would
replace, are now both scored on folds the search did not select on; the gain rule must hold there too.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from app.services import improvement, tuning
from app.services.tuning import SearchResult


def classification(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "outcome": np.where(x + rng.normal(0, 1, n) > 0, "yes", "no")})


def noise(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "outcome": rng.choice(["yes", "no"], n)})


def held_out_rows(splitter, X, y, groups=None):
    return [tuple(test.tolist()) for _, test in splitter.split(X, y, groups)]


class TestFreshFolds:
    def test_a_shuffled_split_is_reseeded(self):
        rng = np.random.default_rng(1)
        y = pd.Series(rng.choice(["yes", "no"], 150))
        X = pd.DataFrame({"a": np.arange(150)})
        search = tuning.cv_splitter("classification", y, repeated=True)
        fresh = tuning.cv_splitter("classification", y, repeated=True, confirmation=True)
        assert held_out_rows(search, X, y.astype(str)) != held_out_rows(fresh, X, y.astype(str))
        assert len(held_out_rows(search, X, y)) == len(held_out_rows(fresh, X, y))  # as many folds, different rows

    def test_a_regression_split_is_reseeded(self):
        y = pd.Series(np.random.default_rng(2).normal(size=150))
        X = pd.DataFrame({"a": np.arange(150)})
        assert held_out_rows(tuning.cv_splitter("regression", y), X, y) != held_out_rows(
            tuning.cv_splitter("regression", y, confirmation=True), X, y
        )

    def test_grouped_classification_is_reseeded_and_still_keeps_groups_whole(self):
        rng = np.random.default_rng(3)
        y = pd.Series(rng.choice(["yes", "no"], 200))
        groups = pd.Series(np.repeat(np.arange(50), 4))
        X = pd.DataFrame({"a": np.arange(200)})
        fresh = tuning.cv_splitter("classification", y, groups=groups, confirmation=True)
        assert held_out_rows(tuning.cv_splitter("classification", y, groups=groups), X, y, groups) != held_out_rows(fresh, X, y, groups)
        for train, test in fresh.split(X, y, groups):
            assert not set(groups.iloc[train]) & set(groups.iloc[test])

    def test_what_cannot_be_reshuffled_changes_its_fold_count_instead(self):
        y = pd.Series(np.random.default_rng(4).normal(size=200))
        groups = pd.Series(np.repeat(np.arange(40), 5))
        assert tuning.cv_splitter("regression", y, groups=groups).get_n_splits() == tuning.CV_FOLDS
        assert tuning.cv_splitter("regression", y, groups=groups, confirmation=True).get_n_splits() == tuning.CV_FOLDS + 1
        assert tuning.cv_splitter("regression", y, ordered=True).get_n_splits() == tuning.CV_FOLDS
        assert tuning.cv_splitter("regression", y, ordered=True, confirmation=True).get_n_splits() == tuning.CV_FOLDS + 1

    def test_forward_folds_still_never_train_on_the_future(self):
        y = pd.Series(np.random.default_rng(5).normal(size=120))
        X = pd.DataFrame({"a": np.arange(120)})
        for train, test in tuning.cv_splitter("regression", y, ordered=True, confirmation=True).split(X, y):
            assert train.max() < test.min()

    def test_the_search_folds_are_exactly_what_they_were(self):
        y = pd.Series(np.random.default_rng(6).choice(["yes", "no"], 150))
        X = pd.DataFrame({"a": np.arange(150)})
        assert held_out_rows(tuning.cv_splitter("classification", y), X, y) == held_out_rows(
            tuning.cv_splitter("classification", y, confirmation=False), X, y
        )

    def test_scoring_a_configuration_on_fresh_folds_gives_a_different_partition(self):
        df = classification()
        X, y = df[["x", "z"]], df["outcome"]
        from app.services import modeling

        build = lambda: modeling.build_pipeline("logistic_regression", ["x", "z"], [])  # noqa: E731
        a = tuning.score_config(build, X, y, "classification")
        b = tuning.score_config(build, X, y, "classification", confirmation=True)
        assert a.fold_scores != b.fold_scores and a.fits == b.fits

    def test_a_wrapper_with_the_old_fixed_signature_still_works_when_confirmation_is_not_asked_for(self, monkeypatch):
        real = tuning.cv_splitter

        def old_signature(task, y, folds=tuning.CV_FOLDS, groups=None, ordered=False, repeated=False):
            return real(task, y, folds, groups, ordered, repeated)

        monkeypatch.setattr(tuning, "cv_splitter", old_signature)
        df = classification()
        from app.services import modeling

        found = tuning.score_config(
            lambda: modeling.build_pipeline("logistic_regression", ["x", "z"], []), df[["x", "z"]], df["outcome"], "classification"
        )
        assert found is not None


class Controlled:
    """Replaces the search and the confirmation scores with known numbers, so the decision is the only thing tested."""

    def __init__(self, monkeypatch, search_score=0.99, challenger=0.75, incumbent=0.70):
        self.confirmation_calls = []
        real_score = tuning.score_config

        def search(build, name, X, y, task, **kwargs):
            return SearchResult(params={"model__C": 10.0}, score=search_score, std=0.01, metric="roc_auc", fits=60,
                                tried=12, planned=12)

        def score(build, X, y, task, **kwargs):
            if kwargs.get("confirmation"):
                tuned = build().get_params().get("model__C") == 10.0
                self.confirmation_calls.append("challenger" if tuned else "incumbent")
                return SearchResult(params={}, score=challenger if tuned else incumbent, std=0.01, metric="roc_auc", fits=5)
            return real_score(build, X, y, task, **kwargs)

        monkeypatch.setattr(tuning, "search", search)
        monkeypatch.setattr(tuning, "score_config", score)


def improve(**kwargs):
    return improvement.improve(
        classification(), target_column="outcome", task="classification",
        model_names=["logistic_regression"], budget=12, **kwargs,
    )


def settings_round(found):
    return next(r for r in found["rounds"] if r["name"] == "hyperparameters")


class TestTheDecision:
    def test_a_winner_that_holds_on_fresh_folds_is_accepted_and_the_trail_shows_both_scores(self, monkeypatch):
        controlled = Controlled(monkeypatch, challenger=0.75, incumbent=0.70)
        found = improve()
        round_ = settings_round(found)
        assert round_["accepted"] is True
        assert round_["detail"]["confirmation"] == {
            "status": "confirmed", "search_score": 0.99, "confirmation_score": 0.75,
            "confirmation_std": 0.01, "incumbent_confirmation_score": 0.70,
        }
        assert "Confirmed on fresh folds: 0.750 against 0.700." in round_["outcome"]
        assert controlled.confirmation_calls == ["challenger", "incumbent"]

    def test_a_winner_that_does_not_hold_is_the_luckiest_of_several_and_is_refused(self, monkeypatch):
        Controlled(monkeypatch, challenger=0.70, incumbent=0.70)
        found = improve()
        round_ = settings_round(found)
        assert round_["accepted"] is False
        assert round_["detail"]["confirmation"]["status"] == "not_confirmed"
        assert "on fresh folds it was 0.700 against 0.700" in round_["outcome"]
        assert "so the defaults were kept." in round_["outcome"]
        assert found["best"]["params"] == {}

    def test_a_refused_winner_does_not_end_up_in_the_final_settings(self, monkeypatch):
        Controlled(monkeypatch, challenger=0.69, incumbent=0.70)
        assert improve()["best"]["params"] == {}

    def test_without_the_confirmation_the_same_winner_would_have_been_accepted(self, monkeypatch):
        monkeypatch.setattr(improvement, "CONFIRM_SEARCH_WINNERS", False)
        controlled = Controlled(monkeypatch, challenger=0.70, incumbent=0.70)
        round_ = settings_round(improve())
        assert round_["accepted"] is True and "confirmation" not in round_["detail"]
        assert controlled.confirmation_calls == []

    def test_a_setting_that_never_beat_the_incumbent_is_not_confirmed_at_all(self, monkeypatch):
        controlled = Controlled(monkeypatch, search_score=0.0)
        round_ = settings_round(improve())
        assert round_["accepted"] is False and "confirmation" not in round_["detail"]
        assert controlled.confirmation_calls == []

    def test_the_cost_is_two_scorings_a_round_whatever_the_budget(self, monkeypatch):
        controlled = Controlled(monkeypatch)
        improve()
        improve()
        assert len(controlled.confirmation_calls) == 4

    def test_a_time_limit_reached_before_the_check_refuses_rather_than_trusting_the_search(self, monkeypatch):
        class Clock:
            now = 1000.0

            def monotonic(self):
                return self.now

        clock = Clock()
        monkeypatch.setattr(improvement, "time", clock)
        monkeypatch.setattr(tuning, "time", clock)
        Controlled(monkeypatch)
        real_search = tuning.search

        def slow_search(*args, **kwargs):
            clock.now += 500.0  # the search itself used up the limit
            return real_search(*args, **kwargs)

        monkeypatch.setattr(tuning, "search", slow_search)
        found = improve(time_limit_seconds=50)
        round_ = settings_round(found)
        assert round_["accepted"] is False
        assert round_["detail"]["confirmation"]["status"] == "skipped_for_time"
        assert "the time limit came before it could be checked" in round_["outcome"]
        assert found["stopped_because"] == "time_limit"


class TestTheOtherModelsRound:
    def run(self, monkeypatch, challenger_confirms: bool):
        Controlled(monkeypatch, challenger=0.75 if challenger_confirms else 0.70, incumbent=0.70)
        return improvement.improve(
            classification(), target_column="outcome", task="classification",
            model_names=["logistic_regression", "random_forest_classifier"], budget=12,
        )

    def test_a_challenger_that_holds_on_fresh_folds_overtakes(self, monkeypatch):
        found = self.run(monkeypatch, True)
        others = next(r for r in found["rounds"] if r["name"] == "other_models")
        if others["accepted"]:
            assert others["detail"]["confirmation"]["status"] == "confirmed"
            assert "Confirmed on fresh folds" in others["outcome"]

    def test_a_challenger_that_does_not_is_refused_and_the_incumbent_stays(self, monkeypatch):
        found = self.run(monkeypatch, False)
        others = next(r for r in found["rounds"] if r["name"] == "other_models")
        assert others["accepted"] is False
        if others["detail"].get("confirmation"):
            assert others["detail"]["confirmation"]["status"] == "not_confirmed"
            assert "led once tuned" in others["outcome"] and "the incumbent was kept" in others["outcome"]


class TestOnDataWithNoSignal:
    """The seeded check the issue asks for, kept small. Per seed, a setting accepted with the confirmation is
    always one that would have been accepted without it, so the count can only fall."""

    SEEDS = range(8)

    def accepted(self, monkeypatch, confirm: bool) -> list[bool]:
        monkeypatch.setattr(improvement, "CONFIRM_SEARCH_WINNERS", confirm)
        out = []
        for seed in self.SEEDS:
            found = improvement.improve(
                noise(seed=seed), target_column="outcome", task="classification",
                model_names=["logistic_regression"], budget=12,
            )
            out.append(settings_round(found)["accepted"])
        return out

    def test_the_confirmation_never_accepts_what_the_old_rule_refused_and_accepts_at_most_one(self, monkeypatch):
        before = self.accepted(monkeypatch, False)
        after = self.accepted(monkeypatch, True)
        assert all(not on or off for on, off in zip(after, before))
        assert sum(after) <= 1
        assert sum(after) <= sum(before)
