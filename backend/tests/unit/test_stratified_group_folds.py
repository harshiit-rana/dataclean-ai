"""Grouped folds that also balance the classes.

A grouped split keeps each entity whole so a model is never scored on a customer it trained on, but
a plain GroupKFold balances only how many rows land in each fold. On a lopsided target with few
entities one fold could end up with almost none of the rarer answer -- and its score, or the cut-off
chosen on it, rested on a handful of cases. StratifiedGroupKFold keeps groups whole and balances the
answers as far as the grouping allows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from app.services import diagnostics, modeling, tuning


def customers(n_groups: int = 40, per: int = 25, rate: float = 0.10, seed: int = 0):
    """Forty customers of unequal size, the positives concentrated in a few of them: the case
    where balancing by rows alone leaves some folds thin."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(per // 2, per * 2, n_groups)
    groups = np.repeat(np.arange(n_groups), sizes)
    positive_groups = rng.choice(n_groups, size=max(2, int(n_groups * rate * 1.5)), replace=False)
    y = np.zeros(len(groups), dtype=int)
    for g in positive_groups:
        rows = np.flatnonzero(groups == g)
        y[rows[: max(1, int(len(rows) * 0.6))]] = 1
    y = pd.Series(np.where(y == 1, "yes", "no"))
    return pd.DataFrame({"x": rng.normal(size=len(y))}), y, pd.Series(groups)


def positive_rates(splitter, X, y, groups):
    return [float((y.iloc[test] == "yes").mean()) for _, test in splitter.split(X, y, groups)]


class TestTheSplitter:
    def test_a_grouped_classification_gets_the_stratified_one(self):
        _, y, groups = customers()
        assert isinstance(tuning.cv_splitter("classification", y, 5, groups), StratifiedGroupKFold)

    def test_a_grouped_regression_is_unchanged(self):
        _, y, groups = customers()
        assert type(tuning.cv_splitter("regression", pd.Series(np.arange(len(y)), dtype=float), 5, groups)) is GroupKFold

    def test_the_ungrouped_and_date_splitters_are_unchanged(self):
        _, y, groups = customers()
        assert isinstance(tuning.cv_splitter("classification", y, 5), StratifiedKFold)
        assert isinstance(tuning.cv_splitter("regression", pd.Series(np.arange(len(y)), dtype=float), 5), KFold)

    def test_it_is_shuffled_with_a_fixed_seed_so_every_candidate_gets_the_same_folds(self):
        X, y, groups = customers()
        a, b = tuning.cv_splitter("classification", y, 5, groups), tuning.cv_splitter("classification", y, 5, groups)
        first = [(tr.tolist(), te.tolist()) for tr, te in a.split(X, y, groups)]
        again = [(tr.tolist(), te.tolist()) for tr, te in b.split(X, y, groups)]
        assert first == again

    def test_the_fold_count_is_still_capped_by_the_number_of_groups(self):
        _, y, _ = customers(n_groups=40)
        few = pd.Series(np.repeat([0, 1, 2], len(y) // 3 + 1)[: len(y)])
        assert tuning.cv_splitter("classification", y, 5, few).n_splits == 3
        assert tuning.cv_splitter("classification", y, 5, pd.Series([0] * len(y))) is None


class TestWholeGroups:
    def test_no_group_is_on_both_sides_of_any_fold(self):
        X, y, groups = customers()
        for train, test in tuning.cv_splitter("classification", y, 5, groups).split(X, y, groups):
            assert not (set(groups.iloc[train]) & set(groups.iloc[test]))

    def test_every_row_is_validated_exactly_once(self):
        X, y, groups = customers()
        seen = np.concatenate([test for _, test in tuning.cv_splitter("classification", y, 5, groups).split(X, y, groups)])
        assert sorted(seen.tolist()) == list(range(len(y)))


class TestBetterBalanced:
    def test_the_positive_rate_varies_less_across_folds_than_with_the_old_splitter(self):
        """The issue's benchmark: forty customers and a 10% positive rate."""
        X, y, groups = customers()
        old = positive_rates(GroupKFold(n_splits=5), X, y, groups)
        new = positive_rates(tuning.cv_splitter("classification", y, 5, groups), X, y, groups)
        assert max(new) - min(new) < max(old) - min(old)
        assert np.std(new) < np.std(old)

    def test_no_fold_is_left_without_the_rarer_answer(self):
        X, y, groups = customers()
        assert min(positive_rates(tuning.cv_splitter("classification", y, 5, groups), X, y, groups)) > 0

    @pytest.mark.parametrize("seed", range(5))
    def test_it_holds_across_different_files(self, seed):
        X, y, groups = customers(seed=seed)
        old = positive_rates(GroupKFold(n_splits=5), X, y, groups)
        new = positive_rates(tuning.cv_splitter("classification", y, 5, groups), X, y, groups)
        assert np.std(new) <= np.std(old) + 1e-9


class TestEveryFoldProducerAgrees:
    def test_the_cross_validated_spread_uses_it(self, monkeypatch):
        import sklearn.model_selection as selection

        seen = {}
        real = selection.cross_val_score

        def spy(*args, **kwargs):
            seen["cv"] = kwargs.get("cv")
            return real(*args, **kwargs)

        monkeypatch.setattr(selection, "cross_val_score", spy)
        X, y, groups = customers()
        found = diagnostics.cross_validate_pipeline(
            lambda: modeling.build_pipeline("logistic_regression", ["x"], []), X, y, "classification", groups=groups
        )
        assert isinstance(seen["cv"], StratifiedGroupKFold) and found["grouped"] is True

    def test_the_spread_for_a_regression_still_uses_the_plain_one(self, monkeypatch):
        import sklearn.model_selection as selection

        seen = {}
        real = selection.cross_val_score
        monkeypatch.setattr(
            selection, "cross_val_score", lambda *a, **k: (seen.update(cv=k.get("cv")), real(*a, **k))[1]
        )
        X, _, groups = customers()
        target = pd.Series(np.random.default_rng(1).normal(size=len(X)))
        diagnostics.cross_validate_pipeline(
            lambda: modeling.build_pipeline("linear_regression", ["x"], []), X, target, "regression", groups=groups
        )
        assert type(seen["cv"]) is GroupKFold

    def test_the_learning_curve_uses_it_too(self, monkeypatch):
        import sklearn.model_selection as selection

        seen = {}
        real = selection.learning_curve
        monkeypatch.setattr(
            selection, "learning_curve", lambda *a, **k: (seen.update(cv=k.get("cv")), real(*a, **k))[1]
        )
        X, y, groups = customers()
        diagnostics.learning_curve_points(
            lambda: modeling.build_pipeline("logistic_regression", ["x"], []), X, y, "classification", groups=groups
        )
        assert isinstance(seen["cv"], StratifiedGroupKFold)

    def test_a_grouped_run_trains_end_to_end(self):
        rng = np.random.default_rng(3)
        base = np.repeat(rng.normal(size=120), 8)
        df = pd.DataFrame(
            {"customer": np.repeat(np.arange(120), 8), "x": base + rng.normal(0, 0.5, 960),
             "y": np.where(base + rng.normal(0, 0.5, 960) > 1.0, "yes", "no")}
        )
        result = modeling.train_candidates(
            df, target_column="y", task="classification", model_names=["logistic_regression", "random_forest_classifier"],
            split_strategy="group", group_column="customer",
        )
        assert result["selection"]["method"] == "cross_validation"
        assert result["deep_diagnostics"]["cross_validation"]["grouped"] is True
