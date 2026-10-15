"""A date-based split has to be forward-in-time all the way down.

The final test set was already the most recent rows, but every fold inside the training rows
was shuffled: model choice, tuning, the cut-off, the cross-validation spread and the learning
curve were all made by training on later rows and validating on earlier ones. These pin that
every one of those now validates each chunk on a model trained only on rows before it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from app.services import improvement, modeling, operating_point, tuning
from app.services.time_folds import ForwardInTimeFolds
from app.services.operating_point import OperatingPoint


@pytest.fixture
def dated() -> pd.DataFrame:
    rng = np.random.default_rng(8)
    n = 400
    signal = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(0, 1, n),
            "joined": pd.date_range("2021-01-01", periods=n, freq="D").astype(str),
            "outcome": np.where(rng.random(n) < 1 / (1 + np.exp(-2 * signal)), "yes", "no"),
        }
    )


@pytest.fixture
def splitters(monkeypatch):
    """Every fold splitter anything asks for, whichever module asked."""
    made = []
    real = tuning.cv_splitter

    def spy(*args, **kwargs):
        splitter = real(*args, **kwargs)
        made.append(splitter)
        return splitter

    monkeypatch.setattr(tuning, "cv_splitter", spy)
    monkeypatch.setattr(operating_point, "cv_splitter", spy)
    return made


def test_an_ordered_splitter_only_ever_validates_forwards():
    y = pd.Series(["yes", "no"] * 50)
    splitter = tuning.cv_splitter("classification", y, ordered=True)
    for train_idx, test_idx in splitter.split(np.zeros(len(y)), y):
        assert test_idx.min() > train_idx.max()


def test_choosing_the_model_uses_forward_folds(dated, splitters):
    modeling.train_candidates(
        dated, target_column="outcome", task="classification",
        model_names=["logistic_regression", "random_forest_classifier"],
        include_deep_diagnostics=False, split_strategy="time", time_column="joined",
    )
    assert splitters and all(isinstance(s, ForwardInTimeFolds) for s in splitters)


def test_the_cut_off_is_chosen_on_forward_folds(dated, splitters):
    result = modeling.train_candidates(
        dated, target_column="outcome", task="classification",
        model_names=["logistic_regression"], include_deep_diagnostics=False,
        split_strategy="time", time_column="joined", point=OperatingPoint("f1"),
    )
    point = result["operating_point"]
    assert point is not None and point["folds"] >= 2
    assert "later than it trained on" in point["chosen_on"]
    assert all(isinstance(s, ForwardInTimeFolds) for s in splitters)
    # The earliest chunk is only ever trained on, so fewer rows inform the choice than train.
    assert point["expected"]["rows"] < result["split"]["train_rows"]


def test_the_shown_cross_validation_is_forward_too(dated):
    result = modeling.train_candidates(
        dated, target_column="outcome", task="classification",
        model_names=["logistic_regression"], include_deep_diagnostics=True,
        split_strategy="time", time_column="joined",
    )
    assert result["deep_diagnostics"]["cross_validation"]["ordered"] is True
    assert result["deep_diagnostics"]["learning_curve"]


def test_the_improvement_loop_searches_on_forward_folds(dated, splitters):
    improvement.improve(
        dated, target_column="outcome", task="classification",
        model_names=["logistic_regression"], budget=4,
        split_strategy="time", time_column="joined",
    )
    assert splitters and all(isinstance(s, ForwardInTimeFolds) for s in splitters)


def test_a_random_split_still_uses_ordinary_folds(dated, splitters):
    modeling.train_candidates(
        dated, target_column="outcome", task="classification",
        model_names=["logistic_regression", "random_forest_classifier"],
        include_deep_diagnostics=False, point=OperatingPoint("f1"),
    )
    assert splitters and all(isinstance(s, StratifiedKFold) for s in splitters)


class TestTheNewSplitterIsUsedEverywhereTheOldOneWas:
    """A fold producer that was never handed the timestamps silently reverts to cutting at row
    positions. Each one the date split reaches is pinned here."""

    def test_the_cross_validated_spread_uses_it(self, dated, monkeypatch):
        import sklearn.model_selection as selection

        seen = []
        real = selection.cross_val_score
        monkeypatch.setattr(
            selection, "cross_val_score", lambda *a, **k: (seen.append(k.get("cv")), real(*a, **k))[1]
        )
        modeling.train_candidates(
            dated, target_column="outcome", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=True, split_strategy="time", time_column="joined",
        )
        assert seen and all(isinstance(cv, ForwardInTimeFolds) for cv in seen)

    def test_the_learning_curve_uses_it(self, dated, monkeypatch):
        import sklearn.model_selection as selection

        seen = []
        real = selection.learning_curve
        monkeypatch.setattr(
            selection, "learning_curve", lambda *a, **k: (seen.append(k.get("cv")), real(*a, **k))[1]
        )
        modeling.train_candidates(
            dated, target_column="outcome", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=True, split_strategy="time", time_column="joined",
        )
        assert seen and all(isinstance(cv, ForwardInTimeFolds) for cv in seen)

    def test_the_calibrator_uses_it(self, dated, splitters):
        result = modeling.train_candidates(
            dated, target_column="outcome", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False, split_strategy="time", time_column="joined", calibration="sigmoid",
        )
        assert result["calibration"]["method"] == "sigmoid"
        assert splitters and all(isinstance(s, ForwardInTimeFolds) for s in splitters)

    def test_the_cut_off_with_calibration_uses_it_inside_the_calibrators_own_folds(self, dated, splitters):
        modeling.train_candidates(
            dated, target_column="outcome", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False, split_strategy="time", time_column="joined",
            calibration="sigmoid", point=OperatingPoint("f1"),
        )
        assert len(splitters) > 5 and all(isinstance(s, ForwardInTimeFolds) for s in splitters)

    def test_the_timestamps_are_the_columns_not_the_row_positions(self, dated, splitters):
        shuffled = dated.sample(frac=1, random_state=3)  # file order says nothing about time
        modeling.train_candidates(
            shuffled, target_column="outcome", task="classification",
            model_names=["logistic_regression", "random_forest_classifier"],
            include_deep_diagnostics=False, split_strategy="time", time_column="joined",
        )
        stamps = pd.to_datetime(shuffled["joined"])
        X_train, _, _, _ = modeling.split_data(
            shuffled, "outcome", "classification", strategy="time", time_column="joined"
        )
        assert splitters
        for splitter in splitters:
            for train, test in splitter.split(X_train):
                assert stamps.loc[X_train.index[train]].max() < stamps.loc[X_train.index[test]].min()

    def test_a_legacy_flag_with_no_timestamps_still_cuts_forwards(self):
        y = pd.Series(["yes", "no"] * 50)
        assert type(tuning.cv_splitter("classification", y, ordered=True)).__name__ == "TimeSeriesSplit"

    def test_a_rule_turns_into_the_new_splitter(self):
        from app.services.time_folds import TimeFoldRule

        y = pd.Series(["yes", "no"] * 50)
        stamps = pd.Series(pd.date_range("2021-01-01", periods=100))
        assert isinstance(tuning.cv_splitter("classification", y, ordered=TimeFoldRule(stamps)), ForwardInTimeFolds)
