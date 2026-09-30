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
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from app.services import improvement, modeling, operating_point, tuning
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
    assert splitters and all(isinstance(s, TimeSeriesSplit) for s in splitters)


def test_the_cut_off_is_chosen_on_forward_folds(dated, splitters):
    result = modeling.train_candidates(
        dated, target_column="outcome", task="classification",
        model_names=["logistic_regression"], include_deep_diagnostics=False,
        split_strategy="time", time_column="joined", point=OperatingPoint("f1"),
    )
    point = result["operating_point"]
    assert point is not None and point["folds"] >= 2
    assert "later than it trained on" in point["chosen_on"]
    assert all(isinstance(s, TimeSeriesSplit) for s in splitters)
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
    assert splitters and all(isinstance(s, TimeSeriesSplit) for s in splitters)


def test_a_random_split_still_uses_ordinary_folds(dated, splitters):
    modeling.train_candidates(
        dated, target_column="outcome", task="classification",
        model_names=["logistic_regression", "random_forest_classifier"],
        include_deep_diagnostics=False, point=OperatingPoint("f1"),
    )
    assert splitters and all(isinstance(s, StratifiedKFold) for s in splitters)
