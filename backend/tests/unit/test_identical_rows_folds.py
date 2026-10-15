"""Copies kept together at the split are kept together in every fold, too.

If the train/test split keeps copies of a row on one side but the folds inside the training rows
do not, model choice and tuning are still made on folds where a row and its copy straddle the
boundary -- the same leak, one level down. The project learned this for grouped splits ("grouped
all the way down"); this pins the same for identical rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, StratifiedKFold

# What an ungrouped random split uses: repeated on a small file when models are compared.
ORDINARY = (StratifiedKFold, RepeatedStratifiedKFold)

from app.services import diagnostics, improvement, modeling, operating_point, tuning
from app.services.operating_point import OperatingPoint


@pytest.fixture
def copies() -> pd.DataFrame:
    rng = np.random.default_rng(12)
    n = 220
    base = pd.DataFrame({"x": rng.normal(0, 1, n).round(4), "z": rng.normal(0, 1, n).round(4)})
    base["y"] = np.where(base["x"] + rng.normal(0, 0.8, n) > 0, "yes", "no")
    return pd.concat([base, base.sample(80, random_state=3)], ignore_index=True)


@pytest.fixture
def folds(monkeypatch):
    """Every fold request anything makes: the splitter, the groups it was given, and the
    labels it was asked to fold, whichever module asked."""
    made: list[dict] = []
    real = tuning.cv_splitter

    def spy(task, y, folds=tuning.CV_FOLDS, groups=None, ordered=False, repeated=False):
        splitter = real(task, y, folds, groups=groups, ordered=ordered, repeated=repeated)
        made.append({"splitter": splitter, "groups": groups, "y": y})
        return splitter

    monkeypatch.setattr(tuning, "cv_splitter", spy)
    monkeypatch.setattr(operating_point, "cv_splitter", spy)
    return made


def assert_no_group_straddles(made: list[dict]) -> None:
    assert made, "no folds were requested, so there is nothing to check"
    for entry in made:
        groups = np.asarray(entry["groups"])
        y = np.asarray(entry["y"])
        for train, test in entry["splitter"].split(np.zeros(len(y)), y, groups):
            assert not set(groups[train]) & set(groups[test])


def train(df, **kwargs):
    return modeling.train_candidates(
        df, target_column="y", task="classification",
        model_names=["logistic_regression", "random_forest_classifier"],
        include_deep_diagnostics=False, **kwargs,
    )


class TestChoosingTheModel:
    def test_every_fold_is_grouped_when_copies_were_kept_together(self, copies, folds):
        train(copies, keep_identical_rows_together=True)
        assert all(isinstance(f["splitter"], StratifiedGroupKFold) for f in folds)
        assert_no_group_straddles(folds)

    def test_no_copy_is_on_both_sides_of_any_fold(self, copies, folds):
        """The issue's check, on the splitter's own indices."""
        result = train(copies, keep_identical_rows_together=True)
        assert result["split"]["keep_duplicates_together"] is True
        assert_no_group_straddles(folds)

    def test_without_the_option_the_folds_are_the_ordinary_ones(self, copies, folds):
        train(copies)
        assert folds and all(isinstance(f["splitter"], ORDINARY) for f in folds)

    def test_the_groups_are_the_training_rows_copy_ids(self, copies, folds):
        train(copies, keep_identical_rows_together=True)
        ids = modeling.identical_row_ids(copies, ["x", "z"], "y")
        X_train, _, _, _ = modeling.split_data(
            copies, "y", "classification", duplicate_columns=["x", "z"]
        )
        expected = ids.loc[list(X_train.index)].to_numpy()
        assert all(np.array_equal(np.asarray(f["groups"]), expected) for f in folds)

    def test_asking_on_a_file_with_no_copies_leaves_the_folds_alone(self, copies, folds):
        train(copies.drop_duplicates(ignore_index=True), keep_identical_rows_together=True)
        assert folds and all(isinstance(f["splitter"], ORDINARY) for f in folds)

    def test_an_entity_grouping_wins_over_copies(self, folds):
        """A grouped split keeps copies together already, since they share an entity; the
        folds are grouped by the entity, not by the copy id."""
        rng = np.random.default_rng(5)
        customers = 60
        base = pd.DataFrame(
            {
                "customer": np.repeat([f"c{i:03d}" for i in range(customers)], 5),
                "x": rng.normal(0, 1, customers * 5).round(3),
            }
        )
        base["y"] = np.repeat(rng.choice(["yes", "no"], customers), 5)
        train(base, split_strategy="group", group_column="customer", keep_identical_rows_together=True)
        entity_groups = {tuple(sorted(set(np.asarray(f["groups"])))) for f in folds}
        assert all(str(g).startswith("c") for groups in entity_groups for g in groups)


class TestTheOtherFoldProducers:
    def test_the_cut_off_is_chosen_on_grouped_folds(self, copies, folds):
        result = train(copies, keep_identical_rows_together=True, point=OperatingPoint("f1"))
        assert result["operating_point"]["folds"] >= 2
        assert_no_group_straddles(folds)

    def test_the_deep_diagnostics_receive_the_groups(self, copies, monkeypatch):
        seen = {}
        real_cv = diagnostics.cross_validate_pipeline
        real_curve = diagnostics.learning_curve_points

        def cv(build, X, y, task, *args, **kwargs):
            seen["cv"] = kwargs.get("groups")
            return real_cv(build, X, y, task, *args, **kwargs)

        def curve(build, X, y, task, *args, **kwargs):
            seen["curve"] = kwargs.get("groups")
            return real_curve(build, X, y, task, *args, **kwargs)

        monkeypatch.setattr(diagnostics, "cross_validate_pipeline", cv)
        monkeypatch.setattr(diagnostics, "learning_curve_points", curve)

        modeling.train_candidates(
            copies, target_column="y", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=True,
            keep_identical_rows_together=True,
        )
        assert seen["cv"] is not None and seen["curve"] is not None
        assert len(seen["cv"]) == len(seen["curve"])

    def test_the_improvement_loop_searches_on_grouped_folds(self, copies, folds):
        improvement.improve(
            copies, target_column="y", task="classification",
            model_names=["logistic_regression"], budget=4, keep_identical_rows_together=True,
        )
        assert folds and all(isinstance(f["splitter"], StratifiedGroupKFold) for f in folds)
        assert_no_group_straddles(folds)

    def test_the_improvement_loop_without_it_is_unchanged(self, copies, folds):
        improvement.improve(
            copies, target_column="y", task="classification",
            model_names=["logistic_regression"], budget=4,
        )
        assert folds and all(isinstance(f["splitter"], ORDINARY) for f in folds)
