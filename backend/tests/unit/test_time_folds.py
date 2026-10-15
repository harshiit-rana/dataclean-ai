"""Forward-in-time folds that keep same-timestamp rows together and can leave a gap.

`TimeSeriesSplit` cuts at row positions. On daily data, or records loaded in batches, that falls in
the middle of a day: a fold trains on part of a day and is validated on the rest of it, which is not
predicting the future. This splitter takes the timestamps explicitly and moves every boundary to the
nearest change of timestamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from app.services.time_folds import ForwardInTimeFolds


def frame(n: int, per_day: int = 1, start: str = "2021-01-01") -> tuple[pd.DataFrame, pd.Series]:
    days = pd.date_range(start, periods=(n + per_day - 1) // per_day)
    stamps = pd.Series(np.repeat(days.to_numpy(), per_day)[:n])
    X = pd.DataFrame({"v": np.arange(n, dtype=float)})
    return X, stamps.set_axis(X.index)


def folds(splitter, X):
    return [(train.tolist(), test.tolist()) for train, test in splitter.split(X)]


class TestAgreesWithTheStandardOneWhereThereAreNoTies:
    @pytest.mark.parametrize("n,k", [(600, 5), (101, 3), (60, 4), (1000, 5)])
    def test_distinct_timestamps_give_exactly_the_time_series_split(self, n, k):
        X, stamps = frame(n)
        assert folds(ForwardInTimeFolds(stamps, n_splits=k), X) == [
            (a.tolist(), b.tolist()) for a, b in TimeSeriesSplit(n_splits=k).split(X)
        ]


class TestNoTimestampStraddlesAFold:
    @pytest.mark.parametrize("per_day", [2, 3, 7, 24])
    def test_no_timestamp_is_in_both_the_training_and_validation_rows_of_any_fold(self, per_day):
        X, stamps = frame(840, per_day)
        for train, test in ForwardInTimeFolds(stamps, n_splits=5).split(X):
            assert not (set(stamps.iloc[train]) & set(stamps.iloc[test]))

    def test_the_standard_splitter_does_straddle_so_the_check_has_teeth(self):
        X, stamps = frame(840, per_day=11)
        straddles = [
            bool(set(stamps.iloc[tr]) & set(stamps.iloc[te])) for tr, te in TimeSeriesSplit(n_splits=5).split(X)
        ]
        assert any(straddles)

    def test_every_validation_row_is_later_than_every_training_row(self):
        X, stamps = frame(840, per_day=5)
        for train, test in ForwardInTimeFolds(stamps, n_splits=5).split(X):
            assert stamps.iloc[train].max() < stamps.iloc[test].min()

    def test_each_row_is_validated_at_most_once_and_chunks_move_forward(self):
        X, stamps = frame(840, per_day=5)
        tests = [t for _, t in ForwardInTimeFolds(stamps, n_splits=5).split(X)]
        flat = np.concatenate(tests)
        assert len(flat) == len(set(flat))
        assert all(a.max() < b.min() for a, b in zip(tests, tests[1:]))

    def test_the_first_chunk_is_only_ever_trained_on(self):
        X, stamps = frame(600)
        first_train = next(iter(ForwardInTimeFolds(stamps, n_splits=5).split(X)))[0]
        validated = set(np.concatenate([t for _, t in ForwardInTimeFolds(stamps, n_splits=5).split(X)]))
        assert not (set(first_train) & validated)

    def test_the_chunks_stay_close_to_equal_size(self):
        X, stamps = frame(840, per_day=7)
        sizes = [len(t) for _, t in ForwardInTimeFolds(stamps, n_splits=5).split(X)]
        assert max(sizes) - min(sizes) <= 14  # a day or two either way


class TestTheGap:
    def test_training_stops_at_least_the_gap_before_validation_begins(self):
        X, stamps = frame(600)
        for train, test in ForwardInTimeFolds(stamps, n_splits=5, gap_days=30).split(X):
            assert stamps.iloc[test].min() - stamps.iloc[train].max() >= pd.Timedelta(days=30)

    def test_it_drops_rows_it_does_not_move_the_validation_chunks(self):
        X, stamps = frame(600)
        with_gap = [t for _, t in ForwardInTimeFolds(stamps, n_splits=5, gap_days=30).split(X)]
        without = [t for _, t in ForwardInTimeFolds(stamps, n_splits=5).split(X)]
        assert [t.tolist() for t in with_gap] == [t.tolist() for t in without]

    def test_the_gap_and_the_tie_rule_hold_together(self):
        X, stamps = frame(840, per_day=7)
        for train, test in ForwardInTimeFolds(stamps, n_splits=5, gap_days=10).split(X):
            assert stamps.iloc[test].min() - stamps.iloc[train].max() >= pd.Timedelta(days=10)

    def test_a_fold_with_nothing_left_to_train_on_is_skipped_not_returned_empty(self):
        X, stamps = frame(600)
        found = list(ForwardInTimeFolds(stamps, n_splits=5, gap_days=250).split(X))
        assert found and all(len(train) > 0 for train, _ in found)
        assert len(found) < 5

    def test_a_gap_longer_than_the_data_leaves_no_folds(self):
        X, stamps = frame(600)
        assert list(ForwardInTimeFolds(stamps, n_splits=5, gap_days=5000).split(X)) == []


class TestTimestampsAreGivenNotGuessed:
    def test_the_order_comes_from_the_timestamps_not_from_the_row_positions(self):
        X, stamps = frame(300)
        shuffled = X.sample(frac=1, random_state=0)
        for train, test in ForwardInTimeFolds(stamps, n_splits=4).split(shuffled):
            before = stamps.loc[shuffled.index[train]].max()
            after = stamps.loc[shuffled.index[test]].min()
            assert before < after

    def test_a_subset_of_the_rows_is_folded_by_the_labels_it_carries(self):
        """As inside a cut-off's own folds: fewer rows, same labels."""
        X, stamps = frame(600)
        subset = X.iloc[100:500]
        for train, test in ForwardInTimeFolds(stamps, n_splits=5).split(subset):
            assert stamps.loc[subset.index[train]].max() < stamps.loc[subset.index[test]].min()

    def test_rows_without_a_readable_date_sort_last_and_count_as_one_timestamp(self):
        X, stamps = frame(300)
        stamps = stamps.astype(object)
        stamps.iloc[-20:] = "not a date"
        for train, test in ForwardInTimeFolds(stamps, n_splits=4).split(X):
            assert not (set(stamps.iloc[train].astype(str)) & set(stamps.iloc[test].astype(str)))

    def test_it_is_deterministic(self):
        X, stamps = frame(600, per_day=4)
        a, b = ForwardInTimeFolds(stamps, 5, 7), ForwardInTimeFolds(stamps, 5, 7)
        assert folds(a, X) == folds(b, X)

    def test_too_few_rows_is_refused_like_the_standard_one(self):
        X, stamps = frame(4)
        with pytest.raises(ValueError):
            list(ForwardInTimeFolds(stamps, n_splits=5).split(X))

    def test_it_works_inside_cross_validation(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        X, stamps = frame(600, per_day=3)
        y = pd.Series(np.arange(600.0) * 2)
        scores = cross_val_score(LinearRegression(), X, y, cv=ForwardInTimeFolds(stamps, n_splits=5), scoring="r2")
        assert len(scores) == 5 and np.all(np.isfinite(scores))

    def test_the_class_is_self_contained_so_the_export_can_copy_its_source(self):
        import inspect

        source = inspect.getsource(ForwardInTimeFolds)
        namespace = {"np": np, "pd": pd}
        exec(source, namespace)  # noqa: S102 -- the text under test
        assert "ForwardInTimeFolds" in namespace
