"""Model training from the registry.

The most important test in this file is `test_imputation_is_fit_on_training_rows_only`.
Everything else is behaviour; that one is the correctness property the whole design exists to
protect, and it is asserted against the fitted pipeline's own learned statistics rather than
against a score, so it can't pass by accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import modeling
from app.services.modeling import ModelingError


@pytest.fixture
def classification_df() -> pd.DataFrame:
    """300 rows where `signal` genuinely carries information about the target, so a working
    model beats the always-guess-the-common-answer baseline and a broken one doesn't."""
    rng = np.random.default_rng(5)
    n = 300
    signal = rng.normal(0, 1, n)
    probability = 1 / (1 + np.exp(-(1.8 * signal)))
    target = np.where(rng.random(n) < probability, "yes", "no")
    return pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(0, 1, n),
            "region": rng.choice(["north", "south", "east"], n),
            "joined": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
            "outcome": target,
        }
    )


@pytest.fixture
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    n = 300
    size = rng.uniform(50, 250, n)
    return pd.DataFrame(
        {
            "size": size,
            "noise": rng.normal(0, 1, n),
            "kind": rng.choice(["flat", "house"], n),
            "price": size * 1000 + rng.normal(0, 5000, n),
        }
    )


class TestFeatureClassification:
    def test_splits_numeric_and_categorical(self, classification_df):
        features = modeling.classify_features(classification_df, "outcome")
        assert set(features["numeric"]) == {"signal", "noise"}
        assert "region" in features["categorical"]

    def test_target_is_never_a_feature(self, classification_df):
        features = modeling.classify_features(classification_df, "outcome")
        assert "outcome" not in features["numeric"] + features["categorical"]

    def test_constant_columns_are_dropped_with_a_reason(self, classification_df):
        classification_df["always_same"] = "x"
        features = modeling.classify_features(classification_df, "outcome")
        dropped = {d["column"]: d["reason"] for d in features["dropped"]}
        assert "always_same" in dropped
        assert "same value" in dropped["always_same"]

    def test_empty_columns_are_dropped(self, classification_df):
        classification_df["nothing"] = None
        features = modeling.classify_features(classification_df, "outcome")
        assert "nothing" in {d["column"] for d in features["dropped"]}

    def test_high_cardinality_text_is_dropped(self, classification_df):
        """A near-unique text column one-hot encodes into hundreds of columns and lets the
        model memorize rows instead of learning anything."""
        classification_df["employee_id"] = [f"E{i}" for i in range(len(classification_df))]
        features = modeling.classify_features(classification_df, "outcome")
        dropped = {d["column"]: d["reason"] for d in features["dropped"]}
        assert "employee_id" in dropped
        assert "too many" in dropped["employee_id"]

    def test_the_category_limit_scales_with_the_row_count(self):
        """"Too many categories" is a statement about how many rows each category gets, not
        an absolute number. A fixed cap of 30 threw away a 36-value department column on an
        809-row file -- 22 rows each, plainly learnable, and the best column in the data."""
        assert modeling.category_limit(809) >= 36
        # Floor: a small file shouldn't lose every categorical column it has.
        assert modeling.category_limit(60) == modeling.MIN_CATEGORIES_TO_ENCODE
        # Ceiling: a large one shouldn't one-hot free text just because it has the rows.
        assert modeling.category_limit(500_000) == modeling.MAX_CATEGORIES_TO_ENCODE

    def test_a_column_the_row_count_can_support_is_kept(self, classification_df):
        codes = [f"dept_{i % 36}" for i in range(len(classification_df))]
        classification_df["department"] = codes
        big = pd.concat([classification_df] * 3, ignore_index=True)  # 900 rows
        features = modeling.classify_features(big, "outcome")
        assert "department" in features["categorical"]

    def test_the_same_column_is_dropped_when_there_are_too_few_rows(self, classification_df):
        small = classification_df.head(60).copy()
        small["department"] = [f"dept_{i % 36}" for i in range(len(small))]
        features = modeling.classify_features(small, "outcome")
        dropped = {d["column"]: d["reason"] for d in features["dropped"]}
        assert "department" in dropped
        assert "60 rows" in dropped["department"]  # the reason names what it was measured against

    def test_user_exclusions_are_honoured_and_explained(self, classification_df):
        features = modeling.classify_features(classification_df, "outcome", ["noise"])
        assert "noise" not in features["numeric"]
        assert any(
            d["column"] == "noise" and "excluded" in d["reason"] for d in features["dropped"]
        )


class TestSplitting:
    def test_holds_back_a_test_set(self, classification_df):
        X_train, X_test, y_train, y_test = modeling.split_data(
            classification_df, "outcome", "classification"
        )
        assert len(X_test) == pytest.approx(len(classification_df) * 0.2, abs=2)
        assert len(X_train) + len(X_test) == len(classification_df)

    def test_train_and_test_rows_never_overlap(self, classification_df):
        X_train, X_test, _y_train, _y_test = modeling.split_data(
            classification_df, "outcome", "classification"
        )
        assert not set(X_train.index) & set(X_test.index)

    def test_rows_without_an_answer_are_excluded(self, classification_df):
        classification_df.loc[:19, "outcome"] = None
        X_train, X_test, _yt, _ys = modeling.split_data(
            classification_df, "outcome", "classification"
        )
        assert len(X_train) + len(X_test) == len(classification_df) - 20

    def test_classification_split_keeps_both_classes_on_each_side(self):
        df = pd.DataFrame(
            {"a": range(200), "target": ["no"] * 180 + ["yes"] * 20}
        )
        _Xt, _Xs, y_train, y_test = modeling.split_data(df, "target", "classification")
        assert set(y_train.astype(str)) == {"yes", "no"}
        assert set(y_test.astype(str)) == {"yes", "no"}

    def test_time_split_puts_the_most_recent_rows_in_the_test_set(self, classification_df):
        """A random split lets the model learn from the future to predict the past. On data
        with a time dimension that inflates the score and cannot happen in production."""
        X_train, X_test, _yt, _ys = modeling.split_data(
            classification_df, "outcome", "classification",
            strategy="time", time_column="joined",
        )
        newest_train = pd.to_datetime(X_train["joined"]).max()
        oldest_test = pd.to_datetime(X_test["joined"]).min()
        assert newest_train < oldest_test

    def test_time_split_needs_a_usable_date_column(self, classification_df):
        with pytest.raises(ModelingError, match="date column"):
            modeling.split_data(
                classification_df, "outcome", "classification", strategy="time"
            )

    def test_time_split_rejects_a_column_that_is_not_dates(self, classification_df):
        with pytest.raises(ModelingError, match="could not be read as dates"):
            modeling.split_data(
                classification_df, "outcome", "classification",
                strategy="time", time_column="region",
            )

    def test_too_few_labelled_rows_is_refused_with_an_explanation(self):
        df = pd.DataFrame({"a": range(20), "target": ["yes", "no"] * 10})
        with pytest.raises(ModelingError, match="too few"):
            modeling.split_data(df, "target", "classification")


class TestPipelineLeakageSafety:
    def test_imputation_is_fit_on_training_rows_only(self):
        """The property this whole module is arranged around.

        The imputer's median must come from the training rows alone. If preprocessing were
        done to the dataframe before splitting, the test rows would have helped choose the
        value written into the training rows, and every score afterwards would be flattering.
        Asserted against the fitted imputer's own learned statistic, not against a score.
        """
        train = pd.DataFrame({"x": [10.0] * 80 + [np.nan] * 20})
        test = pd.DataFrame({"x": [9999.0] * 25})
        labels = pd.Series(["yes", "no"] * 50)

        pipeline = modeling.build_pipeline("logistic_regression", ["x"], [])
        pipeline.fit(train, labels)

        imputer = pipeline.named_steps["prepare"].named_transformers_["numeric"].named_steps["impute"]
        assert imputer.statistics_[0] == 10.0, "the test set's 9999s influenced the fill value"

        # And the wildly different test rows transform without changing anything learned.
        pipeline.predict(test)
        assert imputer.statistics_[0] == 10.0

    def test_scaling_is_fit_on_training_rows_only(self):
        train = pd.DataFrame({"x": np.linspace(0, 10, 100)})
        labels = pd.Series(["yes", "no"] * 50)

        pipeline = modeling.build_pipeline("logistic_regression", ["x"], [])
        pipeline.fit(train, labels)

        scaler = pipeline.named_steps["prepare"].named_transformers_["numeric"].named_steps["scale"]
        assert scaler.mean_[0] == pytest.approx(5.0, abs=0.1)

    def test_categories_only_in_the_test_set_do_not_break_prediction(self):
        train = pd.DataFrame({"kind": ["a", "b"] * 50})
        labels = pd.Series(["yes", "no"] * 50)
        pipeline = modeling.build_pipeline("logistic_regression", [], ["kind"])
        pipeline.fit(train, labels)

        # 'z' was never seen during training -- in production this happens constantly.
        assert len(pipeline.predict(pd.DataFrame({"kind": ["z", "a"]}))) == 2

    def test_unknown_model_is_refused(self):
        with pytest.raises(ModelingError, match="Unknown model"):
            modeling.build_pipeline("neural_net_9000", ["x"], [])

    def test_pipeline_with_no_features_is_refused(self):
        with pytest.raises(ModelingError, match="no usable columns"):
            modeling.build_pipeline("logistic_regression", [], [])


class TestEvaluation:
    def test_classification_reports_imbalance_aware_metrics(self, classification_df):
        result = modeling.train_candidates(
            classification_df,
            target_column="outcome",
            task="classification",
            model_names=["logistic_regression"],
        )
        metrics = result["results"][0]["metrics"]
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert key in metrics, f"{key} missing -- accuracy alone flatters a lopsided target"
        assert metrics["confusion_matrix"]

    def test_regression_reports_error_magnitudes(self, regression_df):
        result = modeling.train_candidates(
            regression_df,
            target_column="price",
            task="regression",
            model_names=["linear_regression"],
        )
        metrics = result["results"][0]["metrics"]
        assert set(metrics) == {"r2", "mae", "rmse"}
        assert metrics["r2"] > 0.9  # price is a near-linear function of size

    def test_multiclass_is_scored_without_a_positive_label(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame(
            {"a": rng.normal(0, 1, 200), "grade": rng.choice(["A", "B", "C"], 200)}
        )
        result = modeling.train_candidates(
            df, target_column="grade", task="classification",
            model_names=["logistic_regression"],
        )
        metrics = result["results"][0]["metrics"]
        assert metrics["positive_label"] is None
        assert len(metrics["labels"]) == 3
        assert "roc_auc" not in metrics  # only meaningful for two classes

    def test_baseline_is_reported_so_a_useless_model_is_visible(self, classification_df):
        """A model that can't beat 'always guess the common answer' is doing nothing, and
        that is easy to miss when accuracy alone looks high."""
        result = modeling.train_candidates(
            classification_df, target_column="outcome", task="classification",
            model_names=["logistic_regression"],
        )
        assert "accuracy" in result["baseline"]
        assert "always answer" in result["baseline"]["strategy"]

    def test_regression_baseline_is_the_mean(self, regression_df):
        result = modeling.train_candidates(
            regression_df, target_column="price", task="regression",
            model_names=["linear_regression"],
        )
        assert "average" in result["baseline"]["strategy"]
        assert result["results"][0]["metrics"]["r2"] > result["baseline"]["r2"]


class TestAssessment:
    """A model that learned nothing must say so. On a 70/30 target a model that has learned
    *nothing at all* still reports 70% accuracy, which reads like success."""

    def test_coin_flip_classifier_is_called_out(self):
        metrics = {"accuracy": 0.70, "roc_auc": 0.51}
        result = modeling.assess_result(metrics, {"accuracy": 0.70}, "classification")
        assert result["verdict"] == "no_better_than_guessing"
        assert "coin flip" in result["headline"]

    def test_the_bar_rises_when_the_test_set_is_small(self):
        """AUC 0.65 means something on 2000 test rows and nothing on 60. A fixed threshold
        reported noise as signal -- this was found by training on data with no signal in it
        and watching the verdict come back 'marginal'."""
        big = modeling.assess_result(
            {"accuracy": 0.7, "roc_auc": 0.65}, {"accuracy": 0.7}, "classification",
            test_rows=2000, class_counts=[1400, 600],
        )
        small = modeling.assess_result(
            {"accuracy": 0.7, "roc_auc": 0.65}, {"accuracy": 0.7}, "classification",
            test_rows=60, class_counts=[42, 18],
        )
        assert big["verdict"] == "marginal"
        assert small["verdict"] == "no_better_than_guessing"
        assert "by luck" in small["headline"]

    def test_a_test_set_too_small_to_judge_gets_no_verdict(self):
        """Found live on the RSNA knee dataset. 4,407 rows, 58 of them labelled, so a 20%
        split left twelve test rows -- and the app called AUC 0.89 on those twelve "useful".
        One row landing differently moves that number by 0.08."""
        result = modeling.assess_result(
            {"accuracy": 0.83, "roc_auc": 0.89}, {"accuracy": 0.58}, "classification",
            test_rows=12, class_counts=[7, 5],
        )
        assert result["verdict"] == "not_enough_test_data"
        assert "0.89" not in result["headline"], "quoting the score re-asserts what it denies"

    def test_it_says_unjudgeable_rather_than_bad(self):
        """A different claim, and the distinction is the point: "we cannot tell" must not be
        delivered as "this does not work", which would be just as unfounded."""
        result = modeling.assess_result(
            {"accuracy": 0.5, "roc_auc": 0.52}, {"accuracy": 0.5}, "classification",
            test_rows=14, class_counts=[7, 7],
        )
        assert result["verdict"] == "not_enough_test_data"
        assert "not a statement that the model is bad" in result["headline"]

    def test_a_rare_class_can_be_too_thin_even_on_enough_rows(self):
        """200 test rows sounds ample until you notice three of them carry the answer."""
        result = modeling.assess_result(
            {"accuracy": 0.98, "roc_auc": 0.91}, {"accuracy": 0.98}, "classification",
            test_rows=200, class_counts=[197, 3],
        )
        assert result["verdict"] == "not_enough_test_data"

    def test_an_adequate_test_set_is_still_judged_normally(self):
        result = modeling.assess_result(
            {"accuracy": 0.88, "roc_auc": 0.91}, {"accuracy": 0.70}, "classification",
            test_rows=400, class_counts=[280, 120],
        )
        assert result["verdict"] == "useful"

    def test_the_noise_floor_is_not_capped(self):
        """The `min(0.75, ...)` that used to sit here fired only on the smallest test sets --
        the exact case the formula exists to be strict about -- and lowered the bar there."""
        assert modeling._auc_noise_threshold([7, 5], model_count=1) > 0.75

    def test_noise_threshold_shrinks_as_evidence_grows(self):
        assert modeling._auc_noise_threshold([42, 18]) > modeling._auc_noise_threshold([1400, 600])
        assert modeling._auc_noise_threshold(None) == 0.60

    def test_accuracy_margin_widens_on_a_small_test_set(self):
        """No AUC to lean on, so beating the baseline by a hair on 40 rows proves nothing."""
        result = modeling.assess_result(
            {"accuracy": 0.45}, {"accuracy": 0.40}, "classification", test_rows=40
        )
        assert result["verdict"] == "no_better_than_guessing"

    def test_strong_classifier_is_called_useful(self):
        result = modeling.assess_result(
            {"accuracy": 0.88, "roc_auc": 0.91}, {"accuracy": 0.70}, "classification"
        )
        assert result["verdict"] == "useful"

    def test_weak_signal_is_marginal_not_useful(self):
        result = modeling.assess_result(
            {"accuracy": 0.72, "roc_auc": 0.64}, {"accuracy": 0.70}, "classification"
        )
        assert result["verdict"] == "marginal"

    def test_multiclass_is_judged_against_the_baseline(self):
        """No AUC available, so the comparison has to be accuracy versus always-guessing."""
        result = modeling.assess_result({"accuracy": 0.41}, {"accuracy": 0.40}, "classification")
        assert result["verdict"] == "no_better_than_guessing"

    def test_regression_at_zero_r2_is_no_better_than_the_mean(self):
        result = modeling.assess_result({"r2": 0.01}, {"r2": 0.0}, "regression")
        assert result["verdict"] == "no_better_than_guessing"
        assert "average" in result["headline"]

    def test_strong_regression_is_useful(self):
        result = modeling.assess_result({"r2": 0.85}, {"r2": 0.0}, "regression")
        assert result["verdict"] == "useful"

    def test_assessment_is_attached_to_a_real_run(self, regression_df):
        result = modeling.train_candidates(
            regression_df, target_column="price", task="regression",
            model_names=["linear_regression"],
        )
        assert result["assessment"]["verdict"] == "useful"

    def test_signal_free_data_is_reported_as_worthless(self):
        """The target is generated independently of every feature, so there is genuinely
        nothing to learn. The app must not present the result as a working model.

        Asserted across many random datasets rather than one: a single seed passing proves
        the threshold happens to sit above that seed's noise, not that it's calibrated. An
        earlier version of this test pinned one seed and hid the fact that the fixed 0.60
        cut-off was letting most noise through.
        """
        verdicts = []
        for seed in range(12):
            rng = np.random.default_rng(seed)
            n = 300
            df = pd.DataFrame(
                {
                    "a": rng.normal(0, 1, n),
                    "b": rng.normal(0, 1, n),
                    "target": rng.choice(["yes", "no"], n, p=[0.3, 0.7]),
                }
            )
            result = modeling.train_candidates(
                df, target_column="target", task="classification",
                model_names=["logistic_regression", "random_forest_classifier"],
            )
            verdicts.append(result["assessment"]["verdict"])

        flagged = verdicts.count("no_better_than_guessing")
        assert flagged >= 10, f"noise slipped through as signal too often: {verdicts}"
        # The failure that actually costs the user something is calling noise a good model.
        assert "useful" not in verdicts


class TestFeatureImportance:
    def test_the_informative_column_outranks_the_noise(self, classification_df):
        result = modeling.train_candidates(
            classification_df, target_column="outcome", task="classification",
            model_names=["random_forest_classifier"],
        )
        ranking = [c["column"] for c in result["results"][0]["column_importance"]]
        assert ranking.index("signal") < ranking.index("noise")

    def test_encoded_columns_roll_up_to_their_source(self):
        """The user can act on "Department", not on "categorical__Department_sales"."""
        importances = [
            {"feature": "categorical__region_north", "importance": 0.25},
            {"feature": "categorical__region_south", "importance": 0.15},
            {"feature": "numeric__signal", "importance": 0.6},
        ]
        rolled = modeling.source_column_importance(importances, ["region", "signal"])
        assert {r["column"]: r["importance"] for r in rolled} == {"region": 0.4, "signal": 0.6}
        assert [r["column"] for r in rolled] == ["signal", "region"]  # ranked, not incidental

    def test_importance_is_normalized(self, classification_df):
        result = modeling.train_candidates(
            classification_df, target_column="outcome", task="classification",
            model_names=["random_forest_classifier"],
        )
        total = sum(f["importance"] for f in result["results"][0]["feature_importance"])
        assert total == pytest.approx(1.0, abs=0.05)


class TestTrainCandidates:
    def test_trains_several_models_on_the_same_split(self, classification_df):
        result = modeling.train_candidates(
            classification_df, target_column="outcome", task="classification",
            model_names=["logistic_regression", "random_forest_classifier"],
        )
        assert len(result["results"]) == 2
        assert result["split"]["train_rows"] + result["split"]["test_rows"] == 300

    def test_picks_a_best_model(self, classification_df):
        result = modeling.train_candidates(
            classification_df, target_column="outcome", task="classification",
            model_names=["logistic_regression", "random_forest_classifier"],
        )
        assert result["best_model"] in {"logistic_regression", "random_forest_classifier"}

    def test_gradient_boosting_trains_when_there_are_categorical_columns(
        self, classification_df
    ):
        """Found live: the one-hot encoder returned a sparse matrix and HistGradientBoosting
        refuses sparse input, so the model the UI calls "typically the most accurate of the
        three" silently failed to train on every dataset that had a categorical column --
        which is nearly all of them. It showed up only as an empty row in the results."""
        assert "region" in modeling.classify_features(classification_df, "outcome")["categorical"]

        result = modeling.train_candidates(
            classification_df, target_column="outcome", task="classification",
            model_names=["gradient_boosting_classifier"],
        )
        assert [r["model"] for r in result["results"]] == ["gradient_boosting_classifier"]
        assert result["results"][0]["metrics"]["accuracy"] > 0

    def test_gradient_boosting_trains_for_regression_too(self, regression_df):
        result = modeling.train_candidates(
            regression_df, target_column="price", task="regression",
            model_names=["gradient_boosting_regressor"],
        )
        assert [r["model"] for r in result["results"]] == ["gradient_boosting_regressor"]

    def test_a_model_for_the_wrong_task_is_ignored(self, classification_df):
        result = modeling.train_candidates(
            classification_df, target_column="outcome", task="classification",
            model_names=["logistic_regression", "linear_regression"],
        )
        assert [r["model"] for r in result["results"]] == ["logistic_regression"]

    def test_excluding_every_feature_is_refused_with_an_explanation(self, classification_df):
        with pytest.raises(ModelingError, match="no usable columns"):
            modeling.train_candidates(
                classification_df, target_column="outcome", task="classification",
                model_names=["logistic_regression"],
                excluded_columns=["signal", "noise", "region", "joined"],
            )

    def test_missing_target_column_is_refused(self, classification_df):
        with pytest.raises(ModelingError, match="not in the data"):
            modeling.train_candidates(
                classification_df, target_column="nope", task="classification",
                model_names=["logistic_regression"],
            )

    def test_registry_covers_both_tasks(self):
        assert len(modeling.models_for_task("classification")) >= 3
        assert len(modeling.models_for_task("regression")) >= 3
        assert all(s.task == "regression" for s in modeling.models_for_task("regression"))


class TestGroupedSplit:
    """When one thing appears in many rows, a random split puts some of its rows in training
    and some in test. The model then scores well by recognising the entity rather than
    learning the pattern -- and that score cannot survive meeting an entity it has never seen.

    The app has detected this for a while and could only advise treating the numbers as
    optimistic. These tests pin down the fix it was pointing at.
    """

    @staticmethod
    def memorisable(customers: int = 80, per: int = 12) -> pd.DataFrame:
        """No learnable pattern whatsoever: the label is a coin flip *per customer*, and one
        column is a per-customer constant. Any score above chance here is memorisation."""
        rng = np.random.default_rng(7)
        return pd.DataFrame(
            {
                "customer_id": np.repeat([f"CUST-{i:03d}" for i in range(customers)], per),
                "account_age_days": np.repeat(
                    rng.integers(100, 9000, customers), per
                ).astype(float),
                "visit_number": list(range(per)) * customers,
                "spend": rng.normal(50, 15, customers * per).round(2),
                "churned": np.repeat(rng.choice(["yes", "no"], customers), per),
            }
        )

    def test_no_entity_appears_on_both_sides(self):
        df = self.memorisable()
        X_train, X_test, _, _ = modeling.split_data(
            df, "churned", "classification", strategy="group", group_column="customer_id"
        )
        train = set(df.loc[X_train.index, "customer_id"])
        test = set(df.loc[X_test.index, "customer_id"])
        assert train and test
        assert not (train & test)

    def test_a_random_split_invents_a_signal_that_a_grouped_split_does_not(self):
        """The whole justification. There is nothing to learn in this data, and only one of
        these two splits says so."""
        df = self.memorisable()
        common = dict(
            target_column="churned",
            task="classification",
            model_names=["random_forest_classifier"],
            include_deep_diagnostics=False,
        )
        random_run = modeling.train_candidates(df, split_strategy="random", **common)
        grouped_run = modeling.train_candidates(
            df, split_strategy="group", group_column="customer_id", **common
        )

        random_auc = random_run["results"][0]["metrics"]["roc_auc"]
        grouped_auc = grouped_run["results"][0]["metrics"]["roc_auc"]
        assert random_auc > grouped_auc + 0.08
        assert grouped_run["assessment"]["verdict"] == "no_better_than_guessing"

    def test_the_grouping_column_is_never_an_input(self):
        """Every test group is unseen by construction, so as a feature it could only ever
        contribute unknown categories -- and if it did carry signal, that signal is exactly
        the memorisation the split exists to prevent."""
        result = modeling.train_candidates(
            self.memorisable(),
            target_column="churned",
            task="classification",
            model_names=["logistic_regression"],
            split_strategy="group",
            group_column="customer_id",
            include_deep_diagnostics=False,
        )
        used = result["features"]["numeric"] + result["features"]["categorical"]
        assert "customer_id" not in used

    def test_cross_validation_is_grouped_too(self):
        """Otherwise a run that carefully grouped its train/test split still reports an
        inflated cross-validated score directly beside the honest one."""
        result = modeling.train_candidates(
            self.memorisable(),
            target_column="churned",
            task="classification",
            model_names=["logistic_regression"],
            split_strategy="group",
            group_column="customer_id",
            include_deep_diagnostics=True,
        )
        assert result["deep_diagnostics"]["cross_validation"]["grouped"] is True

    def test_the_split_reports_entities_not_just_rows(self):
        result = modeling.train_candidates(
            self.memorisable(),
            target_column="churned",
            task="classification",
            model_names=["logistic_regression"],
            split_strategy="group",
            group_column="customer_id",
            include_deep_diagnostics=False,
        )
        split = result["split"]
        assert split["train_groups"] + split["test_groups"] == 80
        assert split["group_column"] == "customer_id"

    def test_too_few_groups_is_refused_with_the_reason(self):
        df = pd.DataFrame(
            {"team": ["a", "b"] * 60, "x": range(120), "outcome": ["yes", "no"] * 60}
        )
        with pytest.raises(modeling.ModelingError, match="too few"):
            modeling.split_data(
                df, "outcome", "classification", strategy="group", group_column="team"
            )

    def test_a_missing_group_column_is_refused(self):
        with pytest.raises(modeling.ModelingError, match="which rows belong"):
            modeling.split_data(
                self.memorisable(), "churned", "classification", strategy="group"
            )

    def test_blank_groups_are_refused_rather_than_lumped_together(self):
        """Treating every blank as one giant entity would silently put unrelated rows in the
        same group, which is a different wrong answer rather than an error."""
        df = self.memorisable()
        df.loc[df.index[:5], "customer_id"] = None
        with pytest.raises(modeling.ModelingError, match="blank"):
            modeling.split_data(
                df, "churned", "classification", strategy="group", group_column="customer_id"
            )
