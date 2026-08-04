"""Model diagnostics: the numbers behind the charts.

These assert properties that would still hold on any correct implementation -- an ROC curve
runs from (0,0) to (1,1), a perfect model's calibration sits on the diagonal, residuals of an
unbiased model centre on zero -- rather than pinning the exact floats a particular sklearn
version happens to produce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import diagnostics, modeling


@pytest.fixture
def binary_scores():
    """A model with real but imperfect skill: positives score higher on average, with
    overlap. Perfect separation would make every curve degenerate and prove nothing."""
    rng = np.random.default_rng(4)
    n = 400
    y = rng.integers(0, 2, n)
    scores = np.clip(rng.normal(0.35, 0.18, n) + y * 0.3, 0.001, 0.999)
    return y, scores


@pytest.fixture
def learnable_df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 400
    signal = rng.normal(0, 1, n)
    probability = 1 / (1 + np.exp(-2.2 * signal))
    return pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(0, 1, n),
            "region": rng.choice(["n", "s"], n),
            "outcome": np.where(rng.random(n) < probability, "yes", "no"),
        }
    )


class TestRocCurve:
    def test_runs_corner_to_corner(self, binary_scores):
        y, scores = binary_scores
        points = diagnostics.roc_curve_points(y, scores)
        assert points[0]["fpr"] == 0.0 and points[0]["tpr"] == 0.0
        assert points[-1]["fpr"] == 1.0 and points[-1]["tpr"] == 1.0

    def test_is_monotonic(self, binary_scores):
        y, scores = binary_scores
        points = diagnostics.roc_curve_points(y, scores)
        fprs = [p["fpr"] for p in points]
        assert fprs == sorted(fprs)

    def test_a_skilful_model_sits_above_the_diagonal(self, binary_scores):
        y, scores = binary_scores
        points = diagnostics.roc_curve_points(y, scores)
        middle = points[len(points) // 2]
        assert middle["tpr"] > middle["fpr"]

    def test_thresholds_stay_json_serializable(self, binary_scores):
        """roc_curve's first threshold is +inf by construction, which is not valid JSON."""
        y, scores = binary_scores
        assert all(np.isfinite(p["threshold"]) for p in diagnostics.roc_curve_points(y, scores))

    def test_long_curves_are_thinned(self):
        rng = np.random.default_rng(1)
        n = 20_000
        y = rng.integers(0, 2, n)
        scores = rng.random(n)
        points = diagnostics.roc_curve_points(y, scores)
        assert len(points) <= diagnostics.MAX_CURVE_POINTS
        # thinning must keep the endpoints, which are the part people read
        assert points[0]["fpr"] == 0.0 and points[-1]["fpr"] == 1.0


class TestPrecisionRecall:
    def test_recall_falls_as_the_threshold_rises(self, binary_scores):
        y, scores = binary_scores
        points = diagnostics.precision_recall_points(y, scores)
        by_threshold = sorted(points, key=lambda p: p["threshold"])
        assert by_threshold[0]["recall"] >= by_threshold[-1]["recall"]

    def test_values_stay_in_range(self, binary_scores):
        y, scores = binary_scores
        for point in diagnostics.precision_recall_points(y, scores):
            assert 0.0 <= point["precision"] <= 1.0
            assert 0.0 <= point["recall"] <= 1.0


class TestCalibration:
    def test_a_perfectly_calibrated_model_lands_on_the_diagonal(self):
        """Scores are literally the probability of a 1, so predicted should track actual."""
        rng = np.random.default_rng(7)
        scores = rng.random(6000)
        y = (rng.random(6000) < scores).astype(int)

        for point in diagnostics.calibration_points(y, scores):
            if point["count"] >= 100:
                assert abs(point["predicted"] - point["actual"]) < 0.08

    def test_an_overconfident_model_is_visibly_off_the_diagonal(self):
        """Ranks cases correctly but its probabilities are far too extreme -- great AUC,
        useless as a probability. This is exactly what calibration exists to reveal."""
        rng = np.random.default_rng(8)
        truth = rng.random(4000)
        y = (rng.random(4000) < truth).astype(int)
        overconfident = np.clip((truth - 0.5) * 3 + 0.5, 0.001, 0.999)

        points = [p for p in diagnostics.calibration_points(y, overconfident) if p["count"] >= 50]
        assert max(abs(p["predicted"] - p["actual"]) for p in points) > 0.1

    def test_empty_bins_are_omitted_rather_than_reported_as_zero(self):
        scores = np.full(200, 0.9)
        y = np.ones(200, dtype=int)
        points = diagnostics.calibration_points(y, scores)
        assert len(points) == 1
        assert all(p["count"] > 0 for p in points)


class TestThresholdTable:
    def test_raising_the_threshold_flags_fewer_and_misses_more(self, binary_scores):
        y, scores = binary_scores
        rows = diagnostics.threshold_table(y, scores)
        assert rows[0]["flagged"] >= rows[-1]["flagged"]
        assert rows[0]["missed"] <= rows[-1]["missed"]

    def test_caught_and_missed_always_account_for_every_positive(self, binary_scores):
        y, scores = binary_scores
        total_positives = int(y.sum())
        for row in diagnostics.threshold_table(y, scores):
            assert row["caught"] + row["missed"] == total_positives

    def test_precision_never_exceeds_one(self, binary_scores):
        y, scores = binary_scores
        assert all(0.0 <= r["precision"] <= 1.0 for r in diagnostics.threshold_table(y, scores))


class TestConfusionAndPerClass:
    def test_confusion_cells_are_row_normalized(self):
        cells = diagnostics.confusion_cells([[8, 2], [3, 7]], ["no", "yes"])
        no_row = [c for c in cells if c["actual"] == "no"]
        assert sum(c["share_of_actual"] for c in no_row) == pytest.approx(1.0)

    def test_confusion_cells_mark_the_diagonal(self):
        cells = diagnostics.confusion_cells([[8, 2], [3, 7]], ["no", "yes"])
        correct = {(c["actual"], c["predicted"]) for c in cells if c["correct"]}
        assert correct == {("no", "no"), ("yes", "yes")}

    def test_per_class_support_sums_to_the_row_count(self):
        y_true = pd.Series(["a"] * 30 + ["b"] * 20)
        y_pred = pd.Series(["a"] * 25 + ["b"] * 25)
        report = diagnostics.per_class_report(y_true, y_pred, ["a", "b"])
        assert sum(r["support"] for r in report) == 50

    def test_a_class_the_model_never_predicts_scores_zero_not_an_error(self):
        """The classic imbalance failure: the rare class is never predicted at all. It has to
        show up as recall 0, not as a divide-by-zero."""
        y_true = pd.Series(["common"] * 95 + ["rare"] * 5)
        y_pred = pd.Series(["common"] * 100)
        report = {r["label"]: r for r in diagnostics.per_class_report(y_true, y_pred, ["common", "rare"])}
        assert report["rare"]["recall"] == 0.0
        assert report["rare"]["support"] == 5


class TestRegressionDiagnostics:
    def test_residuals_of_an_unbiased_model_centre_on_zero(self):
        rng = np.random.default_rng(3)
        actual = rng.normal(100, 20, 500)
        predicted = actual + rng.normal(0, 5, 500)
        points = diagnostics.residual_points(actual, predicted)
        assert abs(np.mean([p["residual"] for p in points])) < 2.0

    def test_a_biased_model_shows_a_shifted_residual_histogram(self):
        rng = np.random.default_rng(21)
        actual = np.linspace(0, 100, 400)
        predicted = actual + 25 + rng.normal(0, 3, 400)  # systematically over-predicts
        bars = diagnostics.residual_histogram(actual, predicted)
        weighted = sum(b["centre"] * b["count"] for b in bars) / sum(b["count"] for b in bars)
        assert weighted == pytest.approx(-25, abs=1)

    def test_a_constant_error_does_not_crash_the_histogram(self):
        """Every residual identical -- numpy cannot split a zero-width range into bins."""
        actual = np.linspace(0, 100, 400)
        bars = diagnostics.residual_histogram(actual, actual + 25)
        assert bars == [{"centre": -25.0, "count": 400}]

    def test_predicted_vs_actual_is_downsampled(self):
        actual = np.linspace(0, 1000, 5000)
        points = diagnostics.predicted_vs_actual_points(actual, actual)
        assert len(points) <= diagnostics.MAX_SCATTER_POINTS

    def test_error_by_range_covers_every_row(self):
        rng = np.random.default_rng(6)
        actual = rng.uniform(0, 100, 300)
        predicted = actual + rng.normal(0, 5, 300)
        rows = diagnostics.error_by_actual_range(actual, predicted)
        assert sum(r["count"] for r in rows) == 300

    def test_error_by_range_surfaces_where_the_model_is_worst(self):
        """Accurate in the middle, badly wrong at the top -- invisible in a single R²."""
        actual = np.linspace(0, 100, 400)
        predicted = np.where(actual > 80, actual + 40, actual)
        rows = diagnostics.error_by_actual_range(actual, predicted)
        assert rows[-1]["mean_absolute_error"] > rows[0]["mean_absolute_error"] * 5

    def test_constant_target_does_not_crash(self):
        actual = np.full(50, 7.0)
        assert diagnostics.error_by_actual_range(actual, actual) == []


class TestFeatureCorrelations:
    def test_finds_a_duplicated_column(self):
        rng = np.random.default_rng(2)
        base = rng.normal(0, 1, 200)
        df = pd.DataFrame({"a": base, "a_copy": base * 2 + 1, "b": rng.normal(0, 1, 200)})
        result = diagnostics.feature_correlations(df, ["a", "a_copy", "b"])
        pairs = {tuple(sorted((p["a"], p["b"]))) for p in result["strong_pairs"]}
        assert ("a", "a_copy") in pairs

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)})
        result = diagnostics.feature_correlations(df, ["a", "b"])
        diagonal = [c for c in result["cells"] if c["row"] == c["column"]]
        assert all(c["correlation"] == pytest.approx(1.0) for c in diagonal)

    def test_needs_at_least_two_numeric_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "text": ["x", "y", "z"]})
        assert diagnostics.feature_correlations(df, ["a", "text"])["cells"] == []


class TestDeepDiagnostics:
    def test_cross_validation_reports_spread_not_just_a_mean(self, learnable_df):
        X = learnable_df[["signal", "noise"]]
        y = learnable_df["outcome"]
        result = diagnostics.cross_validate_pipeline(
            lambda: modeling.build_pipeline("logistic_regression", ["signal", "noise"], []),
            X, y, "classification",
        )
        assert len(result["folds"]) == diagnostics.CV_FOLDS
        assert "std" in result and "spread" in result
        assert result["mean"] > 0.7  # signal is genuinely there

    def test_cross_validation_backs_off_when_a_class_is_tiny(self):
        """Five folds need at least five examples of the rarest class."""
        df = pd.DataFrame({"a": range(100), "target": ["no"] * 97 + ["yes"] * 3})
        result = diagnostics.cross_validate_pipeline(
            lambda: modeling.build_pipeline("logistic_regression", ["a"], []),
            df[["a"]], df["target"], "classification",
        )
        assert result == {} or len(result["folds"]) <= 3

    def test_learning_curve_grows_the_training_set(self, learnable_df):
        points = diagnostics.learning_curve_points(
            lambda: modeling.build_pipeline("logistic_regression", ["signal"], []),
            learnable_df[["signal"]], learnable_df["outcome"], "classification",
        )
        sizes = [p["training_rows"] for p in points]
        assert sizes == sorted(sizes) and len(sizes) == 5
        assert all("validation_score" in p for p in points)

    def test_permutation_importance_ranks_the_real_signal_first(self, learnable_df):
        features = ["signal", "noise"]
        pipeline = modeling.build_pipeline("logistic_regression", features, [])
        pipeline.fit(learnable_df[features], learnable_df["outcome"])
        ranking = diagnostics.permutation_importance_scores(
            pipeline, learnable_df[features], learnable_df["outcome"], "classification"
        )
        assert ranking[0]["column"] == "signal"

    def test_deep_diagnostics_are_skipped_on_large_data(self, monkeypatch):
        monkeypatch.setattr(diagnostics, "MAX_ROWS_FOR_DEEP_DIAGNOSTICS", 10)
        result = diagnostics.compute_deep_diagnostics(
            lambda: None,
            None,
            pd.DataFrame({"a": range(20)}),
            pd.DataFrame({"a": range(5)}),
            pd.Series(range(20)),
            pd.Series(range(5)),
            "regression",
        )
        assert result["skipped"] and "skipped" in result["skipped"]

    def test_one_failing_diagnostic_does_not_lose_the_others(self, learnable_df, monkeypatch):
        def explode(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(diagnostics, "learning_curve_points", explode)
        features = ["signal", "noise"]
        pipeline = modeling.build_pipeline("logistic_regression", features, [])
        X = learnable_df[features]
        y = learnable_df["outcome"]
        pipeline.fit(X, y)

        result = diagnostics.compute_deep_diagnostics(
            lambda: modeling.build_pipeline("logistic_regression", features, []),
            pipeline, X.iloc[:300], X.iloc[300:], y.iloc[:300], y.iloc[300:], "classification",
        )
        assert result["learning_curve"] is None
        assert result["cross_validation"]["folds"]


class TestIntegrationWithTraining:
    def test_a_classification_run_carries_its_charts(self, learnable_df):
        result = modeling.train_candidates(
            learnable_df, target_column="outcome", task="classification",
            model_names=["logistic_regression"],
        )
        charts = result["results"][0]["diagnostics"]
        for key in ("roc_curve", "precision_recall_curve", "calibration", "thresholds",
                    "confusion_cells", "per_class", "class_distribution"):
            assert charts[key], f"{key} missing from the training result"

    def test_a_regression_run_carries_residual_charts(self):
        rng = np.random.default_rng(12)
        n = 300
        size = rng.uniform(50, 250, n)
        df = pd.DataFrame({"size": size, "price": size * 900 + rng.normal(0, 4000, n)})
        result = modeling.train_candidates(
            df, target_column="price", task="regression", model_names=["linear_regression"],
        )
        charts = result["results"][0]["diagnostics"]
        for key in ("predicted_vs_actual", "residuals", "residual_histogram", "error_by_range"):
            assert charts[key], f"{key} missing from the training result"
        assert "roc_curve" not in charts  # meaningless for regression

    def test_deep_diagnostics_run_for_the_chosen_model(self, learnable_df):
        result = modeling.train_candidates(
            learnable_df, target_column="outcome", task="classification",
            model_names=["logistic_regression"],
        )
        deep = result["deep_diagnostics"]
        assert deep["cross_validation"]["folds"]
        assert deep["learning_curve"]
        assert deep["permutation_importance"]

    def test_deep_diagnostics_can_be_turned_off(self, learnable_df):
        result = modeling.train_candidates(
            learnable_df, target_column="outcome", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False,
        )
        assert result["deep_diagnostics"] == {}

    def test_multiclass_gets_a_confusion_matrix_but_no_roc(self):
        rng = np.random.default_rng(13)
        df = pd.DataFrame(
            {"a": rng.normal(0, 1, 240), "grade": rng.choice(["A", "B", "C"], 240)}
        )
        result = modeling.train_candidates(
            df, target_column="grade", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False,
        )
        charts = result["results"][0]["diagnostics"]
        assert len(charts["per_class"]) == 3
        assert charts["confusion_cells"]
        assert "roc_curve" not in charts
