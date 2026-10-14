"""The cut-off is chosen on the probabilities it will be applied to.

If calibration is applied after the line was chosen, the line refers to uncalibrated scores and the
precision and recall in the table are not what the delivered model does. A forest's uncalibrated
scores and its calibrated probabilities sit on different scales, so 0.30 means a different set of
rows in each. These pin that, with calibration on, the line is chosen on out-of-fold probabilities
from a calibrated model, that it reproduces what the table says the test set delivered, and that
the nested cross-validation this needs is bounded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import modeling, operating_point
from app.services.model_card import build_model_card
from app.services.operating_point import OperatingPoint

FOREST = "random_forest_classifier"
LOGISTIC = "logistic_regression"


def frame(n: int = 1600, seed: int = 0) -> pd.DataFrame:
    """A target a forest ranks well and states over-confidently."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {"x": x, "z": rng.normal(0, 1, n), "y": np.where(x + rng.normal(0, 1, n) > 0.3, "yes", "no")}
    )


def train(df, method="none", model=FOREST, objective=("min_recall", 1, 1, 0.8), **kwargs):
    return modeling.train_candidates(
        df, target_column="y", task="classification", model_names=[model],
        include_deep_diagnostics=False, calibration=method, point=OperatingPoint(*objective), **kwargs,
    )


def chosen(result):
    return next(r for r in result["results"] if r["model"] == result["best_model"])


@pytest.fixture(scope="module")
def runs():
    df = frame()
    return df, {method: train(df, method) for method in ("none", "sigmoid")}


class TestTheStoredLineReproducesWhatWasDelivered:
    def test_applied_to_the_calibrated_models_test_probabilities_it_gives_the_reported_figures(self, runs):
        df, results = runs
        result = results["sigmoid"]
        _, X_test, _, y_test = modeling.split_data(df, "y", "classification")
        probabilities = chosen(result)["pipeline"].predict_proba(X_test)[:, 1]
        again = operating_point.metrics_at(
            (y_test == "yes").to_numpy().astype(int), probabilities, result["operating_point"]["threshold"]
        )
        reported = result["operating_point"]["observed"]
        for key in ("precision", "recall", "f1", "flagged", "caught", "missed", "false_alarms"):
            assert again[key] == reported[key], key

    def test_the_line_is_on_the_calibrated_scale_not_the_raw_one(self, runs):
        """The same objective on the same model picks a different number once the probabilities
        are honest -- which is why a line chosen before calibration cannot be reused."""
        _, results = runs
        assert results["sigmoid"]["operating_point"]["threshold"] != results["none"]["operating_point"]["threshold"]

    def test_an_uncalibrated_line_applied_to_the_calibrated_model_would_not_deliver_what_it_promised(self):
        """Logistic regression reweights a rare answer, so its probabilities sit far above the
        truth: a line of 0.46 on them is a very different set of rows from 0.46 on calibrated
        ones. Chosen before calibration and applied after, it promises 81% recall and delivers
        about 55%; chosen on the calibrated probabilities it delivers what it promised."""
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 1600)
        df = pd.DataFrame(
            {"x": x, "z": rng.normal(0, 1, 1600), "y": np.where(x + rng.normal(0, 1, 1600) > 1.0, "yes", "no")}
        )
        results = {m: train(df, m, model=LOGISTIC) for m in ("none", "sigmoid")}
        _, X_test, _, y_test = modeling.split_data(df, "y", "classification")
        truth = (y_test == "yes").to_numpy().astype(int)
        stale = results["none"]["operating_point"]
        wrong = operating_point.metrics_at(
            truth, chosen(results["sigmoid"])["pipeline"].predict_proba(X_test)[:, 1], stale["threshold"]
        )
        promised = stale["expected"]["recall"]
        assert promised - wrong["recall"] > 0.15  # the mismatch this issue exists to prevent

        honest = results["sigmoid"]["operating_point"]
        assert abs(honest["observed"]["recall"] - honest["expected"]["recall"]) < 0.1
        assert honest["threshold"] < stale["threshold"]  # honest probabilities are lower on a rare answer

    def test_the_default_line_on_a_calibrated_model_still_reports_its_probabilities(self):
        result = train(frame(), "sigmoid", objective=("default",))
        point = result["operating_point"]
        assert point["threshold"] == 0.5 and point["calibrated_with"] == "sigmoid"

    def test_each_method_can_be_the_one_the_line_is_chosen_on(self):
        result = train(frame(2000), "isotonic", model=LOGISTIC, objective=("f1",))
        assert result["operating_point"]["calibrated_with"] == "isotonic"


class TestTheResultSaysWhatTheLineRefersTo:
    def test_a_calibrated_line_is_marked_and_explained(self, runs):
        point = runs[1]["sigmoid"]["operating_point"]
        assert point["calibrated_with"] == "sigmoid"
        assert "calibrated (sigmoid) inside each fold" in point["chosen_on"]
        assert "the test set was not consulted" in point["chosen_on"]

    def test_an_uncalibrated_line_is_not(self, runs):
        point = runs[1]["none"]["operating_point"]
        assert point["calibrated_with"] is None and "calibrated" not in point["chosen_on"]

    def test_the_card_states_that_the_line_is_on_calibrated_probabilities(self, runs):
        card = build_model_card(session_id="s", file_name="f.csv", result=runs[1]["sigmoid"], summary="A.", interpretation="B.")
        threshold = runs[1]["sigmoid"]["operating_point"]["threshold"]
        assert "so this line is on calibrated probabilities" in card
        assert f"{threshold} means the model says it is {threshold:.0%} likely, and means it" in card

    def test_the_card_for_an_uncalibrated_line_says_nothing_of_the_kind(self, runs):
        card = build_model_card(session_id="s", file_name="f.csv", result=runs[1]["none"], summary="A.", interpretation="B.")
        assert "calibrated probabilities" not in card


class TestTheNestedCostIsBounded:
    """Calibration inside each of the cut-off's folds multiplies the fits: every outer fold fits
    the calibrator's own folds and then the model once more. That is outer x (inner + 1), and a
    test pins it so a change cannot quietly make it worse."""

    def fits_during_the_cutoff(self, monkeypatch, method):
        counts = {"fits": 0, "in_cutoff": False}
        real_apply = operating_point.apply
        from sklearn.pipeline import Pipeline

        real_fit = Pipeline.fit

        def counting_fit(self, X, y=None, **kwargs):
            if counts["in_cutoff"]:
                counts["fits"] += 1
            return real_fit(self, X, y, **kwargs)

        def watching_apply(**kwargs):
            counts["in_cutoff"] = True
            try:
                return real_apply(**kwargs)
            finally:
                counts["in_cutoff"] = False

        monkeypatch.setattr(Pipeline, "fit", counting_fit)
        monkeypatch.setattr(operating_point, "apply", watching_apply)
        train(frame(800), method, model=LOGISTIC, objective=("f1",))
        return counts["fits"]

    def test_uncalibrated_it_is_one_fit_per_outer_fold(self, monkeypatch):
        assert self.fits_during_the_cutoff(monkeypatch, "none") == 5

    def test_calibrated_it_is_outer_times_inner_plus_one(self, monkeypatch):
        outer = inner = 5
        assert self.fits_during_the_cutoff(monkeypatch, "sigmoid") == outer * (inner + 1)

    def test_the_default_line_needs_no_folds_at_all(self, monkeypatch):
        monkeypatch.setattr(
            operating_point, "out_of_fold_scores",
            lambda *a, **k: pytest.fail("the default line must not cross-validate"),
        )
        result = train(frame(800), "sigmoid", model=LOGISTIC, objective=("default",))
        assert result["operating_point"]["folds"] == 0
