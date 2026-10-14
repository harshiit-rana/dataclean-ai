"""Every chart beside the cross-validation spread speaks the same metric.

The spread was scored in AUC for a two-class target while the learning curve and the permutation
importance were scored in accuracy. On a lopsided target accuracy can stay flat while AUC improves,
so the curve beside an AUC spread measured a different thing about the same model and the two read
as disagreeing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sklearn.inspection as inspection
import sklearn.model_selection as selection

from app.services import diagnostics, modeling, tuning


def two_class(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {"x": x, "z": rng.normal(0, 1, n), "y": np.where(x + rng.normal(0, 1, n) > 0, "yes", "no")}
    )


def three_class(n: int = 450) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {"x": x, "z": rng.normal(0, 1, n), "y": np.digitize(x + rng.normal(0, 1, n), [-0.5, 0.5]).astype(str)}
    )


def regression(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "z": rng.normal(0, 1, n), "y": x * 3 + rng.normal(0, 1, n)})


CASES = [
    ("two classes", two_class, "classification", "roc_auc", "logistic_regression"),
    ("three classes", three_class, "classification", "accuracy", "logistic_regression"),
    ("a number", regression, "regression", "r2", "linear_regression"),
]


@pytest.fixture
def scorers(monkeypatch):
    """The `scoring=` every sklearn fold/permutation helper was asked for."""
    asked: dict[str, list] = {"learning_curve": [], "permutation_importance": [], "cross_val_score": []}
    for module, name in (
        (selection, "learning_curve"),
        (inspection, "permutation_importance"),
        (selection, "cross_val_score"),
    ):
        real = getattr(module, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            asked[_name].append(kwargs.get("scoring"))
            return _real(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)
    return asked


@pytest.mark.parametrize("label,make,task,expected,model", CASES, ids=[c[0] for c in CASES])
class TestEveryChartUsesThePrimaryMetric:
    def _run(self, make, task, model):
        return modeling.train_candidates(
            make(), target_column="y", task=task, model_names=[model],
            include_deep_diagnostics=True,
        )["deep_diagnostics"]

    def test_the_learning_curve_is_scored_in_it(self, label, make, task, expected, model, scorers):
        self._run(make, task, model)
        assert scorers["learning_curve"] == [expected]

    def test_permutation_importance_is_scored_in_it(self, label, make, task, expected, model, scorers):
        self._run(make, task, model)
        assert scorers["permutation_importance"] == [expected]

    def test_the_cross_validation_spread_still_uses_it(self, label, make, task, expected, model, scorers):
        deep = self._run(make, task, model)
        assert deep["cross_validation"]["metric"] == expected
        assert expected in scorers["cross_val_score"]

    def test_all_three_name_the_same_metric_in_their_output(self, label, make, task, expected, model, scorers):
        deep = self._run(make, task, model)
        assert {p["metric"] for p in deep["learning_curve"]} == {expected}
        assert {r["metric"] for r in deep["permutation_importance"]} == {expected}
        assert deep["cross_validation"]["metric"] == expected


class TestTheMetricIsNotChangedByTheTestSet:
    def test_a_single_class_in_the_test_rows_would_have_switched_the_metric_if_it_were_read_from_them(self):
        """Why the metric is fixed from the training rows: chosen from a test set that happens
        to hold one class, 'roc_auc' would silently become 'accuracy'."""
        df = two_class()
        _, _, y_train, y_test = modeling.split_data(df, "y", "classification")
        one_class = y_test[y_test == "yes"]
        assert tuning.scoring_for("classification", y_train) == "roc_auc"
        assert tuning.scoring_for("classification", one_class) == "accuracy"

    def test_deep_diagnostics_pass_the_training_metric(self, monkeypatch):
        seen = {}
        real = diagnostics.permutation_importance_scores

        def spy(pipeline, X, y, task, top_n=15, scoring=None):
            seen["scoring"] = scoring
            return real(pipeline, X, y, task, top_n, scoring)

        monkeypatch.setattr(diagnostics, "permutation_importance_scores", spy)
        modeling.train_candidates(
            two_class(), target_column="y", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=True,
        )
        assert seen["scoring"] == "roc_auc"


class TestTheCardSpeaksIt:
    def test_the_card_names_the_metric_of_the_learning_curve(self):
        from app.services.model_card import build_model_card

        result = modeling.train_candidates(
            two_class(), target_column="y", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=True,
        )
        card = build_model_card(session_id="s", file_name="f.csv", result=result)
        assert "the ROC AUC on unseen rows went from" in card
