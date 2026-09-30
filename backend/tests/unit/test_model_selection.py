"""Choosing between candidate models without the test set.

`train_candidates` used to keep whichever candidate scored highest on the held-out rows, which
makes the reported score the best of several draws on those rows. The first test here is the
property that matters: flip every answer in the test set and the choice must not move. Under
the old code it did -- a flipped test set reverses the ranking, and the worst model won.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import modeling, tuning


@pytest.fixture
def learnable() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    n = 400
    a, b = rng.normal(0, 1, n), rng.normal(0, 1, n)
    # Non-linear, so the tree models and the linear one genuinely differ.
    probability = 1 / (1 + np.exp(-(2.0 * a * b + 0.8 * a)))
    return pd.DataFrame(
        {"a": a, "b": b, "noise": rng.normal(0, 1, n),
         "outcome": np.where(rng.random(n) < probability, "yes", "no")}
    )


ALL = ["logistic_regression", "random_forest_classifier", "gradient_boosting_classifier"]


def _train(df, **kwargs):
    return modeling.train_candidates(
        df, target_column="outcome", task="classification", model_names=ALL,
        include_deep_diagnostics=False, **kwargs,
    )


def test_flipping_every_test_answer_does_not_change_the_choice(learnable, monkeypatch):
    honest = _train(learnable)

    real_split = modeling.split_data

    def flipped_test_answers(*args, **kwargs):
        X_train, X_test, y_train, y_test = real_split(*args, **kwargs)
        return X_train, X_test, y_train, y_test.map({"yes": "no", "no": "yes"})

    monkeypatch.setattr(modeling, "split_data", flipped_test_answers)
    rigged = _train(learnable)

    assert rigged["best_model"] == honest["best_model"]
    # The test set is still *reported* -- flipped answers must show up as a terrible score.
    chosen = lambda r: next(x for x in r["results"] if x["model"] == r["best_model"])  # noqa: E731
    assert chosen(rigged)["metrics"]["roc_auc"] < 0.5 < chosen(honest)["metrics"]["roc_auc"]


def test_the_winner_is_the_best_cross_validated_candidate(learnable):
    result = _train(learnable)
    scores = {r["model"]: r["cv_score"] for r in result["results"]}
    assert all(score is not None for score in scores.values())
    assert result["best_model"] == max(scores, key=scores.get)
    assert result["selection"]["method"] == "cross_validation"
    assert "test set was not consulted" in result["selection"]["chosen_on"]


def test_a_test_set_favourite_does_not_win_against_the_folds(learnable, monkeypatch):
    """Rig both sides directly: the folds prefer logistic regression, the test set prefers
    random forest. The folds decide."""
    real_evaluate = modeling.evaluate

    def test_prefers_forest(pipeline, X_test, y_test, task, positive_label=None):
        metrics = real_evaluate(pipeline, X_test, y_test, task, positive_label)
        is_forest = "RandomForest" in type(pipeline.named_steps["model"]).__name__
        metrics["roc_auc"] = 0.99 if is_forest else 0.51
        return metrics

    def folds_prefer_logistic(build, X_train, y_train, task, *, folds=5, **_kwargs):
        name = type(build().named_steps["model"]).__name__
        score = 0.9 if name == "LogisticRegression" else 0.6
        return tuning.SearchResult(params={}, score=score, std=0.01, metric="roc_auc", fits=folds)

    monkeypatch.setattr(modeling, "evaluate", test_prefers_forest)
    monkeypatch.setattr(tuning, "score_config", folds_prefer_logistic)
    assert _train(learnable)["best_model"] == "logistic_regression"


def test_when_nothing_can_be_folded_the_simplest_is_kept_and_it_says_so(learnable, monkeypatch):
    monkeypatch.setattr(tuning, "score_config", lambda *a, **k: None)
    result = _train(learnable)
    assert result["best_model"] == "logistic_regression"
    assert result["selection"]["method"] == "default_order"
    assert "test-set score" in result["selection"]["chosen_on"]


def test_a_single_candidate_needs_no_choosing(learnable):
    result = modeling.train_candidates(
        learnable, target_column="outcome", task="classification",
        model_names=["logistic_regression"], include_deep_diagnostics=False,
    )
    assert result["selection"]["method"] == "only_candidate"


def test_a_grouped_split_chooses_on_grouped_folds(learnable, monkeypatch):
    """Same rule as everywhere: grouped all the way down, or the choice is made on the leak."""
    learnable["customer"] = [f"c{i % 40}" for i in range(len(learnable))]
    seen_groups = []
    real = tuning.score_config

    def recording(build, X_train, y_train, task, *, folds=5, groups=None, **kwargs):
        seen_groups.append(groups)
        return real(build, X_train, y_train, task, folds=folds, groups=groups, **kwargs)

    monkeypatch.setattr(tuning, "score_config", recording)
    _train(learnable, split_strategy="group", group_column="customer")
    assert seen_groups and all(g is not None for g in seen_groups)


# --- Test rows that are copies of training rows -------------------------------------------


def test_copies_of_training_rows_in_the_test_set_are_counted(learnable):
    """Every row duplicated once: a random split puts many copies on both sides."""
    doubled = pd.concat([learnable, learnable], ignore_index=True)
    result = modeling.train_candidates(
        doubled, target_column="outcome", task="classification",
        model_names=["logistic_regression"], include_deep_diagnostics=False,
    )
    seen = result["split"]["test_rows_seen_in_training"]
    assert 0 < seen <= result["split"]["test_rows"]


def test_no_copies_means_zero(learnable):
    result = modeling.train_candidates(
        learnable, target_column="outcome", task="classification",
        model_names=["logistic_regression"], include_deep_diagnostics=False,
    )
    assert result["split"]["test_rows_seen_in_training"] == 0


def test_a_copy_with_a_different_answer_is_not_a_copy():
    X_train = pd.DataFrame({"a": [1.0, 2.0]})
    X_test = pd.DataFrame({"a": [1.0, 2.0]})
    assert modeling.test_rows_seen_in_training(X_train, X_test, ["yes", "no"], ["no", "no"]) == 1


def test_the_model_card_warns_about_copies():
    from app.services.model_card import _deterministic_cautions

    result = {"split": {"test_rows": 80, "test_rows_seen_in_training": 20}, "task": "regression",
              "features": {}}
    cautions = " ".join(_deterministic_cautions(result, {"diagnostics": {}}))
    assert "20 of the 80 test rows (25%) are exact copies" in cautions
