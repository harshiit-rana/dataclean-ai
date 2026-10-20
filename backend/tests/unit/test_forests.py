"""Forests with balanced class weights on labels like "0"/"1" (#126): scikit-learn 1.9 raises; ours fit, identically."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn import ensemble

from app.services import forests, modeling, predict_export, tuning

rng = np.random.default_rng(0)
X = rng.random((240, 4))
NUMBERS = np.where(X[:, 0] + rng.normal(0, 0.3, 240) > 0.75, "1", "0")
WORDS = np.where(NUMBERS == "1", "yes", "no")


def test_scikit_learn_itself_still_fails_on_whole_number_text_labels():
    """The day this stops failing, forests.py can go."""
    with pytest.raises(ValueError, match="not in class_weight"):
        ensemble.RandomForestClassifier(n_estimators=5, class_weight="balanced").fit(X, NUMBERS)


@pytest.mark.parametrize("name", ["random_forest_classifier", "extra_trees_classifier"])
def test_the_registry_builds_forests_that_fit_on_them(name):
    model = modeling.MODELS[name].build()
    assert isinstance(model, forests.REPLACEMENTS[f"sklearn.ensemble.{modeling.MODELS[name].class_name}"])
    model.fit(X, NUMBERS)
    assert list(model.classes_) == ["0", "1"]
    assert model.get_params()["class_weight"] == "balanced"  # settings untouched, so the export is too


@pytest.mark.parametrize("preset", ["balanced", "balanced_subsample"])
@pytest.mark.parametrize("cls", [forests.RandomForestClassifier, forests.ExtraTreesClassifier])
def test_the_weights_are_exactly_scikit_learns(cls, preset):
    """Same probabilities as scikit-learn's own class on labels it can handle, with and without row weights."""
    settings = {"n_estimators": 25, "class_weight": preset, "random_state": 0}
    original = getattr(ensemble, cls.__name__)
    rows = rng.random(len(X)) + 0.5
    for weights in (None, rows):
        ours = cls(**settings).fit(X, NUMBERS, sample_weight=weights)
        theirs = original(**settings).fit(X, WORDS, sample_weight=weights)
        assert np.allclose(ours.predict_proba(X), theirs.predict_proba(X))


def test_a_zero_one_target_keeps_both_forests_in_the_comparison():
    frame = pd.DataFrame(X, columns=list("abcd"))
    target = pd.Series(NUMBERS.astype(int))  # as read_csv gives it; score_config turns it into text
    for name in ("random_forest_classifier", "extra_trees_classifier"):
        found = tuning.score_config(modeling.MODELS[name].build, frame, target, "classification")
        assert found is not None, f"{name} failed to score on a 0/1 target"


def test_predict_py_ships_the_module_so_a_saved_model_loads():
    assert "app/services/forests.py" in predict_export.runtime_files()
