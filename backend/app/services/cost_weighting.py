"""Training with the user's stated mistake costs, not only choosing the cut-off with them.

The user can say a missed case costs N times a false alarm. That used to move only the cut-off, after a model fitted
to a different trade-off (balanced class weights) had been chosen. `CostWeighted` wraps a classifier so every fit
weights each row by the cost of getting it wrong: a positive row by the cost of missing one, a negative row by the
cost of a false alarm. Because the weighting lives inside the estimator, every fit uses it with nothing else wired
up -- the folds the loop compares on, calibration, the cut-off's out-of-fold scores and the final fit.

The class is self-contained (numpy and scikit-learn only) because `model_export` copies its source into
`train_model.py`, the same way it copies the clipping step.
"""

from __future__ import annotations

import inspect

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class CostWeighted(ClassifierMixin, BaseEstimator):
    """A classifier fitted with each row weighted by the stated cost of getting it wrong.

    Weights are normalised to average one, so they change the trade-off the model fits and not the overall
    strength of any regularisation. They multiply whatever class weighting the model already has.
    """

    def __init__(self, estimator=None, positive_label=None, cost_of_missing=1.0, cost_of_false_alarm=1.0):
        self.estimator = estimator
        self.positive_label = positive_label
        self.cost_of_missing = cost_of_missing
        self.cost_of_false_alarm = cost_of_false_alarm

    def fit(self, X, y):
        from sklearn.base import clone

        answers = np.asarray(y).astype(str)
        weights = np.where(
            answers == str(self.positive_label), float(self.cost_of_missing), float(self.cost_of_false_alarm)
        )
        weights = weights / weights.mean()
        self.estimator_ = clone(self.estimator).fit(X, y, sample_weight=weights)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def source() -> str:
    """The class as text, for `train_model.py`. Read from the class itself rather than kept as a second copy."""
    return inspect.getsource(CostWeighted)
