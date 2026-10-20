"""The random forest and extra trees, able to weight classes evenly when the labels are whole numbers written as text.

scikit-learn 1.9 (1.9.0 and 1.9.1) cannot fit either forest with `class_weight="balanced"` when the labels are text
that reads as a whole number, such as "0" and "1": the forest turns "balanced" into a dict keyed by the labels, and
`compute_class_weight` converts each label with `int()` before looking it up, so it looks for 0 where the key is "0"
and raises "The classes, [0, 1], are not in class_weight". The app hands its classifiers text labels wherever it
cross-validates -- the settings search, fold importance, calibration, the cut-off folds -- so on any target coded
0/1 or 1/2/3 both forests failed every comparison and were silently left out of the model choice and the
improvement loop (#126, found by the accuracy benchmark, #124). Labels such as yes/no or 1.0/2.0 were never hit.

These subclasses compute the same weights through the path that works, `compute_class_weight("balanced", ...)`
(n_rows / (n_classes * rows in the class), counted and multiplied by any row weights the caller passes, exactly as
scikit-learn does), and fit with them as row weights. Their settings are scikit-learn's, so the exported script still
writes `RandomForestClassifier(class_weight='balanced', ...)` from sklearn itself; it trains on the file's own labels,
which never take the broken path. `predict.py` ships this module, so a saved model finds these classes.

Remove the module, and the lookup in `ModelSpec.build`, once scikit-learn fixes `compute_class_weight`.
"""

from __future__ import annotations

import numpy as np
from sklearn import ensemble
from sklearn.utils.class_weight import compute_class_weight


class _EvenClassWeightsAsRowWeights:
    def fit(self, X, y, sample_weight=None):
        preset = self.class_weight
        # "balanced_subsample" with bootstrapping weighs each tree's own sample on a path that works; without
        # bootstrapping it is "balanced".
        if not (preset == "balanced" or (preset == "balanced_subsample" and not self.bootstrap)):
            return super().fit(X, y, sample_weight=sample_weight)
        # As scikit-learn computes them: classes counted by their rows' weights when there are any, then those
        # class weights multiplied into the row weights.
        labels = np.asarray(y)
        classes, row_class = np.unique(labels, return_inverse=True)
        rows = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        weights = compute_class_weight("balanced", classes=classes, y=labels, sample_weight=rows)[row_class]
        if rows is not None:
            weights = weights * rows
        self.class_weight = None
        try:
            return super().fit(X, y, sample_weight=weights)
        finally:
            self.class_weight = preset


class RandomForestClassifier(_EvenClassWeightsAsRowWeights, ensemble.RandomForestClassifier):
    pass


class ExtraTreesClassifier(_EvenClassWeightsAsRowWeights, ensemble.ExtraTreesClassifier):
    pass


# What `ModelSpec.build` makes in place of the scikit-learn class of the same name.
REPLACEMENTS = {
    "sklearn.ensemble.RandomForestClassifier": RandomForestClassifier,
    "sklearn.ensemble.ExtraTreesClassifier": ExtraTreesClassifier,
}
