"""Hyperparameter search, scored where it is safe to score.

The whole reason this module is careful is that the obvious way to "keep improving until the
model is good enough" is also the way to produce a model that only looks good. Try twenty
configurations, keep whichever scored best on the held-out rows, and the held-out score stops
being an estimate of anything -- it is now the best of twenty draws on those particular rows,
and it will not survive contact with new data. On a small test set the effect is large enough
to manufacture a "useful" verdict out of noise.

So the rule here, and in improvement.py which drives it: **searching happens on the training
data only, judged by cross-validation.** The test set is opened once, at the very end, to
report a number nothing was selected against.

The search spaces are small and curated on purpose. A huge grid over a few hundred rows finds
the fold-specific quirk rather than the better model, and it costs minutes to do it. These
cover the parameters that actually move a score on tabular data of this size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fits per search. Randomized rather than exhaustive: with this many candidates a random
# sample gets within noise of the grid optimum for a fraction of the fits.
#
# Twelve, not more, because this runs behind a button. Every extra sample is another five
# fits (one per fold), and past a dozen draws over spaces this small the gain is inside the
# fold noise anyway -- so a longer search buys a slower page rather than a better model.
DEFAULT_BUDGET = 12
CV_FOLDS = 5

# Parameters are set on the `model` step of the Pipeline built by modeling.build_pipeline,
# hence the `model__` prefix sklearn expects.
SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {
        # C is the whole story for a linear model: how much the coefficients are allowed to
        # grow before the penalty bites.
        "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        "model__class_weight": [None, "balanced"],
    },
    "random_forest_classifier": {
        # 800 trees is out: on data this size it lands inside the fold noise of 400 and costs
        # twice as long, and this search is something someone is waiting on.
        "model__n_estimators": [150, 300],
        "model__max_depth": [None, 6, 10, 16],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", None],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    },
    "gradient_boosting_classifier": {
        "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
        "model__max_iter": [100, 200, 400],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [5, 10, 20, 40],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    },
    "linear_regression": {
        "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "random_forest_regressor": {
        # 800 trees is out: on data this size it lands inside the fold noise of 400 and costs
        # twice as long, and this search is something someone is waiting on.
        "model__n_estimators": [150, 300],
        "model__max_depth": [None, 6, 10, 16],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", None],
    },
    "gradient_boosting_regressor": {
        "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
        "model__max_iter": [100, 200, 400],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [5, 10, 20, 40],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    },
}


@dataclass(frozen=True)
class SearchResult:
    """What a search found, and how much to believe it.

    `std` matters as much as `score`: a configuration that wins by less than the spread
    across folds hasn't been shown to be better, and improvement.py refuses those.
    """

    params: dict
    score: float
    std: float
    metric: str
    fits: int

    @property
    def readable_params(self) -> dict:
        """Without the `model__` prefix -- that's a plumbing detail, not something to show."""
        return {k.replace("model__", ""): v for k, v in self.params.items()}


def scoring_for(task: str, y) -> str:
    """The metric to search against.

    Deliberately the same choice cross_validate_pipeline makes, so the number a search
    reports and the number the diagnostics panel reports mean the same thing. AUC on a binary
    target because accuracy is misleading when the classes are lopsided; accuracy when there
    are more than two classes; R² for regression.
    """
    if task == "regression":
        return "r2"
    return "roc_auc" if pd.Series(y).nunique() == 2 else "accuracy"


def cv_splitter(task: str, y, folds: int = CV_FOLDS, groups=None, ordered: bool = False):
    """Stratified for classification, so every fold keeps the class balance -- without it a
    rare class can be missing from a fold entirely and the score for that fold is undefined.

    `groups` overrides that and keeps each entity whole instead. The two cannot both be had,
    and splitting an entity across folds is the one that inflates the score -- so when a run
    is grouped, every configuration in the search is compared on grouped folds. Otherwise the
    loop would pick its winner on exactly the leakage the split was chosen to prevent.

    `ordered` is for a date-based split, and overrides both. The final test set there is the
    most recent rows, but every fold inside the training rows used to be shuffled -- so model
    choice, tuning and the cut-off were all made by training on later rows and validating on
    earlier ones, which rewards whatever exploits the future: the very thing a date-based
    split exists to rule out. Time-ordered folds validate each chunk only on a model trained
    on the rows before it. The rows must arrive in time order, which `split_data` guarantees.

    Returns None when the data can't support even two folds, which is the caller's signal
    that searching isn't possible here.
    """
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

    if ordered:
        usable = min(folds, len(y) - 1)
        return TimeSeriesSplit(n_splits=usable) if usable >= 2 else None

    if groups is not None:
        usable = min(folds, int(pd.Series(groups).nunique()))
        return GroupKFold(n_splits=usable) if usable >= 2 else None

    if task != "classification":
        return KFold(n_splits=folds, shuffle=True, random_state=42)

    counts = pd.Series(y).value_counts()
    # One class in the training data is the degenerate case that reaches here when a rare
    # class is small enough that the stratified split couldn't keep any of it on this side.
    # Every fold would then be single-class, which no classifier will fit.
    if len(counts) < 2:
        return None

    usable = min(folds, int(counts.min()))
    if usable < 2:
        return None
    return StratifiedKFold(n_splits=usable, shuffle=True, random_state=42)


def score_config(
    build: Callable[[], Any],
    X_train: pd.DataFrame,
    y_train,
    task: str,
    *,
    folds: int = CV_FOLDS,
    groups=None,
    ordered: bool = False,
) -> SearchResult | None:
    """Cross-validates one configuration. The unit of comparison everything else uses.

    `build` returns an *unfitted* pipeline each call -- each fold has to fit from scratch, or
    the imputer's medians and the encoder's categories carry over from rows the fold is about
    to be scored on.
    """
    from sklearn.model_selection import cross_val_score

    splitter = cv_splitter(task, y_train, folds, groups, ordered)
    if splitter is None:
        return None

    metric = scoring_for(task, y_train)
    y_input = pd.Series(y_train).astype(str) if task == "classification" else y_train
    try:
        scores = cross_val_score(
            build(), X_train, y_input, cv=splitter, scoring=metric, n_jobs=1, groups=groups
        )
    except Exception as exc:  # noqa: BLE001 -- a config that won't fit is a result, not a crash
        logger.warning("Configuration failed to score: %s", exc, exc_info=True)
        return None

    return SearchResult(
        params={},
        score=round(float(np.mean(scores)), 4),
        std=round(float(np.std(scores)), 4),
        metric=metric,
        fits=len(scores),
    )


def search(
    build: Callable[[], Any],
    model_name: str,
    X_train: pd.DataFrame,
    y_train,
    task: str,
    *,
    budget: int = DEFAULT_BUDGET,
    folds: int = CV_FOLDS,
    groups=None,
    ordered: bool = False,
) -> SearchResult | None:
    """Randomized search over this model's space, cross-validated on the training data.

    Returns None when there is nothing to search (no space for this model, or too few rows
    per class to fold), which the caller reports as "nothing to try" rather than as a failure.
    """
    from sklearn.model_selection import RandomizedSearchCV

    space = SEARCH_SPACES.get(model_name)
    if not space:
        return None

    splitter = cv_splitter(task, y_train, folds, groups, ordered)
    if splitter is None:
        return None

    metric = scoring_for(task, y_train)
    y_input = pd.Series(y_train).astype(str) if task == "classification" else y_train

    # Never sample more candidates than the space actually holds -- otherwise the same
    # configuration gets fitted repeatedly and the budget is spent on duplicates.
    combinations = 1
    for values in space.values():
        combinations *= len(values)
    iterations = min(budget, combinations)

    searcher = RandomizedSearchCV(
        build(),
        space,
        n_iter=iterations,
        cv=splitter,
        scoring=metric,
        random_state=42,
        n_jobs=1,
        refit=False,  # improvement.py refits once, at the end, on everything it kept
        # A combination that can't fit (say `max_features=None` on a degenerate matrix) is
        # scored NaN and loses, rather than aborting the whole search.
        error_score=np.nan,
    )
    try:
        searcher.fit(X_train, y_input, groups=groups)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Search failed for %s: %s", model_name, exc, exc_info=True)
        return None

    index = int(searcher.best_index_)
    std = float(searcher.cv_results_["std_test_score"][index])
    if not np.isfinite(searcher.best_score_):
        return None

    return SearchResult(
        params=dict(searcher.best_params_),
        score=round(float(searcher.best_score_), 4),
        std=round(std, 4),
        metric=metric,
        fits=iterations * splitter.get_n_splits(),
    )
