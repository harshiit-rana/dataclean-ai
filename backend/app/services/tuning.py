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
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

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


# Below this many training rows, comparing models on one 5-fold partition is a coin that can
# land either way: another partition can rank two candidates the other way round. So comparisons
# there repeat the folds with different shuffles and average. Above it one partition is stable
# enough, and repeats would multiply the cost on the files where it is largest.
REPEATED_CV_BELOW_ROWS = 2_000
CV_REPEATS = 3


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
    # The score of each fold, in fold order. Two configurations scored with `score_config` on the
    # same rows are scored on the same folds, so these can be compared fold by fold: how often
    # and by how much one beat the other is far more telling than two means and two spreads.
    fold_scores: tuple[float, ...] = ()
    # How many times the folds were repeated with a different shuffle: `fits` is folds x repeats.
    # One for anything but a small file split at random.
    repeats: int = 1
    # Settings tried, of those the budget allowed, and whether the time limit cut the search short.
    # A search that stopped for time is the best of the settings it got to, which is a prefix of the
    # same sequence a full search would have drawn.
    tried: int = 0
    planned: int = 0
    stopped_for_time: bool = False
    # Which search strategy found it (see `SearchStrategy`), so the trail and the run history can say.
    strategy: str = "random"
    # What each rung of a halving search did: `{"rung", "settings", "folds", "best_score", "kept"}`.
    # Empty for a search that has no rungs.
    rungs: tuple[dict, ...] = ()
    # Why this is not the strategy that was asked for, when it is not (halving with too little to halve).
    note: str = ""

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


def cv_splitter(
    task: str, y, folds: int = CV_FOLDS, groups=None, ordered: bool = False, repeated: bool = False
):
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
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        RepeatedKFold,
        RepeatedStratifiedKFold,
        StratifiedGroupKFold,
        StratifiedKFold,
        TimeSeriesSplit,
    )

    from app.services.time_folds import TimeFoldRule

    if ordered:
        usable = min(folds, len(y) - 1)
        if usable < 2:
            return None
        # A rule carries the timestamps, so the folds keep every timestamp on one side and can
        # leave a gap. The bare flag is for callers that have no timestamps to give (and for
        # tests); it falls back to cutting at row positions, which the app itself never does.
        if isinstance(ordered, TimeFoldRule):
            return ordered.splitter(usable)
        return TimeSeriesSplit(n_splits=usable)

    if groups is not None:
        usable = min(folds, int(pd.Series(groups).nunique()))
        if usable < 2:
            return None
        if task == "classification":
            # Whole groups stay together, and within that the classes are balanced across folds as
            # far as the grouping allows. A plain GroupKFold balances only the number of rows, so
            # on a lopsided target with few entities a fold could end up with almost none of the
            # rarer answer. Best-effort: when a rare answer is concentrated in a few large groups
            # no assignment can balance it, and a fold that cannot be scored is reported as such.
            return StratifiedGroupKFold(n_splits=usable, shuffle=True, random_state=42)
        return GroupKFold(n_splits=usable)

    # `repeated` is asked for by the calls that *compare* models (choosing one, tuning it, the
    # fold spread shown beside the score). On a small file split at random they repeat the folds;
    # grouped and date splits are left alone (repeats need group-aware reshuffling, and time order
    # has only one honest partition). The cut-off and calibration never ask: they need every row
    # validated exactly once.
    repeat = repeated and len(y) < REPEATED_CV_BELOW_ROWS

    if task != "classification":
        if repeat:
            return RepeatedKFold(n_splits=folds, n_repeats=CV_REPEATS, random_state=42)
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
    if repeat:
        return RepeatedStratifiedKFold(n_splits=usable, n_repeats=CV_REPEATS, random_state=42)
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

    splitter = cv_splitter(task, y_train, folds, groups, ordered, repeated=True)
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
        fold_scores=tuple(round(float(s), 6) for s in scores),
        repeats=int(getattr(splitter, "n_repeats", 1)),
    )


class SearchStrategy(Protocol):
    """How a model's settings are searched: which to try, in what order, and when to stop.

    This interface exists because a second strategy (successive halving) is being built, and it was
    written with that second implementation rather than before it, so it holds only what random search
    and halving both need. Nothing speculative: no hooks for strategies that do not exist yet.

    `run` is handed everything already decided -- the unfitted pipeline factory, the space, the data
    with its target already in the form the scorer wants, the folds, the metric, the budget (a count of
    settings the strategy may spend, however it spends them) and the deadline -- and returns the best
    setting it found as a `SearchResult` with its own `name` in `strategy`, or None when nothing could
    be scored. A strategy must check the deadline between pieces of work and report `stopped_for_time`.
    """

    name: str

    def run(
        self,
        build: Callable[[], Any],
        space: dict[str, list[Any]],
        X: pd.DataFrame,
        y,
        *,
        splitter,
        metric: str,
        budget: int,
        deadline: float | None = None,
        groups=None,
    ) -> SearchResult | None: ...


class RandomSearch:
    """Settings drawn at random, each scored on every fold, the best kept.

    The settings are drawn exactly as `RandomizedSearchCV` draws them (the same `ParameterSampler` and
    seed) but scored one at a time, so a search can be stopped part-way: `deadline` is a
    `time.monotonic()` instant, checked before each setting after the first. A search that runs out of
    time returns the best of the settings it reached and says so (`stopped_for_time`). Without a deadline
    the result is the one `RandomizedSearchCV` gave.
    """

    name = "random"

    def run(
        self,
        build: Callable[[], Any],
        space: dict[str, list[Any]],
        X: pd.DataFrame,
        y,
        *,
        splitter,
        metric: str,
        budget: int,
        deadline: float | None = None,
        groups=None,
    ) -> SearchResult | None:
        return _random_search(build, space, X, y, splitter, metric, budget, deadline, groups, self.name)


class SuccessiveHalvingSearch:
    """Many settings scored cheaply, the promising ones scored more, the best scored fully.

    Random search spends the same on every setting, including ones that are plainly poor after one
    fold. Here a wide set of settings is scored on the first few folds, the best third go on to more
    folds, and the best third of those to all of them. The resource is cross-validation folds, not
    training rows: the folds are cut once, by the same splitter random search uses (grouped, forward in
    time, stratified), and each rung simply scores more of them, so a grouped or date split is respected
    at every rung without cutting the data a second time. The survivor ends with a score on *every*
    fold, which is what the improvement loop compares with the incumbent; a score on a few folds is
    never compared with anything.

    The trade-off: the early rungs rank settings on a few folds, which are noisy, and a setting that
    only shows its worth across all of them can be dropped before it gets there. That is the price of
    scoring many settings for little; with a small space or few folds the strategy says so and searches
    at random instead.

    Cost is in fold fits. With five folds, twice the budget in settings and three rungs (1, 2 and 5 folds)
    it spends about two thirds of what random search does for the same budget while looking at twice as
    many settings.
    """

    name = "halving"
    FACTOR = 3
    RUNGS = 3
    # Fewer folds than this leave no cheap rung to start on; fewer settings leave nothing to discard.
    MIN_FOLDS = 3
    MIN_SETTINGS = 6
    # Settings tried for each unit of budget. Halving looks at more because most are scored cheaply.
    SETTINGS_PER_BUDGET = 2

    def run(
        self,
        build: Callable[[], Any],
        space: dict[str, list[Any]],
        X: pd.DataFrame,
        y,
        *,
        splitter,
        metric: str,
        budget: int,
        deadline: float | None = None,
        groups=None,
    ) -> SearchResult | None:
        from sklearn.base import clone
        from sklearn.model_selection import ParameterSampler, cross_val_score

        splits = list(splitter.split(X, y, groups))
        combinations = 1
        for values in space.values():
            combinations *= len(values)
        wanted = min(combinations, self.SETTINGS_PER_BUDGET * max(int(budget), 1))
        if len(splits) < self.MIN_FOLDS or wanted < self.MIN_SETTINGS:
            fallback = _random_search(build, space, X, y, splitter, metric, budget, deadline, groups, RandomSearch.name)
            if fallback is None:
                return None
            reason = (
                f"only {len(splits)} folds" if len(splits) < self.MIN_FOLDS else f"only {wanted} settings to try"
            )
            return _with(fallback, note=f"Searched at random: {reason} is too few to halve over.")

        schedule = self.fold_schedule(len(splits))
        settings = list(ParameterSampler(space, n_iter=wanted, random_state=42))
        template = build()
        folds_scored: dict[int, list[float]] = {i: [] for i in range(len(settings))}
        alive = list(range(len(settings)))
        fits = 0
        stopped_for_time = False
        rungs: list[dict] = []

        def mean_of(i: int) -> float:
            mean = float(np.mean(folds_scored[i])) if folds_scored[i] else float("nan")
            return mean if np.isfinite(mean) else float("-inf")

        def score(i: int, start: int, stop: int) -> None:
            nonlocal fits
            try:
                out = cross_val_score(
                    clone(template).set_params(**settings[i]), X, y, cv=splits[start:stop], scoring=metric,
                    n_jobs=1, error_score=np.nan,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Halving search failed: %s", exc, exc_info=True)
                out = np.full(stop - start, np.nan)
            folds_scored[i].extend(float(v) for v in out)
            fits += stop - start

        for rung, folds_now in enumerate(schedule):
            already = len(folds_scored[alive[0]])
            for i in alive:
                # After the first fit of all, so a search that was started has something to report.
                if deadline is not None and fits > 0 and time.monotonic() >= deadline:
                    stopped_for_time = True
                    break
                score(i, already, folds_now)
            if stopped_for_time:
                break
            ranked = sorted(alive, key=lambda i: (-mean_of(i), i))
            keep = 1 if rung == len(schedule) - 1 else max(1, -(-len(alive) // self.FACTOR))
            rungs.append({
                "rung": rung + 1, "settings": len(alive), "folds": folds_now,
                "best_score": round(mean_of(ranked[0]), 4) if np.isfinite(mean_of(ranked[0])) else None,
                "kept": keep,
            })
            alive = ranked[:keep]

        if stopped_for_time:
            # Settings scored on the same number of folds are the only ones that can be compared.
            deepest = max(len(folds_scored[i]) for i in alive)
            pool = [i for i in alive if len(folds_scored[i]) == deepest]
            winner = max(pool, key=lambda i: (mean_of(i), -i))
            # The leader is finished on the remaining folds, so what is reported is a full score.
            if len(folds_scored[winner]) < len(splits):
                score(winner, len(folds_scored[winner]), len(splits))
        else:
            winner = alive[0]

        values = folds_scored[winner]
        mean = float(np.mean(values)) if values else float("nan")
        if len(values) != len(splits) or not np.isfinite(mean):
            return None
        return SearchResult(
            params=dict(settings[winner]),
            score=round(mean, 4),
            std=round(float(np.std(values)), 4),
            metric=metric,
            fits=fits,
            repeats=int(getattr(splitter, "n_repeats", 1)),
            tried=sum(1 for i in folds_scored if folds_scored[i]),
            planned=len(settings),
            stopped_for_time=stopped_for_time,
            strategy=self.name,
            rungs=tuple(rungs),
        )

    @classmethod
    def fold_schedule(cls, total: int) -> list[int]:
        """How many folds each rung has scored by its end: the last rung is all of them."""
        return sorted({max(1, round(total / cls.FACTOR**j)) for j in range(cls.RUNGS - 1, -1, -1)})


# Where halving starts to pay. Below these the fits are cheap enough that scoring fewer folds saves
# little and only adds noise to the ranking. They are a first guess, to be revisited with benchmark data
# on real files: moving them changes which strategy a run uses, never what either strategy does.
HALVING_MIN_ROWS = 5_000
HALVING_MIN_SETTINGS = 12  # settings in the model's space
HALVING_MIN_BUDGET = 12  # fits the caller allows


def choose_strategy(model_name: str, training_rows: int, budget: int) -> tuple[str, str]:
    """Which strategy to search this model's settings with, and why, in a sentence for the trail.

    Halving when there are many settings, a budget big enough to cut from, and enough rows that a fit
    costs something worth saving; random search otherwise. Pure: the same inputs always give the same
    answer, so the choice can be tested at each boundary and shown to the user with its reason.
    """
    space = SEARCH_SPACES.get(model_name) or {}
    combinations = int(np.prod([len(v) for v in space.values()])) if space else 0
    if combinations < HALVING_MIN_SETTINGS:
        return "random", (
            f"this model has only {combinations} settings to try, fewer than the {HALVING_MIN_SETTINGS} "
            f"needed for halving to have anything to discard"
        )
    if budget < HALVING_MIN_BUDGET:
        return "random", (
            f"a budget of {budget} is below the {HALVING_MIN_BUDGET} where halving pays for its early rungs"
        )
    if training_rows < HALVING_MIN_ROWS:
        return "random", (
            f"{training_rows:,} training rows is below the {HALVING_MIN_ROWS:,} where scoring fewer folds "
            f"saves real time"
        )
    return "halving", (
        f"{training_rows:,} training rows and {combinations:,} possible settings: cheaper to rank them on a "
        f"few folds first and score only the promising ones on all of them"
    )


def _with(result: SearchResult, **changes) -> SearchResult:
    from dataclasses import replace

    return replace(result, **changes)


# The strategies a search can be asked for by name.
STRATEGIES: dict[str, SearchStrategy] = {
    RandomSearch.name: RandomSearch(),
    SuccessiveHalvingSearch.name: SuccessiveHalvingSearch(),
}
DEFAULT_STRATEGY = RandomSearch.name


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
    deadline: float | None = None,
    strategy: str | SearchStrategy = DEFAULT_STRATEGY,
) -> SearchResult | None:
    """Searches this model's settings with a strategy, cross-validated on the training data.

    A thin wrapper: it settles what every strategy needs the same way (this model's space, the folds,
    the metric, the target in the form the scorer wants) and hands the rest to the strategy.

    Returns None when there is nothing to search (no space for this model, or too few rows
    per class to fold), which the caller reports as "nothing to try" rather than as a failure.
    """
    if isinstance(strategy, str):
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown search strategy {strategy!r}; known: {sorted(STRATEGIES)}")
        strategy = STRATEGIES[strategy]

    space = SEARCH_SPACES.get(model_name)
    if not space:
        return None

    splitter = cv_splitter(task, y_train, folds, groups, ordered, repeated=True)
    if splitter is None:
        return None

    metric = scoring_for(task, y_train)
    y_input = pd.Series(y_train).astype(str) if task == "classification" else y_train
    return strategy.run(
        build, space, X_train, y_input,
        splitter=splitter, metric=metric, budget=budget, deadline=deadline, groups=groups,
    )


def _random_search(build, space, X_train, y_input, splitter, metric, budget, deadline, groups, strategy_name):
    from sklearn.base import clone
    from sklearn.model_selection import ParameterSampler, cross_val_score

    # Never sample more candidates than the space actually holds -- otherwise the same
    # configuration gets fitted repeatedly and the budget is spent on duplicates.
    combinations = 1
    for values in space.values():
        combinations *= len(values)
    iterations = min(budget, combinations)

    settings = list(ParameterSampler(space, n_iter=iterations, random_state=42))
    template = build()
    best: tuple[float, float, dict] | None = None  # mean, spread across folds, settings
    tried = 0
    stopped_for_time = False
    for params in settings:
        # After the first, so a search that was started always has something to report; the caller
        # decides whether to start one at all.
        if deadline is not None and tried > 0 and time.monotonic() >= deadline:
            stopped_for_time = True
            break
        try:
            scores = cross_val_score(
                clone(template).set_params(**params), X_train, y_input, cv=splitter, scoring=metric,
                n_jobs=1, groups=groups,
                # A combination that can't fit (say `max_features=None` on a degenerate matrix) is
                # scored NaN and loses, rather than aborting the whole search.
                error_score=np.nan,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Search failed: %s", exc, exc_info=True)
            return None
        tried += 1
        mean = float(np.mean(scores))
        if np.isfinite(mean) and (best is None or mean > best[0]):
            best = (mean, float(np.std(scores)), dict(params))

    if best is None:
        return None

    return SearchResult(
        params=best[2],
        score=round(best[0], 4),
        std=round(best[1], 4),
        metric=metric,
        fits=tried * splitter.get_n_splits(),
        repeats=int(getattr(splitter, "n_repeats", 1)),
        tried=tried,
        planned=len(settings),
        stopped_for_time=stopped_for_time,
        strategy=strategy_name,
    )
