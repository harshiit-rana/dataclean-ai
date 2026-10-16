"""Keep trying to make the model better -- honestly, and knowing when to stop.

The naive version of "don't stop until we reach the goal" is a machine for fooling yourself.
Try enough configurations against the held-out rows and one of them will clear any bar you
set, not because it is better but because it suits those particular rows. The loop below is
built so that cannot happen:

- The test set is split off once and never touched during the search. Every round competes on
  cross-validation folds drawn from the training data alone.
- A round is only accepted if it beats the incumbent by more than the fold-to-fold spread.
  Winning by less than the noise is not winning; it is picking a number out of a hat.
- The loop stops. It stops when the goal is met, when a round stops earning its place, or
  when it runs out of things to try -- and then it says which, because "we tried everything
  and this data will not support what you asked for" is a real answer and a useful one.

At the end the winning configuration is refitted on the whole training set and scored once on
the sealed test set. That number is the honest one, and it is usually a little below the
cross-validated score that won -- which is itself the evidence the loop is behaving.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from app.services import features as feature_recipes, modeling, operating_point, time_folds, tuning

logger = logging.getLogger(__name__)

# A round has to beat the incumbent by more than this multiple of the fold spread to count.
# One standard deviation is deliberately strict: at half of one, ordinary fold noise on a few
# hundred rows clears the bar most of the time, and the loop would "improve" forever.
MIN_GAIN_IN_SDS = 1.0

# ...but on data where folds agree almost perfectly the spread collapses toward zero and the
# rule above would accept a gain of 0.0001. This is the floor underneath it.
MIN_ABSOLUTE_GAIN = 0.002

# sklearn's scorer names read like plumbing. These are what the rounds say out loud.
METRIC_LABELS = {"roc_auc": "AUC", "accuracy": "accuracy", "r2": "R²"}

# Whether a search winner is re-scored on folds it was not selected on before it can replace the
# incumbent. It is a module constant so the effect can be measured with it off (see the commit that
# introduced it and `tests/unit/test_winners_curse.py`); there is no reason to turn it off in use.
CONFIRM_SEARCH_WINNERS = True

# How far behind the leader a model can be and still be worth tuning. Tuning moves a score by
# a few hundredths on data this size; a family that lost the first round by more than this is
# not going to close the gap, and searching it anyway is the single most expensive thing the
# loop can do for the least return.
WORTH_TUNING_WITHIN = 0.05

# Fewer inputs than this and there is nothing worth ranking: removing one of three leaves two.
MIN_INPUTS_TO_SELECT = 4

# The interactions round looks at the numeric inputs that matter most, and only so many: every pair tried is
# another chance for one to look good by luck, so the number tried is bounded and recorded.
INTERACTION_TOP_INPUTS = 4
MAX_INTERACTIONS_TRIED = 8
MAX_INTERACTIONS_KEPT = 2


@dataclass
class Round:
    """One attempt at making it better, kept whether or not it worked.

    The rejected rounds are the interesting ones. "We tried tuning it and tuning did not
    help" tells you something about your data that a silent success would not.
    """

    name: str
    label: str
    what_it_tried: str
    score: float | None = None
    std: float | None = None
    gain: float | None = None
    accepted: bool = False
    outcome: str = ""
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "what_it_tried": self.what_it_tried,
            "score": self.score,
            "std": self.std,
            "gain": self.gain,
            "accepted": self.accepted,
            "outcome": self.outcome,
            "detail": self.detail,
        }


def fold_importance(
    build: Callable[[], object], X: pd.DataFrame, y, task: str, *, groups, ordered, columns: list[str],
    repeats: int = 2,
) -> dict[str, tuple[float, float]]:
    """How much each input matters to the model, measured on validation folds of the training rows.

    For every fold the incumbent is fitted on the fold's training part, scored on its validation part, and
    scored again with one input's values shuffled there; the drop is that input's importance on that fold.
    Returns, per column, the mean of the drops over the folds and their spread. The held-out test rows are
    never touched, and neither is a row the model was fitted on, which is what makes a *negative or zero*
    importance mean something: shuffling the column on rows the model has not seen did not hurt it.
    Uses one partition of the folds, not the repeated ones: it is a ranking, not a comparison.
    """
    from sklearn.metrics import get_scorer

    splitter = tuning.cv_splitter(task, y, groups=groups, ordered=ordered)
    if splitter is None:
        return {}
    y_in = pd.Series(y).astype(str).reset_index(drop=True) if task == "classification" else pd.Series(y).reset_index(drop=True)
    frame = X.reset_index(drop=True)
    group_values = None if groups is None else pd.Series(groups).reset_index(drop=True)
    scorer = get_scorer(tuning.scoring_for(task, y))
    rng = np.random.default_rng(modeling.RANDOM_STATE)
    drops: dict[str, list[float]] = {c: [] for c in columns}
    for train_idx, val_idx in splitter.split(frame, y_in, group_values):
        try:
            model = build().fit(frame.iloc[train_idx], y_in.iloc[train_idx])
            held, truth = frame.iloc[val_idx], y_in.iloc[val_idx]
            base = scorer(model, held, truth)
            for column in columns:
                lost = []
                for _ in range(repeats):
                    shuffled = held.copy()
                    shuffled[column] = rng.permutation(shuffled[column].to_numpy())
                    lost.append(base - scorer(model, shuffled, truth))
                drops[column].append(float(np.mean(lost)))
        except Exception:  # noqa: BLE001 -- a fold with one answer in it cannot be scored; the others can
            logger.debug("A fold could not be used for the input importance", exc_info=True)
    return {c: (float(np.mean(v)), float(np.std(v))) for c, v in drops.items() if v}


def fold_expected_costs(
    build: Callable[[], object], X: pd.DataFrame, y, *, positive_label: str, cost_of_missing: float,
    cost_of_false_alarm: float, groups=None, ordered=False,
) -> list[float] | None:
    """The stated expected cost per row on each validation fold, at the cut-off that minimises it.

    Out-of-fold probabilities for every training row (each from a model that did not train on it), the cut-off
    chosen on all of them the way the app chooses it, then the cost of the misses and false alarms on each
    fold's rows at that cut-off. Fold by fold, so two ways of training can be compared on the same folds.
    Returns None when the folds cannot be made or scored.
    """
    point = operating_point.OperatingPoint(
        objective="cost", cost_of_missing=float(cost_of_missing), cost_of_false_alarm=float(cost_of_false_alarm),
        positive_label=positive_label,
    )
    scores, _ = operating_point.out_of_fold_scores(build, X, y, positive_label, groups, ordered)
    if scores is None:
        return None
    labels = pd.Series(y).astype(str).reset_index(drop=True)
    binary = (labels == str(positive_label)).to_numpy().astype(int)
    covered = ~np.isnan(scores)
    if not covered.any() or binary[covered].min() == binary[covered].max():
        return None
    threshold = operating_point.choose_threshold(binary[covered], scores[covered], point)["threshold"]
    splitter = tuning.cv_splitter("classification", labels, groups=groups, ordered=ordered)
    group_values = None if groups is None else pd.Series(groups).reset_index(drop=True)
    costs = []
    for _, held in splitter.split(X.reset_index(drop=True), labels, group_values):
        held = held[covered[held]]
        if len(held) == 0:
            continue
        at = operating_point.metrics_at(binary[held], scores[held], threshold)
        costs.append((cost_of_missing * at["missed"] + cost_of_false_alarm * at["false_alarms"]) / len(held))
    return costs or None


def _without(features: dict, removed: set[str]) -> dict:
    """The feature lists with these inputs taken out."""
    kept = dict(features)
    for key in ("numeric", "categorical", "frequent", "target_encoded", "clipped"):
        if key in kept and kept[key] is not None:
            kept[key] = [c for c in kept[key] if c not in removed]
    return kept


def chance_level(metric: str, y) -> float | None:
    """What a model that has learned nothing scores on this metric: 0.5 AUC, the commonest answer's share of
    accuracy, 0 R-squared. None for a metric without one."""
    if metric == "roc_auc":
        return 0.5
    if metric == "r2":
        return 0.0
    if metric == "accuracy":
        shares = pd.Series(y).astype(str).value_counts(normalize=True)
        return float(shares.iloc[0]) if len(shares) else None
    return None


def clear_of_chance(score: float, std: float | None, metric: str, y) -> bool:
    """Whether a fold score is above what guessing scores by more than the folds disagree. A change that only
    brings a model up to chance -- a default that overfits noise to below 0.5 AUC, replaced by one that
    predicts the same for everyone -- is not an improvement worth keeping: with no signal every setting is
    chance, and calling one of them better is the luck the confirmation exists to refuse."""
    floor = chance_level(metric, y)
    return floor is None or score - float(std or 0.0) > floor


def _reached(found: tuning.SearchResult) -> dict:
    """What a settings search was and how far it got, for the trail: always the strategy, and the
    settings it reached only when the time limit cut it short."""
    reached: dict = {"strategy": found.strategy}
    if found.rungs:
        reached["rungs"] = list(found.rungs)
    if found.note:
        reached["note"] = found.note
    if found.stopped_for_time:
        reached |= {"tried": found.tried, "planned": found.planned, "stopped_for_time": True}
    return reached


def _is_real_gain(candidate: float, incumbent: float, std: float | None) -> bool:
    """Did this beat the incumbent by more than the measurement is worth?"""
    gain = candidate - incumbent
    if gain <= MIN_ABSOLUTE_GAIN:
        return False
    return gain >= MIN_GAIN_IN_SDS * float(std or 0.0)


def _goal_met(score: float, goal: float | None) -> bool:
    return goal is not None and score >= goal


def improve(
    df: pd.DataFrame,
    *,
    target_column: str,
    task: str,
    model_names: list[str] | None = None,
    excluded_columns: list[str] | None = None,
    split_strategy: str = "random",
    time_column: str | None = None,
    group_column: str | None = None,
    goal: float | None = None,
    budget: int = tuning.DEFAULT_BUDGET,
    on_progress: Callable[[Round], None] | None = None,
    feature_recipes_wanted: list[str] | None = None,
    on_stage: Callable[[str, str], None] | None = None,
    clip_outliers: list[str] | None = None,
    keep_identical_rows_together: bool = False,
    sealed_holdout: bool = False,
    test_size: float | None = None,
    time_gap_days: int | None = None,
    group_columns: list[str] | None = None,
    time_limit_seconds: float | None = None,
    search_strategy: str = "auto",
    cost_limit: int | None = None,
    stated_costs: dict | None = None,
    journal=None,
) -> dict:
    """Runs rounds of increasingly expensive strategies until one of the stopping conditions.

    `stated_costs` is the user's cost objective for the cut-off, `{"cost_of_missing", "cost_of_false_alarm",
    "positive_label"}`, when they gave one. For a two-answer target the last round then asks whether training
    with those costs as row weights lowers the expected cost, fold by fold, beyond the folds' disagreement.

    `cost_limit` is the processing limit (see `modeling.plan_within_limit`): over it, the most expensive
    families are not tried and the settings searches get a smaller budget, and what was reduced is in the
    result's `resource_limits`. The loop always scores on every training row, so it never samples; a file
    too big even for the simplest model is refused.

    `search_strategy` is "random", "halving" or "auto" (the default), which picks one for each settings
    search from the model, the training rows and the budget (`tuning.choose_strategy`) and records why.
    An explicit choice is respected and recorded as the user's.

    `time_limit_seconds` bounds the wall-clock time the search may take, counted from the call. The
    deadline is checked before each round and before each runner-up family in the third, and the
    settings search inside a round stops at it, so the call returns within the limit plus the time
    one setting takes to score. A round that did not run is recorded in the trail as skipped, and
    `stopped_because` is `time_limit`. The one look at the test set at the end is not skipped: it
    is what makes the result a result, and it is a single fit.

    `goal` is a target for the same metric the rounds are judged on -- AUC for a two-class
    target, accuracy for more, R² for regression. None means "get it as good as you can",
    which stops when the rounds stop paying for themselves.

    `on_progress` is called after each round so the caller can stream them; the loop is slow
    enough that watching it work matters. `on_stage(name, description)` is called as each
    round *starts* -- a settings search can run for most of a minute on its own, and a
    stream that only speaks when a round ends looks exactly like one that has hung.

    `feature_recipes_wanted` are the derived inputs the run started with. The fourth round
    tries the opposite -- all of them if none were on, none if some were -- and keeps whichever
    the folds prefer. Tuning is the small lever; on the churn sample it found nothing, which
    is how we know the ceiling is the columns.

    `journal` (a `search_journal.Journal`) records every cross-validation, settings search and importance
    measurement, or replays a journal from an interrupted run of the same search, so that run resumes where it
    stopped instead of starting over (#104). None, the default, calls everything directly.
    """
    def through_journal(kind: str, function: Callable):
        if journal is None:
            return function
        return lambda *args, **kwargs: journal.call(kind, function, *args, **kwargs)

    score_config = through_journal("score_config", tuning.score_config)
    search = through_journal("search", tuning.search)
    importance_of = through_journal("fold_importance", fold_importance)
    expected_costs = through_journal("fold_expected_costs", fold_expected_costs)
    if target_column not in df.columns:
        raise modeling.ModelingError(f"'{target_column}' is not in the data.")

    limit = float(time_limit_seconds) if time_limit_seconds else None
    deadline = None if limit is None else time.monotonic() + limit

    # Mirrors train_candidates: the column being grouped on cannot also be an input, or the
    # loop would be searching for settings that exploit the very memorisation the grouped
    # split exists to prevent.
    withheld = list(excluded_columns or [])
    if split_strategy == "group":
        group_parts = modeling.resolve_group_columns(group_column, group_columns)
        if group_parts:
            df, group_column, to_withhold = modeling.with_group_key(df, group_parts, target_column)
            withheld += [c for c in to_withhold if c not in withheld]

    # The derived inputs in force, and the alternative the fourth round will try. Both frames
    # are built up front because the split is by index and the recipes only ever add columns,
    # so the same rows land on the same side of the split whichever frame is in use.
    raw_columns = [c for c in df.columns if c != target_column]
    proposed = feature_recipes.propose(df, target_column, withheld)
    # Target encoding cross-fits with ordinary folds, which split a group; not offered where rows are grouped.
    proposed, _declined = feature_recipes.without_target_encoding_where_it_leaks(
        proposed, grouped=split_strategy == "group" or keep_identical_rows_together
    )
    wanted_ids = set(feature_recipes_wanted or [])
    recipes = [r for r in proposed if r.id in wanted_ids]
    alternative = [] if recipes else proposed
    frames = {
        "current": (recipes, feature_recipes.apply(df, recipes)),
        "alternative": (alternative, feature_recipes.apply(df, alternative)),
    }
    df = frames["current"][1]

    def inputs_for(which: str) -> tuple[dict, list[str]]:
        chosen, frame = frames[which]
        classified = modeling.classify_features(
            frame, target_column, withheld, chosen, clip_outliers
        )
        return classified, classified["numeric"] + classified["categorical"] + classified["frequent"]

    features, used = inputs_for("current")
    if not used:
        raise modeling.ModelingError(
            "There are no usable columns left to predict from, so there is nothing to improve."
        )

    gap_days = int(time_gap_days or 0) if split_strategy == "time" else 0
    # Split once. Nothing below this line is allowed to look at X_test or y_test.
    duplicate_columns = used if keep_identical_rows_together and split_strategy == "random" else None
    parts = None
    if sealed_holdout:
        # The same sealed rows `train_candidates` takes, by the same rule, so the model this
        # loop settles on is rebuilt on exactly the partitions it searched within. When the file
        # cannot support three parts the ordinary split is used, as it is there.
        try:
            parts = modeling.split_with_sealed_holdout(
                df, target_column, task, strategy=split_strategy, time_column=time_column,
                group_column=group_column, duplicate_columns=duplicate_columns,
                copy_columns=(
                    [str(c) for c in raw_columns] if keep_identical_rows_together else None
                ),
                time_gap_days=gap_days,
            )
            if modeling.sealed_unavailable_reason(parts[4], parts[5], task) is not None:
                parts = None
        except modeling.ModelingError:
            parts = None
    if parts is not None:
        X_train, X_test, _sealed, y_train, y_test, _sealed_y = parts
    else:
        X_train, X_test, y_train, y_test = modeling.split_data(
            df,
            target_column,
            task,
            strategy=split_strategy,
            time_column=time_column,
            group_column=group_column,
            test_size=test_size if test_size else modeling.DEFAULT_TEST_SIZE,
            duplicate_columns=duplicate_columns,
            time_gap_days=gap_days,
        )
    # Captured before the frame is narrowed to the model's inputs, and aligned to the training
    # rows -- every round below compares configurations on these folds.
    train_groups = (
        df.loc[list(X_train.index), group_column]
        if split_strategy == "group" and group_column
        else None
    )
    if train_groups is None and keep_identical_rows_together and split_strategy == "random":
        # Copies were kept together at the split, so the folds keep them together too. Computed
        # once, from the inputs the run started with; the alternative frame the derived-inputs
        # round tries shares this split and these folds.
        copy_ids = modeling.identical_row_ids(df, used, target_column)
        if copy_ids.nunique() < len(copy_ids):
            train_groups = copy_ids.loc[list(X_train.index)]
    train_index, test_index = list(X_train.index), list(X_test.index)
    X_train, X_test = X_train[used], X_test[used]

    # A date-based split validates every fold forwards in time, here as in the final split.
    # The timestamps every fold is cut by for a date split; False otherwise. See time_folds.
    ordered = (
        time_folds.time_rule(df, time_column, gap_days) if split_strategy == "time" and time_column else False
    )
    if tuning.cv_splitter(task, y_train, groups=train_groups, ordered=ordered) is None:
        raise modeling.ModelingError(
            "There are too few rows in the smallest class to cross-validate, so there is no "
            "safe way to compare one configuration against another on this data."
        )

    candidates = [
        spec.name
        for spec in modeling.models_for_task(task)
        if not model_names or spec.name in model_names
    ]
    if not candidates:
        raise modeling.ModelingError("No models are available for this kind of prediction.")

    # Within the processing limit, or reduced to fit it: dearest families first, then the search budget.
    resource_plan = modeling.plan_within_limit(candidates, len(X_train), limit=cost_limit, can_sample=False)
    candidates = resource_plan["models"]
    reductions = list(resource_plan["reductions"])
    if cost_limit:
        folds_now, repeats_now = modeling._selection_folds(len(X_train))
        dearest = max(modeling.MODELS[m].cost_weight for m in candidates)
        affordable = max(2, int(cost_limit // (len(X_train) * folds_now * repeats_now * dearest)))
        if affordable < budget:
            reductions.append({
                "kind": "search_budget", "from": budget, "to": affordable,
                "reason": f"Each setting searched costs about {len(X_train) * folds_now * repeats_now * dearest:,} "
                f"row-fits on {len(X_train):,} rows, so the search tried {affordable} settings instead of {budget} "
                f"to stay within the processing limit.",
            })
            budget = affordable
    resource_limits = (
        {"limit": cost_limit, "estimated_before": resource_plan["estimated_before"],
         "estimated": resource_plan["estimated"], "reductions": reductions}
        if reductions
        else None
    )

    rounds: list[Round] = []
    metric = tuning.scoring_for(task, y_train)
    metric_label = METRIC_LABELS.get(metric, metric)

    def emit(entry: Round) -> None:
        rounds.append(entry)
        if on_progress:
            on_progress(entry)

    def starting(name: str, description: str) -> None:
        if on_stage:
            on_stage(name, description)

    def out_of_time() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    # Set once anything was left undone for want of time, whichever round it happened in.
    cut_short = False
    skipped_for_time: list[str] = []

    def skip_for_time(name: str, label: str, what: str) -> None:
        """Records a round that did not run because the time limit was reached first."""
        skipped_for_time.append(name)
        starting(name, "Skipped: the time limit was reached before this round.")
        emit(
            Round(
                name=name,
                label=label,
                what_it_tried=what,
                accepted=False,
                outcome=f"Not run: the {limit:g}-second time limit was reached first.",
                detail={"skipped": "time_limit"},
            )
        )

    strategies_used: set[str] = set()
    # The best settings each family's search found, accepted or not, for the members of an ensemble.
    searched_settings: dict[str, dict] = {}
    # The stated costs as row weights, once the last round has shown they lower the expected cost.
    training_weights: dict | None = None

    def strategy_for(model: str) -> tuple[str, str]:
        """The strategy for this model's settings search, and the reason, for the trail."""
        if search_strategy != "auto":
            return search_strategy, f"you chose {tuning.STRATEGIES[search_strategy].name} search"
        return tuning.choose_strategy(model, len(X_train), budget)

    def searched(found: tuning.SearchResult, reason: str) -> dict:
        strategies_used.add(found.strategy)
        reached = _reached(found)
        # A halving search that fell back to random says so itself; otherwise the reason it was chosen.
        return {**reached, "strategy_reason": found.note or reason}

    def confirm(found: tuning.SearchResult, model: str) -> dict:
        """Re-scores a search winner, and the incumbent it would replace, on folds the search did not use.

        The winner is the best of several noisy estimates, so its score on the folds it won on is biased
        upward; the incumbent is re-scored on the same fresh folds so the two are compared like for like,
        not a fresh score against an old one. Costs two cross-validations per round, whatever the budget.
        """
        nonlocal cut_short
        if out_of_time():
            cut_short = True
            return {"status": "skipped_for_time", "real": False}
        fresh = dict(groups=train_groups, ordered=ordered, confirmation=True)
        candidate = score_config(build_for(model, found.params), X_train, y_train, task, **fresh)
        incumbent = score_config(build_for(best_name, best_params), X_train, y_train, task, **fresh)
        if candidate is None or incumbent is None:
            return {"status": "unavailable", "real": False}
        beats = _is_real_gain(candidate.score, incumbent.score, candidate.std)
        clear = clear_of_chance(candidate.score, candidate.std, metric, y_train)
        return {
            "status": "confirmed" if beats and clear else "at_chance" if beats else "not_confirmed",
            "real": beats and clear,
            "search_score": found.score,
            "confirmation_score": candidate.score,
            "confirmation_std": candidate.std,
            "incumbent_confirmation_score": incumbent.score,
            "chance": chance_level(metric, y_train),
        }

    def confirmation_detail(check: dict | None) -> dict:
        if not check:
            return {}
        return {"confirmation": {k: v for k, v in check.items() if k != "real"}}

    def not_confirmed_outcome(
        check: dict, led: str = "The best setting looked better", kept: str = "the defaults were kept"
    ) -> str:
        """Why a winner on the search's folds did not replace the incumbent. `led` opens the sentence."""
        if check["status"] == "skipped_for_time":
            return f"{led}, but the time limit came before it could be checked, so {kept}."
        if check["status"] == "unavailable":
            return f"{led}, but it could not be checked on fresh folds, so {kept}."
        if check["status"] == "at_chance":
            return (
                f"{led}, and on fresh folds it beat what it would replace ({check['confirmation_score']:.3f} against "
                f"{check['incumbent_confirmation_score']:.3f}), but it is not clear of the {check['chance']:.3f} that "
                f"guessing scores by more than the folds disagree: it only brought the model up to chance, so {kept}."
            )
        return (
            f"{led} on the folds it was picked on ({check['search_score']:.3f}), but on fresh folds it was "
            f"{check['confirmation_score']:.3f} against {check['incumbent_confirmation_score']:.3f} for what it "
            f"would replace -- not by more than the folds disagree -- so {kept}."
        )

    def time_tail(found) -> str:
        if not found.stopped_for_time:
            return ""
        return f" The time limit stopped it after {found.tried} of {found.planned} settings."

    def build_for(
        name: str, params: dict | None = None, with_features: dict | None = None, weighting: dict | None = None,
    ) -> Callable[[], object]:
        feats = with_features or features
        weighting = weighting or training_weights

        def factory():
            ensemble = modeling.MODELS[name].ensemble
            # A cost-weighted model wraps the one that is tuned, so its searched values are passed to the
            # constructor rather than set on the pipeline's model step.
            as_settings = ensemble or bool(weighting)
            settings = (
                {k.replace("model__", ""): v for k, v in (params or {}).items()} if as_settings else None
            )
            if weighting:
                settings = {**(settings or {}), "cost_weighting": weighting}
            pipeline = modeling.build_pipeline(
                name,
                feats["numeric"],
                feats["categorical"],
                # An ensemble's settings are its members, which are built with it; everything else is tuned by
                # setting the searched values on the pipeline's model step.
                model_params=settings,
                frequent=feats["frequent"],
                target_encoded=feats.get("target_encoded"),
                clip=feats.get("clipped"),
            )
            if params and not as_settings:
                pipeline.set_params(**params)
            return pipeline

        return factory

    # --- Round 1: which model family, judged on folds rather than on the test set ---------
    starting(
        "baseline",
        f"Cross-validating {len(candidates)} model "
        f"{'family' if len(candidates) == 1 else 'families'} on the training rows.",
    )
    #
    # train_candidates picks its winner by test-set score, which is fine for three models
    # looked at once. A loop that keeps going cannot afford even that, so family selection
    # moves onto cross-validation here.
    family_scores: dict[str, float] = {}
    scored: dict[str, dict] = {}
    unscored_for_time: list[str] = []
    for name in candidates:
        # At least one family is always scored: without it there is nothing to improve.
        if scored and out_of_time():
            unscored_for_time.append(name)
            continue
        result = score_config(
            build_for(name), X_train, y_train, task, groups=train_groups, ordered=ordered
        )
        if result is None:
            continue
        family_scores[name] = result.score
        scored[name] = {
            "score": result.score,
            "std": result.std,
            # Folds of one partition, as in train_candidates: repeats are not more independent folds.
            "folds": result.fits // max(result.repeats, 1),
        }

    if not scored:
        raise modeling.ModelingError("None of the models could be scored on this data.")
    # The same rule the plain train uses, so a search and a train cannot start from different
    # models on the same evidence: the simplest family within one standard error of the best.
    pick = modeling.one_standard_error_choice(scored)
    best_name = pick["chosen"]
    best_score, best_std = scored[best_name]["score"], scored[best_name]["std"]

    best_params: dict = {}
    emit(
        Round(
            name="baseline",
            label="Starting point",
            what_it_tried=f"Cross-validated {len(family_scores)} model "
            f"{'family' if len(family_scores) == 1 else 'families'} on the training rows and "
            f"kept the strongest."
            + (
                f" The time limit left {len(unscored_for_time)} more unscored."
                if unscored_for_time
                else ""
            ),
            score=best_score,
            std=best_std,
            gain=None,
            accepted=True,
            outcome=f"{modeling.MODELS[best_name].label} leads at {best_score:.3f} {metric_label}.",
            detail={
                "scores": family_scores,
                "chosen": best_name,
                "rule": pick["rule"],
                "highest_scorer": pick["highest_scorer"],
                **({"unscored_for_time": unscored_for_time} if unscored_for_time else {}),
            },
        )
    )
    cut_short = bool(unscored_for_time)

    stopped_because = ""
    if _goal_met(best_score, goal):
        stopped_because = "goal_reached"

    # --- Round 2: settings for the leading family ----------------------------------------
    if not stopped_because and out_of_time():
        stopped_because = "time_limit"
    if stopped_because == "time_limit":
        skip_for_time("hyperparameters", "Model settings", "Searched this model's settings.")
    if not stopped_because:
        starting("hyperparameters", f"Searching the settings of {modeling.MODELS[best_name].label.lower()}.")
        chosen, why = strategy_for(best_name)
        found = search(
            build_for(best_name), best_name, X_train, y_train, task,
            budget=budget, groups=train_groups, ordered=ordered, deadline=deadline,
            strategy=chosen,
        )
        cut_short = cut_short or bool(found and found.stopped_for_time)
        # Worth it on the folds it was picked on; then it has to hold on fresh ones.
        worth = found is not None and _is_real_gain(found.score, best_score, found.std)
        check = None
        if worth and CONFIRM_SEARCH_WINNERS:
            check = confirm(found, best_name)
            worth = check["real"]
        if found is None:
            emit(
                Round(
                    name="hyperparameters",
                    label="Model settings",
                    what_it_tried="Searched this model's settings.",
                    outcome="This model has no settings worth searching.",
                )
            )
        elif worth:
            gain = round(found.score - best_score, 4)
            best_score, best_std, best_params = found.score, found.std, found.params
            emit(
                Round(
                    name="hyperparameters",
                    label="Model settings",
                    what_it_tried=f"Tried {found.fits} fits across this model's settings, "
                    f"scored on the training folds." + time_tail(found),
                    score=found.score,
                    std=found.std,
                    gain=gain,
                    accepted=True,
                    outcome=f"Better settings found: {metric_label} up {gain:+.3f}."
                    + (
                        f" Confirmed on fresh folds: {check['confirmation_score']:.3f} against "
                        f"{check['incumbent_confirmation_score']:.3f}."
                        if check
                        else ""
                    ),
                    detail={
                        "params": found.readable_params, **searched(found, why), **confirmation_detail(check)
                    },
                )
            )
        else:
            emit(
                Round(
                    name="hyperparameters",
                    label="Model settings",
                    what_it_tried=f"Tried {found.fits} fits across this model's settings, "
                    f"scored on the training folds." + time_tail(found),
                    score=found.score,
                    std=found.std,
                    gain=round(found.score - best_score, 4),
                    accepted=False,
                    outcome=(
                        not_confirmed_outcome(check)
                        if check is not None
                        else "No setting beat the defaults by more than the folds disagree "
                        "with each other, so the defaults were kept."
                    ),
                    detail={
                        "params": found.readable_params, **searched(found, why), **confirmation_detail(check)
                    },
                )
            )

        if _goal_met(best_score, goal):
            stopped_because = "goal_reached"

    # --- Round 3: tune the runners-up too, in case the ranking changes once they're tuned -
    #
    # An untuned gradient booster losing to a tuned logistic regression says nothing about
    # which is better; default settings favour some families more than others.
    if not stopped_because and out_of_time():
        stopped_because = "time_limit"
    if stopped_because == "time_limit" and "other_models" not in skipped_for_time:
        skip_for_time(
            "other_models", "The other models, tuned",
            "Looked at whether any of the models that lost the first round were close enough to be "
            "worth searching.",
        )
    if not stopped_because:
        starting("other_models", "Tuning the runners-up, in case they only lost on their defaults.")
        challengers = [
            name
            for name, score in family_scores.items()
            if name != best_name and score >= best_score - WORTH_TUNING_WITHIN
        ]
        skipped = len(family_scores) - len(challengers) - 1
        improved_challenger = None
        untried_for_time: list[str] = []
        for name in challengers:
            if out_of_time():
                untried_for_time.append(name)
                continue
            chosen, why = strategy_for(name)
            found = search(
                build_for(name), name, X_train, y_train, task,
                budget=budget, groups=train_groups, ordered=ordered, deadline=deadline,
                strategy=chosen,
            )
            if found:
                strategies_used.add(found.strategy)
                searched_settings[name] = found.params  # what an ensemble of it would use
            cut_short = cut_short or bool(found and found.stopped_for_time)
            if found and _is_real_gain(found.score, best_score, found.std):
                if improved_challenger is None or found.score > improved_challenger[1].score:
                    improved_challenger = (name, found)

        cut_short = cut_short or bool(untried_for_time)
        # The best challenger has to hold on fresh folds too, against the incumbent it would replace.
        check3 = None
        failed_check3 = None
        if improved_challenger is not None and CONFIRM_SEARCH_WINNERS:
            check3 = confirm(improved_challenger[1], improved_challenger[0])
            if not check3["real"]:
                failed_check3 = (improved_challenger[0], check3)
                improved_challenger, check3 = None, None
        time_note = (
            f" The time limit stopped it before {len(untried_for_time)} of the "
            f"{len(challengers)} could be tried."
            if untried_for_time
            else ""
        )
        if not challengers:
            emit(
                Round(
                    name="other_models",
                    label="The other models, tuned",
                    what_it_tried="Looked at whether any of the models that lost the first "
                    "round were close enough to be worth searching.",
                    accepted=False,
                    outcome=f"All {skipped} of them were too far behind for tuning to close "
                    f"the gap, so the time went elsewhere.",
                )
            )
        elif improved_challenger:
            name, found = improved_challenger
            gain = round(found.score - best_score, 4)
            best_name, best_score, best_std, best_params = name, found.score, found.std, found.params
            emit(
                Round(
                    name="other_models",
                    label="The other models, tuned",
                    what_it_tried=f"Searched the settings of the {len(challengers)} "
                    f"{'model' if len(challengers) == 1 else 'models'} close enough behind to "
                    f"be worth it, in case they only lost because of their defaults.",
                    score=found.score,
                    std=found.std,
                    gain=gain,
                    accepted=True,
                    outcome=f"{modeling.MODELS[name].label} overtakes it once tuned: "
                    f"{metric_label} up {gain:+.3f}."
                    + (
                        f" Confirmed on fresh folds: {check3['confirmation_score']:.3f} against "
                        f"{check3['incumbent_confirmation_score']:.3f}."
                        if check3
                        else ""
                    )
                    + time_note,
                    detail={
                        "model": name, "params": found.readable_params,
                        **({"untried_for_time": untried_for_time} if untried_for_time else {}),
                        **confirmation_detail(check3),
                    },
                )
            )
        else:
            emit(
                Round(
                    name="other_models",
                    label="The other models, tuned",
                    what_it_tried=f"Searched the settings of the {len(challengers)} "
                    f"{'model' if len(challengers) == 1 else 'models'} close enough behind to "
                    f"be worth it, in case they only lost because of their defaults.",
                    accepted=False,
                    outcome=(
                        not_confirmed_outcome(
                            failed_check3[1], f"{modeling.MODELS[failed_check3[0]].label} led once tuned",
                            "the incumbent was kept",
                        )
                        if failed_check3
                        else "None of them caught up once tuned."
                    )
                    + time_note,
                    detail={
                        **({"untried_for_time": untried_for_time} if untried_for_time else {}),
                        **(confirmation_detail(failed_check3[1]) if failed_check3 else {}),
                    },
                )
            )

        if _goal_met(best_score, goal):
            stopped_because = "goal_reached"

    # --- Round 4: the columns, not the settings ------------------------------------------
    #
    # Tuning moves a score by hundredths. A date column that was being thrown away can move
    # it by tenths. This round tries the derived inputs the other way round from how the run
    # started -- all of them if none were on, none if some were -- with the model and
    # settings that lead, and keeps whichever the folds prefer. "The derived inputs did not
    # help" and "they did" are both findings, and either way the user learns which.
    if not stopped_because and out_of_time():
        stopped_because = "time_limit"
    if stopped_because == "time_limit" and "derived_features" not in skipped_for_time:
        skip_for_time(
            "derived_features", "Derived inputs",
            "Looked for columns that could be turned into inputs -- dates, free text, blanks, "
            "values that repeat.",
        )
    if not stopped_because:
        starting("derived_features", "Trying inputs built from your columns.")
        alt_recipes, alt_frame = frames["alternative"]
        alt_features, alt_used = inputs_for("alternative")
        if alt_recipes == recipes or not proposed:
            emit(
                Round(
                    name="derived_features",
                    label="Derived inputs",
                    what_it_tried="Looked for columns that could be turned into inputs -- "
                    "dates, free text, blanks, values that repeat.",
                    accepted=False,
                    outcome="Nothing in this data can be derived from that isn't already in "
                    "use.",
                )
            )
        else:
            X_train_alt = alt_frame.loc[train_index, alt_used]
            trying_on = bool(alt_recipes)
            what = (
                f"Added {len(alt_recipes)} derived input"
                f"{'s' if len(alt_recipes) != 1 else ''} "
                f"({', '.join(r.id.split(':', 1)[1] for r in alt_recipes)}) and re-scored "
                f"the leading model on the training folds."
                if trying_on
                else f"Removed the {len(recipes)} derived input"
                f"{'s' if len(recipes) != 1 else ''} the run started with and re-scored the "
                f"leading model on the training folds, to check they were earning their place."
            )
            scored = score_config(
                build_for(best_name, best_params, alt_features),
                X_train_alt,
                y_train,
                task,
                groups=train_groups, ordered=ordered,
            )
            if scored is None:
                emit(
                    Round(
                        name="derived_features",
                        label="Derived inputs",
                        what_it_tried=what,
                        accepted=False,
                        outcome="The model could not be scored with them, so they were left as "
                        "they were.",
                    )
                )
            elif _is_real_gain(scored.score, best_score, scored.std):
                gain = round(scored.score - best_score, 4)
                best_score, best_std = scored.score, scored.std
                recipes, features, used = alt_recipes, alt_features, alt_used
                X_train, X_test = alt_frame.loc[train_index, used], alt_frame.loc[test_index, used]
                frames["current"] = (alt_recipes, alt_frame)  # what later rounds add columns to
                emit(
                    Round(
                        name="derived_features",
                        label="Derived inputs",
                        what_it_tried=what,
                        score=scored.score,
                        std=scored.std,
                        gain=gain,
                        accepted=True,
                        outcome=(
                            f"The derived inputs help: {metric_label} up {gain:+.3f}. They stay."
                            if trying_on
                            else f"The model does better without them: {metric_label} up "
                            f"{gain:+.3f}. They were dropped."
                        ),
                        detail={"recipes": [r.id for r in alt_recipes]},
                    )
                )
            else:
                emit(
                    Round(
                        name="derived_features",
                        label="Derived inputs",
                        what_it_tried=what,
                        score=scored.score,
                        std=scored.std,
                        gain=round(scored.score - best_score, 4),
                        accepted=False,
                        outcome=(
                            "They did not move the score by more than the folds disagree with "
                            "each other, so the inputs were left as they were."
                        ),
                        detail={"recipes": [r.id for r in alt_recipes]},
                    )
                )

        if _goal_met(best_score, goal):
            stopped_because = "goal_reached"

    # --- Round 5: which of the inputs earn their place ----------------------------------
    #
    # Every usable column is an input, and a weak or noisy one adds variance, most of all on a small file.
    # Rank the inputs by how much shuffling each hurts the incumbent on validation folds, and see whether
    # the model does better without the ones that did not matter. It is kept only if it clears the same
    # gain rule as every other round; correlated inputs share their importance, so a group is removed
    # and re-scored, and if that fails a smaller one is tried.
    removed_columns: list[dict] = []
    if not stopped_because and out_of_time():
        stopped_because = "time_limit"
    if (
        stopped_because == "time_limit"
        and "feature_selection" not in skipped_for_time
        and len(used) >= MIN_INPUTS_TO_SELECT  # a round that was never going to run was not skipped
    ):
        skip_for_time(
            "feature_selection", "Inputs that earn their place",
            "Checked whether the model does better without the inputs that matter least.",
        )
    if not stopped_because and len(used) >= MIN_INPUTS_TO_SELECT:
        starting("feature_selection", "Checking which inputs earn their place.")
        importance = importance_of(
            build_for(best_name, best_params, features), X_train, y_train, task,
            groups=train_groups, ordered=ordered, columns=used,
        )
        ranked_inputs = sorted(importance, key=lambda c: (importance[c][0], c))
        useless = [c for c in ranked_inputs if importance[c][0] <= 0.0]
        # Never every input, and never more than half: removing most of the columns is a different model.
        useless = useless[: max(len(used) // 2, 0)]
        what = (
            f"Shuffled each of the {len(used)} inputs on validation folds to see which ones the model "
            f"does not miss, and re-scored it without them."
        )
        if not useless:
            emit(
                Round(
                    name="feature_selection",
                    label="Inputs that earn their place",
                    what_it_tried=what,
                    accepted=False,
                    outcome="Every input is missed when it is shuffled, so none was removed.",
                    detail={"importance": {c: round(importance[c][0], 4) for c in ranked_inputs[:10]}},
                )
            )
        else:
            tried_sets = [useless]
            if len(useless) >= 3:
                tried_sets.append(useless[: -(-len(useless) // 3)])  # the lowest-ranked third
            winner = None  # (group, scored) once a smaller set clears the gain rule
            last_scored = None
            shown: dict = {}
            for group in tried_sets:
                reduced = _without(features, set(group))
                reduced_used = [c for c in used if c not in set(group)]
                if not reduced_used:
                    continue
                if out_of_time():
                    cut_short = True
                    break
                last_scored = score_config(
                    build_for(best_name, best_params, reduced),
                    X_train[reduced_used], y_train, task, groups=train_groups, ordered=ordered,
                )
                shown = {
                    "tried": len(group),
                    "removed": [
                        {"column": c, "importance": round(importance[c][0], 4), "spread": round(importance[c][1], 4)}
                        for c in group
                    ],
                }
                if last_scored is not None and _is_real_gain(last_scored.score, best_score, last_scored.std):
                    winner = (group, last_scored, reduced, reduced_used)
                    break

            if winner is not None:
                group, scored, reduced, reduced_used = winner
                gain = round(scored.score - best_score, 4)
                best_score, best_std = scored.score, scored.std
                features, used = reduced, reduced_used
                X_train, X_test = X_train[used], X_test[used]
                removed_columns = shown["removed"]
                emit(
                    Round(
                        name="feature_selection",
                        label="Inputs that earn their place",
                        what_it_tried=what,
                        score=scored.score,
                        std=scored.std,
                        gain=gain,
                        accepted=True,
                        outcome=f"The model does better without {len(group)} of its inputs: "
                        f"{metric_label} up {gain:+.3f}. They were removed.",
                        detail=shown,
                    )
                )
            else:
                emit(
                    Round(
                        name="feature_selection",
                        label="Inputs that earn their place",
                        what_it_tried=what,
                        score=None if last_scored is None else last_scored.score,
                        std=None if last_scored is None else last_scored.std,
                        gain=None if last_scored is None else round(last_scored.score - best_score, 4),
                        accepted=False,
                        outcome=(
                            "The time limit came before the model could be re-scored without them, so all "
                            "of the inputs stayed."
                            if last_scored is None
                            else "The model did not do better by more than the folds disagree without the "
                            "inputs that matter least, so all of them stayed."
                        ),
                        detail=shown,
                    )
                )
        if _goal_met(best_score, goal):
            stopped_because = "goal_reached"

    # --- Round 6: combinations of the inputs that matter ---------------------------------
    #
    # A linear model cannot form a ratio or a product, and some relationships only show as one (spend per
    # visit, tickets per month of tenure). Take the numeric inputs that matter most, try their products and a
    # few ratios, and keep a combination only if it clears the gain rule *and* holds on fresh folds -- the
    # best of several tried is the luckiest of them, which is the winner's curse again. Each candidate is a
    # row-wise recipe, so it is computed before the split and cannot leak.
    kept_interactions: list[dict] = []
    tried_interactions: list[dict] = []
    if not stopped_because and out_of_time():
        stopped_because = "time_limit"
    if (
        stopped_because == "time_limit"
        and "interactions" not in skipped_for_time
        and len([c for c in used if c in features["numeric"]]) >= 2
    ):
        skip_for_time(
            "interactions", "Combinations of inputs",
            "Tried products and ratios of the numeric inputs that matter most.",
        )
    if not stopped_because and len([c for c in used if c in features["numeric"]]) >= 2:
        starting("interactions", "Trying combinations of the inputs that matter most.")
        importance = importance_of(
            build_for(best_name, best_params, features), X_train, y_train, task,
            groups=train_groups, ordered=ordered, columns=[c for c in used if c in features["numeric"]],
        )
        top = [c for c in sorted(importance, key=lambda c: -importance[c][0])
               if importance[c][0] > 0][:INTERACTION_TOP_INPUTS]
        base_frame = frames["current"][1]
        candidates: list[feature_recipes.Recipe] = []
        for i, a in enumerate(top):
            for b in top[i + 1:]:
                found = feature_recipes.interaction_recipe(base_frame, "product", a, b)
                if found is not None:
                    candidates.append(found)
        for a, b in [(top[0], top[1]), (top[1], top[0])] if len(top) >= 2 else []:
            found = feature_recipes.interaction_recipe(base_frame, "ratio", a, b)
            if found is not None:
                candidates.append(found)
        candidates = candidates[:MAX_INTERACTIONS_TRIED]

        what = (
            f"Tried {len(candidates)} products and ratios of the {len(top)} numeric inputs that matter most, "
            f"each scored on the training folds."
            if len(top) >= 2
            else "Looked for numeric inputs that matter enough to combine."
        )

        def with_interaction(recipe):
            frame = feature_recipes.apply(frames["current"][1], [recipe])
            added = {**features, "numeric": [*features["numeric"], *recipe.outputs]}
            columns = [*used, *recipe.outputs]
            return frame, added, columns

        def score_with(recipe, *, fresh_folds: bool = False):
            frame, added, columns = with_interaction(recipe)
            extra = dict(confirmation=True) if fresh_folds else {}
            scored = score_config(
                build_for(best_name, best_params, added), frame.loc[train_index, columns], y_train, task,
                groups=train_groups, ordered=ordered, **extra,
            )
            return scored, frame, added, columns

        # Each candidate is scored once, against the incumbent as it stands now.
        ranked: list[tuple[tuning.SearchResult, feature_recipes.Recipe]] = []
        for recipe in candidates:
            scored, *_ = score_with(recipe)
            tried_interactions.append({"id": recipe.id, "score": None if scored is None else scored.score})
            if scored is not None:
                ranked.append((scored, recipe))
            if out_of_time():
                cut_short = True
                break
        ranked.sort(key=lambda pair: -pair[0].score)

        while ranked and len(kept_interactions) < MAX_INTERACTIONS_KEPT and not out_of_time():
            scored, recipe = ranked.pop(0)
            if kept_interactions:
                # The incumbent has changed since this was scored: score it on top of what was kept.
                scored, *_ = score_with(recipe)
                tried_interactions.append({"id": recipe.id, "score": None if scored is None else scored.score})
                if scored is None:
                    continue
            if not _is_real_gain(scored.score, best_score, scored.std):
                break
            # The best of those tried: confirm it, against the incumbent, on folds that did not pick it.
            winner, frame, added, columns = score_with(recipe, fresh_folds=True)
            incumbent = score_config(
                build_for(best_name, best_params, features), X_train, y_train, task,
                groups=train_groups, ordered=ordered, confirmation=True,
            )
            if (
                winner is None or incumbent is None
                or not _is_real_gain(winner.score, incumbent.score, winner.std)
                or not clear_of_chance(winner.score, winner.std, metric, y_train)
            ):
                break  # the luckiest of those tried did not hold on fresh folds, or only reached chance
            gain = round(scored.score - best_score, 4)
            best_score, best_std = scored.score, scored.std
            recipes = [*recipes, recipe]
            frames["current"] = (recipes, frame)
            features, used = added, columns
            X_train, X_test = frame.loc[train_index, used], frame.loc[test_index, used]
            kept_interactions.append(
                {"id": recipe.id, "output": recipe.outputs[0], "gain": gain,
                 "search_score": scored.score, "confirmation_score": winner.score,
                 "incumbent_confirmation_score": incumbent.score}
            )
            # Only the next two best get a second look, on top of what was just kept.
            ranked = ranked[:2]

        if kept_interactions:
            emit(
                Round(
                    name="interactions",
                    label="Combinations of inputs",
                    what_it_tried=what,
                    score=best_score,
                    std=best_std,
                    gain=round(sum(k["gain"] for k in kept_interactions), 4),
                    accepted=True,
                    outcome=f"{len(kept_interactions)} combination{'s' if len(kept_interactions) != 1 else ''} "
                    f"help: {', '.join(k['output'] for k in kept_interactions)}. They stay.",
                    detail={"tried": len(tried_interactions), "kept": kept_interactions},
                )
            )
        else:
            emit(
                Round(
                    name="interactions",
                    label="Combinations of inputs",
                    what_it_tried=what,
                    accepted=False,
                    outcome=(
                        "Fewer than two numeric inputs matter enough to combine."
                        if len(top) < 2
                        else "None of them helped by more than the folds disagree with each other, so the inputs "
                        "were left as they were."
                    ),
                    detail={"tried": len(tried_interactions)},
                )
            )
        if _goal_met(best_score, goal):
            stopped_because = "goal_reached"

    # --- Round 7: the strongest models averaged ------------------------------------------
    #
    # Two models that make different mistakes can average to fewer than either. Average the incumbent with the
    # strongest runner-up, each with the settings its search found, and keep the average only if it beats the
    # incumbent by more than the folds disagree *and* holds on fresh folds, clear of chance. It is one candidate,
    # but a model built from the winners is still a winner's score, so it is checked the same way.
    ensemble_name = modeling.ENSEMBLE_FOR_TASK.get(task)
    runner_ups = [n for n in sorted(family_scores, key=lambda n: -family_scores[n]) if n != best_name]
    if not stopped_because and out_of_time():
        stopped_because = "time_limit"
    if (
        stopped_because == "time_limit"
        and "ensemble" not in skipped_for_time
        and ensemble_name and runner_ups and best_name != ensemble_name
    ):
        skip_for_time("ensemble", "The strongest models, averaged", "Averaged the two strongest models.")
    if not stopped_because and ensemble_name and runner_ups and best_name != ensemble_name:
        starting("ensemble", "Averaging the two strongest models.")
        partner = runner_ups[0]

        def readable(params: dict) -> dict:
            return {k.replace("model__", ""): v for k, v in (params or {}).items()}

        members = [
            {"model": best_name, "params": readable(best_params)},
            {"model": partner, "params": readable(searched_settings.get(partner, {}))},
        ]
        settings = {"members": members}
        label = f"{modeling.MODELS[best_name].label} and {modeling.MODELS[partner].label.lower()}"
        what = f"Averaged {label}, each with the settings its search found, and scored it on the training folds."
        scored = score_config(
            build_for(ensemble_name, settings), X_train, y_train, task, groups=train_groups, ordered=ordered,
        )
        check: dict | None = None
        accepted = scored is not None and _is_real_gain(scored.score, best_score, scored.std)
        if accepted:
            if out_of_time():
                cut_short = True
                accepted = False
                check = {"status": "skipped_for_time"}
            else:
                fresh = dict(groups=train_groups, ordered=ordered, confirmation=True)
                averaged = score_config(build_for(ensemble_name, settings), X_train, y_train, task, **fresh)
                single = score_config(build_for(best_name, best_params), X_train, y_train, task, **fresh)
                if averaged is None or single is None:
                    accepted, check = False, {"status": "unavailable"}
                else:
                    beats = _is_real_gain(averaged.score, single.score, averaged.std)
                    clear = clear_of_chance(averaged.score, averaged.std, metric, y_train)
                    accepted = beats and clear
                    check = {
                        "status": "confirmed" if accepted else "at_chance" if beats else "not_confirmed",
                        "search_score": scored.score,
                        "confirmation_score": averaged.score,
                        "confirmation_std": averaged.std,
                        "incumbent_confirmation_score": single.score,
                        "chance": chance_level(metric, y_train),
                    }
        detail = {"members": members, **({"confirmation": check} if check else {})}
        if accepted:
            gain = round(scored.score - best_score, 4)
            best_name, best_params = ensemble_name, settings
            best_score, best_std = scored.score, scored.std
            emit(
                Round(
                    name="ensemble",
                    label="The strongest models, averaged",
                    what_it_tried=what,
                    score=scored.score,
                    std=scored.std,
                    gain=gain,
                    accepted=True,
                    outcome=f"The average of {label} beats either alone: {metric_label} up {gain:+.3f}, and it "
                    f"held on fresh folds. It is the model now.",
                    detail=detail,
                )
            )
        else:
            emit(
                Round(
                    name="ensemble",
                    label="The strongest models, averaged",
                    what_it_tried=what,
                    score=None if scored is None else scored.score,
                    std=None if scored is None else scored.std,
                    gain=None if scored is None else round(scored.score - best_score, 4),
                    accepted=False,
                    outcome=(
                        f"The average could not be scored, so {modeling.MODELS[best_name].label.lower()} stayed."
                        if scored is None
                        else not_confirmed_outcome(
                            check, "The average looked better", f"{modeling.MODELS[best_name].label.lower()} stayed"
                        )
                        if check is not None
                        else f"The average did not beat {modeling.MODELS[best_name].label.lower()} by more than the "
                        f"folds disagree, so the single model stayed."
                    ),
                    detail=detail,
                )
            )
        if _goal_met(best_score, goal):
            stopped_because = "goal_reached"

    # --- Round 8: training with the stated costs ------------------------------------------
    #
    # The user said a missed case costs N times a false alarm. Until now that only moved the cut-off, after a
    # model fitted with balanced class weights -- a different trade-off -- had been chosen. Train the incumbent
    # again with each row weighted by the cost of getting it wrong, and compare the expected cost per row at the
    # best cut-off, fold by fold on the same folds. Kept only when it is lower by more than the folds disagree.
    two_answers = task == "classification" and pd.Series(y_train).nunique() == 2
    if stated_costs and two_answers and not stopped_because and out_of_time():
        stopped_because = "time_limit"
    if stopped_because == "time_limit" and stated_costs and two_answers and "cost_weighting" not in skipped_for_time:
        skip_for_time("cost_weighting", "Training with your costs", "Trained with the stated costs as row weights.")
    if stated_costs and two_answers and not stopped_because:
        starting("cost_weighting", "Training with the costs you stated.")
        positive = str(
            stated_costs.get("positive_label") or sorted(pd.Series(y_train).astype(str).unique())[-1]
        )
        weights = {
            "positive_label": positive,
            "cost_of_missing": float(stated_costs.get("cost_of_missing") or 1.0),
            "cost_of_false_alarm": float(stated_costs.get("cost_of_false_alarm") or 1.0),
        }
        costs = dict(
            positive_label=positive, cost_of_missing=weights["cost_of_missing"],
            cost_of_false_alarm=weights["cost_of_false_alarm"], groups=train_groups, ordered=ordered,
        )
        plain = expected_costs(build_for(best_name, best_params), X_train, y_train, **costs)
        weighted = expected_costs(build_for(best_name, best_params, weighting=weights), X_train, y_train, **costs)
        what = (
            f"Trained {modeling.MODELS[best_name].label.lower()} again with each row weighted by the cost you "
            f"stated for getting it wrong (a missed case {weights['cost_of_missing']:g}, a false alarm "
            f"{weights['cost_of_false_alarm']:g}), and compared the expected cost per row at the best cut-off on "
            f"the same folds."
        )
        gap = None
        if plain and weighted and len(plain) == len(weighted):
            from app.services import uncertainty

            folds = len(plain)
            # Lower cost is better, so compare the negatives: a positive difference means weighting helped.
            gap = uncertainty.fold_difference(
                [-c for c in weighted], [-c for c in plain],
                test_train_ratio=1 / (folds - 1) if folds > 1 else None,
            )
        detail = {
            "weights": weights,
            "expected_cost": {
                "without": None if not plain else round(float(np.mean(plain)), 4),
                "with": None if not weighted else round(float(np.mean(weighted)), 4),
            },
            "paired_difference": gap,
        }
        helped = gap is not None and gap["low"] > 0
        if helped:
            training_weights = weights
        emit(
            Round(
                name="cost_weighting",
                label="Training with your costs",
                what_it_tried=what,
                accepted=helped,
                outcome=(
                    f"Training with your costs lowers the expected cost per row from "
                    f"{detail['expected_cost']['without']:.3f} to {detail['expected_cost']['with']:.3f}, by more than "
                    f"the folds disagree, so the model is trained with them."
                    if helped
                    else "The folds could not be compared, so the model is trained as it was."
                    if gap is None
                    else f"Training with your costs changed the expected cost per row from "
                    f"{detail['expected_cost']['without']:.3f} to {detail['expected_cost']['with']:.3f}, not by more "
                    f"than the folds disagree, so the model is trained as it was; the cut-off still uses your costs."
                ),
                detail=detail,
            )
        )

    if not stopped_because:
        stopped_because = (
            "goal_reached" if _goal_met(best_score, goal)
            else "time_limit" if cut_short
            else "goal_out_of_reach" if goal is not None
            else "no_further_gains"
        )

    # --- The one look at the test set ----------------------------------------------------
    final_pipeline = build_for(best_name, best_params)()
    final_pipeline.fit(X_train, y_train)
    test_metrics = modeling.evaluate(final_pipeline, X_test, y_test, task)

    return {
        "rounds": [r.as_dict() for r in rounds],
        "metric": metric,
        "goal": goal,
        "stopped_because": stopped_because,
        "time_limit_seconds": limit,
        "skipped_for_time": skipped_for_time,
        "search_strategy": search_strategy,
        "search_strategies": sorted(strategies_used),
        "resource_limits": resource_limits,
        "best": {
            "model": best_name,
            "label": modeling.MODELS[best_name].label,
            "params": {
                **{k.replace("model__", ""): v for k, v in best_params.items()},
                # Rebuilt with the weighting when the last round kept it.
                **({"cost_weighting": training_weights} if training_weights else {}),
            },
            "cv_score": best_score,
            "cv_std": best_std,
            # The derived inputs the winner uses, as ids, so the caller rebuilds it the same way.
            "feature_recipes": [r.id for r in recipes],
            # Inputs the fifth round removed because the model did better without them, with how little
            # each mattered; the caller withholds them from the rebuild.
            "removed_columns": removed_columns,
        },
        "test_metrics": test_metrics,
        "pipeline": final_pipeline,
        "features": features,
        "split": {
            "strategy": split_strategy,
            "time_column": time_column if split_strategy == "time" else None,
            "group_column": group_column if split_strategy == "group" else None,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
    }
