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

import pandas as pd

from app.services import features as feature_recipes, modeling, time_folds, tuning

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

# How far behind the leader a model can be and still be worth tuning. Tuning moves a score by
# a few hundredths on data this size; a family that lost the first round by more than this is
# not going to close the gap, and searching it anyway is the single most expensive thing the
# loop can do for the least return.
WORTH_TUNING_WITHIN = 0.05


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
    search_strategy: str = tuning.DEFAULT_STRATEGY,
) -> dict:
    """Runs rounds of increasingly expensive strategies until one of the stopping conditions.

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
    """
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

    def time_tail(found) -> str:
        if not found.stopped_for_time:
            return ""
        return f" The time limit stopped it after {found.tried} of {found.planned} settings."

    def build_for(
        name: str, params: dict | None = None, with_features: dict | None = None
    ) -> Callable[[], object]:
        feats = with_features or features

        def factory():
            pipeline = modeling.build_pipeline(
                name,
                feats["numeric"],
                feats["categorical"],
                frequent=feats["frequent"],
                clip=feats.get("clipped"),
            )
            if params:
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
        result = tuning.score_config(
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
        found = tuning.search(
            build_for(best_name), best_name, X_train, y_train, task,
            budget=budget, groups=train_groups, ordered=ordered, deadline=deadline,
            strategy=search_strategy,
        )
        cut_short = cut_short or bool(found and found.stopped_for_time)
        if found is None:
            emit(
                Round(
                    name="hyperparameters",
                    label="Model settings",
                    what_it_tried="Searched this model's settings.",
                    outcome="This model has no settings worth searching.",
                )
            )
        elif _is_real_gain(found.score, best_score, found.std):
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
                    outcome=f"Better settings found: {metric_label} up {gain:+.3f}.",
                    detail={"params": found.readable_params, **_reached(found)},
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
                    outcome="No setting beat the defaults by more than the folds disagree "
                    "with each other, so the defaults were kept.",
                    detail={"params": found.readable_params, **_reached(found)},
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
            found = tuning.search(
                build_for(name), name, X_train, y_train, task,
                budget=budget, groups=train_groups, ordered=ordered, deadline=deadline,
                strategy=search_strategy,
            )
            cut_short = cut_short or bool(found and found.stopped_for_time)
            if found and _is_real_gain(found.score, best_score, found.std):
                if improved_challenger is None or found.score > improved_challenger[1].score:
                    improved_challenger = (name, found)

        cut_short = cut_short or bool(untried_for_time)
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
                    f"{metric_label} up {gain:+.3f}." + time_note,
                    detail={
                        "model": name, "params": found.readable_params,
                        **({"untried_for_time": untried_for_time} if untried_for_time else {}),
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
                    outcome="None of them caught up once tuned." + time_note,
                    detail={"untried_for_time": untried_for_time} if untried_for_time else {},
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
            scored = tuning.score_config(
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
        "best": {
            "model": best_name,
            "label": modeling.MODELS[best_name].label,
            "params": {k.replace("model__", ""): v for k, v in best_params.items()},
            "cv_score": best_score,
            "cv_std": best_std,
            # The derived inputs the winner uses, as ids, so the caller rebuilds it the same way.
            "feature_recipes": [r.id for r in recipes],
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
