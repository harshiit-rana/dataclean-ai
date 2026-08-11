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
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from app.services import modeling, tuning

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
) -> dict:
    """Runs rounds of increasingly expensive strategies until one of the stopping conditions.

    `goal` is a target for the same metric the rounds are judged on -- AUC for a two-class
    target, accuracy for more, R² for regression. None means "get it as good as you can",
    which stops when the rounds stop paying for themselves.

    `on_progress` is called after each round so the caller can stream them; the loop is slow
    enough that watching it work matters.
    """
    if target_column not in df.columns:
        raise modeling.ModelingError(f"'{target_column}' is not in the data.")

    # Mirrors train_candidates: the column being grouped on cannot also be an input, or the
    # loop would be searching for settings that exploit the very memorisation the grouped
    # split exists to prevent.
    withheld = list(excluded_columns or [])
    if split_strategy == "group" and group_column and group_column not in withheld:
        withheld.append(group_column)

    features = modeling.classify_features(df, target_column, withheld)
    if not features["numeric"] and not features["categorical"]:
        raise modeling.ModelingError(
            "There are no usable columns left to predict from, so there is nothing to improve."
        )

    # Split once. Nothing below this line is allowed to look at X_test or y_test.
    X_train, X_test, y_train, y_test = modeling.split_data(
        df,
        target_column,
        task,
        strategy=split_strategy,
        time_column=time_column,
        group_column=group_column,
    )
    used = features["numeric"] + features["categorical"]
    # Captured before the frame is narrowed to the model's inputs, and aligned to the training
    # rows -- every round below compares configurations on these folds.
    train_groups = (
        df.loc[list(X_train.index), group_column]
        if split_strategy == "group" and group_column
        else None
    )
    X_train, X_test = X_train[used], X_test[used]

    if tuning.cv_splitter(task, y_train, groups=train_groups) is None:
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

    def build_for(name: str, params: dict | None = None) -> Callable[[], object]:
        def factory():
            pipeline = modeling.build_pipeline(name, features["numeric"], features["categorical"])
            if params:
                pipeline.set_params(**params)
            return pipeline

        return factory

    # --- Round 1: which model family, judged on folds rather than on the test set ---------
    #
    # train_candidates picks its winner by test-set score, which is fine for three models
    # looked at once. A loop that keeps going cannot afford even that, so family selection
    # moves onto cross-validation here.
    family_scores: dict[str, float] = {}
    best_name, best_score, best_std = "", float("-inf"), 0.0
    for name in candidates:
        result = tuning.score_config(
            build_for(name), X_train, y_train, task, groups=train_groups
        )
        if result is None:
            continue
        family_scores[name] = result.score
        if result.score > best_score:
            best_name, best_score, best_std = name, result.score, result.std

    if not best_name:
        raise modeling.ModelingError("None of the models could be scored on this data.")

    best_params: dict = {}
    emit(
        Round(
            name="baseline",
            label="Starting point",
            what_it_tried=f"Cross-validated {len(family_scores)} model "
            f"{'family' if len(family_scores) == 1 else 'families'} on the training rows and "
            f"kept the strongest.",
            score=best_score,
            std=best_std,
            gain=None,
            accepted=True,
            outcome=f"{modeling.MODELS[best_name].label} leads at {best_score:.3f} {metric_label}.",
            detail={"scores": family_scores, "chosen": best_name},
        )
    )

    stopped_because = ""
    if _goal_met(best_score, goal):
        stopped_because = "goal_reached"

    # --- Round 2: settings for the leading family ----------------------------------------
    if not stopped_because:
        found = tuning.search(
            build_for(best_name), best_name, X_train, y_train, task,
            budget=budget, groups=train_groups,
        )
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
                    f"scored on the training folds.",
                    score=found.score,
                    std=found.std,
                    gain=gain,
                    accepted=True,
                    outcome=f"Better settings found: {metric_label} up {gain:+.3f}.",
                    detail={"params": found.readable_params},
                )
            )
        else:
            emit(
                Round(
                    name="hyperparameters",
                    label="Model settings",
                    what_it_tried=f"Tried {found.fits} fits across this model's settings, "
                    f"scored on the training folds.",
                    score=found.score,
                    std=found.std,
                    gain=round(found.score - best_score, 4),
                    accepted=False,
                    outcome="No setting beat the defaults by more than the folds disagree "
                    "with each other, so the defaults were kept.",
                    detail={"params": found.readable_params},
                )
            )

        if _goal_met(best_score, goal):
            stopped_because = "goal_reached"

    # --- Round 3: tune the runners-up too, in case the ranking changes once they're tuned -
    #
    # An untuned gradient booster losing to a tuned logistic regression says nothing about
    # which is better; default settings favour some families more than others.
    if not stopped_because:
        challengers = [
            name
            for name, score in family_scores.items()
            if name != best_name and score >= best_score - WORTH_TUNING_WITHIN
        ]
        skipped = len(family_scores) - len(challengers) - 1
        improved_challenger = None
        for name in challengers:
            found = tuning.search(
                build_for(name), name, X_train, y_train, task,
                budget=budget, groups=train_groups,
            )
            if found and _is_real_gain(found.score, best_score, found.std):
                if improved_challenger is None or found.score > improved_challenger[1].score:
                    improved_challenger = (name, found)

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
                    f"{metric_label} up {gain:+.3f}.",
                    detail={"model": name, "params": found.readable_params},
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
                    outcome="None of them caught up once tuned.",
                )
            )

        if _goal_met(best_score, goal):
            stopped_because = "goal_reached"

    if not stopped_because:
        stopped_because = "goal_reached" if _goal_met(best_score, goal) else (
            "goal_out_of_reach" if goal is not None else "no_further_gains"
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
        "best": {
            "model": best_name,
            "label": modeling.MODELS[best_name].label,
            "params": {k.replace("model__", ""): v for k, v in best_params.items()},
            "cv_score": best_score,
            "cv_std": best_std,
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
