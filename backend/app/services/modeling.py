"""Model training from a fixed registry. The counterpart to services/operations.py.

Same rule as the cleaning operations, for the same reasons: the app **selects** a model
family and assembles a scikit-learn `Pipeline` from typed parts. It never generates or
executes model code, so there is no sandbox to harden and no way for a bad LLM response to
turn into arbitrary execution. The LLM's only job here is to explain the result in English.

The part that matters most is where the preprocessing lives. Filling a column with its mean
across the whole file and *then* splitting into train and test means the test rows helped
produce the values the model trained on, and every score comes out flattering. So imputation,
scaling and encoding are steps **inside** the pipeline: `fit` sees only the training fold,
and the test set is transformed with statistics it had no part in computing. This is the
reason cleaning deliberately leaves those operations alone (see `ml_readiness`'s
DATASET_STATISTIC_OPERATIONS) rather than baking them into cleaned.csv.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.services import diagnostics, features as feature_recipes, operating_point

logger = logging.getLogger(__name__)

# Below this there isn't enough data to hold out a test set that means anything.
MIN_ROWS_TO_TRAIN = 50
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42  # fixed so a rerun of the same session reproduces the same numbers

# One-hot encoding a column with hundreds of categories produces hundreds of near-empty
# columns and a model that memorizes rows. Anything above the limit is dropped from the
# features with a note rather than silently exploding the matrix.
#
# The limit scales with the row count, because "too many categories" is a statement about
# how many rows each category gets, not an absolute number. A fixed cap of 30 threw away
# Department_Region (36 values) on an 809-row file -- roughly 22 rows per category, plainly
# learnable, and the most informative column in the data. The floor keeps a small file from
# losing every categorical column; the ceiling keeps a large one from one-hotting free text.
ROWS_PER_CATEGORY = 10
MIN_CATEGORIES_TO_ENCODE = 10
MAX_CATEGORIES_TO_ENCODE = 50


def category_limit(rows: int) -> int:
    """How many distinct values a categorical column may have and still be worth encoding."""
    return max(MIN_CATEGORIES_TO_ENCODE, min(MAX_CATEGORIES_TO_ENCODE, rows // ROWS_PER_CATEGORY))


class ModelingError(Exception):
    """Raised when the data can't support training. Carries a message written for the user."""


@dataclass(frozen=True)
class ModelSpec:
    """How to construct one model, described rather than coded.

    Held as (module, class, params) instead of a callable so that the model the app fits and
    the model the exported Python script fits are generated from the same three values. A
    hand-written code template beside a hand-written constructor is a drift waiting to
    happen: the export would keep claiming `n_estimators=200` long after the real one
    changed, and nobody would notice until someone tried to reproduce a result.
    """

    name: str
    label: str
    description: str
    task: str  # "classification" | "regression"
    module: str
    class_name: str
    params: dict
    # True when the model's coefficients/importances are directly readable. Used to decide
    # whether a feature-importance review is offered for this model.
    explains_features: bool = True

    def settings(self, overrides: dict | None = None) -> dict:
        """Defaults, with anything the tuner found layered on top.

        One place decides this, because the app and the exported script must build the same
        estimator -- a tuned model whose script quietly reverts to the defaults would report
        a score the script cannot reproduce."""
        return {**self.params, **(overrides or {})}

    def build(self, overrides: dict | None = None) -> Any:
        import importlib

        cls = getattr(importlib.import_module(self.module), self.class_name)
        return cls(**self.settings(overrides))

    @property
    def import_line(self) -> str:
        return f"from {self.module} import {self.class_name}"

    def constructor_source(self, overrides: dict | None = None) -> str:
        arguments = ", ".join(f"{k}={v!r}" for k, v in self.settings(overrides).items())
        return f"{self.class_name}({arguments})"


MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in [
        ModelSpec(
            "logistic_regression",
            "Logistic regression",
            "Simple and fully explainable. A good baseline: if something more complex can't "
            "beat it, the extra complexity isn't earning its keep.",
            "classification",
            "sklearn.linear_model",
            "LogisticRegression",
            # balanced weights: on a lopsided target an unweighted fit learns to always
            # answer with the common class, which scores well and is useless.
            {"max_iter": 1000, "class_weight": "balanced", "random_state": RANDOM_STATE},
        ),
        ModelSpec(
            "random_forest_classifier",
            "Random forest",
            "Handles messy, non-linear relationships without much tuning. Usually strong "
            "out of the box.",
            "classification",
            "sklearn.ensemble",
            "RandomForestClassifier",
            {
                "n_estimators": 200,
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
            },
        ),
        ModelSpec(
            "gradient_boosting_classifier",
            "Gradient boosting",
            "Typically the most accurate of the three on tabular data, and the slowest to "
            "train.",
            "classification",
            "sklearn.ensemble",
            "HistGradientBoostingClassifier",
            {"random_state": RANDOM_STATE},
            explains_features=False,
        ),
        ModelSpec(
            "linear_regression",
            "Ridge regression",
            "Simple and fully explainable. A good baseline to judge the others against.",
            "regression",
            "sklearn.linear_model",
            "Ridge",
            {"random_state": RANDOM_STATE},
        ),
        ModelSpec(
            "random_forest_regressor",
            "Random forest",
            "Handles non-linear relationships without much tuning.",
            "regression",
            "sklearn.ensemble",
            "RandomForestRegressor",
            {"n_estimators": 200, "random_state": RANDOM_STATE, "n_jobs": -1},
        ),
        ModelSpec(
            "gradient_boosting_regressor",
            "Gradient boosting",
            "Typically the most accurate on tabular data, and the slowest to train.",
            "regression",
            "sklearn.ensemble",
            "HistGradientBoostingRegressor",
            {"random_state": RANDOM_STATE},
            explains_features=False,
        ),
    ]
}


def models_for_task(task: str) -> list[ModelSpec]:
    return [spec for spec in MODELS.values() if spec.task == task]


# --- feature selection ---------------------------------------------------------------------


def classify_features(
    df: pd.DataFrame,
    target_column: str,
    excluded: list[str] | None = None,
    recipes: list[feature_recipes.Recipe] | None = None,
    clip: list[str] | None = None,
) -> dict:
    """Splits the columns into what the model can actually use.

    Returns numeric/categorical/frequent feature lists plus the columns being left out and
    why, so the UI can show its reasoning instead of silently dropping things.

    The reason is the product. A column of radiology reports used to be dropped as an
    "identifier" because nearly every value was distinct -- the right action for the wrong
    reason, and the wrong reason is the defect. Free text and dates are now named as what
    they are, with what to do about them, whether or not a recipe was chosen.

    `recipes` are the derived inputs in force (see features.py). A date or text column that
    one of them replaces is dropped with that as the reason; a `frequent_categories` column
    goes into its own bucket for the pipeline to encode.

    `clip` names numeric inputs to clip to training-learned limits (see clipping.py). Only
    columns that are numeric inputs are kept -- a name that is not one is ignored rather than
    failing the run, the same way a stale recipe id is.
    """
    excluded = set(excluded or [])
    recipes = list(recipes or [])
    replaced = feature_recipes.derived_from(recipes)
    frequent_set = set(feature_recipes.frequent_columns(recipes))
    derived_outputs = {o for r in recipes for o in r.outputs}

    numeric: list[str] = []
    categorical: list[str] = []
    frequent: list[str] = []
    dropped: list[dict] = []
    limit = category_limit(len(df))

    for column in df.columns:
        if column == target_column:
            continue
        if column in excluded:
            dropped.append({"column": column, "reason": "You excluded this column."})
            continue

        series = df[column]
        if series.notna().sum() == 0:
            dropped.append({"column": column, "reason": "Every value is empty."})
            continue
        if series.nunique(dropna=True) <= 1 and column not in derived_outputs:
            dropped.append(
                {"column": column, "reason": "Every row has the same value, so it explains nothing."}
            )
            continue

        if column in replaced:
            recipe = replaced[column]
            dropped.append(
                {
                    "column": column,
                    "reason": f"Used through its derived inputs instead "
                    f"({', '.join(recipe.outputs)}).",
                }
            )
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric.append(column)
            continue

        if column in frequent_set:
            frequent.append(column)
            continue

        if feature_recipes.looks_like_dates(series):
            dropped.append(
                {
                    "column": column,
                    "reason": "Dates. A model can't use a date as a category -- nearly every "
                    "row is a different day. Turned into parts (year, month, weekday, days "
                    "since) it becomes usable; that is offered under derived inputs.",
                }
            )
            continue

        if feature_recipes.looks_like_free_text(series):
            dropped.append(
                {
                    "column": column,
                    "reason": "Free text. A model can't read prose as a category -- nearly "
                    "every row is different -- but how much was written often carries "
                    "signal, and its length is offered under derived inputs.",
                }
            )
            continue

        distinct = int(series.nunique(dropna=True))
        if distinct > limit:
            keepable = feature_recipes.frequent_categories_recipe(column, series, limit)
            dropped.append(
                {
                    "column": column,
                    "reason": f"{distinct} different values -- too many for {len(df)} rows to "
                    f"turn into model inputs without the model simply memorizing rows. "
                    + (
                        f"Its {keepable.evidence['top_values']} most common values cover "
                        f"{keepable.evidence['coverage']:.0%} of rows, though; keeping just "
                        f"those is offered under derived inputs."
                        if keepable
                        else "Usually an ID or a name."
                    ),
                }
            )
            continue
        categorical.append(column)

    return {
        "numeric": numeric,
        "categorical": categorical,
        "frequent": frequent,
        "clipped": [c for c in dict.fromkeys(clip or []) if c in numeric],
        "dropped": dropped,
        "derived": [r.as_dict() for r in recipes],
    }


# --- splitting ----------------------------------------------------------------------------


def time_cut(times: pd.Series, test_size: float, column: str = "the date column") -> int:
    """Where to cut rows already sorted by time, so no timestamp is on both sides.

    A cut at a row count falls wherever the count lands. On daily data, or records loaded in
    batches, that is usually in the middle of a day: the model trains on part of a day and is
    tested on the rest of it, which is not predicting the future. So the cut moves to the
    nearest point where the timestamp changes -- the one that leaves the test fraction closest
    to the one asked for, the larger training side on a tie. Unreadable dates, which sort last,
    count as one timestamp.

    If a single timestamp holds more than half the rows there is no honest split by time
    ("everything happened on the same day"), and it is refused rather than approximated.

    `model_export._split_block` writes these same few lines into `train_model.py`; a test runs
    the script and compares the partitions.
    """
    rows = len(times)
    starts = [i for i, repeat in enumerate(times.duplicated().tolist()) if not repeat]
    sizes = [end - start for start, end in zip(starts, starts[1:] + [rows])]
    biggest = max(sizes)
    if biggest * 2 > rows:
        crowded = sizes.index(biggest)
        when = times.iloc[starts[crowded]]
        label = "an unreadable date" if pd.isna(when) else str(when)
        raise ModelingError(
            f"{biggest:,} of the {rows:,} rows share the timestamp {label} in '{column}', so "
            f"they cannot be kept together and still leave a before and an after. Use a "
            f"column with more distinct dates, or a random split."
        )
    boundaries = [start for start in starts if 0 < start < rows]
    target = int(rows * (1 - test_size))
    return min(boundaries, key=lambda boundary: (abs(boundary - target), -boundary))


def split_data(
    df: pd.DataFrame,
    target_column: str,
    task: str,
    *,
    strategy: str = "random",
    time_column: str | None = None,
    group_column: str | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
):
    """Holds out a test set. Returns (X_train, X_test, y_train, y_test).

    `strategy="time"` sorts by `time_column` and takes the most recent rows as the test set,
    cutting at a timestamp boundary so rows sharing a timestamp stay together (see `time_cut`).
    When the data has a time dimension a random split lets the model learn from the future to
    predict the past, which inflates the score and can't happen in production.

    `strategy="group"` keeps every row belonging to one entity on one side of the split. When
    the same customer, patient or machine appears in many rows, a random split puts some of
    their rows in training and some in test, and the model scores well by recognising the
    entity rather than learning the pattern. The app has detected this for a while and could
    only advise treating the scores as optimistic; this is the fix it was pointing at.
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    labelled = df[df[target_column].notna()]
    if len(labelled) < MIN_ROWS_TO_TRAIN:
        raise ModelingError(
            f"Only {len(labelled)} rows have a value for '{target_column}'. That is too few "
            f"to train a model and still hold back a test set worth trusting."
        )

    features = labelled.drop(columns=[target_column])
    labels = labelled[target_column]

    if strategy == "time":
        if not time_column or time_column not in labelled.columns:
            raise ModelingError("A time-based split needs a date column to order the rows by.")
        order = pd.to_datetime(labelled[time_column], errors="coerce", format="mixed")
        if order.notna().sum() < len(order) * 0.9:
            raise ModelingError(
                f"'{time_column}' could not be read as dates for enough rows to order by."
            )
        ranked = order.sort_values(kind="stable").index
        cut = time_cut(order.loc[ranked], test_size, time_column)
        train_idx, test_idx = ranked[:cut], ranked[cut:]
        return (
            features.loc[train_idx],
            features.loc[test_idx],
            labels.loc[train_idx],
            labels.loc[test_idx],
        )

    if strategy == "group":
        if not group_column or group_column not in labelled.columns:
            raise ModelingError(
                "A grouped split needs the column that says which rows belong to the same "
                "thing."
            )
        groups = labelled[group_column]
        if groups.isna().any():
            raise ModelingError(
                f"'{group_column}' is blank on some rows, so they cannot be assigned to a "
                f"group. Fill or drop those rows before splitting on it."
            )
        distinct = int(groups.nunique())
        # Two groups can only ever produce a 50/50 split, and one produces none at all. The
        # number here is low deliberately -- it is the point below which the split becomes
        # arithmetically impossible, not a judgement about how much data is enough.
        if distinct < 5:
            raise ModelingError(
                f"'{group_column}' has only {distinct} distinct "
                f"{'value' if distinct == 1 else 'values'}, which is too few to hold some "
                f"back as a test set. Use a random split, or group on something else."
            )
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
        train_pos, test_pos = next(splitter.split(features, labels, groups=groups))
        return (
            features.iloc[train_pos],
            features.iloc[test_pos],
            labels.iloc[train_pos],
            labels.iloc[test_pos],
        )

    # Stratify classification splits so a rare class isn't absent from one side entirely.
    stratify = None
    if task == "classification":
        counts = labels.value_counts()
        if counts.min() >= 2:
            stratify = labels

    return train_test_split(
        features, labels, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
    )


# --- pipeline -----------------------------------------------------------------------------


def build_pipeline(
    model_name: str,
    numeric: list[str],
    categorical: list[str],
    model_params: dict | None = None,
    frequent: list[str] | None = None,
    clip: list[str] | None = None,
):
    """Assembles preprocessing + model into one Pipeline.

    Everything that learns from data -- the imputer's medians, the scaler's means, the
    encoder's category list -- lives inside this object, so `fit(X_train)` never sees the
    test rows. That is the whole point; doing any of it to the dataframe beforehand would
    leak.

    `frequent` are high-cardinality columns kept under a `frequent_categories` recipe: the
    most common values get an input each and the rest are pooled. Which values are common is
    learned, so it is learned here, on the training fold.

    `clip` are numeric inputs whose extreme values are pulled back to limits learned from the
    training fold, before the imputer and scaler so the statistics they learn describe the
    clipped values. Cleaning offers the same fix computed over the whole file, which leaks.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    spec = MODELS.get(model_name)
    if spec is None:
        raise ModelingError(f"Unknown model: {model_name}")

    from app.services.clipping import IQRClipper

    clipped = [c for c in (clip or []) if c in set(numeric)]
    plain = [c for c in numeric if c not in set(clipped)]

    transformers = []
    if plain:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                plain,
            )
        )
    if clipped:
        transformers.append(
            (
                "numeric_clipped",
                Pipeline(
                    [
                        ("clip", IQRClipper()),
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                clipped,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        # Categories seen only in the test set must not blow up prediction.
                        # sparse_output=False because HistGradientBoosting refuses sparse
                        # input -- without it the strongest of the three models silently
                        # fails to train on any dataset that has a categorical column. The
                        # matrix is safe to keep dense: MAX_CATEGORIES_TO_ENCODE caps how
                        # wide the encoding can get.
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=0.01,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )

    if frequent:
        transformers.append(
            (
                "frequent",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                max_categories=feature_recipes.MAX_FREQUENT_CATEGORIES,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                frequent,
            )
        )

    if not transformers:
        raise ModelingError("There are no usable columns left to predict from.")

    return Pipeline(
        [("prepare", ColumnTransformer(transformers)), ("model", spec.build(model_params))]
    )


# --- evaluation ---------------------------------------------------------------------------


def evaluate(pipeline, X_test, y_test, task: str, positive_label: str | None = None) -> dict:
    """Metrics appropriate to the task. For classification the imbalance-aware ones are
    included alongside accuracy, because accuracy alone flatters a lopsided target.

    `positive_label` says which of two answers counts as "a case" for precision and recall.
    It defaults to the last label in sorted order -- right for yes/no and 0/1, wrong for
    churned/retained -- so the user can name it."""
    from sklearn import metrics as sk

    predictions = pipeline.predict(X_test)

    if task == "regression":
        return {
            "r2": round(float(sk.r2_score(y_test, predictions)), 4),
            "mae": round(float(sk.mean_absolute_error(y_test, predictions)), 4),
            "rmse": round(float(np.sqrt(sk.mean_squared_error(y_test, predictions))), 4),
        }

    labels = sorted(pd.Series(y_test).astype(str).unique())
    binary = len(labels) == 2
    average = "binary" if binary else "macro"
    pos_label = resolve_positive_label(labels, positive_label) if binary else None

    y_true = pd.Series(y_test).astype(str)
    y_pred = pd.Series(predictions).astype(str)

    scores = {
        "accuracy": round(float(sk.accuracy_score(y_true, y_pred)), 4),
        "precision": round(
            float(sk.precision_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)), 4
        ),
        "recall": round(
            float(sk.recall_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)), 4
        ),
        "f1": round(
            float(sk.f1_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)), 4
        ),
    }

    # ROC AUC needs probabilities and only makes sense for two classes.
    if binary and hasattr(pipeline, "predict_proba"):
        try:
            classes = [str(c) for c in pipeline.classes_]
            probabilities = pipeline.predict_proba(X_test)[:, classes.index(pos_label)]
            scores["roc_auc"] = round(float(sk.roc_auc_score(y_true == pos_label, probabilities)), 4)
        except Exception:  # noqa: BLE001 -- a missing AUC must not sink the whole evaluation
            logger.debug("ROC AUC could not be computed", exc_info=True)

    scores["confusion_matrix"] = sk.confusion_matrix(y_true, y_pred, labels=labels).tolist()
    scores["labels"] = labels
    scores["positive_label"] = pos_label
    return scores


def resolve_positive_label(labels: list[str], requested: str | None) -> str:
    """The requested label if it is one of the two, else the conventional last-in-sorted-order.
    A label that isn't in the data is ignored rather than raised: it usually means the target
    was recoded during cleaning after the plan was drawn up."""
    if requested is not None and str(requested) in labels:
        return str(requested)
    return labels[-1]


def baseline_score(y_train, y_test, task: str) -> dict:
    """What you get without a model at all -- always guess the most common class, or always
    guess the mean. A model that can't beat this is not doing anything, and that fact is
    easy to miss when accuracy alone looks high."""
    from sklearn import metrics as sk

    if task == "regression":
        guess = float(pd.Series(y_train).mean())
        predictions = np.full(len(y_test), guess)
        return {
            "strategy": f"always predict the average ({guess:,.2f})",
            "r2": round(float(sk.r2_score(y_test, predictions)), 4),
            "mae": round(float(sk.mean_absolute_error(y_test, predictions)), 4),
        }

    common = pd.Series(y_train).astype(str).value_counts().index[0]
    y_true = pd.Series(y_test).astype(str)
    return {
        "strategy": f"always answer {common!r}",
        "accuracy": round(float((y_true == common).mean()), 4),
    }


# One-tailed z for a 5% false-alarm rate spread across k candidate models (0.05 / k). We
# report the *best* of several models, and the maximum of several noisy estimates runs high,
# so the bar has to rise with the number of candidates or the winner is often just the
# luckiest. Tabulated rather than pulling in scipy for five numbers.
_Z_BY_CANDIDATE_COUNT = {1: 1.645, 2: 1.96, 3: 2.13, 4: 2.24, 5: 2.33}

# Below this there is no verdict to give. Not "the model is bad" -- *unjudgeable*, which is a
# different statement and the only honest one. Found live on the RSNA knee dataset: 4,407 rows
# of which 58 carried a label, so a 20% split left twelve test rows, and the app called AUC
# 0.89 on those twelve "useful". One row landing differently moves that number by 0.08. A
# verdict a single row can overturn is not a verdict.
MIN_TEST_ROWS_TO_JUDGE = 30
MIN_TEST_ROWS_PER_CLASS_TO_JUDGE = 10


def _auc_noise_threshold(class_counts: list[int] | None, model_count: int = 2) -> float:
    """The AUC a model with no skill can reach by luck alone on a test set this small.

    Under the null (no relationship at all) the standard error of AUC is
    sqrt((n1 + n2 + 1) / (12 * n1 * n2)). This matters more than it sounds: on 60 test rows,
    models trained on pure noise reached a median AUC of 0.57 and a maximum of 0.69 across 25
    random datasets, so a fixed 0.60 cut-off would have called most of that a real signal.

    Falls back to a plain 0.60 when the class counts aren't known.

    Deliberately uncapped. There used to be a `min(0.75, ...)` here, which sounds cautious and
    is the exact opposite: the formula raises the bar precisely when evidence is thinnest, so
    a ceiling only ever fires on the smallest test sets and *lowers* the bar there. On twelve
    test rows it pulled the threshold from 0.79 down to 0.75.
    """
    if not class_counts or len(class_counts) != 2 or min(class_counts) < 1:
        return 0.60
    n1, n2 = class_counts
    standard_error = float(np.sqrt((n1 + n2 + 1) / (12 * n1 * n2)))
    z = _Z_BY_CANDIDATE_COUNT.get(max(1, model_count), 2.33)
    return 0.5 + z * standard_error


def _too_small_to_judge(
    test_rows: int | None, class_counts: list[int] | None
) -> dict | None:
    """Is there enough held-out data to say anything at all?

    Separate from the noise floor on purpose. The noise floor asks "could luck produce this
    score?"; this asks the prior question, "is there enough evidence here for any answer to
    mean something?" A model can clear the noise floor on twelve rows and still be
    unjudgeable, because the interval around that estimate spans most of the usable range.
    """
    if not test_rows:
        return None

    rarest = min(class_counts) if class_counts and len(class_counts) == 2 else None
    if test_rows >= MIN_TEST_ROWS_TO_JUDGE and (
        rarest is None or rarest >= MIN_TEST_ROWS_PER_CLASS_TO_JUDGE
    ):
        return None

    if rarest is not None and rarest < MIN_TEST_ROWS_PER_CLASS_TO_JUDGE:
        detail = (
            f"only {rarest} of the {test_rows} test rows carry the rarer answer, and at least "
            f"{MIN_TEST_ROWS_PER_CLASS_TO_JUDGE} are needed before a score means anything"
        )
    else:
        detail = (
            f"{test_rows} test rows is below the {MIN_TEST_ROWS_TO_JUDGE} needed before a "
            f"score means anything"
        )

    return {
        "verdict": "not_enough_test_data",
        "headline": f"There isn't enough held-back data to judge this model -- {detail}. "
        f"Whatever it scored, a single row landing differently would move it. This is not a "
        f"statement that the model is bad; it is a statement that nothing here can tell you "
        f"either way. Get more labelled rows before trusting any number on this page.",
    }


def assess_result(
    metrics: dict,
    baseline: dict,
    task: str,
    *,
    test_rows: int | None = None,
    class_counts: list[int] | None = None,
    model_count: int = 2,
) -> dict:
    """Is this model actually worth anything?

    The same principle as the readiness verdict: a tool that always hands back a model is
    less useful than one willing to say the model learned nothing. It is genuinely easy to
    miss -- on a 70/30 target a model that has learned *nothing at all* still reports 70%
    accuracy, which reads like success. So the comparison against "just guess" is made here
    rather than left for the user to work out, and it is made against what luck alone could
    have produced on a test set of this size.
    """
    # Asked before anything else. Every verdict below is a claim about what the test set
    # shows, so if the test set is too small to show anything, none of them can be made --
    # including "no better than guessing", which is just as much a claim as "useful".
    unjudgeable = _too_small_to_judge(test_rows, class_counts)
    if unjudgeable is not None:
        return unjudgeable

    if task == "regression":
        r2 = float(metrics.get("r2", 0.0))
        # R² of 0 is exactly the performance of always predicting the mean; below 0 is worse.
        if r2 < 0.05:
            return {
                "verdict": "no_better_than_guessing",
                "headline": "This model is no better than always predicting the average. The "
                "columns you have don't explain the target -- more or different data is "
                "needed, not a different model.",
            }
        if r2 < 0.3:
            return {
                "verdict": "marginal",
                "headline": f"This model explains only about {r2:.0%} of the variation. It "
                f"picks up a real signal but a weak one; treat its predictions as a hint, "
                f"not an answer.",
            }
        return {
            "verdict": "useful",
            "headline": f"This model explains about {r2:.0%} of the variation in the target, "
            f"clearly better than guessing the average.",
        }

    accuracy = float(metrics.get("accuracy", 0.0))
    baseline_accuracy = float(baseline.get("accuracy", 0.0))
    auc = metrics.get("roc_auc")

    if auc is not None:
        # AUC is the honest measure on a lopsided target: 0.5 is a coin flip regardless of
        # how the classes are distributed, which accuracy cannot tell you.
        noise_floor = _auc_noise_threshold(class_counts, model_count)
        sample_note = (
            f" On only {test_rows} test rows, anything up to about {noise_floor:.2f} can "
            f"happen by luck, so this is not evidence of a real pattern."
            if test_rows
            else ""
        )
        if auc < noise_floor:
            return {
                "verdict": "no_better_than_guessing",
                "headline": f"This model is within the range of random guessing (AUC "
                f"{auc:.2f}, where 0.50 is a coin flip). Its {accuracy:.0%} accuracy comes "
                f"from the answers being lopsided, not from anything it learned.{sample_note}",
            }
        if auc < 0.70:
            return {
                "verdict": "marginal",
                "headline": f"This model finds a weak signal (AUC {auc:.2f}, above the "
                f"{noise_floor:.2f} that luck alone could produce here). Better than chance, "
                f"but not enough to rely on for individual decisions.",
            }
        return {
            "verdict": "useful",
            "headline": f"This model separates the two outcomes meaningfully (AUC {auc:.2f}) "
            f"and beats the {baseline_accuracy:.0%} you would get by always answering the "
            f"same thing.",
        }

    strategy = baseline.get("strategy", "always giving the same answer")
    # Same idea without an AUC to lean on: how far above the baseline could this land by luck
    # given the number of test rows?
    margin = 0.02
    if test_rows and test_rows > 0:
        p = min(max(baseline_accuracy, 0.01), 0.99)
        margin = max(margin, 1.96 * float(np.sqrt(p * (1 - p) / test_rows)))

    if accuracy <= baseline_accuracy + margin:
        return {
            "verdict": "no_better_than_guessing",
            "headline": f"At {accuracy:.0%} accuracy this model does no better than the "
            f"{baseline_accuracy:.0%} you would get by choosing to {strategy}.",
        }
    if accuracy < baseline_accuracy + max(0.10, margin * 2):
        return {
            "verdict": "marginal",
            "headline": f"At {accuracy:.0%} accuracy this model is only slightly ahead of the "
            f"{baseline_accuracy:.0%} baseline.",
        }
    return {
        "verdict": "useful",
        "headline": f"At {accuracy:.0%} accuracy this model is clearly ahead of the "
        f"{baseline_accuracy:.0%} baseline.",
    }


def feature_importance(pipeline, top_n: int = 15) -> list[dict]:
    """Per-feature contribution, normalized to sum to 1 so the numbers are comparable across
    model types. Returns [] for models that don't expose it."""
    try:
        prepare = pipeline.named_steps["prepare"]
        model = pipeline.named_steps["model"]
        names = [str(n) for n in prepare.get_feature_names_out()]

        if hasattr(model, "feature_importances_"):
            weights = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coefficients = np.asarray(model.coef_, dtype=float)
            # Multi-class coefficients are one row per class; magnitude across classes is
            # the closest honest summary of "how much does this feature matter".
            weights = np.abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
        else:
            return []

        if len(weights) != len(names):
            return []
        total = float(weights.sum())
        if total <= 0:
            return []

        ranked = sorted(
            ({"feature": n, "importance": round(float(w / total), 4)} for n, w in zip(names, weights)),
            key=lambda item: item["importance"],
            reverse=True,
        )
        return ranked[:top_n]
    except Exception:  # noqa: BLE001 -- importance is a nice-to-have, never worth a 500
        logger.warning("Feature importance could not be computed", exc_info=True)
        return []


def source_column_importance(importances: list[dict], columns: list[str]) -> list[dict]:
    """Rolls one-hot encoded features back up to the column they came from, so the user sees
    "Department: 0.22" rather than nine separate `Department_*` rows they can't act on."""
    totals: dict[str, float] = {column: 0.0 for column in columns}
    for item in importances:
        name = item["feature"].split("__", 1)[-1]
        match = max(
            (c for c in columns if name == c or name.startswith(f"{c}_")),
            key=len,
            default=None,
        )
        if match:
            totals[match] += item["importance"]

    return sorted(
        ({"column": c, "importance": round(v, 4)} for c, v in totals.items() if v > 0),
        key=lambda item: item["importance"],
        reverse=True,
    )


# Above this many training rows, selection uses three folds instead of five. It fits every
# candidate once per fold, and on a large file the extra folds cost more than they tell.
ROWS_FOR_FEWER_SELECTION_FOLDS = 20_000
SELECTION_FOLDS_LARGE = 3


def test_rows_seen_in_training(X_train, X_test, y_train, y_test) -> int:
    """How many test rows are exact copies -- inputs and answer -- of a training row.

    A copy of a training row in the test set is a question the model has already been shown
    the answer to, and every one of them flatters the score. The contradictory-labels check
    finds identical inputs with *different* answers; nothing looked for identical inputs with
    the same answer on both sides of the split. They can be genuine repeats, or duplicates
    the cleaning step wasn't asked to remove -- the app can't tell which, so it counts them
    and says so rather than guessing.
    """
    try:
        train = X_train.assign(__answer__=pd.Series(y_train).to_numpy())
        test = X_test.assign(__answer__=pd.Series(y_test).to_numpy())
        # Rows as tuples of strings, so NaN matches NaN and 1 matches 1.0 the way a reader
        # comparing the two files would expect.
        seen = set(train.astype(str).itertuples(index=False, name=None))
        return int(sum(row in seen for row in test.astype(str).itertuples(index=False, name=None)))
    except Exception:  # noqa: BLE001 -- a diagnostic, never a reason to fail a run
        logger.warning("Could not compare test rows with training rows", exc_info=True)
        return 0


def choose_on_training_folds(
    trained, build_for, X_train, y_train, task: str, *, groups=None, ordered: bool = False
) -> dict:
    """Which of the trained candidates to keep, decided without the test set.

    This used to be `max(trained, key=<test-set score>)`: fit every candidate, score each on
    the held-out rows, keep the highest. That makes the reported score the best of several
    draws on those particular rows -- the same selection-on-the-test-set the improvement loop
    was built to avoid, done once on every plain train. The verdict partly compensated by
    raising its bar with the number of candidates, but compensating for a biased choice is
    not the same as not making one.

    So each candidate is cross-validated on the training rows -- grouped folds for a grouped
    split, the folds the improvement loop uses -- and the best cross-validated score wins. The
    test set is then opened once, for the winner's reported number, which nothing was chosen
    against. Ties go to the simpler model, in registry order.

    When the training rows can't be folded at all (a rare class too small to spread across
    two folds), there is no honest basis for a choice. The simplest candidate is kept and the
    result says so, rather than quietly falling back to the test set.
    """
    from app.services import tuning

    order = {name: i for i, name in enumerate(MODELS)}
    ranked = sorted(trained, key=lambda r: order.get(r["model"], len(order)))
    if len(ranked) == 1:
        return {
            "chosen": ranked[0]["model"],
            "method": "only_candidate",
            "metric": None,
            "folds": 0,
            "scores": {},
            "chosen_on": "the only model trained",
        }

    folds = SELECTION_FOLDS_LARGE if len(X_train) > ROWS_FOR_FEWER_SELECTION_FOLDS else tuning.CV_FOLDS
    scores: dict[str, dict] = {}
    for entry in ranked:
        found = tuning.score_config(
            build_for(entry["model"]), X_train, y_train, task, folds=folds, groups=groups,
            ordered=ordered,
        )
        if found is not None:
            scores[entry["model"]] = {"score": found.score, "std": found.std, "folds": found.fits}

    if not scores:
        return {
            "chosen": ranked[0]["model"],
            "method": "default_order",
            "metric": None,
            "folds": 0,
            "scores": {},
            "chosen_on": (
                "no candidate could be cross-validated on the training rows, so the simplest "
                "one was kept -- choosing by test-set score instead would make the reported "
                "number the best of several draws on those rows"
            ),
        }

    chosen = max(scores, key=lambda name: (scores[name]["score"], -order.get(name, 0)))
    metric = tuning.scoring_for(task, y_train)
    return {
        "chosen": chosen,
        "method": "cross_validation",
        "metric": metric,
        "folds": scores[chosen]["folds"],
        "scores": scores,
        "chosen_on": (
            f"cross-validated {metric} over {scores[chosen]['folds']} folds of the training "
            f"rows; the test set was not consulted"
        ),
    }


def clip_limits(pipeline, clipped: list[str]) -> dict[str, dict]:
    """The limits the fitted clipping step learned, by column, for the card to quote.

    A limit that could not be formed (a column with no spread) is None: that column was passed
    through untouched, and the card says so rather than printing infinity.
    """
    try:
        step = (
            pipeline.named_steps["prepare"]
            .named_transformers_["numeric_clipped"]
            .named_steps["clip"]
        )
    except (KeyError, AttributeError):
        return {}

    def finite(value) -> float | None:
        return round(float(value), 6) if np.isfinite(value) else None

    return {
        column: {"lower": finite(lower), "upper": finite(upper)}
        for column, lower, upper in zip(clipped, step.lower_, step.upper_)
    }


def train_candidates(
    df: pd.DataFrame,
    *,
    target_column: str,
    task: str,
    model_names: list[str],
    excluded_columns: list[str] | None = None,
    split_strategy: str = "random",
    time_column: str | None = None,
    group_column: str | None = None,
    include_deep_diagnostics: bool = True,
    model_params: dict[str, dict] | None = None,
    point: operating_point.OperatingPoint | None = None,
    feature_recipes_wanted: list[str] | None = None,
    clip_outliers: list[str] | None = None,
) -> dict:
    """Trains each requested model on the same split and returns everything needed to compare
    them. A model that fails to fit is reported rather than aborting the others.

    `model_params` carries settings the improvement loop found, keyed by model name. Running
    the tuned model back through this function rather than reporting the loop's own numbers
    is deliberate: the result then has the same shape as any other, so the charts, the model
    card, the joblib and the exported script all work on it without knowing it was tuned.

    `point` is the trade-off the user stated for where the cut-off should sit. It is applied
    to the chosen model only, and chosen on the training folds -- see operating_point.py.

    `feature_recipes_wanted` are the ids of the derived inputs the user ticked. They are
    re-proposed from the data rather than trusted, and the stateless ones are added to the
    frame before the split -- a row's weekday does not depend on any other row, so nothing
    leaks. See features.py.

    `clip_outliers` are numeric inputs to clip to limits learned from the training fold, inside
    the pipeline. See clipping.py."""
    point = point or operating_point.OperatingPoint()
    if target_column not in df.columns:
        raise ModelingError(f"'{target_column}' is not in the data.")

    # The column being grouped on cannot also be an input. Every group in the test set is one
    # the model has never seen by construction, so as a feature it can only ever contribute
    # unseen categories -- and if it did carry signal, that signal would be the memorisation
    # the grouped split exists to prevent.
    withheld = list(excluded_columns or [])
    if split_strategy == "group" and group_column and group_column not in withheld:
        withheld.append(group_column)

    recipes = feature_recipes.select(df, target_column, withheld, feature_recipes_wanted)
    df = feature_recipes.apply(df, recipes)

    features = classify_features(df, target_column, withheld, recipes, clip_outliers)
    if not features["numeric"] and not features["categorical"] and not features["frequent"]:
        raise ModelingError(
            "There are no usable columns left to predict from. Every other column was "
            "excluded, empty, constant, or too unique to learn from."
        )

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column,
        task,
        strategy=split_strategy,
        time_column=time_column,
        group_column=group_column,
    )
    used = features["numeric"] + features["categorical"] + features["frequent"]
    X_train, X_test = X_train[used], X_test[used]

    results = []
    for name in model_names:
        spec = MODELS.get(name)
        if spec is None or spec.task != task:
            continue
        tuned = (model_params or {}).get(name)
        try:
            pipeline = build_pipeline(
                name,
                features["numeric"],
                features["categorical"],
                tuned,
                features["frequent"],
                clip=features.get("clipped"),
            )
            pipeline.fit(X_train, y_train)
            scores = evaluate(pipeline, X_test, y_test, task, point.positive_label)
            importances = feature_importance(pipeline)
            results.append(
                {
                    "model": name,
                    "label": spec.label,
                    "description": spec.description,
                    "metrics": scores,
                    "tuned_params": tuned or {},
                    "feature_importance": importances,
                    "column_importance": source_column_importance(importances, used),
                    # Curves, confusion matrix, residuals -- one predict() over the test set,
                    # so it's affordable for every candidate rather than the winner alone.
                    "diagnostics": diagnostics.compute_diagnostics(
                        pipeline, X_test, y_test, task, scores
                    ),
                    "pipeline": pipeline,  # stripped before the response is serialized
                    "failed": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 -- one model failing must not lose the rest
            logger.warning("Model %s failed to train: %s", name, exc, exc_info=True)
            results.append(
                {
                    "model": name,
                    "label": spec.label,
                    "description": spec.description,
                    "metrics": {},
                    "feature_importance": [],
                    "column_importance": [],
                    "diagnostics": {},
                    "pipeline": None,
                    "failed": str(exc),
                }
            )

    trained = [r for r in results if r["pipeline"] is not None]
    if not trained:
        raise ModelingError(
            "None of the models could be trained on this data. "
            + (results[0]["failed"] if results else "")
        )

    key = "r2" if task == "regression" else ("roc_auc" if "roc_auc" in trained[0]["metrics"] else "f1")
    selection_groups = (
        df.loc[list(X_train.index), group_column]
        if split_strategy == "group" and group_column
        else None
    )
    selection = choose_on_training_folds(
        trained,
        lambda name: (
            lambda: build_pipeline(
                name,
                features["numeric"],
                features["categorical"],
                (model_params or {}).get(name),
                features["frequent"],
                clip=features.get("clipped"),
            )
        ),
        X_train,
        y_train,
        task,
        groups=selection_groups,
        ordered=split_strategy == "time",
    )
    best = next(r for r in trained if r["model"] == selection["chosen"])
    if features.get("clipped"):
        features["clip_limits"] = clip_limits(best["pipeline"], features["clipped"])
    for entry in results:
        scored = selection["scores"].get(entry["model"])
        entry["cv_score"] = scored["score"] if scored else None
        entry["cv_std"] = scored["std"] if scored else None
    baseline = baseline_score(y_train, y_test, task)

    # The expensive tier, for the chosen model only. `build` hands back an *unfitted*
    # pipeline each time, because cross-validation and the learning curve must fit from
    # scratch on each fold -- reusing a fitted one would score it on rows it trained on.
    deep: dict = {}
    if include_deep_diagnostics:
        deep = diagnostics.compute_deep_diagnostics(
            lambda: build_pipeline(
                best["model"],
                features["numeric"],
                features["categorical"],
                (model_params or {}).get(best["model"]),
                features["frequent"],
                clip=features.get("clipped"),
            ),
            best["pipeline"],
            X_train,
            X_test,
            y_train,
            y_test,
            task,
            # The training rows' groups, row for row with what cross-validation is handed.
            groups=selection_groups,
            ordered=split_strategy == "time",
        )

    # Where the cut-off sits, for a two-class model. Chosen on the training folds against
    # what the user said matters, then reported against the test set. The default is still
    # recorded, so the card and the script can say plainly that nothing was chosen.
    chosen_point = None
    if task == "classification" and best["metrics"].get("positive_label") is not None:
        train_groups = (
            df.loc[list(X_train.index), group_column]
            if split_strategy == "group" and group_column
            else None
        )
        chosen_point = operating_point.apply(
            build=lambda: build_pipeline(
                best["model"],
                features["numeric"],
                features["categorical"],
                (model_params or {}).get(best["model"]),
                features["frequent"],
                clip=features.get("clipped"),
            ),
            pipeline=best["pipeline"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            point=point,
            positive_label=best["metrics"]["positive_label"],
            groups=train_groups,
            ordered=split_strategy == "time",
        )

    return {
        "task": task,
        "target_column": target_column,
        "features": features,
        "operating_point": chosen_point,
        "split": {
            "strategy": split_strategy,
            "time_column": time_column if split_strategy == "time" else None,
            "group_column": group_column if split_strategy == "group" else None,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            # The rule that held the rows out, as requested: what the test-set ledger keys a
            # look on. Not derived from the row counts, which move with the data.
            "requested_test_size": DEFAULT_TEST_SIZE,
            "seed": RANDOM_STATE,
            # The fraction actually held out. A time split moves its cut to a timestamp
            # boundary, so this can differ from the 20% asked for.
            "test_fraction": round(len(X_test) / (len(X_train) + len(X_test)), 4),
            "test_rows_seen_in_training": test_rows_seen_in_training(X_train, X_test, y_train, y_test),
            # How many entities ended up on each side. The row counts alone hide the thing
            # that matters about a grouped split: 800 training rows drawn from 4 customers is
            # not 800 independent observations.
            "train_groups": (
                int(df.loc[list(X_train.index), group_column].nunique())
                if split_strategy == "group" and group_column
                else None
            ),
            "test_groups": (
                int(df.loc[list(X_test.index), group_column].nunique())
                if split_strategy == "group" and group_column
                else None
            ),
        },
        "baseline": baseline,
        "primary_metric": key,
        "results": results,
        "best_model": best["model"],
        # How the winner was chosen. The test set is not part of it -- see
        # choose_on_training_folds.
        "selection": {k: v for k, v in selection.items() if k != "scores"},
        "deep_diagnostics": deep,
        # Correlated features split the credit between them in any importance ranking, so
        # both look unimportant and neither gets investigated. Worth showing next to it.
        "feature_correlations": diagnostics.feature_correlations(
            X_train, features["numeric"]
        ),
        # Class counts come from the *test* set: the verdict is about how much the held-out
        # evidence can actually support.
        "assessment": assess_result(
            best["metrics"],
            baseline,
            task,
            test_rows=int(len(y_test)),
            class_counts=(
                pd.Series(y_test).astype(str).value_counts().tolist()
                if task == "classification"
                else None
            ),
            # Still the number of candidates, although the winner is no longer chosen on the
            # test set -- which is what this correction for picking the best of several draws
            # was compensating for. Loosening a guard is its own decision, not a side effect
            # of fixing the thing it guarded against; until it is made, the bar stays strict.
            model_count=len(trained),
        ),
    }
