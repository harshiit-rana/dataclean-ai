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

from app.services import diagnostics

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

    def build(self) -> Any:
        import importlib

        cls = getattr(importlib.import_module(self.module), self.class_name)
        return cls(**self.params)

    @property
    def import_line(self) -> str:
        return f"from {self.module} import {self.class_name}"

    def constructor_source(self) -> str:
        arguments = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
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
    df: pd.DataFrame, target_column: str, excluded: list[str] | None = None
) -> dict:
    """Splits the columns into what the model can actually use.

    Returns numeric/categorical feature lists plus the columns being left out and why, so the
    UI can show its reasoning instead of silently dropping things.
    """
    excluded = set(excluded or [])
    numeric: list[str] = []
    categorical: list[str] = []
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
        if series.nunique(dropna=True) <= 1:
            dropped.append(
                {"column": column, "reason": "Every row has the same value, so it explains nothing."}
            )
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric.append(column)
            continue

        distinct = int(series.nunique(dropna=True))
        if distinct > limit:
            dropped.append(
                {
                    "column": column,
                    "reason": f"{distinct} different values -- too many for {len(df)} rows to "
                    f"turn into model inputs without the model simply memorizing rows. "
                    f"Usually an ID, a name, or free text.",
                }
            )
            continue
        categorical.append(column)

    return {"numeric": numeric, "categorical": categorical, "dropped": dropped}


# --- splitting ----------------------------------------------------------------------------


def split_data(
    df: pd.DataFrame,
    target_column: str,
    task: str,
    *,
    strategy: str = "random",
    time_column: str | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
):
    """Holds out a test set. Returns (X_train, X_test, y_train, y_test).

    `strategy="time"` sorts by `time_column` and takes the most recent rows as the test set.
    When the data has a time dimension a random split lets the model learn from the future to
    predict the past, which inflates the score and can't happen in production.
    """
    from sklearn.model_selection import train_test_split

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
        ranked = order.sort_values().index
        cut = int(len(ranked) * (1 - test_size))
        train_idx, test_idx = ranked[:cut], ranked[cut:]
        return (
            features.loc[train_idx],
            features.loc[test_idx],
            labels.loc[train_idx],
            labels.loc[test_idx],
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


def build_pipeline(model_name: str, numeric: list[str], categorical: list[str]):
    """Assembles preprocessing + model into one Pipeline.

    Everything that learns from data -- the imputer's medians, the scaler's means, the
    encoder's category list -- lives inside this object, so `fit(X_train)` never sees the
    test rows. That is the whole point; doing any of it to the dataframe beforehand would
    leak.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    spec = MODELS.get(model_name)
    if spec is None:
        raise ModelingError(f"Unknown model: {model_name}")

    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
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

    if not transformers:
        raise ModelingError("There are no usable columns left to predict from.")

    return Pipeline([("prepare", ColumnTransformer(transformers)), ("model", spec.build())])


# --- evaluation ---------------------------------------------------------------------------


def evaluate(pipeline, X_test, y_test, task: str) -> dict:
    """Metrics appropriate to the task. For classification the imbalance-aware ones are
    included alongside accuracy, because accuracy alone flatters a lopsided target."""
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
    pos_label = labels[-1] if binary else None

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
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            scores["roc_auc"] = round(float(sk.roc_auc_score(y_true == labels[-1], probabilities)), 4)
        except Exception:  # noqa: BLE001 -- a missing AUC must not sink the whole evaluation
            logger.debug("ROC AUC could not be computed", exc_info=True)

    scores["confusion_matrix"] = sk.confusion_matrix(y_true, y_pred, labels=labels).tolist()
    scores["labels"] = labels
    scores["positive_label"] = pos_label
    return scores


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


def _auc_noise_threshold(class_counts: list[int] | None, model_count: int = 2) -> float:
    """The AUC a model with no skill can reach by luck alone on a test set this small.

    Under the null (no relationship at all) the standard error of AUC is
    sqrt((n1 + n2 + 1) / (12 * n1 * n2)). This matters more than it sounds: on 60 test rows,
    models trained on pure noise reached a median AUC of 0.57 and a maximum of 0.69 across 25
    random datasets, so a fixed 0.60 cut-off would have called most of that a real signal.

    Falls back to a plain 0.60 when the class counts aren't known.
    """
    if not class_counts or len(class_counts) != 2 or min(class_counts) < 1:
        return 0.60
    n1, n2 = class_counts
    standard_error = float(np.sqrt((n1 + n2 + 1) / (12 * n1 * n2)))
    z = _Z_BY_CANDIDATE_COUNT.get(max(1, model_count), 2.33)
    return min(0.75, 0.5 + z * standard_error)


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


def train_candidates(
    df: pd.DataFrame,
    *,
    target_column: str,
    task: str,
    model_names: list[str],
    excluded_columns: list[str] | None = None,
    split_strategy: str = "random",
    time_column: str | None = None,
    include_deep_diagnostics: bool = True,
) -> dict:
    """Trains each requested model on the same split and returns everything needed to compare
    them. A model that fails to fit is reported rather than aborting the others."""
    if target_column not in df.columns:
        raise ModelingError(f"'{target_column}' is not in the data.")

    features = classify_features(df, target_column, excluded_columns)
    if not features["numeric"] and not features["categorical"]:
        raise ModelingError(
            "There are no usable columns left to predict from. Every other column was "
            "excluded, empty, constant, or too unique to learn from."
        )

    X_train, X_test, y_train, y_test = split_data(
        df, target_column, task, strategy=split_strategy, time_column=time_column
    )
    used = features["numeric"] + features["categorical"]
    X_train, X_test = X_train[used], X_test[used]

    results = []
    for name in model_names:
        spec = MODELS.get(name)
        if spec is None or spec.task != task:
            continue
        try:
            pipeline = build_pipeline(name, features["numeric"], features["categorical"])
            pipeline.fit(X_train, y_train)
            scores = evaluate(pipeline, X_test, y_test, task)
            importances = feature_importance(pipeline)
            results.append(
                {
                    "model": name,
                    "label": spec.label,
                    "description": spec.description,
                    "metrics": scores,
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
    best = max(trained, key=lambda r: r["metrics"].get(key, float("-inf")))
    baseline = baseline_score(y_train, y_test, task)

    # The expensive tier, for the chosen model only. `build` hands back an *unfitted*
    # pipeline each time, because cross-validation and the learning curve must fit from
    # scratch on each fold -- reusing a fitted one would score it on rows it trained on.
    deep: dict = {}
    if include_deep_diagnostics:
        deep = diagnostics.compute_deep_diagnostics(
            lambda: build_pipeline(best["model"], features["numeric"], features["categorical"]),
            best["pipeline"],
            X_train,
            X_test,
            y_train,
            y_test,
            task,
        )

    return {
        "task": task,
        "target_column": target_column,
        "features": features,
        "split": {
            "strategy": split_strategy,
            "time_column": time_column if split_strategy == "time" else None,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "baseline": baseline,
        "primary_metric": key,
        "results": results,
        "best_model": best["model"],
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
            model_count=len(trained),
        ),
    }
