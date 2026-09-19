"""Generates a standalone Python script that rebuilds the trained model.

The point of the whole exercise. A score in a web page is something you have to trust; a
script you can read, run and change is something you own. It also makes the app's claims
checkable -- if the exported script doesn't reproduce the reported numbers, one of them is
wrong, and that should be discoverable.

Nothing here is written by the LLM. The script is templated from the same `ModelSpec` the
app fits with, so the constructor in the file is the constructor that ran. A hand-written
code template beside a hand-written pipeline drifts silently; this cannot.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.services.modeling import (
    DEFAULT_TEST_SIZE,
    MODELS,
    RANDOM_STATE,
    ModelSpec,
)


def _literal(value) -> str:
    return repr(value)


def _preprocessing_block(numeric: list[str], categorical: list[str]) -> tuple[list[str], list[str]]:
    """Returns (transformer entries, notes). Mirrors modeling.build_pipeline exactly."""
    entries: list[str] = []
    if numeric:
        entries.append(
            "        (\n"
            '            "numeric",\n'
            "            Pipeline([\n"
            '                ("impute", SimpleImputer(strategy="median")),\n'
            '                ("scale", StandardScaler()),\n'
            "            ]),\n"
            "            NUMERIC_FEATURES,\n"
            "        ),"
        )
    if categorical:
        entries.append(
            "        (\n"
            '            "categorical",\n'
            "            Pipeline([\n"
            '                ("impute", SimpleImputer(strategy="most_frequent")),\n'
            '                ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01,\n'
            "                                         sparse_output=False)),\n"
            "            ]),\n"
            "            CATEGORICAL_FEATURES,\n"
            "        ),"
        )
    return entries, []


def _split_block(split: dict, task: str) -> str:
    if split.get("strategy") == "time" and split.get("time_column"):
        return f'''# Time-based split: train on older rows, test on the most recent.
# A random split here would let the model learn from the future to predict the past,
# which inflates every score and cannot happen once it is deployed.
order = pd.to_datetime(labelled[{_literal(split["time_column"])}], errors="coerce", format="mixed")
ranked = order.sort_values().index
cut = int(len(ranked) * {1 - DEFAULT_TEST_SIZE})
X_train, X_test = features.loc[ranked[:cut]], features.loc[ranked[cut:]]
y_train, y_test = labels.loc[ranked[:cut]], labels.loc[ranked[cut:]]'''

    if split.get("strategy") == "group" and split.get("group_column"):
        column = split["group_column"]
        return f'''# Grouped split: every row belonging to one entity stays on one side.
# The same {column} appearing in both training and test lets the model score well by
# recognising the entity rather than learning the pattern. That inflates every number
# below and cannot help once it meets an entity it has never seen.
groups = labelled[{_literal(column)}]
splitter = GroupShuffleSplit(n_splits=1, test_size={DEFAULT_TEST_SIZE}, random_state={RANDOM_STATE})
train_pos, test_pos = next(splitter.split(features, labels, groups=groups))
X_train, X_test = features.iloc[train_pos], features.iloc[test_pos]
y_train, y_test = labels.iloc[train_pos], labels.iloc[test_pos]'''

    stratify = (
        "\n# Stratified so a rare class isn't absent from one side of the split entirely.\n"
        "stratify = labels if labels.value_counts().min() >= 2 else None"
        if task == "classification"
        else "\nstratify = None"
    )
    return f"""{stratify}
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size={DEFAULT_TEST_SIZE},
    random_state={RANDOM_STATE},
    stratify=stratify,
)"""


def _evaluation_block(task: str, point: dict | None = None) -> str:
    if task == "regression":
        return '''predictions = pipeline.predict(X_test)
print(f"R2:   {r2_score(y_test, predictions):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, predictions):.4f}")
print(f"RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.4f}")'''

    # classification_report rounds to two decimals, which is fine to read and useless for
    # checking the numbers against what the app reported. Print the exact value too.
    default = '''predictions = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print()
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, predictions))'''

    if not point or point.get("objective") in (None, "default"):
        return default

    # The cut-off is a declared constant, not re-derived here. The app chose it on
    # out-of-fold predictions over the training rows; this script's job is to reproduce the
    # model that was reported, and the number it was reported at is part of that.
    return default + '''

# --- At the chosen cut-off -------------------------------------------------------------
# The figures above use scikit-learn's default line of 0.50. The app chose THRESHOLD
# against the stated trade-off, on cross-validated predictions over the training rows only,
# so the numbers below are measured at a line the test set had no say in.
positive_index = [str(c) for c in pipeline.classes_].index(POSITIVE_LABEL)
scores = pipeline.predict_proba(X_test)[:, positive_index]
flagged = scores >= THRESHOLD
is_case = y_test.astype(str).to_numpy() == POSITIVE_LABEL
caught = int((flagged & is_case).sum())
print()
print(f"At cut-off {THRESHOLD} (flag as {POSITIVE_LABEL!r} when P >= THRESHOLD):")
print(f"  flagged {int(flagged.sum())} of {len(y_test)} rows")
print(f"  caught  {caught} of {int(is_case.sum())} real cases"
      f" (recall {caught / max(int(is_case.sum()), 1):.4f})")
print(f"  correct when flagged: {caught / max(int(flagged.sum()), 1):.4f} (precision)")'''


def _metric_imports(task: str) -> str:
    if task == "regression":
        return "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score"
    return (
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix"
    )


def build_training_script(result: dict, *, file_name: str = "cleaned.csv") -> str:
    """Renders a runnable script reproducing the chosen model from `result`."""
    model_name = result.get("best_model")
    spec: ModelSpec | None = MODELS.get(model_name)
    if spec is None:
        raise ValueError(f"No such model in the registry: {model_name!r}")

    features = result.get("features", {})
    numeric = features.get("numeric", [])
    categorical = features.get("categorical", [])
    excluded = features.get("excluded", [])
    split = result.get("split", {})
    task = result.get("task", "classification")
    target = result.get("target_column", "target")

    best = next(
        (r for r in result.get("results", []) if r.get("model") == model_name), {}
    )
    metrics = best.get("metrics", {})
    point = result.get("operating_point")
    has_point = bool(point and point.get("objective") not in (None, "default"))
    scored = ", ".join(
        f"{k} {v}" for k, v in metrics.items() if isinstance(v, (int, float))
    ) or "not recorded"

    entries, _notes = _preprocessing_block(numeric, categorical)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    dropped_note = "\n".join(
        f"#   - {d['column']}: {d['reason']}" for d in features.get("dropped", [])
    ) or "#   (none)"

    encoder_import = (
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler"
        if numeric and categorical
        else "from sklearn.preprocessing import OneHotEncoder"
        if categorical
        else "from sklearn.preprocessing import StandardScaler"
    )
    # Each strategy needs a different splitter, and only the one actually used is imported. A
    # time-based split orders the rows itself and imports nothing. Leaving an unused import in
    # a file people are meant to read and edit is a small lie about what the script does.
    strategy = split.get("strategy")
    if strategy == "time" and split.get("time_column"):
        split_import = ""
    elif strategy == "group" and split.get("group_column"):
        split_import = "\nfrom sklearn.model_selection import GroupShuffleSplit"
    else:
        split_import = "\nfrom sklearn.model_selection import train_test_split"

    return f'''"""Reproduces the model DataClean AI trained on {file_name}.

Generated {generated_at}. Everything below is exactly what the app ran -- the same
preprocessing, the same split, the same model settings -- so running this should give the
same numbers it reported:

    {scored}

Preprocessing lives INSIDE the pipeline on purpose. Imputing or scaling the dataframe before
splitting would let the test rows help choose the values written into the training rows, and
every score afterwards would be flattering. Inside the pipeline, fit() sees the training fold
only.

Run it with:
    pip install scikit-learn pandas joblib
    python train_model.py {file_name}

Columns left out of the model, and why:
{dropped_note}
"""

import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
{encoder_import}{split_import}
from sklearn.pipeline import Pipeline
{_metric_imports(task)}
{spec.import_line}

TARGET = {_literal(target)}
NUMERIC_FEATURES = {_literal(numeric)}
CATEGORICAL_FEATURES = {_literal(categorical)}
EXCLUDED_BY_CHOICE = {_literal(excluded)}
{_threshold_constants(point) if has_point else ""}
path = sys.argv[1] if len(sys.argv) > 1 else {_literal(file_name)}
df = pd.read_csv(path)

# Rows with no answer cannot be trained on, and the gap must never be filled in -- a guessed
# label is indistinguishable from a real one once it is in the file.
labelled = df[df[TARGET].notna()]
features = labelled[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
labels = labelled[TARGET]

{_split_block(split, task)}

pipeline = Pipeline([
    ("prepare", ColumnTransformer([
{chr(10).join(entries)}
    ])),
    ("model", {spec.constructor_source(best.get("tuned_params"))}),
])

pipeline.fit(X_train, y_train)

{_evaluation_block(task, point if has_point else None)}

joblib.dump(pipeline, "model.joblib")
print("\\nSaved the fitted pipeline to model.joblib")
print("Load it later with: pipeline = joblib.load('model.joblib')")
'''


def _threshold_constants(point: dict) -> str:
    return (
        f"\n# Where the cut-off sits. Chosen by the app so that {point.get('description')},\n"
        f"# on out-of-fold predictions over the training rows -- never on the test set.\n"
        f"THRESHOLD = {_literal(point.get('threshold'))}\n"
        f"POSITIVE_LABEL = {_literal(str(point.get('positive_label')))}\n"
    )


def build_prediction_snippet(result: dict) -> str:
    """A short example of using the saved model, for the model card. Loading a pickle and
    getting the column order wrong is the usual first stumble."""
    features = result.get("features", {})
    columns = features.get("numeric", []) + features.get("categorical", [])
    point = result.get("operating_point")
    has_point = bool(point and point.get("objective") not in (None, "default"))

    predict = f"print(pipeline.predict(new_rows[{columns!r}]))"
    if has_point:
        # The joblib file holds the model, not the line. predict() would answer at 0.50 and
        # quietly discard the trade-off the model was reported against.
        predict = (
            f"# The model was reported at a cut-off of {point['threshold']}, not the default 0.50.\n"
            f"# predict() uses 0.50, so threshold the probability yourself:\n"
            f"positive = [str(c) for c in pipeline.classes_].index({str(point['positive_label'])!r})\n"
            f"scores = pipeline.predict_proba(new_rows[{columns!r}])[:, positive]\n"
            f"print(scores >= {point['threshold']})"
        )
    return f'''import joblib
import pandas as pd

pipeline = joblib.load("model.joblib")

# Column names and order must match training. The pipeline handles the rest -- missing
# values, scaling and encoding are all inside it.
new_rows = pd.DataFrame([{{{", ".join(f"{c!r}: ..." for c in columns[:4])}}}])
{predict}'''
