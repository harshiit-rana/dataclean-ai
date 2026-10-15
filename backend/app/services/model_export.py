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

import inspect
import textwrap
from datetime import datetime, timezone

from app.services import clipping, features as feature_recipes, modeling, time_folds
from app.services.modeling import (
    DEFAULT_TEST_SIZE,
    MODELS,
    RANDOM_STATE,
    SEALED_DEV_SIZE,
    SEALED_FINAL_SIZE,
    ModelSpec,
)


def _literal(value) -> str:
    return repr(value)


def _preprocessing_block(
    numeric: list[str],
    categorical: list[str],
    frequent: list[str] | None = None,
    clipped: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Returns (transformer entries, notes). Mirrors modeling.build_pipeline exactly, in the
    same order, so the fitted model has its columns laid out the same way as the app's."""
    entries: list[str] = []
    clipped = [c for c in (clipped or []) if c in set(numeric)]
    plain = [c for c in numeric if c not in set(clipped)]
    if plain:
        entries.append(
            "        (\n"
            '            "numeric",\n'
            "            Pipeline([\n"
            '                ("impute", SimpleImputer(strategy="median")),\n'
            '                ("scale", StandardScaler()),\n'
            "            ]),\n"
            "            [c for c in NUMERIC_FEATURES if c not in CLIPPED_FEATURES],\n"
            "        ),"
        )
    if clipped:
        entries.append(
            "        (\n"
            '            "numeric_clipped",\n'
            "            Pipeline([\n"
            '                ("clip", IQRClipper()),\n'
            '                ("impute", SimpleImputer(strategy="median")),\n'
            '                ("scale", StandardScaler()),\n'
            "            ]),\n"
            "            CLIPPED_FEATURES,\n"
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
    if frequent:
        entries.append(
            "        (\n"
            '            "frequent",\n'
            "            Pipeline([\n"
            '                ("impute", SimpleImputer(strategy="most_frequent")),\n'
            '                ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",\n'
            f"                                         max_categories={feature_recipes.MAX_FREQUENT_CATEGORIES},\n"
            "                                         sparse_output=False)),\n"
            "            ]),\n"
            "            FREQUENT_FEATURES,\n"
            "        ),"
        )
    return entries, []


def _function_source(function) -> str:
    """A function's source, for copying into the script. See `_clipper_block` for why the app's
    own text is used rather than a hand-written twin."""
    return inspect.getsource(function).rstrip()


def _split_block(split: dict, task: str) -> str:
    """The code that divides the rows, for the script.

    A sealed final holdout is taken first, by the app's own function (copied from its source),
    and the ordinary split then divides what is left -- the same order the app uses, so the
    script reproduces all three partitions. The sealed rows are set aside and never touched
    here; opening them is an explicit act in the app.
    """
    if not split.get("sealed"):
        # The fraction the run asked for, so a file that held back 30% is cut the same way here.
        return _ordinary_split_block(split, task, split.get("requested_test_size") or DEFAULT_TEST_SIZE)

    copies = bool(split.get("seal_key_columns"))
    strategy = split.get("strategy") or "random"
    helper = (
        _function_source(modeling.split_keeping_identical_rows_together) + "\n\n" if copies else ""
    )
    seal = f"""{helper}{_function_source(modeling.sealed_holdout_rows)}

# A sealed final holdout, taken out before anything else happens to the rows. It depends only on
# the data, the seed and how the rows are divided -- not on which columns are used or which
# model is fitted -- so nothing done during development can move it. The app opens it only when
# you ask; this script never does.
SEAL_KEY_COLUMNS = {_literal(split.get("seal_key_columns"))}
final_index = sealed_holdout_rows(
    labelled, TARGET, {task == "classification"}, {_literal(strategy)},
    {_literal(split.get("time_column"))}, {_literal(split.get("group_column"))},
    {SEALED_FINAL_SIZE}, {RANDOM_STATE}, SEAL_KEY_COLUMNS,
)
labelled = labelled.drop(index=final_index)
features = features.drop(index=final_index)
labels = labels.drop(index=final_index)
print(f"Sealed {{len(final_index)}} rows as a final holdout; they are not used below.")
"""
    return seal + "\n" + _ordinary_split_block(
        split, task, SEALED_DEV_SIZE / (1 - SEALED_FINAL_SIZE), helper_defined=copies
    )


def _gap_lines(split: dict) -> str:
    """The embargo, when the run left one: training rows less than the gap before the first test row
    are dropped, as `modeling.split_data` does."""
    gap = int(split.get("gap_days") or 0)
    if not gap:
        return ""
    return f"""

# An embargo of {gap} days: a row just before the cut has an answer partly decided by what happens in
# the test period, so training stops at least that long before the first test row.
GAP = pd.Timedelta(days={gap})
before = (order.loc[ranked[:cut]] <= times.iloc[cut] - GAP).to_numpy()
X_train, y_train = X_train.loc[ranked[:cut][before]], y_train.loc[ranked[:cut][before]]"""


def _ordinary_split_block(
    split: dict, task: str, test_size: float, helper_defined: bool = False
) -> str:
    if split.get("strategy") == "time" and split.get("time_column"):
        return f'''# Time-based split: train on older rows, test on the most recent.
# A random split here would let the model learn from the future to predict the past,
# which inflates every score and cannot happen once it is deployed.
order = pd.to_datetime(labelled[{_literal(split["time_column"])}], errors="coerce", format="mixed")
ranked = order.sort_values(kind="stable").index
# Rows sharing a timestamp stay on one side: the cut moves to the nearest point where the
# timestamp changes, rather than falling in the middle of a day.
times = order.loc[ranked]
starts = [i for i, repeat in enumerate(times.duplicated().tolist()) if not repeat]
boundaries = [start for start in starts if 0 < start < len(ranked)]
target = int(len(ranked) * {1 - test_size})
cut = min(boundaries, key=lambda boundary: (abs(boundary - target), -boundary))
X_train, X_test = features.loc[ranked[:cut]], features.loc[ranked[cut:]]
y_train, y_test = labels.loc[ranked[:cut]], labels.loc[ranked[cut:]]''' + _gap_lines(split)

    if split.get("strategy") == "group" and split.get("group_column"):
        column = split["group_column"]
        return f'''# Grouped split: every row belonging to one entity stays on one side.
# The same {column} appearing in both training and test lets the model score well by
# recognising the entity rather than learning the pattern. That inflates every number
# below and cannot help once it meets an entity it has never seen.
groups = labelled[{_literal(column)}]
splitter = GroupShuffleSplit(n_splits=1, test_size={test_size}, random_state={RANDOM_STATE})
train_pos, test_pos = next(splitter.split(features, labels, groups=groups))
X_train, X_test = features.iloc[train_pos], features.iloc[test_pos]
y_train, y_test = labels.iloc[train_pos], labels.iloc[test_pos]'''

    stratify = (
        "\n# Stratified so a rare class isn't absent from one side of the split entirely.\n"
        "stratify = labels if labels.value_counts().min() >= 2 else None"
        if task == "classification"
        else "\nstratify = None"
    )
    standard = f"""{stratify}
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size={test_size},
    random_state={RANDOM_STATE},
    stratify=stratify,
)"""

    if not split.get("keep_duplicates_together"):
        return standard

    # Exact copies stay on one side. The function is the app's own, copied from its source
    # rather than rewritten, so this script cuts the same rows the app did; where there are no
    # copies the split is the ordinary one, as it is in the app.
    helper = "" if helper_defined else _function_source(modeling.split_keeping_identical_rows_together) + "\n\n"
    return f"""{helper}# Rows that are exact copies of each other -- the model's inputs and the answer -- stay on one
# side of the split. A copy in the test set is a question the model was already shown the
# answer to, and it would flatter every number below.
duplicate_key = pd.util.hash_pandas_object(
    labelled[NUMERIC_FEATURES + CATEGORICAL_FEATURES + FREQUENT_FEATURES + [TARGET]], index=False
)
if duplicate_key.nunique() < len(duplicate_key):
    train_pos, test_pos = split_keeping_identical_rows_together(
        labels, duplicate_key.to_numpy(), {task == "classification"}, {test_size}, {RANDOM_STATE}
    )
    X_train, X_test = features.iloc[train_pos], features.iloc[test_pos]
    y_train, y_test = labels.iloc[train_pos], labels.iloc[test_pos]
else:
{textwrap.indent(standard.strip(chr(10)), "    ")}"""


def _evaluation_block(task: str, point: dict | None = None, brier: bool = False) -> str:
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
    if brier:
        # How far the probabilities are from what happened, which accuracy cannot say. Printed
        # to four places so it can be checked against the number the app reported.
        default += '''

positive_index = [str(c) for c in pipeline.classes_].index(POSITIVE_LABEL)
probabilities = pipeline.predict_proba(X_test)[:, positive_index]
print()
print(f"Brier score: {brier_score_loss(y_test.astype(str) == POSITIVE_LABEL, probabilities):.4f}"
      "  (lower is better; 0 is perfect)")'''

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


def _metric_imports(task: str, brier: bool = False) -> str:
    if task == "regression":
        return "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score"
    names = "accuracy_score, " + ("brier_score_loss, " if brier else "") + "classification_report, confusion_matrix"
    return f"from sklearn.metrics import {names}"


def _calibration_applied(result: dict) -> dict | None:
    """The calibration report, when the model delivered is a calibrated one."""
    report = result.get("calibration") or {}
    return report if report.get("method") in ("sigmoid", "isotonic") else None


def _calibration_block(result: dict, report: dict) -> str:
    """Wraps the pipeline in the calibrator the app fitted, on the same folds.

    The folds are the ones every other step on the training rows uses (`tuning.cv_splitter`):
    stratified for a plain split, whole groups for a grouped split or copies kept together,
    forward in time for a date split. They are listed explicitly rather than handed over as a
    splitter, because scikit-learn gives a splitter's `split` no groups. A script that rebuilt
    them differently would calibrate on a different mapping and quietly report different
    numbers, which is the one thing this file exists not to do.
    """
    split = result.get("split", {})
    strategy = split.get("strategy")
    folds = int(report.get("folds") or 0)
    ordered = strategy == "time" and bool(split.get("time_column"))
    grouped = strategy == "group" and bool(split.get("group_column"))
    copies = bool(split.get("keep_duplicates_together")) and not ordered and not grouped

    if ordered:
        # The app's own splitter, copied by source (see `_time_folds_block`), given the parsed
        # dates the split block already made: every fold validates on later rows, no timestamp
        # is on both sides, and the gap the run asked for is left.
        splitter = (
            f"ForwardInTimeFolds(order.loc[X_train.index], n_splits={folds}, "
            f"gap_days={int(split.get('gap_days') or 0)})"
        )
        arguments = "X_train, fold_labels"
        why = (
            "Forward in time: each fold is calibrated on rows later than the ones it trained on,\n"
            "# and a timestamp is never split across the two."
        )
    elif grouped:
        splitter = f"StratifiedGroupKFold(n_splits={folds}, shuffle=True, random_state=42)"
        arguments = "X_train, fold_labels, groups.loc[X_train.index]"
        why = (
            "Whole groups: no entity is in both the part a fold trains on and the part it is calibrated on,\n"
            "# and within that the answers are balanced across folds as far as the grouping allows."
        )
    elif copies:
        splitter = f"StratifiedGroupKFold(n_splits={folds}, shuffle=True, random_state=42)"
        arguments = "X_train, fold_labels, duplicate_key.loc[X_train.index]"
        why = (
            "Copies together: a row and its exact copy are never on opposite sides of a fold,\n"
            "# and within that the answers are balanced across folds as far as the grouping allows."
        )
    else:
        splitter = f"StratifiedKFold(n_splits={folds}, shuffle=True, random_state=42)"
        arguments = "X_train, fold_labels"
        why = "Stratified, so each fold has both answers."

    return f'''
# --- Probability calibration ------------------------------------------------------------
# The app corrected this model's probabilities with a {report["method"]} calibration, so that "70%
# likely" happens about 70% of the time. It is fitted on cross-validation folds of the
# TRAINING rows only -- each row scored by a model that never saw it -- and the test rows take
# no part. {why}
CALIBRATION_METHOD = {_literal(report["method"])}
fold_labels = y_train.astype(str)
calibration_folds = list({splitter}.split({arguments}))
pipeline = CalibratedClassifierCV(
    base_pipeline, method=CALIBRATION_METHOD, cv=calibration_folds, ensemble={bool(report.get("ensemble"))},
)
'''


def _calibration_imports(result: dict, report: dict) -> str:
    split = result.get("split", {})
    strategy = split.get("strategy")
    ordered = strategy == "time" and bool(split.get("time_column"))
    grouped = strategy == "group" and bool(split.get("group_column"))
    copies = bool(split.get("keep_duplicates_together")) and not ordered and not grouped
    if ordered:
        return "from sklearn.calibration import CalibratedClassifierCV"
    splitter = "StratifiedGroupKFold" if (grouped or copies) else "StratifiedKFold"
    return f"from sklearn.calibration import CalibratedClassifierCV\nfrom sklearn.model_selection import {splitter}"


def _uses_time_folds(result: dict, calibrated: dict | None) -> bool:
    """Whether the script needs the forward-in-time splitter: a calibrated model on a date split."""
    split = result.get("split", {})
    return bool(calibrated) and split.get("strategy") == "time" and bool(split.get("time_column"))


def _time_folds_block() -> str:
    """The app's forward-in-time splitter, copied by source like the other pieces the script
    cannot import from the app."""
    return (
        "\n"
        + _function_source(time_folds.ForwardInTimeFolds)
        + "\n"
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
    frequent = features.get("frequent", [])
    excluded = features.get("excluded", [])
    recipes = [
        feature_recipes.Recipe(**r) for r in features.get("derived", []) if isinstance(r, dict)
    ]
    split = result.get("split", {})
    task = result.get("task", "classification")
    target = result.get("target_column", "target")

    best = next(
        (r for r in result.get("results", []) if r.get("model") == model_name), {}
    )
    metrics = best.get("metrics", {})
    point = result.get("operating_point")
    has_point = bool(point and point.get("objective") not in (None, "default"))
    calibrated = _calibration_applied(result) if task == "classification" else None
    positive_label = metrics.get("positive_label")
    # Printed whenever the app reported it, so the probabilities can be checked, not just the
    # answers. Needs the positive answer named, which only a two-class model has.
    brier = task == "classification" and "brier_score" in metrics and positive_label is not None
    scored = ", ".join(
        f"{k} {v}" for k, v in metrics.items() if isinstance(v, (int, float))
    ) or "not recorded"

    clipped = [c for c in features.get("clipped", []) if c in set(numeric)]
    entries, _notes = _preprocessing_block(numeric, categorical, frequent, clipped)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    dropped_note = "\n".join(
        f"#   - {d['column']}: {d['reason']}" for d in features.get("dropped", [])
    ) or "#   (none)"

    needs_time_folds = _uses_time_folds(result, calibrated)
    clipper_imports = "import numpy as np\n" if (clipped or needs_time_folds) else ""
    base_import = (
        "from sklearn.base import BaseEstimator, TransformerMixin\n" if clipped else ""
    )
    encodes = bool(categorical or frequent)
    encoder_import = (
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler"
        if numeric and encodes
        else "from sklearn.preprocessing import OneHotEncoder"
        if encodes
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
{clipper_imports}import pandas as pd
{base_import}from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
{encoder_import}{split_import}
from sklearn.pipeline import Pipeline
{_metric_imports(task, brier)}
{_calibration_imports(result, calibrated) + chr(10) if calibrated else ""}{spec.import_line}
{_clipper_block(clipped)}{_time_folds_block() if needs_time_folds else ""}
TARGET = {_literal(target)}
NUMERIC_FEATURES = {_literal(numeric)}
CATEGORICAL_FEATURES = {_literal(categorical)}
FREQUENT_FEATURES = {_literal(frequent)}
CLIPPED_FEATURES = {_literal(clipped)}
EXCLUDED_BY_CHOICE = {_literal(excluded)}
{_threshold_constants(point) if has_point else _positive_label_constant(positive_label) if brier else ""}
path = sys.argv[1] if len(sys.argv) > 1 else {_literal(file_name)}
df = pd.read_csv(path)
{_derived_block(recipes)}
# Rows with no answer cannot be trained on, and the gap must never be filled in -- a guessed
# label is indistinguishable from a real one once it is in the file.
labelled = df[df[TARGET].notna()]
features = labelled[NUMERIC_FEATURES + CATEGORICAL_FEATURES + FREQUENT_FEATURES]
labels = labelled[TARGET]

{_split_block(split, task)}

{"base_pipeline" if calibrated else "pipeline"} = Pipeline([
    ("prepare", ColumnTransformer([
{chr(10).join(entries)}
    ])),
    ("model", {spec.constructor_source(best.get("tuned_params"))}),
])
{_calibration_block(result, calibrated) if calibrated else ""}
pipeline.fit(X_train, y_train)

{_evaluation_block(task, point if has_point else None, brier)}

joblib.dump(pipeline, "model.joblib")
print("\\nSaved the fitted pipeline to model.joblib")
print("Load it later with: pipeline = joblib.load('model.joblib')"){_clipper_load_note(clipped)}
'''


def _clipper_block(clipped: list[str]) -> str:
    """The clipping step as source, when a run used it.

    Copied from `clipping.IQRClipper` itself rather than rewritten, so the script cannot drift
    from the app: a second copy by hand is how "reproduces the app's numbers" stops being true.
    It needs numpy and scikit-learn, both already required.
    """
    if not clipped:
        return ""
    return (
        "\n# --- Clipping extreme values, learned from the training rows ------------------------\n"
        "# Offered in the app for inputs with values far outside the rest. The limits are\n"
        "# learned in fit(), so inside the pipeline they come from the training fold only.\n"
        "# This is the same class the app used.\n\n"
        + clipping.source().rstrip()
        + "\n"
    )


def _clipper_load_note(clipped: list[str]) -> str:
    if not clipped:
        return ""
    return (
        "\nprint(\"This model contains the IQRClipper step defined at the top of this file; \"\n"
        "      \"define or import that class before calling joblib.load.\")"
    )


def _derived_block(recipes: list) -> str:
    """The derived inputs, as the pandas lines that produce them.

    Every one is row-wise -- a row's weekday and its text length depend on nothing but that
    row -- so running them before the split cannot leak. The column-level recipe
    (frequent_categories) is not here: it learns which values are common, so it lives inside
    the pipeline where it only ever sees the training fold.
    """
    if not recipes:
        return ""
    lines = feature_recipes.as_pandas(recipes)
    reasons = "\n".join(f"#   - {r.column}: {r.reason}" for r in recipes)
    return (
        "\n# --- Derived inputs -------------------------------------------------------------\n"
        "# Columns the raw file could not offer a model, turned into ones it can. Each is\n"
        "# row-wise, so deriving before the split leaks nothing.\n"
        f"{reasons}\n"
        f"{lines}\n"
    )


def _positive_label_constant(label) -> str:
    return (
        f"\n# Which answer counts as a case, for the Brier score below.\n"
        f"POSITIVE_LABEL = {_literal(str(label))}\n"
    )


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
    columns = (
        features.get("numeric", [])
        + features.get("categorical", [])
        + features.get("frequent", [])
    )
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
    calibration_note = (
        f"\n# The probabilities are calibrated ({_calibration_applied(result)['method']}), fitted on the training rows only,\n"
        "# so \"70% likely\" means about 70%. That also moves the default 0.50 line: on a rare\n"
        "# answer predict() flags fewer rows than an uncalibrated model would.\n"
        if _calibration_applied(result)
        else ""
    )
    derived_note = (
        "\n# Some inputs are derived (see train_model.py): "
        + ", ".join(sorted({o for r in features.get("derived", []) for o in r.get("outputs", [])}))
        + ".\n# Build them the same way on a new row before predicting.\n"
        if any(r.get("outputs") for r in features.get("derived", []))
        else ""
    )
    clip_note = (
        "\n# This model contains an IQRClipper step. Copy that class from train_model.py (it\n"
        "# needs numpy and scikit-learn only) into this file before loading, or joblib cannot\n"
        "# find it.\n"
        if features.get("clipped")
        else ""
    )
    return f'''import joblib
import pandas as pd
{clip_note}
pipeline = joblib.load("model.joblib")
{calibration_note}{derived_note}
# Column names and order must match training. The pipeline handles the rest -- missing
# values, scaling and encoding are all inside it.
new_rows = pd.DataFrame([{{{", ".join(f"{c!r}: ..." for c in columns[:4])}}}])
{predict}'''
