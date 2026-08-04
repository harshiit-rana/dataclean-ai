"""Target-aware analysis: the checks that only become possible once we know which column
the user is ultimately trying to predict.

Everything in `profiling.py` looks at a column in isolation -- is it empty, is it the wrong
type, does it have extreme values. Those questions have the same answer whatever the data is
for. The moment the user names a **target column**, a second class of problem becomes
detectable, and it is the class that actually ruins models:

- The target itself must never be imputed. Filling a missing answer manufactures a training
  label out of nothing, and the model dutifully learns the guess.
- A target with one class, or a minority class of four rows, cannot support a model at all.
  It is better to say so than to hand back a confident, useless artifact.
- **Leakage**: a column that predicts the target almost perfectly is nearly always a column
  recorded *after* the outcome was known (`termination_date` when predicting attrition,
  `invoice_paid_at` when predicting default). It inflates every offline score and the model
  collapses in production. Missingness leaks the same way -- `termination_date` is empty for
  exactly the people who stayed -- so emptiness is tested as well as value.

Like `profiling.py`, this module is entirely deterministic pandas/numpy. No LLM is involved,
and every `description` is written for someone who has never heard the word "entropy".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.models.graph_state import ColumnIssue, GoalContext
from app.services.profiling import generate_issue_id

logger = logging.getLogger(__name__)

# A model needs enough rows to have anything to generalize from, and enough examples of the
# rarer answer to have seen it more than anecdotally.
MIN_ROWS_FOR_MODELLING = 50
MIN_ROWS_PER_CLASS = 10

SEVERE_IMBALANCE_SHARE = 0.05  # minority class below this share of rows
NOTABLE_IMBALANCE_SHARE = 0.20

# Agreement thresholds. Deliberately high: a warning that fires on ordinary strong
# correlation trains users to ignore it, and a leakage warning is only useful if it is
# believed the one time it matters.
LEAKAGE_AGREEMENT_HIGH = 0.95
LEAKAGE_AGREEMENT_MEDIUM = 0.80
LEAKAGE_CORRELATION_HIGH = 0.98
LEAKAGE_CORRELATION_MEDIUM = 0.90

# Above this many distinct values a column trivially "predicts" anything -- a row-unique ID
# maps one-to-one onto every target by construction. Those are already reported as
# high_cardinality; testing them here would produce nothing but false alarms.
MAX_CATEGORIES_FOR_LEAKAGE = 20
NUMERIC_BINS = 10

# Numeric with more distinct values than this reads as a quantity to predict rather than a
# set of labels to choose between.
MAX_CLASSES_FOR_NUMERIC_TARGET = 20

# Operations that write a value where one was missing. Applied to the target column they
# fabricate training labels, which is never recoverable downstream.
IMPUTING_OPERATIONS = frozenset(
    {"fill_median", "fill_mean", "fill_mode", "fill_constant"}
)

# Operations whose written value is derived from a statistic over the whole file (the mean of
# every row, the outlier bounds of every row). Baking those into the CSV before a train/test
# split means test-set information reaches the training rows, and every offline score comes
# out flattering. Not wrong -- but the user should know it before approving, and it belongs
# in the audit trail.
DATASET_STATISTIC_OPERATIONS = frozenset({"fill_median", "fill_mean", "fill_mode", "cap_outliers"})


def infer_task_type(series: pd.Series, declared: str = "auto") -> str | None:
    """Returns "classification" or "regression", or None if the column is unusable as a
    target. A user-declared type always wins -- a 1-5 star rating is legitimately either."""
    if declared in {"classification", "regression"}:
        return declared

    non_null = series.dropna()
    if non_null.empty:
        return None
    if pd.api.types.is_numeric_dtype(non_null) and non_null.nunique() > MAX_CLASSES_FOR_NUMERIC_TARGET:
        return "regression"
    return "classification"


def _discretize(series: pd.Series, max_categories: int | None = MAX_CATEGORIES_FOR_LEAKAGE) -> pd.Series | None:
    """Reduces a column to a small set of labels so two columns of any type can be compared.
    Numbers become deciles; text stays as-is. Returns None when the column has too many
    distinct values for the comparison to mean anything."""
    non_null = series.dropna()
    if non_null.empty:
        return None

    if pd.api.types.is_numeric_dtype(non_null):
        if non_null.nunique() <= NUMERIC_BINS:
            return non_null.astype(str)
        try:
            return pd.qcut(non_null, q=NUMERIC_BINS, duplicates="drop").astype(str)
        except (ValueError, IndexError, TypeError):
            return None

    as_text = non_null.astype(str)
    if max_categories is not None and as_text.nunique() > max_categories:
        return None
    return as_text


def _agreement(a: pd.Series, b: pd.Series) -> float:
    """How completely one column determines the other, on a 0-1 scale.

    This is normalized mutual information with the *smaller* of the two entropies as the
    denominator, which is the property we want: it reaches 1.0 when knowing `a` tells you `b`
    exactly, even when `a` has far more distinct values than `b`. Written out longhand rather
    than pulled from scikit-learn -- it's a dozen lines, and this project has no other reason
    to carry that dependency.
    """
    joint = pd.crosstab(a, b).to_numpy(dtype=float)
    total = joint.sum()
    if total <= 0:
        return 0.0

    pxy = joint / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    expected = px @ py

    observed = pxy > 0
    mutual_information = float((pxy[observed] * np.log(pxy[observed] / expected[observed])).sum())
    entropy_a = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    entropy_b = float(-(py[py > 0] * np.log(py[py > 0])).sum())

    smaller = min(entropy_a, entropy_b)
    if smaller <= 1e-12:  # one of the columns is constant -- nothing to be predicted
        return 0.0
    return max(0.0, min(1.0, mutual_information / smaller))


def _leakage_issue(
    column: str,
    target: str,
    score: float,
    *,
    high_cut: float,
    medium_cut: float,
    via_missingness: bool,
) -> ColumnIssue | None:
    if score < medium_cut:
        return None
    severity = "high" if score >= high_cut else "medium"
    # Leakage outranks ordinary cleaning: fixing formatting in a column that shouldn't be in
    # the model at all is wasted effort.
    priority = 0.97 if severity == "high" else 0.85

    if via_missingness:
        description = (
            f"Whether '{column}' is filled in or empty lines up almost exactly with "
            f"'{target}' ({score:.0%} agreement). Even if you never use the value itself, "
            f"the fact that a value exists gives away the answer -- which usually means "
            f"'{column}' is only recorded once '{target}' is already known."
        )
    else:
        description = (
            f"Column '{column}' predicts '{target}' almost perfectly ({score:.0%} agreement). "
            f"That is usually a sign it is recorded after the outcome is known, rather than "
            f"something you would have available at prediction time. Left in, it will make "
            f"your model look far more accurate than it will be in practice."
        )

    return ColumnIssue(
        issue_id=generate_issue_id(),
        column_name=column,
        issue_type="target_leakage",
        severity=severity,
        description=description,
        stats={
            "target_column": target,
            "agreement": round(float(score), 4),
            "via_missingness": via_missingness,
        },
        priority_score=priority,
    )


def detect_leakage(df: pd.DataFrame, target_column: str, task_type: str) -> list[ColumnIssue]:
    """Flags columns that predict the target suspiciously well, by value or by emptiness."""
    if target_column not in df.columns:
        return []

    labelled = df[df[target_column].notna()]
    if len(labelled) < 10:  # too few labelled rows for any agreement number to mean anything
        return []

    target_values = labelled[target_column]
    target_binned = _discretize(target_values, max_categories=None)

    issues: list[ColumnIssue] = []
    for column in df.columns:
        if column == target_column:
            continue
        series = labelled[column]

        # A numeric feature against a numeric target is better served by plain correlation:
        # it is stronger evidence than binned agreement and far easier to explain.
        if (
            task_type == "regression"
            and pd.api.types.is_numeric_dtype(series)
            and pd.api.types.is_numeric_dtype(target_values)
        ):
            pair = pd.DataFrame({"a": series, "b": target_values}).dropna()
            if len(pair) >= 10 and pair["a"].nunique() > 1 and pair["b"].nunique() > 1:
                correlation = abs(float(pair["a"].corr(pair["b"])))
                if not np.isnan(correlation):
                    issue = _leakage_issue(
                        column,
                        target_column,
                        correlation,
                        high_cut=LEAKAGE_CORRELATION_HIGH,
                        medium_cut=LEAKAGE_CORRELATION_MEDIUM,
                        via_missingness=False,
                    )
                    if issue:
                        issues.append(issue)
        elif target_binned is not None:
            binned = _discretize(series)
            if binned is not None:
                aligned = binned.index.intersection(target_binned.index)
                if len(aligned) >= 10:
                    score = _agreement(binned.loc[aligned], target_binned.loc[aligned])
                    issue = _leakage_issue(
                        column,
                        target_column,
                        score,
                        high_cut=LEAKAGE_AGREEMENT_HIGH,
                        medium_cut=LEAKAGE_AGREEMENT_MEDIUM,
                        via_missingness=False,
                    )
                    if issue:
                        issues.append(issue)

        # Missingness leakage, tested separately: this is how the classic offender shows up,
        # and a high-cardinality column skipped above still gets caught here because
        # "empty or not" is only ever two categories.
        if target_binned is None:
            continue
        empty = series.isna()
        if 0 < int(empty.sum()) < len(series):
            score = _agreement(empty.astype(str), target_binned)
            issue = _leakage_issue(
                column,
                target_column,
                score,
                high_cut=LEAKAGE_AGREEMENT_HIGH,
                medium_cut=LEAKAGE_AGREEMENT_MEDIUM,
                via_missingness=True,
            )
            if issue:
                issues.append(issue)

    return issues


def detect_target_issues(df: pd.DataFrame, target_column: str, task_type: str) -> list[ColumnIssue]:
    """Problems with the answer column itself: missing answers, and answers too lopsided
    (or too uniform) to learn from."""
    if target_column not in df.columns:
        return []

    issues: list[ColumnIssue] = []
    series = df[target_column]
    n = len(df)
    null_count = int(series.isna().sum())

    if null_count:
        issues.append(
            ColumnIssue(
                issue_id=generate_issue_id(),
                column_name=target_column,
                issue_type="target_missing",
                severity="high",
                description=(
                    f"'{target_column}' is the column you want to predict, and {null_count} "
                    f"of {n} rows have no value for it. Those rows can't teach the model "
                    f"anything, and the gap must never be filled in -- a guessed answer is "
                    f"indistinguishable from a real one once it's in the file, and the model "
                    f"learns the guess."
                ),
                stats={
                    "null_count": null_count,
                    "row_count": n,
                    "null_rate": round(null_count / n, 4) if n else 0.0,
                    "is_target": True,
                },
                priority_score=0.99,
            )
        )

    non_null = series.dropna()
    if non_null.empty:
        return issues

    if task_type == "regression":
        if pd.api.types.is_numeric_dtype(non_null) and float(non_null.std() or 0.0) == 0.0:
            issues.append(
                ColumnIssue(
                    issue_id=generate_issue_id(),
                    column_name=target_column,
                    issue_type="target_imbalance",
                    severity="high",
                    description=(
                        f"Every row has the same value for '{target_column}'. There is "
                        f"nothing varying for a model to explain."
                    ),
                    stats={"distinct_values": 1, "is_target": True},
                    priority_score=1.0,
                )
            )
        return issues

    counts = non_null.astype(str).value_counts()
    labelled = int(counts.sum())

    if len(counts) < 2:
        issues.append(
            ColumnIssue(
                issue_id=generate_issue_id(),
                column_name=target_column,
                issue_type="target_imbalance",
                severity="high",
                description=(
                    f"Every row has the same value for '{target_column}' "
                    f"({counts.index[0]!r}). A model can only learn to tell things apart if "
                    f"the data contains more than one answer."
                ),
                stats={"distinct_values": 1, "is_target": True},
                priority_score=1.0,
            )
        )
        return issues

    minority_label = str(counts.index[-1])
    minority_count = int(counts.iloc[-1])
    minority_share = minority_count / labelled if labelled else 0.0

    if minority_share < SEVERE_IMBALANCE_SHARE or minority_count < MIN_ROWS_PER_CLASS:
        severity, priority = "high", 0.93
        description = (
            f"'{target_column}' is very lopsided: only {minority_count} of {labelled} rows "
            f"are {minority_label!r} ({minority_share:.1%}). A model can score "
            f"{1 - minority_share:.0%} 'accurate' by always guessing the common answer and "
            f"never once getting the rare one right -- so accuracy is the wrong thing to "
            f"measure here."
        )
    elif minority_share < NOTABLE_IMBALANCE_SHARE:
        severity, priority = "medium", 0.6
        description = (
            f"'{target_column}' is unbalanced: {minority_count} of {labelled} rows are "
            f"{minority_label!r} ({minority_share:.1%}). Worth knowing before you pick how "
            f"to measure the model -- plain accuracy will flatter it."
        )
    else:
        return issues

    issues.append(
        ColumnIssue(
            issue_id=generate_issue_id(),
            column_name=target_column,
            issue_type="target_imbalance",
            severity=severity,
            description=description,
            stats={
                "class_counts": {str(k): int(v) for k, v in counts.items()},
                "minority_label": minority_label,
                "minority_count": minority_count,
                "minority_share": round(minority_share, 4),
                "is_target": True,
            },
            priority_score=priority,
        )
    )
    return issues


def augment_issues(
    df: pd.DataFrame, issues: list[ColumnIssue], goal_context: GoalContext | None
) -> list[ColumnIssue]:
    """Single entry point used by the profiling node: adds the target-aware findings to the
    generic ones, and removes the generic finding that the target-aware one supersedes.

    Returns the original list unchanged when no target column was given, so a user who is
    not building a model sees exactly the behaviour they saw before.
    """
    target = (goal_context.target_column or "").strip() if goal_context else ""
    if not target:
        return issues
    if target not in df.columns:
        logger.warning("Target column %r is not in the uploaded data; skipping target checks", target)
        return issues

    task_type = infer_task_type(df[target], goal_context.task_type if goal_context else "auto")
    if task_type is None:
        return issues

    # "missing_values on the target" is now reported as target_missing, which carries the
    # much more important instruction not to fill it. Two cards for one fact would invite
    # the user to approve the wrong one.
    kept = [
        issue
        for issue in issues
        if not (issue.column_name == target and issue.issue_type == "missing_values")
    ]

    kept += detect_target_issues(df, target, task_type)
    kept += detect_leakage(df, target, task_type)
    kept.sort(key=lambda i: i.priority_score, reverse=True)
    return kept


def enforce_target_safety(
    operation: str, params: dict, goal_context: GoalContext | None
) -> tuple[str | None, dict, str | None]:
    """Corrects a proposed fix that would damage the modelling target, and attaches a warning
    explaining why. Returns (operation, params, warning); a None operation means the fix has
    been blocked and the card is informational only.

    This runs when the suggestion card is *built*, not when it executes, so what the user
    approves is what actually runs -- silently substituting an operation at execution time
    would make the approval meaningless.
    """
    target = (goal_context.target_column or "").strip() if goal_context else ""
    column = str(params.get("column") or "")

    if not target or column != target:
        if target and operation in DATASET_STATISTIC_OPERATIONS:
            return (
                operation,
                params,
                "This fills in values calculated from the whole file. If you split the data "
                "into training and test sets afterwards, the test rows will have helped "
                "produce the numbers written into the training rows, and your scores will "
                "come out better than they should. Safer to do this inside the model "
                "pipeline, where it can be recalculated on the training rows alone.",
            )
        return operation, params, None

    if operation in IMPUTING_OPERATIONS:
        return (
            "drop_rows_with_nulls",
            {**params, "column": target},
            f"'{target}' is the column you're trying to predict, so its missing values "
            f"can't be filled in -- that would invent the answers the model is meant to "
            f"learn. Changed to dropping those rows instead, which is the only safe option.",
        )

    if operation == "drop_column":
        return (
            None,
            params,
            f"'{target}' is the column you're trying to predict. Removing it would leave "
            f"nothing to build a model against, so this fix has been blocked. If you meant "
            f"to predict something else, change the target on the upload step.",
        )

    if operation in {"cap_outliers", "drop_outlier_rows"}:
        return (
            operation,
            params,
            f"This changes the answers themselves, not the inputs. Extreme values in "
            f"'{target}' are often the cases you most want the model to get right -- check "
            f"that these are genuine errors before approving.",
        )

    return operation, params, None


def assess_readiness(df: pd.DataFrame, goal_context: GoalContext | None) -> dict:
    """A verdict on whether this data can support the model the user asked for.

    Deliberately willing to answer "no". A tool that always produces a model is less useful
    than one that can say the data won't support it and explain what is missing.
    """
    target = (goal_context.target_column or "").strip() if goal_context else ""
    row_count, column_count = int(len(df)), int(df.shape[1])

    report: dict = {
        "target_column": target or None,
        "task_type": None,
        "row_count": row_count,
        "column_count": column_count,
        "verdict": "not_applicable",
        "blockers": [],
        "warnings": [],
        "notes": [],
        "target_summary": None,
        "remaining_issues": {"high": 0, "medium": 0, "low": 0, "total": 0},
    }

    if not target:
        report["notes"].append(
            "No prediction target was set for this session, so there is nothing to check "
            "model readiness against. You can still export the cleaned file."
        )
        return report

    if target not in df.columns:
        report["verdict"] = "blocked"
        report["blockers"].append(
            f"'{target}' is no longer in the data -- it was renamed or removed during "
            f"cleaning. Undo that change, or start again with the column you want to predict."
        )
        return report

    task_type = infer_task_type(df[target], goal_context.task_type if goal_context else "auto")
    report["task_type"] = task_type

    if row_count < MIN_ROWS_FOR_MODELLING:
        report["blockers"].append(
            f"Only {row_count} rows are left. That is too few to train a model that will "
            f"hold up on data it hasn't seen -- around {MIN_ROWS_FOR_MODELLING} is a bare "
            f"minimum, and more is much better."
        )
    if column_count < 2:
        report["blockers"].append(
            "There are no columns left to predict from -- only the target itself remains."
        )

    series = df[target]
    null_count = int(series.isna().sum())
    if null_count:
        report["blockers"].append(
            f"{null_count} rows still have no value for '{target}'. Drop those rows before "
            f"building a model -- they cannot be used, and they must not be filled in."
        )

    non_null = series.dropna()
    if task_type == "classification":
        counts = non_null.astype(str).value_counts()
        report["target_summary"] = {
            "class_counts": {str(k): int(v) for k, v in counts.head(20).items()},
            "distinct_values": int(counts.size),
            "null_count": null_count,
        }
        if counts.size < 2:
            report["blockers"].append(
                f"Every remaining row has the same value for '{target}'. There is nothing "
                f"for a model to tell apart."
            )
        elif int(counts.iloc[-1]) < MIN_ROWS_PER_CLASS:
            report["warnings"].append(
                f"The rarest answer, {str(counts.index[-1])!r}, appears only "
                f"{int(counts.iloc[-1])} times. A model will struggle to learn it, and any "
                f"score you get for it will be unreliable."
            )
        elif int(counts.iloc[-1]) / int(counts.sum()) < NOTABLE_IMBALANCE_SHARE:
            report["warnings"].append(
                f"The answers are unbalanced ({int(counts.iloc[-1])} of {int(counts.sum())} "
                f"are {str(counts.index[-1])!r}). Measure with something other than plain "
                f"accuracy."
            )
    elif task_type == "regression" and pd.api.types.is_numeric_dtype(non_null) and len(non_null):
        report["target_summary"] = {
            "min": float(non_null.min()),
            "max": float(non_null.max()),
            "mean": float(non_null.mean()),
            "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
            "null_count": null_count,
        }

    # Re-run the detectors against the data as it stands now, so the verdict reflects the
    # cleaning that has actually happened rather than what was found at the start.
    from app.services import profiling  # local import: profiling must not depend on this module

    remaining = augment_issues(df, profiling.detect_issues(df), goal_context)
    report["remaining_issues"] = {
        "high": sum(1 for i in remaining if i.severity == "high"),
        "medium": sum(1 for i in remaining if i.severity == "medium"),
        "low": sum(1 for i in remaining if i.severity == "low"),
        "total": len(remaining),
    }

    for issue in remaining:
        if issue.issue_type == "target_leakage" and issue.severity == "high":
            report["warnings"].append(
                f"'{issue.column_name}' still predicts '{target}' almost perfectly. If it "
                f"won't be available at the moment you need a prediction, drop it."
            )

    if report["blockers"]:
        report["verdict"] = "blocked"
    elif report["warnings"]:
        report["verdict"] = "warnings"
    else:
        report["verdict"] = "ready"
        report["notes"].append(
            f"{row_count} rows, {column_count - 1} columns to predict from, and no missing "
            f"values in '{target}'."
        )
    return report
