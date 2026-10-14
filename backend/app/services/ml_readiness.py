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
from app.services import readiness_checks
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
# Fixes that read a statistic from the whole file before any split. drop_outlier_rows was
# missing: it decides which rows to remove by limits computed over every row, test rows
# included, exactly as cap_outliers decides where to clip.
DATASET_STATISTIC_OPERATIONS = frozenset(
    {"fill_median", "fill_mean", "fill_mode", "cap_outliers", "drop_outlier_rows"}
)

# What the card says when one of those is proposed for a model input and not offered. Written
# in code, never by the model, and specific to the operation: the alternative differs.
LEFT_FOR_THE_MODEL_FILL = (
    "This is left for the model. It fills each column's gaps itself, using the training rows "
    "only, so nothing in the rows it is later tested on can shape the numbers it writes. "
    "Filling the gaps here would calculate the value from every row, test rows included, and "
    "your scores would come out better than they should. If you need a filled-in file for "
    "something other than the model, ask for it in the chat after cleaning."
)
LEFT_FOR_THE_MODEL_EXTREMES = (
    "This is left for the model. Limits for what counts as extreme would be calculated here "
    "from every row, test rows included, and your scores would come out better than they "
    "should. If extreme values are hurting the model, tick the option in the model plan that "
    "clips them using the training rows only. If you need the file itself changed, ask for it "
    "in the chat after cleaning."
)


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
    operation: str,
    params: dict,
    goal_context: GoalContext | None,
    *,
    source: str = "card",
) -> tuple[str | None, dict, str | None]:
    """Corrects a proposed fix that would damage the modelling target, and attaches a warning
    explaining why. Returns (operation, params, warning); a None operation means the fix has
    been blocked and the card is informational only.

    This runs when the suggestion card is *built*, not when it executes, so what the user
    approves is what actually runs -- silently substituting an operation at execution time
    would make the approval meaningless.

    `source` is "card" for a proposal the app is making and "chat" for something the user
    typed. With a target named, a card is not offered a fix that computes its value from the
    whole file (a fill, an outlier limit) on a model input: the model pipeline does the same
    job on the training rows alone, so offering the whole-file version asks the user to
    approve a leak the app can avoid. A typed instruction is the user's call, and is carried
    out with the warning.
    """
    target = (goal_context.target_column or "").strip() if goal_context else ""
    column = str(params.get("column") or "")

    if not target or column != target:
        if target and operation in DATASET_STATISTIC_OPERATIONS:
            if source == "card":
                return (
                    None,
                    params,
                    LEFT_FOR_THE_MODEL_FILL
                    if operation in {"fill_median", "fill_mean", "fill_mode"}
                    else LEFT_FOR_THE_MODEL_EXTREMES,
                )
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


def _add_rule_checks(log: readiness_checks.CheckLog, df: pd.DataFrame, rules) -> None:
    """Business rules the user has approved, as checks.

    Run before anything target-related and outside every early return, because a rule
    violation is a fact about the data rather than about the model: `age = -5` is wrong
    whether or not anyone is predicting anything, and a session with no target still wants to
    hear about it. This is also why they are not in `_TARGET_DEPENDENT_CHECKS`.
    """
    if not rules:
        return
    from app.services import business_rules  # local import: keeps the dependency one-way

    for check in business_rules.check_all(df, list(rules)):
        log.add(check)


def assess_readiness(
    df: pd.DataFrame, goal_context: GoalContext | None, rules=None
) -> dict:
    """A verdict on whether this data can support the model the user asked for, with the
    working shown.

    Deliberately willing to answer "no". A tool that always produces a model is less useful
    than one that can say the data won't support it and explain what is missing.

    Every check runs, or records why it could not. This used to return early -- no target
    meant six later checks silently never happened, and the result was indistinguishable from
    one where they had all passed. `blockers`, `warnings` and `notes` are now derived views
    over `checks` rather than the primary record, so the summary and the evidence cannot
    drift apart.
    """
    target = (goal_context.target_column or "").strip() if goal_context else ""
    row_count, column_count = int(len(df)), int(df.shape[1])
    log = readiness_checks.CheckLog()

    report: dict = {
        "target_column": target or None,
        "task_type": None,
        "row_count": row_count,
        # Rows carrying an answer. Equal to row_count until a target is named and some rows
        # turn out not to have one -- and every surface that describes what will be modelled
        # reads this rather than row_count, because three of them used to say "4,407 rows"
        # about a file that could train on 58.
        "trainable_row_count": row_count,
        "column_count": column_count,
        "verdict": "not_applicable",
        "blockers": [],
        "warnings": [],
        "notes": [],
        "target_summary": None,
        "remaining_issues": {"high": 0, "medium": 0, "low": 0, "total": 0},
        "checks": [],
        "check_counts": {},
    }

    _add_rule_checks(log, df, rules)

    # --- is there a question to answer at all? -------------------------------------------
    if not target:
        no_target = log.add(
            readiness_checks.Check(
                name="target_named",
                label="A column to predict",
                status="not_run",
                finding="No prediction target was set for this session, so there is nothing "
                "to check model readiness against. You can still export the cleaned file.",
            )
        )
        for name, label in _TARGET_DEPENDENT_CHECKS:
            log.skip(name, label, "target_named", "No target column was named.")
        report["checks"] = log.as_dicts()
        report["check_counts"] = log.counts
        # Rule violations still reach the summary here. The verdict stays not_applicable --
        # it is a statement about model readiness, and no model was asked for -- but a broken
        # rule is worth saying regardless of why someone uploaded the file.
        report["warnings"] = log.warnings
        report["blockers"] = log.blockers
        report["notes"].append(no_target.finding)
        return report

    log.add(
        readiness_checks.Check(
            name="target_named",
            label="A column to predict",
            status="passed",
            finding=f"Predicting '{target}'.",
        )
    )

    if target not in df.columns:
        log.add(
            readiness_checks.Check(
                name="target_present",
                label="The target is still in the data",
                status="failed",
                finding=f"'{target}' is no longer in the data -- it was renamed or removed "
                f"during cleaning.",
                evidence=f"The file now has {column_count} columns and '{target}' is not "
                f"among them.",
                remedy="Undo that change, or start again with the column you want to predict.",
                severity="blocker",
            )
        )
        for name, label in _TARGET_DEPENDENT_CHECKS:
            if name == "target_present":
                continue
            log.skip(name, label, "target_present", f"'{target}' is not in the data.")
        return _finish(report, log, row_count, column_count, target)

    log.add(
        readiness_checks.Check(
            name="target_present",
            label="The target is still in the data",
            status="passed",
            finding=f"'{target}' survived cleaning.",
        )
    )

    task_type = infer_task_type(df[target], goal_context.task_type if goal_context else "auto")
    report["task_type"] = task_type

    # --- enough to learn from ------------------------------------------------------------
    #
    # Counted on rows that carry an answer, not on rows in the file. A row with no target is
    # not training data -- it is dropped before fitting -- so measuring the file overstates
    # the evidence by exactly the number of rows that cannot be used. Found live on the RSNA
    # knee dataset, where this check reported "4,407 rows, comfortably above the 50 needed"
    # while the check two lines below it reported 4,349 rows with no answer. Both were reading
    # the same frame and only one of them was describing what would be trained on.
    trainable_rows = int(df[target].notna().sum())
    unlabelled = row_count - trainable_rows
    report["trainable_row_count"] = trainable_rows
    if trainable_rows < MIN_ROWS_FOR_MODELLING:
        shortfall = (
            f"Only {trainable_rows:,} of the {row_count:,} rows have a value for '{target}'."
            if unlabelled
            else f"Only {trainable_rows:,} rows are left."
        )
        log.add(
            readiness_checks.Check(
                name="enough_rows",
                label="Enough rows to learn from",
                status="failed",
                finding=shortfall,
                evidence=f"That is too few to train a model that will hold up on data it "
                f"hasn't seen -- around {MIN_ROWS_FOR_MODELLING} is a bare minimum, and more "
                f"is much better. Rows with no answer cannot be trained on.",
                remedy="Undo a fix that dropped rows, or start from a larger extract."
                if not unlabelled
                else f"Find the answers for the remaining rows. Filling '{target}' in with a "
                f"guess would not help -- it would train the model on invented answers.",
                severity="blocker",
            )
        )
    elif unlabelled:
        log.add(
            readiness_checks.Check(
                name="enough_rows",
                label="Enough rows to learn from",
                status="warning",
                finding=f"{trainable_rows:,} rows carry a value for '{target}' and can be "
                f"trained on, out of {row_count:,} in the file.",
                evidence=f"The other {unlabelled:,} rows are above the "
                f"{MIN_ROWS_FOR_MODELLING} minimum but will be dropped before fitting, so "
                f"every score comes from the smaller number.",
                remedy="Judge the results against the rows that were usable, not the file size.",
                severity="warning",
            )
        )
    else:
        log.add(
            readiness_checks.Check(
                name="enough_rows",
                label="Enough rows to learn from",
                status="passed",
                finding=f"{trainable_rows:,} rows, comfortably above the "
                f"{MIN_ROWS_FOR_MODELLING} needed to hold back a test set that means anything.",
            )
        )

    feature_columns = [c for c in df.columns if c != target]
    if column_count < 2:
        log.add(
            readiness_checks.Check(
                name="has_features",
                label="Something to predict from",
                status="failed",
                finding="There are no columns left to predict from -- only the target itself "
                "remains.",
                remedy="Undo the fixes that dropped columns.",
                severity="blocker",
            )
        )
    else:
        log.add(
            readiness_checks.Check(
                name="has_features",
                label="Something to predict from",
                status="passed",
                finding=f"{len(feature_columns)} columns available as model inputs.",
            )
        )

    # --- the target itself ---------------------------------------------------------------
    series = df[target]
    null_count = int(series.isna().sum())
    if null_count:
        log.add(
            readiness_checks.Check(
                name="target_complete",
                label="Every row has an answer",
                status="failed",
                finding=f"{null_count:,} rows still have no value for '{target}'.",
                evidence="They cannot be trained on, and the gap must never be filled in -- a "
                "guessed label is indistinguishable from a real one once it is in the file.",
                remedy=f"Drop the rows where '{target}' is empty.",
                severity="blocker",
            )
        )
    else:
        log.add(
            readiness_checks.Check(
                name="target_complete",
                label="Every row has an answer",
                status="passed",
                finding=f"No missing values in '{target}'.",
            )
        )

    non_null = series.dropna()
    _check_target_shape(log, report, task_type, target, non_null, null_count)

    # --- rows that undermine each other ---------------------------------------------------
    if log.passed("has_features"):
        log.add(readiness_checks.check_contradictory_labels(df, target, feature_columns))
        log.add(readiness_checks.check_repeated_entities(df, target))
    else:
        log.skip(
            "contradictory_labels",
            "Rows that contradict each other",
            "has_features",
            "There are no input columns to compare rows on.",
        )
        log.skip(
            "repeated_entities",
            "Rows that belong together",
            "has_features",
            "There are no columns left to look for repeated entities in.",
        )

    # --- everything the detectors still find ----------------------------------------------
    #
    # Re-run against the data as it stands now, so the verdict reflects the cleaning that has
    # actually happened rather than what was found at the start.
    from app.services import profiling  # local import: profiling must not depend on this module

    remaining = augment_issues(df, profiling.detect_issues(df), goal_context)
    report["remaining_issues"] = {
        "high": sum(1 for i in remaining if i.severity == "high"),
        "medium": sum(1 for i in remaining if i.severity == "medium"),
        "low": sum(1 for i in remaining if i.severity == "low"),
        "total": len(remaining),
    }

    leaks = [
        i for i in remaining if i.issue_type == "target_leakage" and i.severity == "high"
    ]
    if leaks:
        named = ", ".join(f"'{i.column_name}'" for i in leaks)
        log.add(
            readiness_checks.Check(
                name="leakage",
                label="Columns that give away the answer",
                status="failed",
                finding=f"{named} still predicts '{target}' almost perfectly.",
                evidence=leaks[0].description,
                remedy="If it won't be available at the moment you need a prediction, drop it.",
                severity="warning",
            )
        )
    else:
        log.add(
            readiness_checks.Check(
                name="leakage",
                label="Columns that give away the answer",
                status="passed",
                finding=f"No column predicts '{target}' closely enough to look like it was "
                f"recorded after the outcome was known.",
                evidence=f"Every remaining column was tested against '{target}' on both its "
                f"values and its missingness.",
            )
        )

    other = report["remaining_issues"]["total"]
    if other:
        log.add(
            readiness_checks.Check(
                name="remaining_issues",
                label="Data quality issues still present",
                status="warning" if report["remaining_issues"]["high"] else "passed",
                finding=f"{other} issue{'' if other == 1 else 's'} still detected in the data.",
                evidence=f"{report['remaining_issues']['high']} high, "
                f"{report['remaining_issues']['medium']} medium, "
                f"{report['remaining_issues']['low']} low severity.",
                remedy="None of them block a model, but the cleaning box can still address them."
                if report["remaining_issues"]["high"]
                else "",
                severity="warning" if report["remaining_issues"]["high"] else None,
            )
        )
    else:
        log.add(
            readiness_checks.Check(
                name="remaining_issues",
                label="Data quality issues still present",
                status="passed",
                finding="The profiler finds nothing else wrong with the data as it stands.",
            )
        )

    return _finish(report, log, row_count, column_count, target)


# Checks that cannot even be attempted without a usable target column. Listed here so a
# session with no target still produces a full transcript, saying of each one that it did not
# run and why, rather than an empty report that reads like a clean bill of health.
_TARGET_DEPENDENT_CHECKS: tuple[tuple[str, str], ...] = (
    ("target_present", "The target is still in the data"),
    ("enough_rows", "Enough rows to learn from"),
    ("has_features", "Something to predict from"),
    ("target_complete", "Every row has an answer"),
    ("target_shape", "The answers can be told apart"),
    ("contradictory_labels", "Rows that contradict each other"),
    ("repeated_entities", "Rows that belong together"),
    ("leakage", "Columns that give away the answer"),
    ("remaining_issues", "Data quality issues still present"),
)


def _check_target_shape(
    log: readiness_checks.CheckLog,
    report: dict,
    task_type: str | None,
    target: str,
    non_null: pd.Series,
    null_count: int,
) -> None:
    """Whether the answers themselves can support a model: enough classes, enough of each."""
    label = "The answers can be told apart"

    if task_type == "classification":
        counts = non_null.astype(str).value_counts()
        report["target_summary"] = {
            "class_counts": {str(k): int(v) for k, v in counts.head(20).items()},
            "distinct_values": int(counts.size),
            "null_count": null_count,
        }
        if counts.size < 2:
            log.add(
                readiness_checks.Check(
                    name="target_shape",
                    label=label,
                    status="failed",
                    finding=f"Every remaining row has the same value for '{target}'.",
                    evidence="There is nothing for a model to tell apart.",
                    remedy="Undo whichever fix removed the other answers.",
                    severity="blocker",
                )
            )
            return

        rarest_count = int(counts.iloc[-1])
        rarest_name = str(counts.index[-1])
        total = int(counts.sum())
        if rarest_count < MIN_ROWS_PER_CLASS:
            log.add(
                readiness_checks.Check(
                    name="target_shape",
                    label=label,
                    status="failed",
                    finding=f"The rarest answer, {rarest_name!r}, appears only "
                    f"{rarest_count} times.",
                    evidence=f"{counts.size} distinct answers across {total:,} rows. A model "
                    f"will struggle to learn a class this rare, and any score you get for it "
                    f"will be unreliable.",
                    remedy="Collect more examples of it, or merge it with a related answer.",
                    severity="warning",
                )
            )
            return
        if rarest_count / total < NOTABLE_IMBALANCE_SHARE:
            log.add(
                readiness_checks.Check(
                    name="target_shape",
                    label=label,
                    status="failed",
                    finding=f"The answers are unbalanced: {rarest_count:,} of {total:,} are "
                    f"{rarest_name!r}.",
                    evidence=f"A model that always answered with the most common class would "
                    f"score {1 - rarest_count / total:.0%} accuracy while learning nothing.",
                    remedy="Measure with something other than plain accuracy -- the model "
                    "page uses AUC for exactly this reason.",
                    severity="warning",
                )
            )
            return

        log.add(
            readiness_checks.Check(
                name="target_shape",
                label=label,
                status="passed",
                finding=f"{counts.size} distinct answers, the rarest appearing "
                f"{rarest_count:,} times ({rarest_count / total:.0%} of rows).",
            )
        )
        return

    if task_type == "regression":
        if pd.api.types.is_numeric_dtype(non_null) and len(non_null):
            report["target_summary"] = {
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "mean": float(non_null.mean()),
                "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                "null_count": null_count,
            }
            spread = float(non_null.std()) if len(non_null) > 1 else 0.0
            if spread == 0.0:
                log.add(
                    readiness_checks.Check(
                        name="target_shape",
                        label=label,
                        status="failed",
                        finding=f"Every row has the same value for '{target}'.",
                        evidence="There is no variation to explain.",
                        severity="blocker",
                    )
                )
                return
            log.add(
                readiness_checks.Check(
                    name="target_shape",
                    label=label,
                    status="passed",
                    finding=f"'{target}' ranges from {float(non_null.min()):,.3g} to "
                    f"{float(non_null.max()):,.3g}.",
                )
            )
            return

    log.skip(
        "target_shape",
        label,
        "target_present",
        f"'{target}' has no usable values, so its shape could not be assessed.",
    )


def _finish(
    report: dict, log: readiness_checks.CheckLog, rows: int, columns: int, target: str
) -> dict:
    """Derives the verdict and the flat summary lists from the checks that ran."""
    report["checks"] = log.as_dicts()
    report["check_counts"] = log.counts
    report["blockers"] = log.blockers
    report["warnings"] = log.warnings

    if report["blockers"]:
        report["verdict"] = "blocked"
    elif report["warnings"]:
        report["verdict"] = "warnings"
    else:
        report["verdict"] = "ready"
        report["notes"].append(
            f"{rows} rows, {columns - 1} columns to predict from, and no missing "
            f"values in '{target}'."
        )
    return report
